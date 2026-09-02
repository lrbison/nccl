#include <nccl.h>
#include "shim_core.h"
#include "kv_store_client.h"
#include <cstdio>
#include <vector>

using namespace nccl_checkpoint;

static int g_CommCheckpointCount = 0;

static int checkpointTimeoutSeconds() {
  const char* env = getenv("NCCL_CHECKPOINT_KVS_TIMEOUT");
  if (env == nullptr || env[0] == '\0') return 300;
  char* end = nullptr;
  long value = strtol(env, &end, 10);
  if (end == env || value <= 0) return 300;
  return (int)value;
}

static bool deadlineExpired(std::chrono::steady_clock::time_point deadline) {
  return std::chrono::steady_clock::now() >= deadline;
}

static ncclResult_t waitCommReady(ncclComm_t comm) {
  using get_async_t = ncclResult_t (*)(ncclComm_t, ncclResult_t*);
  static get_async_t real_ncclCommGetAsyncError = nullptr;
  NCCLCHECK(resolveRealFunction("ncclCommGetAsyncError", &real_ncclCommGetAsyncError));

  ncclResult_t ret = ncclSuccess;
  NCCLCHECK(real_ncclCommGetAsyncError(comm, &ret));
  while (ret == ncclInProgress) {
    usleep(10);
    NCCLCHECK(real_ncclCommGetAsyncError(comm, &ret));
  }
  return ret;
}

static ncclResult_t commGetRuntimeState(ncclComm_t comm, CommRecord* record) {
  using version_t = ncclResult_t (*)(int*);
  static version_t real_ncclGetVersion = nullptr;

  if (comm == nullptr || record == nullptr) return ncclInvalidArgument;
  if (comm->endMagic != NCCL_MAGIC) {
    int runtimeVersion = 0;
    if (resolveRealFunction("ncclGetVersion", &real_ncclGetVersion) == ncclSuccess) {
      (void)real_ncclGetVersion(&runtimeVersion);
    }
    WARN("Detected memory sentinel mismatch. Likely NCCL runtime version (%d) is not compatible with the "
         "libnccl-checkpoint-shim.so version (%d), or the communicator has experienced memory corruption.",
         runtimeVersion, NCCL_VERSION_CODE);
    return ncclInternalError;
  }

  record->comm = comm;
  record->rank = comm->rank;
  record->nranks = comm->nRanks;
  record->cudaDev = comm->cudaDev;
  record->commHash = comm->commHash;
  return ncclSuccess;
}

static ncclResult_t validateActiveCount(const CommRecord& record, int expectedCount, const char* phase) {
  using count_t = ncclResult_t (*)(ncclComm_t, int*);
  static count_t real_ncclCommMaskCountActive = nullptr;
  NCCLCHECK(resolveRealFunction("ncclCommMaskCountActive", &real_ncclCommMaskCountActive));

  int activeCount = 0;
  NCCLCHECK(real_ncclCommMaskCountActive(record.comm, &activeCount));
  if (activeCount != expectedCount) {
    WARN("ncclCheckpoint%s: comm %p rank %d/%d active count is %d, expected %d", phase, record.comm, record.rank,
         record.nranks, activeCount, expectedCount);
    return ncclInvalidUsage;
  }
  return ncclSuccess;
}

static ncclResult_t updateCommRuntimeState(CommRecord* record) {
  NCCLCHECK(waitCommReady(record->comm));
  NCCLCHECK(commGetRuntimeState(record->comm, record));
  if (record->rank < 0 || record->rank >= record->nranks) {
    WARN("ncclCheckpoint: comm %p has invalid local rank %d for nRanks %d", record->comm, record->rank,
         record->nranks);
    return ncclInternalError;
  }
  g_commRegistry.updateRuntime(*record);
  return ncclSuccess;
}

static ncclResult_t maskCommToSelf(CommRecord* record) {
  using reshape_t = ncclResult_t (*)(ncclComm_t, int, const int*, int, const int*, int, int);
  static reshape_t real_ncclCommReshape = nullptr;
  NCCLCHECK(resolveRealFunction("ncclCommReshape", &real_ncclCommReshape));

  NCCLCHECK(updateCommRuntimeState(record));
  NCCLCHECK(validateActiveCount(*record, record->nranks, "Prepare"));

  std::vector<int> leaveList;
  leaveList.reserve(record->nranks - 1);
  for (int rank = 0; rank < record->nranks; rank++) {
    if (rank != record->rank) leaveList.push_back(rank);
  }

  INFO(NCCL_CHECKPOINT, "prepare mask comm %p rank=%d/%d hash=0x%016llx", record->comm, record->rank,
       record->nranks, (unsigned long long)record->commHash);
  NCCLCHECK(real_ncclCommReshape(record->comm, -1, nullptr, 0, leaveList.data(), (int)leaveList.size(),
                                 NCCL_COMM_RESHAPE_LOCAL_ONLY));
  NCCLCHECK(validateActiveCount(*record, 1, "Prepare"));
  return ncclSuccess;
}

static ncclResult_t quiesceComm(const CommRecord& record) {
  using quiesce_t = ncclResult_t (*)(ncclComm_t);
  static quiesce_t real_ncclCommQuiesce = nullptr;
  NCCLCHECK(resolveRealFunction("ncclCommQuiesce", &real_ncclCommQuiesce));

  INFO(NCCL_CHECKPOINT, "prepare quiesce comm %p rank=%d/%d hash=0x%016llx", record.comm, record.rank,
       record.nranks, (unsigned long long)record.commHash);
  NCCLCHECK(real_ncclCommQuiesce(record.comm));
  return ncclSuccess;
}

static ncclResult_t quiesceNet(void) {
  using quiesce_t = ncclResult_t (*)();
  static quiesce_t real_ncclNetQuiesce = nullptr;
  NCCLCHECK(resolveRealFunction("ncclNetQuiesce", &real_ncclNetQuiesce));

  INFO(NCCL_CHECKPOINT, "prepare quiesce NET providers");
  NCCLCHECK(real_ncclNetQuiesce());
  return ncclSuccess;
}

static ncclResult_t rediscoverCommDevices(const CommRecord& record) {
  using rediscover_t = ncclResult_t (*)(ncclComm_t);
  static rediscover_t real_ncclCommRediscoverDevices = nullptr;
  NCCLCHECK(resolveRealFunction("ncclCommRediscoverDevices", &real_ncclCommRediscoverDevices));

  INFO(NCCL_CHECKPOINT, "restore rediscover devices comm %p rank=%d/%d hash=0x%016llx", record.comm, record.rank,
       record.nranks, (unsigned long long)record.commHash);
  NCCLCHECK(real_ncclCommRediscoverDevices(record.comm));
  return ncclSuccess;
}

static void uidKeyForComm(const CommRecord& record, char* key, size_t keySize) {
  snprintf(key, keySize, "restore/%016llx/join_uid", (unsigned long long)record.commHash);
}

struct RestoreOp {
  CommRecord record;
  bool needsJoin = false;
  bool survivor = false;
};

static ncclResult_t startSurvivorJoin(KVStoreClient& kv, RestoreOp* op) {
  using get_uid_t = ncclResult_t (*)(ncclComm_t, ncclUniqueId*, int);
  static get_uid_t real_ncclCommGetUniqueId_v2 = nullptr;
  NCCLCHECK(resolveRealFunction("ncclCommGetUniqueId_v2", &real_ncclCommGetUniqueId_v2));

  ncclUniqueId uid;
  NCCLCHECK(real_ncclCommGetUniqueId_v2(op->record.comm, &uid, NCCL_UNIQUE_ID_RESHAPE));

  char key[64];
  uidKeyForComm(op->record, key, sizeof(key));
  if (!kv.set(key, &uid, sizeof(uid))) {
    WARN("ncclCheckpointRestore: failed to publish join uid %s", key);
    return ncclInternalError;
  }
  TRACE(NCCL_CHECKPOINT, "published join uid for comm %p key %s", op->record.comm, key);
  return ncclSuccess;
}

static ncclResult_t startJoinerConnect(KVStoreClient& kv, RestoreOp* op) {
  using init_t = ncclResult_t (*)(ncclComm_t*, int, ncclUniqueId, int, ncclConfig_t*);
  static init_t real_ncclCommInitRankConfig = nullptr;
  NCCLCHECK(resolveRealFunction("ncclCommInitRankConfig", &real_ncclCommInitRankConfig));

  char key[64];
  uidKeyForComm(op->record, key, sizeof(key));

  ncclUniqueId uid;
  size_t outLen = 0;
  TRACE(NCCL_CHECKPOINT, "waiting for join uid for comm %p key %s", op->record.comm, key);
  if (!kv.get(key, &uid, sizeof(uid), &outLen) || outLen != sizeof(uid)) {
    WARN("ncclCheckpointRestore: failed to get join uid %s", key);
    return ncclInternalError;
  }

  ncclConfig_t config = NCCL_CONFIG_INITIALIZER;
  config.commInitFlags = NCCL_COMM_INIT_REUSE_EXISTING;

  ncclComm_t comm = op->record.comm;
  NCCLCHECK(real_ncclCommInitRankConfig(&comm, op->record.nranks, uid, op->record.rank, &config));
  if (comm != op->record.comm) {
    WARN("ncclCheckpointRestore: existing-comm join replaced comm %p with %p", op->record.comm, comm);
    return ncclInternalError;
  }
  return ncclSuccess;
}

static ncclResult_t completeReshape(RestoreOp* op) {
  using reshape_t = ncclResult_t (*)(ncclComm_t, int, const int*, int, const int*, int, int);
  static reshape_t real_ncclCommReshape = nullptr;
  NCCLCHECK(resolveRealFunction("ncclCommReshape", &real_ncclCommReshape));

  if (op->survivor) {
    std::vector<int> joiners;
    joiners.reserve(op->record.nranks - 1);
    for (int rank = 1; rank < op->record.nranks; rank++) joiners.push_back(rank);

    auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(checkpointTimeoutSeconds());
    while (true) {
      ncclResult_t ret = real_ncclCommReshape(op->record.comm, 0, joiners.data(), (int)joiners.size(), nullptr, 0,
                                              NCCL_COMM_RESHAPE_DEFAULT);
      if (ret != ncclResourceNotReady) return ret;
      if (deadlineExpired(deadline)) {
        WARN("ncclCheckpointRestore: timed out waiting for reshape joiners on comm %p", op->record.comm);
        return ncclInternalError;
      }
      usleep(1000);
    }
  }

  return real_ncclCommReshape(op->record.comm, 0, nullptr, 0, nullptr, 0, NCCL_COMM_RESHAPE_DEFAULT);
}

static ncclResult_t completeAllReshapes(std::vector<RestoreOp>& ops) {
  for (RestoreOp& op : ops) {
    if (op.needsJoin) NCCLCHECK(completeReshape(&op));
  }
  return ncclSuccess;
}

static ncclResult_t restoreAllComms(KVStoreClient& kv, std::vector<CommRecord>& comms) {
  std::vector<RestoreOp> ops;
  ops.reserve(comms.size());

  for (CommRecord& record : comms) {
    NCCLCHECK(updateCommRuntimeState(&record));
    RestoreOp op;
    op.record = record;
    op.needsJoin = record.nranks > 1;
    op.survivor = record.rank == 0;
    ops.push_back(op);
  }

  for (RestoreOp& op : ops) {
    NCCLCHECK(rediscoverCommDevices(op.record));
  }

  for (RestoreOp& op : ops) {
    if (!op.needsJoin) continue;
    if (op.survivor) {
      INFO(NCCL_CHECKPOINT, "restore accept comm %p rank=%d/%d hash=0x%016llx", op.record.comm, op.record.rank,
           op.record.nranks, (unsigned long long)op.record.commHash);
      NCCLCHECK(startSurvivorJoin(kv, &op));
    } else {
      INFO(NCCL_CHECKPOINT, "restore connect comm %p rank=%d/%d hash=0x%016llx", op.record.comm, op.record.rank,
           op.record.nranks, (unsigned long long)op.record.commHash);
      NCCLCHECK(startJoinerConnect(kv, &op));
    }
  }

  NCCLCHECK(completeAllReshapes(ops));

  for (CommRecord& record : comms) {
    NCCLCHECK(updateCommRuntimeState(&record));
    NCCLCHECK(validateActiveCount(record, record.nranks, "Restore"));
  }
  return ncclSuccess;
}

extern "C" ncclResult_t ncclCheckpointPrepare(void) {
  std::vector<CommRecord> comms = g_commRegistry.snapshotActive();
  if (comms.empty()) {
    g_CommCheckpointCount = 0;
    markCheckpointPrepared();
    return ncclSuccess;
  }

  int devSave = 0;
  CUDACHECK(cudaGetDevice(&devSave));

  ncclResult_t ret = ncclSuccess;
  for (CommRecord& record : comms) {
    if (record.cudaDev >= 0) {
      ret = cudaSetDevice(record.cudaDev) == cudaSuccess ? ncclSuccess : ncclUnhandledCudaError;
      if (ret != ncclSuccess) break;
    }
    ret = maskCommToSelf(&record);
    if (ret != ncclSuccess) break;
    ret = quiesceComm(record);
    if (ret != ncclSuccess) break;
  }

  if (ret == ncclSuccess) ret = quiesceNet();

  CUDACHECK(cudaSetDevice(devSave));
  NCCLCHECK(ret);
  g_CommCheckpointCount = (int)comms.size();
  markCheckpointPrepared();
  INFO(NCCL_CHECKPOINT, "prepare complete for %d communicator(s)", g_CommCheckpointCount);
  return ncclSuccess;
}

extern "C" ncclResult_t ncclCheckpointRestore(void) {
  if (g_CommCheckpointCount == 0) {
    clearCheckpointPrepared();
    return ncclSuccess;
  }

  INFO(NCCL_CHECKPOINT, "starting restore for %d communicator(s)", g_CommCheckpointCount);
  KVStoreClient kv;
  if (!kv.connect_from_env()) {
    WARN("ncclCheckpointRestore: failed to connect to KVS; set NCCL_CHECKPOINT_KVS_PATH");
    return ncclInvalidArgument;
  }

  std::vector<CommRecord> comms = g_commRegistry.snapshotActive();
  if (comms.empty()) {
    g_CommCheckpointCount = 0;
    clearCheckpointPrepared();
    return ncclSuccess;
  }

  int devSave = 0;
  CUDACHECK(cudaGetDevice(&devSave));
  ncclResult_t ret = restoreAllComms(kv, comms);
  CUDACHECK(cudaSetDevice(devSave));
  NCCLCHECK(ret);

  g_CommCheckpointCount = 0;
  clearCheckpointPrepared();
  INFO(NCCL_CHECKPOINT, "restore complete");
  return ncclSuccess;
}
