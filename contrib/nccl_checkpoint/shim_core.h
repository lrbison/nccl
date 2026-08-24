#pragma once
/*
 * Included by shim.cc and shim_checkpoint.cc.
 */

#include <atomic>
#include <chrono>
#include <cstdarg>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <vector>
#include <dlfcn.h>
#include <nccl.h>
#include <cuda_runtime.h>
#include <sys/syscall.h>
#include <unistd.h>

/* Internal NCCL headers used:
 *   checks.h - NCCLCHECK/CUDACHECK helpers
 *   comm.h   - ncclComm runtime fields
 */
#include "comm.h"
#undef WARN
#undef INFO
#undef TRACE

#define NCCL_CHECKPOINT_MAJOR 0
#define NCCL_CHECKPOINT_MINOR 1
#define NCCL_CHECKPOINT_PATCH 0
#define NCCL_CHECKPOINT_VERSION_CODE NCCL_VERSION(NCCL_CHECKPOINT_MAJOR, NCCL_CHECKPOINT_MINOR, NCCL_CHECKPOINT_PATCH)

namespace nccl_checkpoint {

enum NcclCheckpointLogLevel {
  ncclCheckpointLogInfo = 1,
  ncclCheckpointLogTrace = 2,
};

static inline bool ncclCheckpointShouldLog(NcclCheckpointLogLevel level) {
  static int enabled = -1;
  if (__builtin_expect(enabled != -1, 1)) return enabled >= static_cast<int>(level);

  enabled = 0;
  const char* debug = getenv("NCCL_DEBUG");
  if (debug != nullptr && strcmp(debug, "INFO") == 0) {
    enabled = ncclCheckpointLogInfo;
  } else if (debug != nullptr && strcmp(debug, "TRACE") == 0) {
    const char* subsys = getenv("NCCL_DEBUG_SUBSYS");
    if (subsys != nullptr && (strstr(subsys, "ALL") != nullptr || strstr(subsys, "CHECKPOINT") != nullptr)) {
      enabled = ncclCheckpointLogTrace;
    }
  }

  return enabled >= static_cast<int>(level);
}

static inline uint64_t ncclCheckpointGetTid() {
  return (uint64_t)syscall(SYS_gettid);
}

static inline const char* ncclCheckpointHostName() {
  static char hostname[1024] = "";
  if (hostname[0] == '\0') {
    if (gethostname(hostname, sizeof(hostname)) != 0) {
      strncpy(hostname, "unknown", sizeof(hostname));
    }
    hostname[sizeof(hostname) - 1] = '\0';
    for (int i = 0; hostname[i] != '\0'; i++) {
      if (hostname[i] == '.') {
        hostname[i] = '\0';
        break;
      }
    }
  }
  return hostname;
}

static inline double ncclCheckpointTimestampMs() {
  static auto epoch = std::chrono::steady_clock::now();
  auto delta = std::chrono::steady_clock::now() - epoch;
  return std::chrono::duration_cast<std::chrono::duration<double>>(delta).count() * 1000;
}

static inline void ncclCheckpointLog(NcclCheckpointLogLevel level, const char* func, int line, const char* fmt, ...) {
  if (!ncclCheckpointShouldLog(level)) return;

  int cudaDev = 0;
  (void)cudaGetDevice(&cudaDev);

  flockfile(stderr);
  if (level == ncclCheckpointLogInfo) {
    fprintf(stderr, "%s:%d:%d [%d] NCCL INFO NCCL Checkpoint: ", ncclCheckpointHostName(), getpid(),
            (int)ncclCheckpointGetTid(), cudaDev);
  } else {
    fprintf(stderr, "%s:%d:%d [%d] %f %s:%d NCCL TRACE NCCL Checkpoint: ", ncclCheckpointHostName(), getpid(),
            (int)ncclCheckpointGetTid(), cudaDev, ncclCheckpointTimestampMs(), func, line);
  }

  va_list args;
  va_start(args, fmt);
  vfprintf(stderr, fmt, args);
  va_end(args);
  fprintf(stderr, "\n");
  funlockfile(stderr);
}

static inline void ncclCheckpointWarn(const char* file, const char* func, int line, const char* fmt, ...) {
  int cudaDev = 0;
  (void)cudaGetDevice(&cudaDev);

  flockfile(stderr);
  fprintf(stderr, "%s:%d:%d [%d] %s:%d (%s) NCCL WARN NCCL Checkpoint: ", ncclCheckpointHostName(), getpid(),
          (int)ncclCheckpointGetTid(), cudaDev, file, line, func);

  va_list args;
  va_start(args, fmt);
  vfprintf(stderr, fmt, args);
  va_end(args);
  fprintf(stderr, "\n");
  funlockfile(stderr);
}

#define WARN(...) ::nccl_checkpoint::ncclCheckpointWarn(__FILE__, __func__, __LINE__, __VA_ARGS__)
#define INFO(_dummy, ...) \
  ::nccl_checkpoint::ncclCheckpointLog(::nccl_checkpoint::ncclCheckpointLogInfo, nullptr, 0, __VA_ARGS__)
#define TRACE(_dummy, ...) \
  ::nccl_checkpoint::ncclCheckpointLog(::nccl_checkpoint::ncclCheckpointLogTrace, __func__, __LINE__, __VA_ARGS__)

template <typename FnT>
static inline ncclResult_t resolveRealFunction(const char* name, FnT* fn) {
  if (*fn != nullptr) return ncclSuccess;

  dlerror();
  void* sym = dlsym(RTLD_NEXT, name);
  if (sym == nullptr) {
    const char* error = dlerror();
    WARN("failed to resolve %s with dlsym(RTLD_NEXT): %s", name, error != nullptr ? error : "unknown error");
    return ncclInternalError;
  }
  *fn = reinterpret_cast<FnT>(sym);
  return ncclSuccess;
}

enum CommUserState {
  comm_user_active = 1,
  comm_user_finalized = 2,
  comm_user_destroyed = 3,
};

struct CommRecord {
  ncclComm_t comm = nullptr;
  uint64_t sequence = 0;
  uint64_t commHash = 0;
  int nranks = -1;
  int rank = -1;
  int cudaDev = -1;
  CommUserState userState = comm_user_active;
};

class CommRegistry {
 public:
  ncclResult_t track(ncclComm_t comm, int nranks = -1, int rank = -1, int cudaDev = -1);
  void markState(ncclComm_t comm, CommUserState userState);
  void remove(ncclComm_t comm);
  void updateRuntime(const CommRecord& record);
  std::vector<CommRecord> snapshotActive() const;

 private:
  mutable std::mutex mtx_;
  uint64_t nextSequence_ = 1;
  std::vector<CommRecord> comms_;
};

extern CommRegistry g_commRegistry;

bool isCheckpointPrepared();
void markCheckpointPrepared();
void clearCheckpointPrepared();

}  // namespace nccl_checkpoint

extern "C" ncclResult_t ncclCheckpointPrepare(void);
extern "C" ncclResult_t ncclCheckpointRestore(void);
extern "C" ncclResult_t ncclCheckpointGetVersion(int* checkpointVersion, int* ncclVersion);
