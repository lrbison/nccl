#include <nccl.h>
#include "shim_core.h"
#include <dlfcn.h>
#include <cstdio>
#include <vector>

using namespace nccl_checkpoint;

extern "C" ncclResult_t ncclCheckpointGetVersion(int* checkpointVersion, int* ncclVersion) {
  if (checkpointVersion == nullptr || ncclVersion == nullptr) return ncclInvalidArgument;
  *checkpointVersion = NCCL_CHECKPOINT_VERSION_CODE;
  *ncclVersion = NCCL_VERSION_CODE;
  return ncclSuccess;
}

static ncclResult_t trackCreatedComm(ncclComm_t comm, int nranks = -1, int rank = -1) {
  if (comm == nullptr) return ncclSuccess;

  int cudaDev = -1;
  cudaError_t err = cudaGetDevice(&cudaDev);
  if (err != cudaSuccess) cudaDev = -1;
  return g_commRegistry.track(comm, nranks, rank, cudaDev);
}

extern "C" ncclResult_t ncclCommInitRank(ncclComm_t* comm, int nranks, ncclUniqueId commId, int rank) {
  using real_t = ncclResult_t (*)(ncclComm_t*, int, ncclUniqueId, int);
  static real_t real_ncclCommInitRank = nullptr;
  NCCLCHECK(resolveRealFunction("ncclCommInitRank", &real_ncclCommInitRank));

  ncclResult_t ret = real_ncclCommInitRank(comm, nranks, commId, rank);
  if ((ret == ncclSuccess || ret == ncclInProgress) && comm != nullptr) {
    NCCLCHECK(trackCreatedComm(*comm, nranks, rank));
  }
  return ret;
}

extern "C" ncclResult_t ncclCommInitRankConfig(ncclComm_t* comm, int nranks, ncclUniqueId commId, int rank,
                                               ncclConfig_t* config) {
  using real_t = ncclResult_t (*)(ncclComm_t*, int, ncclUniqueId, int, ncclConfig_t*);
  static real_t real_ncclCommInitRankConfig = nullptr;
  NCCLCHECK(resolveRealFunction("ncclCommInitRankConfig", &real_ncclCommInitRankConfig));

  ncclResult_t ret = real_ncclCommInitRankConfig(comm, nranks, commId, rank, config);
  if ((ret == ncclSuccess || ret == ncclInProgress) && comm != nullptr) {
    NCCLCHECK(trackCreatedComm(*comm, nranks, rank));
  }
  return ret;
}

extern "C" ncclResult_t ncclCommInitRankScalable(ncclComm_t* newcomm, int nranks, int myrank, int nId,
                                                 ncclUniqueId* commIds, ncclConfig_t* config) {
  using real_t = ncclResult_t (*)(ncclComm_t*, int, int, int, ncclUniqueId*, ncclConfig_t*);
  static real_t real_ncclCommInitRankScalable = nullptr;
  NCCLCHECK(resolveRealFunction("ncclCommInitRankScalable", &real_ncclCommInitRankScalable));

  ncclResult_t ret = real_ncclCommInitRankScalable(newcomm, nranks, myrank, nId, commIds, config);
  if ((ret == ncclSuccess || ret == ncclInProgress) && newcomm != nullptr) {
    NCCLCHECK(trackCreatedComm(*newcomm, nranks, myrank));
  }
  return ret;
}

extern "C" ncclResult_t ncclCommSplit(ncclComm_t comm, int color, int key, ncclComm_t* newcomm,
                                      ncclConfig_t* config) {
  using real_t = ncclResult_t (*)(ncclComm_t, int, int, ncclComm_t*, ncclConfig_t*);
  static real_t real_ncclCommSplit = nullptr;
  NCCLCHECK(resolveRealFunction("ncclCommSplit", &real_ncclCommSplit));

  ncclResult_t ret = real_ncclCommSplit(comm, color, key, newcomm, config);
  if ((ret == ncclSuccess || ret == ncclInProgress) && color != NCCL_SPLIT_NOCOLOR && newcomm != nullptr) {
    NCCLCHECK(trackCreatedComm(*newcomm));
  }
  return ret;
}

extern "C" ncclResult_t ncclCommAbort(ncclComm_t comm) {
  using real_t = ncclResult_t (*)(ncclComm_t);
  static real_t real_ncclCommAbort = nullptr;
  NCCLCHECK(resolveRealFunction("ncclCommAbort", &real_ncclCommAbort));

  ncclResult_t ret = real_ncclCommAbort(comm);
  if (ret == ncclSuccess || ret == ncclInProgress) {
    g_commRegistry.markState(comm, comm_user_destroyed);
    g_commRegistry.remove(comm);
  }
  return ret;
}

extern "C" ncclResult_t ncclCommFinalize(ncclComm_t comm) {
  using real_t = ncclResult_t (*)(ncclComm_t);
  static real_t real_ncclCommFinalize = nullptr;
  NCCLCHECK(resolveRealFunction("ncclCommFinalize", &real_ncclCommFinalize));

  ncclResult_t ret = real_ncclCommFinalize(comm);
  if (ret == ncclSuccess || ret == ncclInProgress) {
    g_commRegistry.markState(comm, comm_user_finalized);
  }
  return ret;
}

extern "C" ncclResult_t ncclCommDestroy(ncclComm_t comm) {
  using real_t = ncclResult_t (*)(ncclComm_t);
  static real_t real_ncclCommDestroy = nullptr;
  NCCLCHECK(resolveRealFunction("ncclCommDestroy", &real_ncclCommDestroy));

  ncclResult_t ret = real_ncclCommDestroy(comm);
  if (ret == ncclSuccess || ret == ncclInProgress) {
    g_commRegistry.markState(comm, comm_user_destroyed);
    g_commRegistry.remove(comm);
  }
  return ret;
}

extern "C" ncclResult_t ncclCommInitAll(ncclComm_t* comm, int ndev, const int* devlist) {
  using real_t = ncclResult_t (*)(ncclComm_t*, int, const int*);
  static real_t real_ncclCommInitAll = nullptr;
  NCCLCHECK(resolveRealFunction("ncclCommInitAll", &real_ncclCommInitAll));

  ncclResult_t ret = real_ncclCommInitAll(comm, ndev, devlist);
  if ((ret == ncclSuccess || ret == ncclInProgress) && comm != nullptr) {
    for (int i = 0; i < ndev; i++) {
      if (comm[i] == nullptr) continue;
      int cudaDev = devlist != nullptr ? devlist[i] : i;
      NCCLCHECK(g_commRegistry.track(comm[i], ndev, i, cudaDev));
    }
  }
  return ret;
}
