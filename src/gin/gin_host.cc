/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * See LICENSE.txt for more license information
 *************************************************************************/

#include "comm.h"
#include "gin.h"
#include "param.h"
#include "graph.h"
#include "transport.h"
#include "register_inline.h"
#include "alloc.h"
#include "gin/gin_host.h"
#include "gin/gin_host_proxy.h"
#include "compiler.h"
#include <cmath>

NCCL_PARAM(GinEnable, "GIN_ENABLE", 1);

// Backend version compatibility. Index: backend version. Value: min compatible NCCL version
const int proxyBackendMinVersions[] = {0, NCCL_VERSION(2, 30, 3), NCCL_VERSION(2, 30, 5)};
const int gdakiBackendMinVersions[] = {0, NCCL_VERSION(2, 30, 3), NCCL_VERSION(2, 30, 5)};
const int gpiBackendMinVersions[] = {0, NCCL_VERSION(2, 30, 5)};
const int efaGdaBackendMinVersions[] = {0, NCCL_VERSION(2, 31, 0)};

ncclResult_t ncclGetGinType(struct ncclComm* comm, ncclGinType_t* ginType) {
  if (comm == nullptr || ginType == nullptr) return ncclInternalError;

  *ginType = comm->globalGinSupport != NCCL_GIN_CONNECTION_FULL ? NCCL_GIN_TYPE_NONE :
                                                                  comm->sharedRes->ginState.backends[0].ginType;
  return ncclSuccess;
}

ncclResult_t ncclGetRailedGinType(struct ncclComm* comm, ncclGinType_t* ginType) {
  if (comm == nullptr || ginType == nullptr) return ncclInternalError;

  *ginType = comm->globalGinSupport == NCCL_GIN_CONNECTION_NONE ? NCCL_GIN_TYPE_NONE :
                                                                  comm->sharedRes->ginState.backends[0].ginType;
  return ncclSuccess;
}

static void ginProgressWriteLock(struct ncclGinState* ginState) {
  // This logic assumes just 1 writer. That's okay for this use case.
  ginState->writePending.store(true);
  ginState->devCommRwMutex.lock();
}

static void ginProgressWriteUnlock(struct ncclGinState* ginState) {
  ginState->devCommRwMutex.unlock();
  ginState->writePending.store(false);
}

// Per-thread progress worker. Thread t owns GIN connections t, t+proxyNthreads, t+2*proxyNthreads, ...
// across all devComms.
void* ncclGinProgress(struct ncclGinState* ginState, int threadIdx) {
  if (ncclOsCpuCount(ginState->cpuAffinity)) {
    ncclOsSetAffinity(ginState->cpuAffinity);
  }
  while (1) {
    if (ginState->proxyThreadStopSignal.load()) return NULL;
    // Back off while the main thread needs to modify the devComms list.
    if (ginState->writePending.load()) {
      std::this_thread::yield();
      continue;
    }
    {
      std::shared_lock<std::shared_timed_mutex> rlock(ginState->devCommRwMutex);
      struct ncclGinStateDevComm* dc = ginState->devComms;
      while (dc) {
        if (!dc->connected) {
          dc = dc->next;
          continue;
        }
        struct ncclGinBackendState* backend = &ginState->backends[dc->backendIndex];
        for (int commIdx = threadIdx; commIdx < dc->connectionCount; commIdx += ginState->proxyNthreads) {
          if (dc->devHandles[commIdx] != NULL && dc->devHandles[commIdx]->needsProxyProgress) {
            ncclResult_t ret = backend->ncclGin->ginProgress(dc->ginCtx[commIdx]);
            if (ret != ncclSuccess) {
              COMPILER_ATOMIC_STORE(&ginState->asyncResult, ret, std::memory_order_release);
              INFO_LOC(NCCL_ALL, "-> %d [GIN Progress Thread %d]", ret, threadIdx);
              return NULL;
            }
          }
        }
        dc = dc->next;
      }
    }
    std::this_thread::yield();
  }
}

NCCL_PARAM(GinNconnections, "GIN_NCONNECTIONS", -2);
NCCL_PARAM(GinProxyNthreads, "GIN_PROXY_NTHREADS", 1);
extern int64_t ncclParamDevApiJit();

ncclResult_t ncclGinConnectOnce(struct ncclComm* comm) {
  ncclTeam_t ginTeam;
  struct ncclGinState* ginState = &comm->sharedRes->ginState;

  if (ginState->connected) return ncclSuccess;

  ncclResult_t ret = ncclSuccess;
  if (ncclParamGinEnable() == 0) {
    WARN("GIN is disabled.");
    return ncclInternalError;
  }

  if (!ginState->supported) {
    WARN("GIN not supported.");
    return ncclInvalidUsage;
  }

  ginState->ginConnectionType = comm->globalGinSupport;

  if (!comm->symmetricSupport) {
    WARN("Communicator does not support symmetric memory!");
    return ncclInternalError;
  }

  int nLocalGinDevs;
  int localGinDevs[NCCL_TOPO_MAX_NODES];
  NCCLCHECK(ncclTopoGetLocalGinDevs(comm, localGinDevs, &nLocalGinDevs));
  if (nLocalGinDevs <= 0) {
    WARN("No local GIN-capable devices found.");
    return ncclInvalidUsage;
  }
  if (nLocalGinDevs > NCCL_GIN_MAX_CONNECTIONS) {
    INFO(NCCL_NET | NCCL_INIT | NCCL_GRAPH,
         "WARNING. Found %d local devices, but GIN supports at most %d connections. Using the first %d connections.",
         nLocalGinDevs, NCCL_GIN_MAX_CONNECTIONS, NCCL_GIN_MAX_CONNECTIONS);
  }

  void** handles = NULL;
  char* allHandles = NULL;
  void** listenComms = NULL;
  int ndev = 0;
  struct ncclGinBackendState* backend = NULL;
  int* ginCommCountHandles = NULL;

  NCCLCHECKGOTO(ncclCalloc(&listenComms, ginState->numActiveBackends), ret, fail);
  NCCLCHECKGOTO(ncclCalloc(&ginCommCountHandles, comm->nRanks), ret, fail);
  NCCLCHECKGOTO(ncclCalloc(&allHandles, (size_t)comm->nRanks * NCCL_NET_HANDLE_MAXSIZE), ret, fail);
  NCCLCHECKGOTO(ncclCalloc(&handles, comm->nRanks), ret, fail);

  // Connect the maximum supported connection type. Any future devComm may request
  // up to this connection type.
  ginTeam = ncclTeamWorld(comm);
  if (ginState->ginConnectionType != NCCL_GIN_CONNECTION_FULL) {
    ginTeam = {
      .nRanks = comm->nRanks / comm->contiguousRanksPerHost,
      .rank = comm->rank / comm->contiguousRanksPerHost,
      .stride = comm->contiguousRanksPerHost,
    };
  }
  for (int r = 0; r < ginTeam.nRanks; r++) {
    int worldRank = ncclTeamRankToWorld(comm, ginTeam, r);
    handles[r] = allHandles + worldRank * NCCL_NET_HANDLE_MAXSIZE;
  }

  for (int backendIdx = 0; backendIdx < ginState->numActiveBackends; backendIdx++) {
    backend = &ginState->backends[backendIdx];

    NCCLCHECKGOTO(backend->ncclGin->devices(&ndev), ret, fail);
    if (ndev <= 0) {
      WARN("No GIN-capable devices found.");
      ret = ncclInternalError;
      goto fail;
    }

    backend->ginCommCount = nLocalGinDevs;
    if (backend->ginVersion < 13) {
      // We only support one context per connection, so create as many connections as possible.
      backend->ginCommCount = NCCL_GIN_MAX_CONNECTIONS;
    }

    // Resolve the number of GIN progress threads. Default 1; Max NCCL_GIN_MAX_CONNECTIONS.
    ginState->proxyNthreads = 1;
    if (ncclParamGinProxyNthreads() > 1) ginState->proxyNthreads = ncclParamGinProxyNthreads();
    ginState->proxyNthreads = std::min<int>(NCCL_GIN_MAX_CONNECTIONS, ginState->proxyNthreads);

    if (ncclParamGinNconnections() != -2) backend->ginCommCount = ncclParamGinNconnections();
    backend->ginCommCount = std::min<int>(NCCL_GIN_MAX_CONNECTIONS, backend->ginCommCount);
    // Ensure ginCommCount >= proxyNthreads before AllGather.
    if (backend->ginCommCount < ginState->proxyNthreads) {
      backend->ginCommCount = ginState->proxyNthreads;
      INFO(NCCL_INIT, "GIN: increased ginCommCount to %d to match GIN_PROXY_NTHREADS", backend->ginCommCount);
    }

    ginCommCountHandles[comm->rank] = backend->ginCommCount;
    NCCLCHECKGOTO(bootstrapAllGather(comm->bootstrap, ginCommCountHandles, sizeof(int)), ret, fail);
    for (int r = 0; r < comm->nRanks; r++) {
      backend->ginCommCount = std::min(backend->ginCommCount, ginCommCountHandles[r]);
    }
    if (backend->ginCommCount <= 0) {
      WARN("No GIN connections available across communicator.");
      ret = ncclInvalidUsage;
      goto fail;
    }
    // After cross-rank min, proxyNthreads may exceed ginCommCount if ranks disagree
    // on NCCL_GIN_PROXY_NTHREADS (atypical — env vars are normally uniform across a job).
    // Extra threads simply idle in the stride loop; no correctness issue.

    for (int commIdx = 0; commIdx < backend->ginCommCount; commIdx++) {
      NCCLCHECKGOTO(backend->ncclGin->listen(backend->ginInstance, localGinDevs[commIdx % nLocalGinDevs],
                                             allHandles + NCCL_NET_HANDLE_MAXSIZE * comm->rank,
                                             &listenComms[backendIdx]),
                    ret, fail);

      NCCLCHECKGOTO(backend->ncclGin->getProperties(localGinDevs[commIdx % nLocalGinDevs], backend->ginProps + commIdx),
                    ret, fail);

      NCCLCHECKGOTO(bootstrapAllGather(comm->bootstrap, allHandles, NCCL_NET_HANDLE_MAXSIZE), ret, fail);

      NCCLCHECKGOTO(backend->ncclGin->connect(backend->ginInstance, handles, ginTeam.nRanks, ginTeam.rank,
                                              listenComms[backendIdx], backend->ginComms + commIdx),
                    ret, fail);

      NCCLCHECKGOTO(backend->ncclGin->closeListen(listenComms[backendIdx]), ret, fail);
      listenComms[backendIdx] = NULL;
    }
  }

exit:
  free(handles);
  free(allHandles);
  free(ginCommCountHandles);
  free(listenComms);
  if (ret == ncclSuccess) ginState->connected = true;
  return ret;
fail:
  for (int backendIdx = 0; backendIdx < ginState->numActiveBackends; backendIdx++) {
    backend = &ginState->backends[backendIdx];

    if (listenComms[backendIdx] != NULL) {
      NCCLCHECKIGNORE(backend->ncclGin->closeListen(listenComms[backendIdx]), ret);
    }

    for (int commIdx = 0; commIdx < backend->ginCommCount; commIdx++) {
      if (backend->ginComms[commIdx] != NULL) {
        NCCLCHECKIGNORE(backend->ncclGin->closeColl(backend->ginComms[commIdx]), ret);
        backend->ginComms[commIdx] = NULL;
      }
    }
  }
  goto exit;
}

ncclResult_t ncclGinValidateSignalRequest(struct ncclDevCommRequirements const* reqs,
                                          struct ncclGinBackendState* backend) {
  if (reqs->ginStrongSignalsRequired && !backend->supportsStrongSignals) {
    WARN("GIN strong signals are required, but the GIN plugin does not support them.");
    return ncclInvalidUsage;
  }

  if (reqs->ginVaSignalsRequired && !backend->supportsVASignals) {
    WARN("GIN VA signals are required, but the GIN plugin does not support them.");
    return ncclInvalidUsage;
  }

  return ncclSuccess;
}

static ncclResult_t ncclGinGetBackendVersion(uint32_t deviceCodeVersion, struct ncclGinBackendState* backend,
                                             int* backendVersion) {
  const int* backendVersionArray;
  int nVersions;

  switch (backend->ginType) {
  case NCCL_GIN_TYPE_PROXY:
    backendVersionArray = proxyBackendMinVersions;
    nVersions = sizeof(proxyBackendMinVersions) / sizeof(int);
    break;
  case NCCL_GIN_TYPE_GDAKI:
    backendVersionArray = gdakiBackendMinVersions;
    nVersions = sizeof(gdakiBackendMinVersions) / sizeof(int);
    break;
  case NCCL_GIN_TYPE_GPI:
    backendVersionArray = gpiBackendMinVersions;
    nVersions = sizeof(gpiBackendMinVersions) / sizeof(int);
    break;
  case NCCL_GIN_TYPE_EFA_GDA:
    backendVersionArray = efaGdaBackendMinVersions;
    nVersions = sizeof(efaGdaBackendMinVersions) / sizeof(int);
    break;
  default:
    WARN("Cannot get backend version for unsupported GIN type %d", backend->ginType);
    return ncclInternalError;
  }

  *backendVersion = 0;
  if (ncclParamDevApiJit() == 1) {
    // JIT: device code version is the latest version.
    *backendVersion = nVersions - 1;
  } else {
    // Non-JIT: device code version matches the version passed by the caller.
    for (int i = 0; i < nVersions; i++) {
      if (deviceCodeVersion >= (uint32_t)backendVersionArray[i]) *backendVersion = i;
      else break;
    }
  }
  return ncclSuccess;
}

static ncclResult_t ncclGinDevCommAppend(struct ncclGinState* ginState, struct ncclGinStateDevComm* ginStateDevComm) {
  bool locked = ginState->proxyThreadsCreated;
  if (locked) ginProgressWriteLock(ginState);
  struct ncclGinStateDevComm* last = ginState->devComms;
  if (last) {
    while (last->next) last = last->next;
    last->next = ginStateDevComm;
  } else {
    ginState->devComms = ginStateDevComm;
  }
  if (locked) ginProgressWriteUnlock(ginState);
  return ncclSuccess;
}

static ncclResult_t ncclGinDevCommStoreDynamic(struct ncclGinStateDevComm* ginStateDevComm,
                                               struct ncclDevComm* devComm) {
  ncclDevCommDynamic_t dynamicState = {};
  dynamicState.ginConnectionCount = ginStateDevComm->connectionCount;
  for (int n = 0; n < ginStateDevComm->connectionCount; n++) {
    dynamicState.ginNetDeviceTypes[n] = ginStateDevComm->devHandles[n]->netDeviceType;
    dynamicState.ginHandles[n] = ginStateDevComm->devHandles[n]->handle;
    if (devComm != NULL) {
      devComm->ginNetDeviceTypes[n] = dynamicState.ginNetDeviceTypes[n];
      devComm->ginHandles[n] = dynamicState.ginHandles[n];
    }
  }
  if (devComm != NULL) devComm->ginConnectionCount = ginStateDevComm->connectionCount;
  NCCLCHECK(ncclCudaMemcpy(ginStateDevComm->dynamicState, &dynamicState, 1));
  return ncclSuccess;
}

static ncclResult_t ncclGinStartProgressThreads(struct ncclComm* comm) {
  struct ncclGinState* ginState = &comm->sharedRes->ginState;
  if (ginState->proxyThreadsCreated) return ncclSuccess;

  ginState->cpuAffinity = comm->cpuAffinity;
  ginState->proxyThreadStopSignal.store(false);
  ginState->proxyThreadsCreated = true;
  for (int t = 0; t < ginState->proxyNthreads; t++) {
    ginState->thread[t] = std::thread([ginState, t] { ncclGinProgress(ginState, t); });
    ncclSetThreadName(ginState->thread[t], "NCCL GIN P%d-%d", comm->cudaDev, t);
  }
  return ncclSuccess;
}

static ncclResult_t ncclGinStopProgressThreads(struct ncclGinState* ginState) {
  if (!ginState->proxyThreadsCreated) return ncclSuccess;

  ginState->proxyThreadStopSignal.store(true);
  for (int t = 0; t < ginState->proxyNthreads; t++) {
    if (ginState->thread[t].joinable()) ginState->thread[t].join();
  }
  ginState->proxyThreadsCreated = false;
  ginState->proxyThreadStopSignal.store(false);
  return ncclSuccess;
}

static ncclResult_t ncclGinDevCommCreateContexts(struct ncclComm* comm, struct ncclGinStateDevComm* ginStateDevComm,
                                                 struct ncclDevComm* devComm) {
  struct ncclGinState* ginState = &comm->sharedRes->ginState;
  struct ncclGinBackendState* backend = &ginState->backends[ginStateDevComm->backendIndex];
  ncclResult_t ret = ncclSuccess;
  bool needsProxyProgress = false;
  int nContextsTotal = ginStateDevComm->devContextCount;

  if (!ginState->connected || backend->ginCommCount <= 0) return ncclInternalError;
  bool legacyBackendContextModel = backend->ginVersion < 13;
  INFO(NCCL_INIT,
       "GIN devComm context selection: plugin api version %d, GPU context layout version %d, requested contexts %d, "
       "connections %d, legacy context model %d, deferred %d",
       backend->ginVersion, ginStateDevComm->backendVersion, ginStateDevComm->devContextCount, backend->ginCommCount,
       legacyBackendContextModel ? 1 : 0, ginStateDevComm->deferred ? 1 : 0);
  if (legacyBackendContextModel) {
    if (ginStateDevComm->deferred) {
      WARN("Deferred GIN devComm setup requires a GIN plugin with backend version 13 or newer.");
      return ncclInvalidUsage;
    }
    nContextsTotal = backend->ginCommCount;
  }
  if (devComm != NULL) {
    devComm->ginContextCount =
      legacyBackendContextModel ? backend->ginCommCount : ginStateDevComm->devContextCount;
  }

  nContextsTotal = ROUNDUP(nContextsTotal, backend->ginCommCount);
  ginStateDevComm->contextCount = nContextsTotal;
  ginStateDevComm->connectionCount = backend->ginCommCount;
  int nContextsPerComm = nContextsTotal / backend->ginCommCount;
  INFO(NCCL_INIT,
       "devCommCreate: creating %d contexts: %d GIN connections with %d contexts each (%d contexts total requested)",
       nContextsTotal, backend->ginCommCount, nContextsPerComm, ginStateDevComm->devContextCount);

  ncclGinConfig_t ginConfig = {
    ginStateDevComm->ginSignalCount, ginStateDevComm->ginCounterCount, nContextsPerComm,
    ginStateDevComm->ginQueueDepth,  ginStateDevComm->ginTrafficClass, ginStateDevComm->backendVersion,
    ginStateDevComm->rankStride,
  };

  for (int n = 0; n < backend->ginCommCount; n++) {
    NCCLCHECKGOTO(backend->ncclGin->createContext(backend->ginComms[n], &ginConfig, &ginStateDevComm->ginCtx[n],
                                                  &ginStateDevComm->devHandles[n]),
                  ret, fail);
    if (ginStateDevComm->ginCtx[n] == NULL || ginStateDevComm->devHandles[n] == NULL ||
        ginStateDevComm->devHandles[n]->handle == NULL) {
      WARN("GIN plugin %s returned invalid context for connection %d: ginCtx=%p devHandle=%p handle=%p",
           backend->ncclGin->name, n, ginStateDevComm->ginCtx[n], ginStateDevComm->devHandles[n],
           ginStateDevComm->devHandles[n] ? ginStateDevComm->devHandles[n]->handle : NULL);
      ret = ncclInternalError;
      goto fail;
    }
    if (ginStateDevComm->devHandles[n]->needsProxyProgress) needsProxyProgress = true;
  }

  ginStateDevComm->connected = true;
  NCCLCHECKGOTO(ncclGinDevCommStoreDynamic(ginStateDevComm, devComm), ret, fail);
  if (needsProxyProgress) NCCLCHECKGOTO(ncclGinStartProgressThreads(comm), ret, fail);
  return ncclSuccess;

fail:
  for (int n = 0; n < NCCL_GIN_MAX_CONNECTIONS; n++) {
    if (ginStateDevComm->ginCtx[n]) {
      backend->ncclGin->destroyContext(ginStateDevComm->ginCtx[n]);
      ginStateDevComm->ginCtx[n] = NULL;
      ginStateDevComm->devHandles[n] = NULL;
    }
  }
  ginStateDevComm->connected = false;
  return ret;
}

static ncclResult_t ginDevCommSetupWithBackend(struct ncclComm* comm, struct ncclDevCommRequirements const* reqs,
                                               struct ncclDevComm* devComm, uint32_t deviceCodeVersion,
                                               struct ncclGinBackendState* backend) {
  struct ncclGinState* ginState = &comm->sharedRes->ginState;
  struct ncclGinStateDevComm* ginStateDevComm = NULL;
  ncclDevCommDynamic_t dynamicState = {};
  ncclResult_t ret = ncclSuccess;
  int connectedStride = comm->globalGinSupport == NCCL_GIN_CONNECTION_FULL ? 1 : comm->contiguousRanksPerHost;
  int requestedStride = 1;

  devComm->backendIndex = (uint8_t)(backend - ginState->backends);
  devComm->ginSignalCount = reqs->ginSignalCount;
  devComm->ginCounterCount = reqs->ginCounterCount;
  // Legacy signals default to what is specified in DevCommRequirements
  devComm->ginStrongLegacySignals = reqs->ginStrongSignalsRequired;

  if (!reqs->ginExclusiveContexts) {
    // TODO: check if a shared devComm in the list could match our requirements.
  }

  if (reqs->ginConnectionType == NCCL_GIN_CONNECTION_CUSTOM_STRIDE) {
    requestedStride = reqs->ginCustomStride;
  } else if (reqs->ginConnectionType == NCCL_GIN_CONNECTION_RAIL) {
    requestedStride = ncclTeamRail(comm).stride;
  }

  if (requestedStride == 0) {
    WARN("Cannot create DevComm with a GIN rank stride of 0. To disable GIN, set reqs->ginConnectionType to "
         "NCCL_GIN_CONNECTION_NONE.");
    ret = ncclInvalidUsage;
    goto end;
  }
  if (requestedStride > ncclTeamRail(comm).stride) {
    // Hierarchical barriers assume GIN is at least RAIL connected.
    WARN("Cannot create DevComm with a GIN rank stride %d greater than the rail team stride %d", requestedStride,
         ncclTeamRail(comm).stride);
    ret = ncclInvalidUsage;
    goto end;
  }
  if (requestedStride % connectedStride != 0) {
    WARN("Cannot create DevComm with the requested GIN rank stride %d, this comm only supports strides that are "
         "multiples of %d",
         requestedStride, connectedStride);
    ret = ncclInvalidUsage;
    goto end;
  }

  NCCLCHECKGOTO(ncclCalloc(&ginStateDevComm, 1), ret, end);
  NCCLCHECKGOTO(ncclGinGetBackendVersion(deviceCodeVersion, backend, &ginStateDevComm->backendVersion), ret, end);

  ginStateDevComm->backendIndex = (int)(backend - ginState->backends);
  ginStateDevComm->devContextCount = reqs->ginContextCount;
  ginStateDevComm->ginSignalCount = reqs->ginSignalCount;
  ginStateDevComm->ginCounterCount = reqs->ginCounterCount;
  ginStateDevComm->ginQueueDepth = reqs->ginQueueDepth;
  ginStateDevComm->ginTrafficClass =
    reqs->ginTrafficClass != NCCL_CONFIG_UNDEF_INT ? reqs->ginTrafficClass : comm->config.trafficClass;
  ginStateDevComm->rankStride = requestedStride / connectedStride;
  ginStateDevComm->deferred = comm->joinDeferred;

  devComm->ginConnectionStride = connectedStride;
  devComm->ginConnectionStride_rcp32 = idivRcp32(connectedStride);
  devComm->ginContextStride = requestedStride;
  devComm->ginContextCount = backend->ginVersion < 13 && ginState->connected ?
                               backend->ginCommCount :
                               ginStateDevComm->devContextCount;
  devComm->ginConnectionCount = ginState->connected ? backend->ginCommCount : 1;
  dynamicState.ginConnectionCount = devComm->ginConnectionCount;
  NCCLCHECKGOTO(ncclCudaCalloc(&devComm->dynamicState, 1, comm->memManager), ret, end);
  NCCLCHECKGOTO(ncclCudaMemcpy(devComm->dynamicState, &dynamicState, 1), ret, end);
  ginStateDevComm->dynamicState = devComm->dynamicState;

  if (ginState->connected) {
    NCCLCHECKGOTO(ncclGinDevCommCreateContexts(comm, ginStateDevComm, devComm), ret, end);
  }
  NCCLCHECKGOTO(ncclGinDevCommAppend(ginState, ginStateDevComm), ret, end);
  ginStateDevComm = NULL;

end:
  if (ret != ncclSuccess) {
    devComm->backendIndex = 0;
    devComm->ginConnectionCount = 0;
    devComm->ginContextCount = 0;
    devComm->ginConnectionStride = 0;
    devComm->ginConnectionStride_rcp32 = 0;
    devComm->ginContextStride = 0;
    memset(devComm->ginNetDeviceTypes, 0, sizeof(devComm->ginNetDeviceTypes));
    memset(devComm->ginHandles, 0, sizeof(devComm->ginHandles));
    if (ginStateDevComm != NULL) {
      for (int n = 0; n < NCCL_GIN_MAX_CONNECTIONS; n++) {
        if (ginStateDevComm->ginCtx[n]) backend->ncclGin->destroyContext(ginStateDevComm->ginCtx[n]);
      }
      free(ginStateDevComm);
    }
    if (devComm->dynamicState) {
      NCCLCHECKIGNORE(ncclCudaFree(devComm->dynamicState, comm->memManager), ret);
      devComm->dynamicState = NULL;
    }
  }
  return ret;
}

ncclResult_t ncclGinDevCommSetup(struct ncclComm* comm, struct ncclDevCommRequirements const* reqs,
                                 struct ncclDevComm* devComm, uint32_t deviceCodeVersion) {
  struct ncclGinState* ginState = &comm->sharedRes->ginState;
  ncclGinType_t reqGinType = reqs->ginType;
  int64_t envGinType = ncclParamGinType();

  if (reqGinType == NCCL_GIN_TYPE_NONE && envGinType > NCCL_GIN_TYPE_NONE) {
    reqGinType = (ncclGinType_t)envGinType;
  }

  if (reqGinType >= NCCL_GIN_MAX_TYPES) {
    WARN("Invalid GIN type requested (%d)", reqGinType);
    return ncclInvalidUsage;
  }

  for (int i = 0; i < ginState->numActiveBackends; i++) {
    struct ncclGinBackendState* candidate = &ginState->backends[i];

    if (reqGinType != NCCL_GIN_TYPE_NONE && candidate->ginType != reqGinType) {
      continue;
    }
    if (ncclSuccess != ncclGinValidateSignalRequest(reqs, candidate)) {
      continue;
    }

    if (ncclSuccess == ginDevCommSetupWithBackend(comm, reqs, devComm, deviceCodeVersion, candidate)) {
      return ncclSuccess;
    }

    INFO(NCCL_INIT, "GIN: DevComm setup failed with backend type %d", candidate->ginType);
  }

  WARN("GIN: DevComm setup failed on all available backends");
  return ncclInternalError;
}

ncclResult_t ncclGinDevCommDisconnectAll(struct ncclComm* comm) {
  ncclResult_t ret = ncclSuccess;
  struct ncclGinState* ginState = &comm->sharedRes->ginState;

  NCCLCHECK(ncclGinStopProgressThreads(ginState));

  for (struct ncclGinStateDevComm* dc = ginState->devComms; dc != NULL; dc = dc->next) {
    if (!dc->connected) continue;
    struct ncclGinBackendState* backend = &ginState->backends[dc->backendIndex];
    for (int n = 0; n < NCCL_GIN_MAX_CONNECTIONS; n++) {
      if (dc->ginCtx[n] != NULL) {
        NCCLCHECKGOTO(backend->ncclGin->destroyContext(dc->ginCtx[n]), ret, exit);
        dc->ginCtx[n] = NULL;
        dc->devHandles[n] = NULL;
      }
    }
    dc->connected = false;
    dc->connectionCount = 0;
  }

  for (int backendIdx = 0; backendIdx < ginState->numActiveBackends; backendIdx++) {
    struct ncclGinBackendState* backend = &ginState->backends[backendIdx];
    for (int n = 0; n < NCCL_GIN_MAX_CONNECTIONS; n++) {
      if (backend->ginComms[n] != NULL) {
        NCCLCHECKGOTO(backend->ncclGin->closeColl(backend->ginComms[n]), ret, exit);
        backend->ginComms[n] = NULL;
      }
    }
    backend->ginCommCount = 0;
  }
  ginState->connected = false;

exit:
  return ret;
}

ncclResult_t ncclGinDevCommConnectAll(struct ncclComm* comm) {
  ncclResult_t ret = ncclSuccess;
  struct ncclGinState* ginState = &comm->sharedRes->ginState;

  if (ginState->devComms == NULL) return ncclSuccess;
  NCCLCHECKGOTO(ncclGinConnectOnce(comm), ret, exit);
  for (struct ncclGinStateDevComm* dc = ginState->devComms; dc != NULL; dc = dc->next) {
    if (dc->connected) continue;
    NCCLCHECKGOTO(ncclGinDevCommCreateContexts(comm, dc, NULL), ret, exit);
  }

exit:
  return ret;
}

ncclResult_t ncclGinDevCommFree(struct ncclComm* comm, struct ncclDevComm const* devComm) {
  // Find the resource associated with this devComm. Use the gin handle as key.
  struct ncclGinState* ginState = &comm->sharedRes->ginState;

  struct ncclGinStateDevComm *dc = ginState->devComms, *prevDc = NULL;
  bool locked = ginState->proxyThreadsCreated;
  if (locked) ginProgressWriteLock(ginState);
  while (1) {
    if (dc == NULL) {
      if (locked) ginProgressWriteUnlock(ginState);
      WARN("Dev comm not found\n");
      return ncclInternalError;
    }
    if ((devComm->dynamicState != NULL && dc->dynamicState == devComm->dynamicState) ||
        (dc->devHandles[0] != NULL && dc->devHandles[0]->handle == devComm->ginHandles[0]))
      break;
    prevDc = dc;
    dc = dc->next;
  }

  // Remove from linked list
  if (prevDc) prevDc->next = dc->next;
  else ginState->devComms = dc->next;
  if (locked) ginProgressWriteUnlock(ginState);

  struct ncclGinBackendState* backend = &ginState->backends[dc->backendIndex];
  // The devComm is now unreachable by any progress thread; safe to destroy
  // its contexts while the workers keep progressing the rest of the list.
  for (int n = 0; n < NCCL_GIN_MAX_CONNECTIONS; n++) {
    if (dc->ginCtx[n] != NULL) NCCLCHECK(backend->ncclGin->destroyContext(dc->ginCtx[n]));
  }
  if (dc->dynamicState) NCCLCHECK(ncclCudaFree(dc->dynamicState, comm->memManager));
  free(dc);
  return ncclSuccess;
}

ncclResult_t ncclGinHostFinalize(struct ncclComm* comm) {
  struct ncclGinState* ginState = &comm->sharedRes->ginState;
  if (!ginState->connected) return ncclSuccess;

  if (ginState->proxyThreadsCreated) {
    ginState->proxyThreadStopSignal.store(true);
    for (int t = 0; t < ginState->proxyNthreads; t++) {
      if (ginState->thread[t].joinable()) ginState->thread[t].join();
    }
  }

  for (int backendIdx = 0; backendIdx < ginState->numActiveBackends; backendIdx++) {
    struct ncclGinBackendState* backend = &ginState->backends[backendIdx];
    for (int commIdx = 0; commIdx < backend->ginCommCount; commIdx++) {
      if (backend->ginComms[commIdx] != NULL) {
        NCCLCHECK(backend->ncclGin->closeColl(backend->ginComms[commIdx]));
        backend->ginComms[commIdx] = NULL;
      }
    }
  }
  memset((void*)ginState, 0, sizeof(*ginState));
  return ncclSuccess;
}

ncclResult_t ncclGinRegister(struct ncclComm* comm, void* address, size_t size,
                             void* ginHostWins[NCCL_GIN_MAX_CONNECTIONS * NCCL_GIN_MAX_ACTIVE_BACKENDS],
                             ncclGinWindow_t ginDevWins[NCCL_GIN_MAX_CONNECTIONS * NCCL_GIN_MAX_ACTIVE_BACKENDS],
                             int winFlags, bool multiSegment, int memType) {
  struct ncclGinState* ginState = &comm->sharedRes->ginState;
  int mrFlags = (winFlags & NCCL_WIN_STRICT_ORDERING) ? NCCL_NET_MR_FLAG_FORCE_SO : 0;
  for (int backendIdx = 0; backendIdx < ginState->numActiveBackends; backendIdx++) {
    struct ncclGinBackendState* backend = &ginState->backends[backendIdx];
    if (multiSegment) {
      // Multi-segment GIN registration requires DMABUF support on all GIN connections
      for (int commIdx = 0; commIdx < backend->ginCommCount; commIdx++) {
        if (!(backend->ginProps[commIdx].ptrSupport & NCCL_PTR_DMABUF)) {
          WARN(
            "Window registration of addresses that span multiple physical segments requires DMABUF support with GIN.");
          return ncclInvalidArgument;
        }
      }
    }
    for (int commIdx = 0; commIdx < backend->ginCommCount; commIdx++) {
      int slot = backendIdx * NCCL_GIN_MAX_CONNECTIONS + commIdx;
      NCCLCHECK(backend->ncclGin->regMrSym(backend->ginComms[commIdx], address, size, memType, mrFlags,
                                           &ginHostWins[slot], &ginDevWins[slot]));
      if (ginHostWins[slot] == NULL) {
        WARN("rank %d - GIN Symmetric register failed: buff %p, size %ld", comm->rank, address, size);
        return ncclSystemError;
      }
    }
  }
  return ncclSuccess;
}

ncclResult_t ncclGinDeregister(struct ncclComm* comm,
                               void* ginHostWins[NCCL_GIN_MAX_CONNECTIONS * NCCL_GIN_MAX_ACTIVE_BACKENDS]) {
  struct ncclGinState* ginState = &comm->sharedRes->ginState;
  for (int backendIdx = 0; backendIdx < ginState->numActiveBackends; backendIdx++) {
    struct ncclGinBackendState* backend = &ginState->backends[backendIdx];
    for (int commIdx = 0; commIdx < backend->ginCommCount; commIdx++) {
      int slot = backendIdx * NCCL_GIN_MAX_CONNECTIONS + commIdx;
      NCCLCHECK(backend->ncclGin->deregMrSym(backend->ginComms[commIdx], ginHostWins[slot]));
    }
  }
  return ncclSuccess;
}

ncclResult_t ncclGinQueryLastError(struct ncclGinState* ginState, bool* hasError) {
  *hasError = false;
  std::shared_lock<std::shared_timed_mutex> rlock(ginState->devCommRwMutex);
  struct ncclGinStateDevComm* dc = ginState->devComms;
  while (dc) {
    if (!dc->connected) {
      dc = dc->next;
      continue;
    }
    struct ncclGinBackendState* backend = &ginState->backends[dc->backendIndex];
    for (int commIdx = 0; commIdx < dc->connectionCount; commIdx++) {
      NCCLCHECK(backend->ncclGin->queryLastError(dc->ginCtx[commIdx], hasError));
      if (*hasError) return ncclSuccess;
    }
    dc = dc->next;
  }
  return ncclSuccess;
}
