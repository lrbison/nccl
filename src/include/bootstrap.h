/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2015-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * See LICENSE.txt for more license information
 *************************************************************************/

#ifndef NCCL_BOOTSTRAP_H_
#define NCCL_BOOTSTRAP_H_

#include "nccl.h"
#include "comm.h"
#include "ras.h"
#include <thread>

struct ncclBootstrapHandle {
  uint64_t magic;
  union ncclSocketAddress addr;
  int nRanks; // number of existing ranks
  int flags;
  int leaderRank;
};
static_assert(sizeof(struct ncclBootstrapHandle) <= sizeof(ncclUniqueId),
              "Bootstrap handle is too large to fit inside NCCL unique ID");

ncclResult_t bootstrapNetInit();
ncclResult_t bootstrapNetReset();
ncclResult_t bootstrapNetRediscover();
ncclResult_t bootstrapCreateRoot(struct ncclBootstrapHandle* handle, bool idFromEnv);
ncclResult_t bootstrapCreateListenSocket(struct ncclBootstrapHandle* handle, struct ncclComm* comm,
                                         volatile uint32_t* abortFlag, struct ncclSocket** listenSock);
ncclResult_t bootstrapGetUniqueId(struct ncclBootstrapHandle* handle, struct ncclComm* comm);
ncclResult_t bcastGrowHandle(struct ncclBootstrapHandle* handle, struct ncclComm* parent, bool isRoot);
ncclResult_t bootstrapInit(int nHandles, void* handle, struct ncclComm* comm, struct ncclComm* parent);
ncclResult_t bootstrapSplit(uint64_t magic, struct ncclComm* comm, struct ncclComm* parent, int color, int key,
                            int* parentRanks);
ncclResult_t bootstrapJoinInitLocalState(struct ncclComm* comm);
ncclResult_t bootstrapQuiesceLocalAddresses(struct ncclComm* comm);
ncclResult_t bootstrapRediscoverLocalAddresses(struct ncclComm* comm);
ncclResult_t bootstrapUpdateRankMask(void* commState);
ncclResult_t bootstrapAllGather(void* commState, void* allData, int size);
ncclResult_t bootstrapSend(void* commState, int peer, int tag, void* data, int size);
ncclResult_t bootstrapRecv(void* commState, int peer, int tag, void* data, int size);
ncclResult_t bootstrapBarrier(void* commState, int rank, int nranks, int tag);
ncclResult_t bootstrapBroadcast(void* commState, int rank, int nranks, int root, void* bcastData, int size);
ncclResult_t bootstrapIntraNodeBarrier(void* commState, int* ranks, int rank, int nranks, int tag);
ncclResult_t bootstrapIntraNodeAllGather(void* commState, int* ranks, int rank, int nranks, void* allData, int size);
ncclResult_t bootstrapIntraNodeBroadcast(void* commState, int* ranks, int rank, int nranks, int root, void* bcastData,
                                         int size);
ncclResult_t bootstrapClose(void* commState);
ncclResult_t bootstrapAbort(void* commState);

typedef enum {
  ncclCommMaskPolicyLocal = 1,
} ncclCommMaskPolicy_t;

typedef struct ncclCommMaskUpdateConfig_t {
  uint8_t reserved[8];
  ncclCommMaskPolicy_t policy;
} ncclCommMaskUpdateConfig_t;

enum ncclJoinPeerStatus {
  ncclJoinPeerStatusUninit = 0,
  ncclJoinPeerStatusPeerConnecting = 1,
  ncclJoinPeerStatusPeerReady = 2,
  ncclJoinPeerStatusLeaderCanceled = 3,
  ncclJoinPeerStatusLeaderCommitted = 4,
};

union ringConnectInfo {
  union ncclSocketAddress addr;
  char handle[NCCL_NET_HANDLE_MAXSIZE];
};

struct bootstrapRingDataPacked {
  union ringConnectInfo ringAddress;
  union ncclSocketAddress peerAddress;
  union ncclSocketAddress peerProxy;
  uint64_t peerUDS;
  struct rasRankInit rasRank;
};

struct ncclReshapeJoinerConn {
  volatile int joinStatus;
  uint64_t hostHash;
  struct ncclSocket sock;
  struct bootstrapRingDataPacked ringData;
};

struct ncclReshapeLeaderState {
  volatile uint32_t threadStop;
  volatile uint32_t peerThreadCount;
  volatile ncclResult_t threadResult;
  struct ncclReshapeJoinerConn* joiners;
  struct ncclSocket* listenSock;
  std::thread* thread;
};

ncclResult_t ncclReshapeLeaderStateCreate(ncclComm_t comm);
ncclResult_t ncclReshapeLeaderStateDestroy(ncclComm_t comm);
ncclResult_t bootstrapReshapeCreateRoot(struct ncclBootstrapHandle* handle, struct ncclComm* comm);
ncclResult_t bootstrapReshapeJoinerConnect(ncclComm_t* comm, ncclUniqueId uniqueId, int rankHint, int useExistingComm,
                                           const ncclConfig_t* commConfig, struct ncclSocket** sock);
ncclResult_t bootstrapReshapeJoinerComplete(ncclComm_t comm, struct ncclSocket** sock);
ncclResult_t bootstrapReshapeSurvivorsCommit(ncclComm_t comm, struct ncclReshapeLeaderState* state, int leaderRank,
                                             const int* joinList, int joinListLen, const int* leaveList,
                                             int leaveListLen, int flags);
ncclResult_t ncclCommJoinCreateFresh(ncclComm_t* comm, int nranks, int rank, uint64_t magic, uint64_t commHash,
                                     const ncclConfig_t* config, const struct ncclPeerInfo* peerInfo,
                                     const struct ncclJoinAllGatherInfo* allGatherData, const void* topoXml,
                                     size_t topoXmlSize);
#endif
