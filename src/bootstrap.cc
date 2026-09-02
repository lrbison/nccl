/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2016-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * See LICENSE.txt for more license information
 *************************************************************************/

#include "nccl.h"
#include "core.h"
#include "utils.h"
#include "bootstrap.h"
#include "net.h"
#include "proxy.h"
#include "transport.h"
#include "param.h"
#include "rank_mask.h"
#include "ras.h"
#include <mutex>
#include "os.h"
#include <thread>
#include <chrono>

#define BOOTSTRAP_N_CHECK_ABORT 10000
#define BOOTSTRAP_TAG_CONNECT (0x1 << 31)
#define BOOTSTRAP_TAG_ALLGATHER (0x1 << 30)
#define BOOTSTRAP_TAG_COMMSPLIT (0x1 << 29)
#define BOOTSTRAP_TAG_INTRANODE_ALLGATHER (0x1 << 28)
#define BOOTSTRAP_TAG_GROW_BOUNDARY (0x1 << 27)
#define BOOTSTRAP_TAG_ACTIVE_ALLGATHER (0x1 << 26)

#define BOOTSTRAP_INIT_TIME_CREATE 0
#define BOOTSTRAP_INIT_TIME_SEND 1
#define BOOTSTRAP_INIT_TIME_RECV 2
#define BOOTSTRAP_INIT_TIME_RING 3
#define BOOTSTRAP_INIT_TIME_TOTAL 4
#define BOOTSTRAP_INIT_TIME_DELAY 5
#define BOOTSTRAP_INIT_TIME_N 6
#define BOOTSTRAP_INIT_ROOT_WAIT 0
#define BOOTSTRAP_INIT_ROOT_SEND 1
#define BOOTSTRAP_INIT_ROOT_RECV 2
#define BOOTSTRAP_INIT_ROOT_N 3
#define BOOTSTRAP_PROF_OPEN(time) \
  do { \
    time = clockNano(); \
  } while (0)
#define BOOTSTRAP_PROF_CLOSE(time) \
  do { \
    time = clockNano() - time; \
  } while (0)

#define BOOTSTRAP_PID(i, n) (((i) + (n)) % (n))
// returns the first rank associated to the root. must have root >=0
// if root >= n_roots, it does NOT assume periodicity
static int firstRankFromRoot(int root, int n_ranks, int nRoots, int offset) {
  if (root == -1) return 0;
  // only distribute the n_ranks - offset on the roots
  n_ranks -= offset;
  return offset + root * (n_ranks / nRoots) + std::min(root, n_ranks % nRoots);
}
// returns the root of a rank, must have rank >=0
// if rank >= n_ranks, it does NOT assume periodicity
static int rootIdFromRank(int rank, int nRanks, int nRoots, int offset) {
  // ranks < offset have no root (id = -1), ranks above the offset will get assigned to their respective root
  if (nRoots == 0 || rank < offset) return -1;
  nRanks -= offset;
  rank -= offset;
  int rmr = nRanks % nRoots; // rank mod root
  int rpr = nRanks / nRoots; // rank per root
  int D = rmr * (rpr + 1);
  if (rank < D) return rank / (rpr + 1);
  else return (rank - D) / rpr + rmr;
}
// return the number of child for a root, root will be periodized
static int nRankFromRoot(int root, int nRanks, int nRoots, int offset) {
  if (root == -1) return 0;
  nRanks -= offset;
  int ir = BOOTSTRAP_PID(root, nRoots);
  int rmr = nRanks % nRoots; // rank mod root
  int rpr = nRanks / nRoots; // rank per root
  return rpr + ((ir < rmr) ? 1 : 0);
}
// return the local id of a given rank for a given root
// root will be periodize, rank will not
static int localIdFromRoot(int rank, int root, int nRanks, int nRoots, int offset) {
  // any rank for root -1 has a local id that is the rank id
  if (root == -1) return rank;
  int ir = BOOTSTRAP_PID(root, nRoots);
  return rank - firstRankFromRoot(ir, nRanks, nRoots, offset);
}
// Check if the given rank is the first rank from the root
static int isFirstFromRoot(int rank, int root, int nRanks, int nRoots, int offset) {
  return (rank == firstRankFromRoot(root, nRanks, nRoots, offset));
}

static ncclResult_t ncclReshapeLeaderConnStart(struct ncclReshapeLeaderState* state, int rank, struct ncclSocket* sock,
                                               struct ncclReshapeJoinerConn** connOut) {
  struct ncclReshapeJoinerConn* conn = NULL;
  int expected = ncclJoinPeerStatusUninit;

  if (state == NULL || state->joiners == NULL || sock == NULL || connOut == NULL) return ncclInvalidArgument;
  *connOut = NULL;
  conn = state->joiners + rank;
  if (!COMPILER_ATOMIC_COMPARE_EXCHANGE(&conn->joinStatus, &expected,
                                        static_cast<int>(ncclJoinPeerStatusPeerConnecting),
                                        std::memory_order_acq_rel, std::memory_order_acquire)) {
    return ncclSuccess;
  }
  (void)ncclSocketClose(&conn->sock);
  conn->hostHash = 0;
  memset(&conn->ringData, 0, sizeof(conn->ringData));
  memcpy(&conn->sock, sock, sizeof(conn->sock));
  conn->sock.abortFlag = &state->threadStop;
  *connOut = conn;
  return ncclSuccess;
}

static ncclResult_t ncclReshapeLeaderStateFree(struct ncclReshapeLeaderState* state, int nRanks) {
  if (state == NULL) return ncclSuccess;
  if (state->thread != NULL) {
    delete state->thread;
    state->thread = NULL;
  }
  if (state->listenSock != NULL) {
    (void)ncclSocketClose(state->listenSock);
    free(state->listenSock);
    state->listenSock = NULL;
  }
  if (state->joiners != NULL) {
    for (int rank = 0; rank < nRanks; rank++) {
      (void)ncclSocketClose(&state->joiners[rank].sock);
    }
  }
  free(state->joiners);
  free(state);
  return ncclSuccess;
}

ncclResult_t ncclReshapeLeaderStateDestroy(ncclComm_t comm) {
  struct ncclReshapeLeaderState* state = comm ? comm->reshapeLeader : NULL;

  if (state == NULL) return ncclSuccess;
  COMPILER_ATOMIC_STORE(&state->threadStop, uint32_t(1), std::memory_order_release);
  if (state->thread != NULL && state->thread->joinable()) state->thread->join();
  while (COMPILER_ATOMIC_LOAD(&state->peerThreadCount, std::memory_order_acquire) != 0) {
    std::this_thread::sleep_for(std::chrono::microseconds(1));
  }
  comm->reshapeLeader = NULL;
  return ncclReshapeLeaderStateFree(state, comm->nRanks);
}

ncclResult_t ncclReshapeLeaderStateCreate(ncclComm_t comm) {
  ncclResult_t ret = ncclSuccess;
  struct ncclReshapeLeaderState* state = NULL;

  if (comm == NULL) return ncclInvalidArgument;
  if (comm->reshapeLeader != NULL) return ncclSuccess;

  NCCLCHECKGOTO(ncclCalloc(&state, 1), ret, fail);
  state->threadResult = ncclInProgress;
  NCCLCHECKGOTO(ncclCalloc(&state->joiners, comm->nRanks), ret, fail);

  comm->reshapeLeader = state;
  return ncclSuccess;

fail:
  if (state != NULL) (void)ncclReshapeLeaderStateFree(state, comm->nRanks);
  return ret;
}

struct bootstrapRootArgs {
  struct ncclSocket* listenSock;
  uint64_t magic;
};

struct ncclReshapeLeaderThreadArgs {
  struct ncclSocket* listenSock;
  ncclComm_t comm;
  struct ncclReshapeLeaderState* state;
};

struct ncclReshapeLeaderPeerThreadArgs {
  struct ncclSocket* sock;
  ncclComm_t comm;
  struct ncclReshapeLeaderState* state;
};

static void ncclCommReshapeLeaderThread(void* rargs);
static void bootstrapReshapeLeaderAcceptPeer(void* rargs);
static ncclResult_t bootstrapReshapeLeaderWaitForCommit(ncclComm_t comm, struct ncclReshapeLeaderState* state, int rank,
                                                        struct ncclReshapeJoinerConn* conn);
static ncclResult_t bootstrapReshapeLeaderStartPeerThread(ncclComm_t comm, struct ncclReshapeLeaderState* state,
                                                          struct ncclSocket** sockPtr);
static ncclResult_t bootstrapReshapeAcceptSendAllRingData(ncclComm_t comm, struct ncclSocket* sock);
static ncclResult_t bootstrapReshapeConnectSendRingData(ncclComm_t comm, int rankHint, struct ncclSocket* sock);
static ncclResult_t bootstrapReshapeConnectRecvAllRingData(ncclComm_t comm, int nRanks, struct ncclSocket* sock);
static ncclResult_t getUDS(uint64_t* peerUDS);
static ncclResult_t netGetDevice(int rank, struct ncclComm* comm, int* dev);

/* Init functions */
static char bootstrapNetIfName[MAX_IF_NAME_SIZE + 1];
static union ncclSocketAddress bootstrapNetIfAddr;
static int bootstrapNetInitDone = 0;
static int bootstrapNetDevOOB = -1;
static std::mutex bootstrapNetMutex;

NCCL_PARAM(BootstrapNetEnable, "OOB_NET_ENABLE", 0);

ncclResult_t bootstrapNetInit() {
  if (bootstrapNetInitDone == 0) {
    std::lock_guard<std::mutex> lock(bootstrapNetMutex);
    if (bootstrapNetInitDone == 0) {
      const char* env = ncclGetEnv("NCCL_COMM_ID");
      int nIfs = 0;
      if (env) {
        union ncclSocketAddress remoteAddr;
        if (ncclSocketGetAddrFromString(&remoteAddr, env) != ncclSuccess) {
          WARN("Invalid NCCL_COMM_ID, please use format: <ipv4>:<port> or [<ipv6>]:<port> or <hostname>:<port>");
          return ncclInvalidArgument;
        }
        NCCLCHECK(ncclFindInterfaceMatchSubnet(bootstrapNetIfName, &bootstrapNetIfAddr, &remoteAddr, MAX_IF_NAME_SIZE,
                                               &nIfs));
        if (nIfs <= 0) {
          WARN("NET/Socket : No usable listening interface found");
          return ncclSystemError;
        }
      } else {
        NCCLCHECK(ncclFindInterfaces(bootstrapNetIfName, &bootstrapNetIfAddr, MAX_IF_NAME_SIZE, 1, &nIfs));
        if (nIfs <= 0) {
          WARN("Bootstrap : no socket interface found");
          return ncclInvalidUsage;
        }
      }
      char line[SOCKET_NAME_MAXLEN + MAX_IF_NAME_SIZE + 2];
      snprintf(line, sizeof(line), " %s:", bootstrapNetIfName);
      ncclSocketToString(&bootstrapNetIfAddr, line + strlen(line));
      INFO(NCCL_BOOTSTRAP, "Bootstrap: Using%s", line);
      bootstrapNetInitDone = 1;
    }
  }
  return ncclSuccess;
}

ncclResult_t bootstrapNetRediscover() {
  bootstrapNetInitDone = 0;
  bootstrapNetDevOOB = -1;
  memset(bootstrapNetIfName, 0, sizeof(bootstrapNetIfName));
  memset(&bootstrapNetIfAddr, 0, sizeof(bootstrapNetIfAddr));
  NCCLCHECK(bootstrapNetInit());
  return ncclSuccess;
}

/* Socket Interface Selection type */
enum bootstrapInterface_t {
  findSubnetIf = -1,
  dontCareIf = -2
};

// check abort function
static ncclResult_t checkAbort(volatile uint32_t* flag, int* cntr) {
  if ((*cntr % BOOTSTRAP_N_CHECK_ABORT) == 0) {
    if (flag && COMPILER_ATOMIC_LOAD(flag, std::memory_order_acquire)) {
      TRACE(NCCL_BOOTSTRAP, "bootstrap: abort called");
      return ncclInternalError;
    }
  }
  *cntr = (*cntr + 1) % BOOTSTRAP_N_CHECK_ABORT;
  return ncclSuccess;
}
// send/recv functions
static ncclResult_t netReg(ncclNet_t* net, void* comm, void* data, int size, void** handle) {
  NCCLCHECK(net->regMr(comm, data, size, NCCL_PTR_HOST, handle));
  return ncclSuccess;
}
static ncclResult_t netDereg(ncclNet_t* net, void* comm, void** handle) {
  NCCLCHECK(net->deregMr(comm, *handle));
  *handle = NULL;
  return ncclSuccess;
}
static ncclResult_t netIsend(ncclNet_t* net, void* sendComm, void* data, int size, void* dataHandle, int tag,
                             void** sendReq, int* done) {
  if (*done) return ncclSuccess;
  if (!*sendReq) {
    NCCLCHECK(net->isend(sendComm, data, (size_t)size, tag, dataHandle, NULL, sendReq));
  }
  if (*sendReq) {
    NCCLCHECK(net->test(*sendReq, done, NULL));
    if (*done) {
      *sendReq = NULL;
    }
  }
  return ncclSuccess;
}
static ncclResult_t netIrecv(ncclNet_t* net, void* recvComm, void* data, int size, void* dataHandle, int tag,
                             void** recvReq, int* done) {
  if (*done) return ncclSuccess;
  if (!*recvReq) {
    size_t size64 = size;
    NCCLCHECK(net->irecv(recvComm, 1, &data, &size64, &tag, &dataHandle, NULL, recvReq));
  }
  if (*recvReq) {
    NCCLCHECK(net->test(*recvReq, done, NULL));
    if (*done) {
      *recvReq = NULL;
    }
  }
  return ncclSuccess;
}
static ncclResult_t netSendRecv(ncclNet_t* net, void* sendComm, void* sendData, int sendSize, void* sendDataHandle,
                                void* recvComm, void* recvData, int recvSize, void* recvDataHandle, int tag,
                                volatile uint32_t* abortFlag) {
  int abortCounter = 0;
  int doneSend = 0, doneRecv = 0;
  void *sendReq = NULL, *recvReq = NULL;
  do {
    NCCLCHECK(checkAbort(abortFlag, &abortCounter));
    if (!doneRecv) {
      NCCLCHECK(netIrecv(net, recvComm, recvData, recvSize, recvDataHandle, tag, &recvReq, &doneRecv));
    }
    if (!doneSend) {
      NCCLCHECK(netIsend(net, sendComm, sendData, sendSize, sendDataHandle, tag, &sendReq, &doneSend));
    }
  } while (!doneSend || !doneRecv);
  return ncclSuccess;
}

// Additional socket based functions, first send the size, then send the message
static ncclResult_t socketSend(struct ncclSocket* sock, void* data, int size) {
  NCCLCHECK(ncclSocketSend(sock, &size, sizeof(int)));
  if (size > 0) NCCLCHECK(ncclSocketSend(sock, data, size));
  return ncclSuccess;
}
static ncclResult_t socketRecv(struct ncclSocket* sock, void* data, int size) {
  int recvSize;
  NCCLCHECK(ncclSocketRecv(sock, &recvSize, sizeof(int)));
  if (recvSize > size) {
    WARN("Message truncated : received %d bytes instead of %d", recvSize, size);
    return ncclInternalError;
  }
  int actualSize = std::min(recvSize, size);
  if (actualSize > 0) NCCLCHECK(ncclSocketRecv(sock, data, actualSize));
  return ncclSuccess;
}
static ncclResult_t socketSendRecv(struct ncclSocket* sendSock, void* sendData, int sendSize,
                                   struct ncclSocket* recvSock, void* recvData, int recvSize) {
  int senderRecvSize;
  NCCLCHECK(ncclSocketSendRecv(sendSock, &sendSize, sizeof(int), recvSock, &senderRecvSize, sizeof(int)));
  if (senderRecvSize > recvSize) {
    WARN("Message truncated : received %d bytes instead of %d", senderRecvSize, recvSize);
    return ncclInternalError;
  }
  NCCLCHECK(ncclSocketSendRecv(sendSock, sendData, sendSize, recvSock, recvData, std::min(recvSize, senderRecvSize)));
  return ncclSuccess;
}

static ncclResult_t socketDoubleSendRecv(struct ncclSocketOp ops[4]) {
  // ops synchronously exchange size then asynchronously exchange data in send->recv->send->recv order
  int senderRecvSize1, senderRecvSize2;
  NCCLCHECK(ncclSocketSendRecv(ops[0].sock, &ops[0].size, sizeof(int), ops[1].sock, &senderRecvSize1, sizeof(int)));
  NCCLCHECK(ncclSocketSendRecv(ops[2].sock, &ops[2].size, sizeof(int), ops[3].sock, &senderRecvSize2, sizeof(int)));
  if (senderRecvSize1 > ops[1].size || senderRecvSize2 > ops[3].size) {
    WARN("Message truncated : received %d,%d bytes instead of %d,%d", senderRecvSize1, senderRecvSize2, ops[1].size,
         ops[3].size);
    return ncclInternalError;
  }
  ops[1].size = std::min(ops[1].size, senderRecvSize1);
  ops[3].size = std::min(ops[3].size, senderRecvSize2);
  NCCLCHECK(ncclSocketMultiOp(ops, 4));
  return ncclSuccess;
}

struct extInfo {
  int rank;                                  // rank of the process reaching out
  int nranks;                                // total number of ranks
  int iroot;                                 // current root index
  int nroots;                                // total number of roots
  int offset;                                // offset for rank distribution
  union ncclSocketAddress listenRootAddress; // address of my listenSocket for the root
  union ringConnectInfo connectInfo;
};
#define NET_HANDLE(h, rank) ((h) + (rank * NCCL_NET_HANDLE_MAXSIZE))
#define BOOTSTRAP_HANDLE(h, i) ((struct ncclBootstrapHandle*)((char*)h + i * NCCL_UNIQUE_ID_BYTES))

struct bootstrapJoinCommInfo {
  int32_t nranks;
  int32_t nChannels;
  uint64_t magic;
  uint64_t commHash;
  uint64_t topoXmlSize;
};

static ncclResult_t ringDataPackUnpack(bool isPack, struct bootstrapRingDataPacked* ringData, int nranks, int* offsets,
                                       union ncclSocketAddress* peerAddresses, union ncclSocketAddress* peerProxy,
                                       uint64_t* peerUDS, struct rasRankInit* rasRanks,
                                       union ringConnectInfo* ringAddresses) {
  /* offsets refers to the offset within the addresses, the packed array is always packed. */
  for (int jslot = 0; jslot < nranks; ++jslot) {
    int pos = offsets ? offsets[jslot] : jslot;
    if (isPack) {
      if (ringAddresses) memcpy(&(ringData[jslot].ringAddress), ringAddresses + pos, sizeof(union ringConnectInfo));
      if (peerAddresses) memcpy(&(ringData[jslot].peerAddress), peerAddresses + pos, sizeof(union ncclSocketAddress));
      if (peerProxy) memcpy(&(ringData[jslot].peerProxy), peerProxy + pos, sizeof(union ncclSocketAddress));
      if (peerUDS) memcpy(&(ringData[jslot].peerUDS), peerUDS + pos, sizeof(uint64_t));
      if (rasRanks) memcpy(&(ringData[jslot].rasRank), rasRanks + pos, sizeof(*rasRanks));
    } else {
      if (ringAddresses) memcpy(ringAddresses + pos, &(ringData[jslot].ringAddress), sizeof(union ringConnectInfo));
      if (peerAddresses) memcpy(peerAddresses + pos, &(ringData[jslot].peerAddress), sizeof(union ncclSocketAddress));
      if (peerProxy) memcpy(peerProxy + pos, &(ringData[jslot].peerProxy), sizeof(union ncclSocketAddress));
      if (peerUDS) memcpy(peerUDS + pos, &(ringData[jslot].peerUDS), sizeof(uint64_t));
      if (rasRanks) memcpy(rasRanks + pos, &(ringData[jslot].rasRank), sizeof(*rasRanks));
    }
  }
  return ncclSuccess;
}

static ncclResult_t rootSend(union ncclSocketAddress* addr, uint64_t magic, union ringConnectInfo* info) {
  ncclResult_t res = ncclSuccess;
  struct ncclSocket sock;
  NCCLCHECKGOTO(ncclSocketInit(&sock, addr, magic, ncclSocketTypeBootstrap), res, fail);
  NCCLCHECKGOTO(ncclSocketConnect(&sock), res, fail);
  NCCLCHECKGOTO(socketSend(&sock, info, sizeof(union ringConnectInfo)), res, fail);
  NCCLCHECK(ncclSocketClose(&sock));
  return res;
fail:
  (void)ncclSocketClose(&sock);
  return res;
}
static void* bootstrapRoot(void* rargs) {
  uint64_t timers[BOOTSTRAP_INIT_ROOT_N] = {0};
  struct bootstrapRootArgs* args = (struct bootstrapRootArgs*)rargs;
  struct ncclSocket* listenSock = args->listenSock;
  uint64_t magic = args->magic;
  ncclResult_t res = ncclSuccess;
  int nranks = 0, c = 0;
  int iroot = 0, nroots = 0, localId = 0;
  int nrecv = 0, n2send = 0, offset = 0;
  struct extInfo info;
  union ringConnectInfo* rankInfo = NULL;
  union ncclSocketAddress* rankAddressesRoot = NULL; // for initial rank <-> root information exchange
  // get zeros for comparison
  char zeroHandle[NCCL_NET_HANDLE_MAXSIZE];
  union ncclSocketAddress zeroAddress;
  union ringConnectInfo zeroInfo;
  memset(&zeroAddress, 0, sizeof(union ncclSocketAddress));
  memset(&zeroHandle, 0, NCCL_NET_HANDLE_MAXSIZE);
  memset(&zeroInfo, 0, sizeof(union ringConnectInfo));
  ncclOsSetFilesLimit();

  TRACE(NCCL_BOOTSTRAP, "BEGIN");
  BOOTSTRAP_PROF_OPEN(timers[BOOTSTRAP_INIT_ROOT_WAIT]);
  /* Receive addresses from all ranks */
  do {
    struct ncclSocket sock;
    NCCLCHECKGOTO(ncclSocketInit(&sock), res, out);
    NCCLCHECKGOTO(ncclSocketAccept(&sock, listenSock), res, out);
    NCCLCHECKGOTO(socketRecv(&sock, &info, sizeof(info)), res, out);
    NCCLCHECKGOTO(ncclSocketClose(&sock), res, out);

    if (c == 0) {
      BOOTSTRAP_PROF_CLOSE(timers[BOOTSTRAP_INIT_ROOT_WAIT]);
      BOOTSTRAP_PROF_OPEN(timers[BOOTSTRAP_INIT_ROOT_RECV]);
      nranks = info.nranks;
      iroot = info.iroot;
      nroots = info.nroots;
      offset = info.offset;
      // if the number of root > 1, we will receive one extra info from the first local_id of the next root
      n2send = nRankFromRoot(iroot, nranks, nroots, offset);
      // offset>0 automatically means that we need to switch to the multiroot logic
      nrecv = n2send + ((offset > 0 || nroots > 1) ? 1 : 0);
      NCCLCHECKGOTO(ncclCalloc(&rankInfo, nrecv), res, out);
      NCCLCHECKGOTO(ncclCalloc(&rankAddressesRoot, nrecv), res, out);
    }

    if (nranks != info.nranks || nroots != info.nroots || iroot != info.iroot || offset != info.offset) {
      WARN("Bootstrap Root : mismatch in info from procs, nranks %d vs %d, nroots %d vs %d, iroot %d vs %d, offset %d "
           "vs %d",
           nranks, info.nranks, nroots, info.nroots, iroot, info.iroot, offset, info.offset);
      goto out;
    }

    localId = localIdFromRoot(info.rank, iroot, nranks, nroots, offset);
    if (localId < 0 || localId >= nrecv) {
      WARN("Bootstrap Root : localId %d is out of range", localId);
      goto out;
    }
    if (memcmp(&zeroAddress, &rankAddressesRoot[localId], sizeof(union ncclSocketAddress)) != 0 ||
        memcmp(&zeroInfo, &rankInfo[localId], sizeof(union ringConnectInfo)) != 0) {
      WARN("Bootstrap Root : rank %d of %d ranks has already checked in", info.rank, nranks);
      goto out;
    }
    // if the previous has already checked in, send the newly received handle, if not save the handle for later
    // if we have more than 1 root, I do not own the previous of local_id = 0
    // if we have prev > n2send, we do not send anything
    int prev = (nroots > 1) ? (localId - 1) : BOOTSTRAP_PID(localId - 1, nrecv);
    if (prev >= 0 && prev < n2send &&
        memcmp(&zeroAddress, &rankAddressesRoot[prev], sizeof(union ncclSocketAddress)) != 0) {
      NCCLCHECKGOTO(rootSend(&rankAddressesRoot[prev], magic, &info.connectInfo), res, out);
    } else {
      memcpy(&rankInfo[localId], &info.connectInfo, sizeof(union ringConnectInfo));
    }
    // if the next rank has checked in, send the newly received info, if not save the addr for later
    // for nroots >=1, I will always own the information of the next connection
    // if the local_id id must be [0 ; n2send[ otherwise we do not answer
    int next = BOOTSTRAP_PID(localId + 1, nrecv);
    if (localId >= 0 && localId < n2send && memcmp(&zeroInfo, &rankInfo[next], sizeof(union ringConnectInfo)) != 0) {
      NCCLCHECKGOTO(rootSend(&info.listenRootAddress, magic, &rankInfo[next]), res, out);
    } else {
      memcpy(rankAddressesRoot + localId, &info.listenRootAddress, sizeof(union ncclSocketAddress));
    }
    ++c;
    TRACE(NCCL_BOOTSTRAP, "Received connect from rank %d total %d/%d", info.rank, c, nrecv);
  } while (c < nrecv);
  TRACE(NCCL_BOOTSTRAP, "COLLECTED ALL %d HANDLES", nrecv);
  BOOTSTRAP_PROF_CLOSE(timers[BOOTSTRAP_INIT_ROOT_RECV]);

  // send the remaining info to the ranks who haven't received anything
  BOOTSTRAP_PROF_OPEN(timers[BOOTSTRAP_INIT_ROOT_SEND]);
  // here we need to send info only to my own local process
  for (int r = 0; r < n2send; ++r) {
    // use nrecv to periodize: if 1 root, we will send the first one to the last one,
    // if >1 roots we will send the additional one we have received
    int next = BOOTSTRAP_PID(r + 1, nrecv);
    if (memcmp(&zeroAddress, &rankAddressesRoot[r], sizeof(union ncclSocketAddress)) != 0 &&
        memcmp(&zeroInfo, &rankInfo[next], sizeof(union ringConnectInfo)) != 0) {
      NCCLCHECKGOTO(rootSend(&rankAddressesRoot[r], magic, &rankInfo[next]), res, out);
    }
  }
  BOOTSTRAP_PROF_CLOSE(timers[BOOTSTRAP_INIT_ROOT_SEND]);
  TRACE(NCCL_BOOTSTRAP | NCCL_PROFILE, "Root timings (wait %f, recv %f, send %f)",
        timers[BOOTSTRAP_INIT_ROOT_WAIT] / 1e9, timers[BOOTSTRAP_INIT_ROOT_RECV] / 1e9,
        timers[BOOTSTRAP_INIT_ROOT_SEND] / 1e9);
out:
  if (listenSock != NULL) {
    (void)ncclSocketClose(listenSock);
    free(listenSock);
  }
  if (rankInfo) free(rankInfo);
  if (rankAddressesRoot) free(rankAddressesRoot);
  free(rargs);

  TRACE(NCCL_BOOTSTRAP, "DONE");
  return NULL;
}

ncclResult_t bootstrapCreateRoot(struct ncclBootstrapHandle* handle, bool idFromEnv) {
  ncclResult_t ret = ncclSuccess;
  struct ncclSocket* listenSock = NULL;
  struct bootstrapRootArgs* args = NULL;
  std::thread thread;

  NCCLCHECK(ncclCalloc(&listenSock, 1));
  NCCLCHECKGOTO(ncclSocketInit(listenSock, &handle->addr, handle->magic, ncclSocketTypeBootstrap, NULL, 0), ret, fail);
  NCCLCHECKGOTO(ncclSocketListen(listenSock), ret, fail);
  NCCLCHECKGOTO(ncclSocketGetAddr(listenSock, &handle->addr), ret, fail);

  NCCLCHECKGOTO(ncclCalloc(&args, 1), ret, fail);
  args->listenSock = listenSock;
  args->magic = handle->magic;
  thread = std::thread(bootstrapRoot, args);
  ncclSetThreadName(thread, "NCCL BootstrapR");
  thread.detach();
exit:
  return ret;
fail:
  if (listenSock) free(listenSock);
  if (args) free(args);
  goto exit;
}

ncclResult_t bootstrapReshapeCreateRoot(struct ncclBootstrapHandle* handle, ncclComm_t comm) {
  ncclResult_t ret = ncclSuccess;
  struct ncclSocket* listenSock = NULL;
  struct ncclReshapeLeaderThreadArgs* args = NULL;
  struct ncclReshapeLeaderState* state = comm ? comm->reshapeLeader : NULL;

  if (handle == NULL || comm == NULL || state == NULL) return ncclInvalidArgument;
  memset(handle, 0, sizeof(*handle));
  handle->magic = comm->magic;
  handle->nRanks = comm->nRanks;
  handle->flags = NCCL_UNIQUE_ID_RESHAPE;
  handle->leaderRank = comm->rank;
  memcpy(&handle->addr, &bootstrapNetIfAddr, sizeof(union ncclSocketAddress));
  if (state->listenSock != NULL) {
    memcpy(&handle->addr, &state->listenSock->addr, sizeof(handle->addr));
    return ncclSuccess;
  }
  if (state->thread != NULL && state->thread->joinable()) {
    state->thread->join();
    ncclResult_t threadResult = COMPILER_ATOMIC_LOAD(&state->threadResult, std::memory_order_acquire);
    if (threadResult != ncclSuccess && threadResult != ncclInProgress) return threadResult;
    delete state->thread;
    state->thread = NULL;
  }
  COMPILER_ATOMIC_STORE(&state->threadStop, uint32_t(0), std::memory_order_release);
  COMPILER_ATOMIC_STORE(&state->threadResult, ncclInProgress, std::memory_order_release);

  NCCLCHECKGOTO(ncclCalloc(&listenSock, 1), ret, fail);
  NCCLCHECKGOTO(ncclSocketInit(listenSock, &handle->addr, handle->magic, ncclSocketTypeJoinRoot, comm->abortFlag, 1),
                ret, fail);
  NCCLCHECKGOTO(ncclSocketListen(listenSock), ret, fail);
  NCCLCHECKGOTO(ncclSocketGetAddr(listenSock, &handle->addr), ret, fail);
  state->listenSock = listenSock;

  NCCLCHECKGOTO(ncclCalloc(&args, 1), ret, fail);
  args->listenSock = listenSock;
  args->comm = comm;
  args->state = state;
  NEW_NOTHROW_GOTO(state->thread, std::thread, ret, fail);
  STDTHREADCREATE_GOTO(*state->thread, ncclCommReshapeLeaderThread, ret, fail, args);
  ncclSetThreadName(*state->thread, "NCCL CommReshapeR");
exit:
  return ret;
fail:
  if (state) state->listenSock = NULL;
  if (listenSock) {
    (void)ncclSocketClose(listenSock);
    free(listenSock);
  }
  if (args) free(args);
  if (state && state->thread != NULL && !state->thread->joinable()) {
    delete state->thread;
    state->thread = NULL;
  }
  goto exit;
}

static ncclResult_t ncclCommReshapeAcceptCheckCompat(ncclComm_t comm, int rankHint) {
  if (rankHint < 0 || rankHint >= comm->nRanks) {
    WARN("ReshapeAccept rank hint %d is invalid for communicator size %d", rankHint, comm->nRanks);
    return ncclInvalidUsage;
  }
  /* TODO: validate topology, window, GIN, and transport compatibility for the target rank. */
  return ncclSuccess;
}

ncclResult_t bootstrapReshapeJoinerConnect(ncclComm_t* comm, ncclUniqueId uniqueId, int rankHint, int useExistingComm,
                                           const ncclConfig_t* commConfig, struct ncclSocket** sockOut) {
  struct ncclSocket* sock = NULL;
  struct ncclBootstrapHandle handle;
  ncclResult_t res = ncclSuccess;
  struct bootstrapJoinCommInfo commInfo;
  struct ncclPeerInfo* peerInfo = NULL;
  struct ncclJoinAllGatherInfo* allGatherData = NULL;
  void* topoXml = NULL;
  uint64_t joinerHostHash = 0;
  volatile uint32_t cancelFlag = 0;

  if (comm == NULL || sockOut == NULL) return ncclInvalidArgument;
  *sockOut = NULL;
  memset(&commInfo, 0, sizeof(commInfo));
  memcpy(&handle, &uniqueId, sizeof(handle));

  NCCLCHECKGOTO(ncclCalloc(&sock, 1), res, fail);
  NCCLCHECKGOTO(ncclSocketInit(sock, &handle.addr, handle.magic, ncclSocketTypeJoinRoot, &cancelFlag), res, fail);
  NCCLCHECKGOTO(ncclSocketConnect(sock), res, fail);
  NCCLCHECKGOTO(ncclSocketRecv(sock, &commInfo, sizeof(commInfo)), res, fail);
  joinerHostHash = getHostHash() + commInfo.commHash;
  NCCLCHECKGOTO(ncclSocketSend(sock, &rankHint, sizeof(rankHint)), res, fail);
  NCCLCHECKGOTO(ncclSocketSend(sock, &joinerHostHash, sizeof(joinerHostHash)), res, fail);

  if (commInfo.nranks <= 0 || commInfo.nChannels <= 0 || commInfo.topoXmlSize == 0 || rankHint < 0 ||
      rankHint >= commInfo.nranks) {
    WARN("ReshapeConnect received invalid deferred init info nranks %d nChannels %d topoXmlSize %lu rankHint %d",
         commInfo.nranks, commInfo.nChannels, commInfo.topoXmlSize, rankHint);
    res = ncclInvalidUsage;
    goto fail;
  }
  NCCLCHECKGOTO(ncclCalloc(&peerInfo, commInfo.nranks), res, fail);
  NCCLCHECKGOTO(ncclCalloc(&allGatherData, commInfo.nranks), res, fail);
  NCCLCHECKGOTO(ncclCalloc(&topoXml, commInfo.topoXmlSize), res, fail);
  NCCLCHECKGOTO(ncclSocketRecv(sock, peerInfo, sizeof(*peerInfo) * commInfo.nranks), res, fail);
  NCCLCHECKGOTO(ncclSocketRecv(sock, allGatherData, sizeof(*allGatherData) * commInfo.nranks), res, fail);
  NCCLCHECKGOTO(ncclSocketRecv(sock, topoXml, commInfo.topoXmlSize), res, fail);
  if (!useExistingComm) {
    NCCLCHECKGOTO(ncclCommJoinCreateFresh(comm, commInfo.nranks, rankHint, commInfo.magic, commInfo.commHash,
                                          commConfig, peerInfo, allGatherData, topoXml, commInfo.topoXmlSize),
                  res, fail);
  }

  if (comm == NULL || *comm == NULL) {
    WARN("ReshapeConnect did not create a pending communicator");
    res = ncclInternalError;
    goto fail;
  }
  sock->abortFlag = (*comm)->abortFlag;
  NCCLCHECKGOTO(bootstrapReshapeConnectSendRingData(*comm, rankHint, sock), res, fail);
  *sockOut = sock;
  sock = NULL;

exit:
  free(peerInfo);
  free(allGatherData);
  free(topoXml);
  return res;
fail:
  free(peerInfo);
  peerInfo = NULL;
  free(allGatherData);
  allGatherData = NULL;
  free(topoXml);
  topoXml = NULL;
  if (sock != NULL) {
    (void)ncclSocketClose(sock);
    free(sock);
    sock = NULL;
  }
  goto exit;
}

static ncclResult_t bootstrapReshapeLeaderWaitForCommit(ncclComm_t comm, struct ncclReshapeLeaderState* state, int rank,
                                                        struct ncclReshapeJoinerConn* conn) {
  ncclResult_t res = ncclSuccess;
  int joinerReady = 0;

  if (comm == NULL || state == NULL || conn == NULL || rank < 0 || rank >= comm->nRanks) return ncclInvalidArgument;
  NCCLCHECKGOTO(ncclSocketRecv(&conn->sock, &joinerReady, sizeof(joinerReady)), res, fail);
  if (joinerReady == 0) {
    WARN("Reshape joiner rank %d indicated it was not ready", rank);
    res = ncclRemoteError;
    goto fail;
  }
  COMPILER_ATOMIC_STORE(&conn->joinStatus, static_cast<int>(ncclJoinPeerStatusPeerReady), std::memory_order_release);

  while (COMPILER_ATOMIC_LOAD(&state->threadStop, std::memory_order_acquire) == 0 &&
         COMPILER_ATOMIC_LOAD(comm->abortFlag, std::memory_order_acquire) == 0) {
    int status = COMPILER_ATOMIC_LOAD(&conn->joinStatus, std::memory_order_acquire);
    if (status == ncclJoinPeerStatusLeaderCommitted) break;
    if (status == ncclJoinPeerStatusLeaderCanceled) {
      res = ncclRemoteError;
      goto exit;
    }
    std::this_thread::sleep_for(std::chrono::microseconds(1));
  }
  if (COMPILER_ATOMIC_LOAD(&state->threadStop, std::memory_order_acquire) != 0 ||
      COMPILER_ATOMIC_LOAD(comm->abortFlag, std::memory_order_acquire) != 0) {
    goto exit;
  }

  NCCLCHECKGOTO(bootstrapReshapeAcceptSendAllRingData(comm, &conn->sock), res, fail);

exit:
  (void)ncclSocketClose(&conn->sock);
  COMPILER_ATOMIC_STORE(&conn->joinStatus, static_cast<int>(ncclJoinPeerStatusUninit), std::memory_order_release);
  return res;
fail:
  if (res == ncclSuccess) res = ncclInternalError;
  goto exit;
}

static void bootstrapReshapeLeaderAcceptPeer(void* rargs) {
  struct ncclReshapeLeaderPeerThreadArgs* args = (struct ncclReshapeLeaderPeerThreadArgs*)rargs;
  ncclResult_t res = ncclSuccess;
  ncclComm_t comm = NULL;
  struct ncclReshapeLeaderState* state = NULL;
  struct ncclSocket* sock = NULL;
  struct bootstrapJoinCommInfo commInfo;
  int32_t rankHint = -1;
  uint64_t joinerHostHash = 0;
  struct ncclReshapeJoinerConn* conn = NULL;

  if (args == NULL) return;
  comm = args->comm;
  state = args->state;
  sock = args->sock;
  args->sock = NULL;
  if (comm == NULL || state == NULL || sock == NULL) {
    res = ncclInvalidArgument;
    goto fail;
  }
  sock->abortFlag = &state->threadStop;
  memset(&commInfo, 0, sizeof(commInfo));
  commInfo.nranks = comm->nRanks;
  commInfo.nChannels = comm->nChannels;
  commInfo.magic = comm->magic;
  commInfo.commHash = comm->commHash;
  commInfo.topoXmlSize = comm->joinInitData.topoXmlSize;
  NCCLCHECKGOTO(ncclSocketSend(sock, &commInfo, sizeof(commInfo)), res, fail);
  NCCLCHECKGOTO(ncclSocketRecv(sock, &rankHint, sizeof(rankHint)), res, fail);
  NCCLCHECKGOTO(ncclSocketRecv(sock, &joinerHostHash, sizeof(joinerHostHash)), res, fail);

  if (rankHint < 0 || rankHint >= comm->nRanks) {
    WARN("Invalid rankHint %d from reshape joiner", rankHint);
    res = ncclInvalidUsage;
    goto fail;
  }

  NCCLCHECKGOTO(ncclReshapeLeaderConnStart(state, rankHint, sock, &conn), res, fail);
  if (conn == NULL) {
    INFO(NCCL_INIT, "reshape joiner rank %d already has a pending connection", rankHint);
    goto exit;
  }
  conn->hostHash = joinerHostHash;
  free(sock);
  sock = NULL;

  // LAR-TODO-LATER: recv additional peer capability so we can check compatibility beyond rank placement.
  NCCLCHECKGOTO(ncclCommReshapeAcceptCheckCompat(comm, rankHint /*expect more args..*/), res, fail);

  if (comm->peerInfo == NULL || comm->joinInitData.allGatherData == NULL || comm->joinInitData.topoXml == NULL ||
      comm->joinInitData.topoXmlSize == 0) {
    WARN("ReshapeAccept missing peer, graph, or topology manifest information");
    res = ncclInvalidUsage;
    goto fail;
  }
  NCCLCHECKGOTO(ncclSocketSend(&conn->sock, comm->peerInfo, sizeof(*comm->peerInfo) * comm->nRanks), res, fail);
  NCCLCHECKGOTO(ncclSocketSend(&conn->sock, comm->joinInitData.allGatherData,
                               sizeof(*comm->joinInitData.allGatherData) * comm->nRanks),
                res, fail);
  NCCLCHECKGOTO(ncclSocketSend(&conn->sock, comm->joinInitData.topoXml, comm->joinInitData.topoXmlSize), res, fail);

  // LAR-TODO-LATER: we should send the joiner the windows we know about at this
  // time.  Then each time the survivors register or destroy a window, we should
  // clear that flag and send an update about the new window meta data, and mark
  // the peer not ready until the joiner confirms it again has matching windows.
  // This way we can more quickly verify that all joiners are ready without
  // waiting to exchange windows with each one.

  NCCLCHECKGOTO(ncclSocketRecv(&conn->sock, &conn->ringData, sizeof(conn->ringData)), res, fail);
  NCCLCHECKGOTO(bootstrapReshapeLeaderWaitForCommit(comm, state, rankHint, conn), res, fail);
  goto exit;

fail:
  if (conn != NULL) {
    (void)ncclSocketClose(&conn->sock);
    COMPILER_ATOMIC_STORE(&conn->joinStatus, static_cast<int>(ncclJoinPeerStatusUninit), std::memory_order_release);
  } else if (sock != NULL) {
    (void)ncclSocketClose(sock);
    free(sock);
    sock = NULL;
  }
exit:
  if (sock != NULL) {
    (void)ncclSocketClose(sock);
    free(sock);
    sock = NULL;
  }
  if (args != NULL) {
    free(args);
  }
  if (state != NULL) (void)COMPILER_ATOMIC_SUB_FETCH(&state->peerThreadCount, (uint32_t)1, std::memory_order_acq_rel);
  return;
}

static ncclResult_t bootstrapReshapeLeaderStartPeerThread(ncclComm_t comm, struct ncclReshapeLeaderState* state,
                                                          struct ncclSocket** sockPtr) {
  ncclResult_t res = ncclSuccess;
  struct ncclReshapeLeaderPeerThreadArgs* args = NULL;
  std::thread thread;
  bool counted = false;

  if (comm == NULL || state == NULL || sockPtr == NULL || *sockPtr == NULL) return ncclInvalidArgument;
  NCCLCHECKGOTO(ncclCalloc(&args, 1), res, fail);
  args->comm = comm;
  args->state = state;
  args->sock = *sockPtr;
  *sockPtr = NULL;
  (void)COMPILER_ATOMIC_FETCH_ADD(&state->peerThreadCount, (uint32_t)1, std::memory_order_acq_rel);
  counted = true;
  STDTHREADCREATE_GOTO(thread, bootstrapReshapeLeaderAcceptPeer, res, fail, args);
  ncclSetThreadName(thread, "NCCL CommReshapeP");
  thread.detach();
  return ncclSuccess;

fail:
  if (counted) (void)COMPILER_ATOMIC_SUB_FETCH(&state->peerThreadCount, (uint32_t)1, std::memory_order_acq_rel);
  if (args != NULL) {
    if (args->sock != NULL) {
      (void)ncclSocketClose(args->sock);
      free(args->sock);
    }
    free(args);
  }
  return res;
}

static ncclResult_t bootstrapReshapeLeaderSignalJoiners(struct ncclReshapeLeaderState* state, const int* joinList,
                                                        int joinListLen, ncclResult_t result) {
  ncclResult_t res = result;
  if (state == NULL || joinList == NULL || joinListLen <= 0) return ncclSuccess;
  for (int i = 0; i < joinListLen; i++) {
    int rank = joinList[i];
    struct ncclReshapeJoinerConn* conn = state->joiners + rank;
    if (result == ncclSuccess) {
      int expected = ncclJoinPeerStatusPeerReady;
      if (!COMPILER_ATOMIC_COMPARE_EXCHANGE(&conn->joinStatus, &expected,
                                            static_cast<int>(ncclJoinPeerStatusLeaderCommitted),
                                            std::memory_order_acq_rel, std::memory_order_acquire)) {
        WARN("bootstrapReshapeLeaderSignalJoiners: joiner rank %d status %d was not ready to commit", rank, expected);
        if (res == ncclSuccess) res = ncclRemoteError;
      }
    } else {
      if (COMPILER_ATOMIC_LOAD(&conn->joinStatus, std::memory_order_acquire) != ncclJoinPeerStatusUninit) {
        COMPILER_ATOMIC_STORE(&conn->joinStatus, static_cast<int>(ncclJoinPeerStatusLeaderCanceled),
                              std::memory_order_release);
        (void)ncclSocketClose(&conn->sock);
      }
    }
  }
  return res;
}

static ncclResult_t bootstrapReshapeLeaderWaitJoiners(struct ncclReshapeLeaderState* state, const int* joinList,
                                                      int joinListLen, ncclResult_t result) {
  ncclResult_t res = result;
  if (state == NULL || joinList == NULL || joinListLen <= 0) return ncclSuccess;
  for (int i = 0; i < joinListLen; i++) {
    int rank = joinList[i];
    struct ncclReshapeJoinerConn* conn = state->joiners + rank;
    while (COMPILER_ATOMIC_LOAD(&conn->joinStatus, std::memory_order_acquire) != ncclJoinPeerStatusUninit) {
      std::this_thread::sleep_for(std::chrono::microseconds(1));
    }
  }
  return res;
}

static void ncclCommReshapeLeaderThread(void* rargs) {
  struct ncclReshapeLeaderThreadArgs* args = (struct ncclReshapeLeaderThreadArgs*)rargs;
  ncclResult_t res = ncclSuccess;
  struct ncclSocket* sock = NULL;
  ncclComm_t comm = NULL;
  struct ncclReshapeLeaderState* state = NULL;

  if (args == NULL) return;
  comm = args->comm;
  state = args->state;
  if (comm == NULL || state == NULL) goto exit;

  INFO(NCCL_INIT, "comm %p rank %d reshape leader thread START", comm, comm->rank);
  while (COMPILER_ATOMIC_LOAD(&state->threadStop, std::memory_order_acquire) == 0 &&
         COMPILER_ATOMIC_LOAD(comm->abortFlag, std::memory_order_acquire) == 0) {
    int accepted = 0;
    if (sock == NULL) {
      NCCLCHECKGOTO(ncclCalloc(&sock, 1), res, fail);
      NCCLCHECKGOTO(ncclSocketInit(sock), res, fail);
    }
    NCCLCHECKGOTO(ncclSocketAccept(sock, args->listenSock), res, fail);
    NCCLCHECKGOTO(ncclSocketReady(sock, &accepted), res, fail);
    if (accepted) {
      NCCLCHECKGOTO(bootstrapReshapeLeaderStartPeerThread(comm, state, &sock), res, fail);
    }
    std::this_thread::sleep_for(std::chrono::microseconds(1));
  }

exit:
  if (sock != NULL) {
    (void)ncclSocketClose(sock);
    free(sock);
  }
  if (args && args->listenSock) {
    (void)ncclSocketClose(args->listenSock);
    free(args->listenSock);
    if (state && state->listenSock == args->listenSock) state->listenSock = NULL;
  }
  if (state) {
    if (res != ncclSuccess && res != ncclInProgress) {
      COMPILER_ATOMIC_STORE(&state->threadResult, res, std::memory_order_release);
    }
    INFO(NCCL_INIT, "comm %p rank %d reshape leader thread STOP result %d", comm, comm ? comm->rank : -1, res);
  }
  free(args);
  return;
fail:
  if (res == ncclSuccess) res = ncclInternalError;
  if (state) COMPILER_ATOMIC_STORE(&state->threadResult, res, std::memory_order_release);
  goto exit;
}

ncclResult_t bootstrapGetUniqueId(struct ncclBootstrapHandle* handle, struct ncclComm* comm) {
  memset(handle, 0, sizeof(ncclBootstrapHandle));

  const char* env = ncclGetEnv("NCCL_COMM_ID");
  if (env) {
    // If comm is provided (grow operation), NCCL_COMM_ID should not be set
    if (comm) {
      WARN("ncclCommGetUniqueId should not be called when NCCL_COMM_ID is set");
      return ncclInvalidUsage;
    }
    // Normal init: use NCCL_COMM_ID from environment
    INFO(NCCL_ENV, "NCCL_COMM_ID set by environment to %s", env);
    if (ncclSocketGetAddrFromString(&handle->addr, env) != ncclSuccess) {
      WARN("Invalid NCCL_COMM_ID, please use format: <ipv4>:<port> or [<ipv6>]:<port> or <hostname>:<port>");
      return ncclInvalidArgument;
    }
    handle->magic = NCCL_MAGIC;
    handle->flags = NCCL_UNIQUE_ID_DEFAULT;
    handle->leaderRank = -1;
  } else {
    if (comm) {
      // comm->childCount will be increment in ncclCommGrow for all existing ranks, use +1 here
      handle->magic = hashCombine(comm->magic, comm->childCount + 1);
    } else {
      NCCLCHECK(getRandomData(&handle->magic, sizeof(handle->magic)));
    }
    handle->nRanks = comm ? comm->nRanks : 0;
    handle->flags = NCCL_UNIQUE_ID_DEFAULT;
    handle->leaderRank = comm ? comm->rank : -1;
    memcpy(&handle->addr, &bootstrapNetIfAddr, sizeof(union ncclSocketAddress));
    NCCLCHECK(bootstrapCreateRoot(handle, false));
  }

  return ncclSuccess;
}

ncclResult_t bcastGrowHandle(struct ncclBootstrapHandle* handle, struct ncclComm* parent, bool isRoot) {
  if (!parent || !handle) {
    WARN("bcastGrowHandle: parent comm and handle must be provided");
    return ncclInvalidArgument;
  }

  // Single rank parent already has the handle, no need to broadcast
  if (parent->nRanks == 1) return ncclSuccess;
  if (isRoot) {
    NCCLCHECK(bootstrapSend(parent->bootstrap, 0, BOOTSTRAP_TAG_GROW_BOUNDARY, handle,
                            sizeof(struct ncclBootstrapHandle)));
    NCCLCHECK(bootstrapSend(parent->bootstrap, parent->nRanks - 1, BOOTSTRAP_TAG_GROW_BOUNDARY, handle,
                            sizeof(struct ncclBootstrapHandle)));
  } else {
    NCCLCHECK(bootstrapRecv(parent->bootstrap, -1, BOOTSTRAP_TAG_GROW_BOUNDARY, handle,
                            sizeof(struct ncclBootstrapHandle)));
  }

  return ncclSuccess;
}

struct unexConn {
  int peer;
  int tag;
  struct ncclSocket sock;
  struct unexConn* next;
};

struct bootstrapRing_t {
  union {
    struct {
      void *sendComm, *recvComm;
      ncclNetDeviceHandle_t *sendDevHandle, *recvDevHandle;
    } net;
    struct {
      struct ncclSocket recv;
      struct ncclSocket send;
    } socket;
  };
};
struct bootstrapListen_t {
  struct ncclSocket peerSocket; // socket for peers to contact me in P2P
  union {
    struct {
      int dev;
      void* comm;
      char handle[NCCL_NET_HANDLE_MAXSIZE];
    } net;
    struct ncclSocket socket; // socket to be used for the ring
  };
};

struct bootstrapState {
  struct bootstrapRing_t ring;
  struct bootstrapListen_t listen;
  ncclNet_t* net;
  union ringConnectInfo* ringAddresses;
  uint64_t* peerProxyAddressesUDS;
  union ncclSocketAddress* peerProxyAddresses;
  union ncclSocketAddress* peerP2pAddresses;
  struct rasRankInit* rasRanks;
  struct unexConn* unexpectedConnections;
  int cudaDev;
  int rank;
  int nranks;
  const ncclCommMaskValue_t* activeRankMask;
  bool ringConnected;
  uint64_t magic;
  volatile uint32_t* abortFlag;
};
#define STATE_RING(s, f) (s->ring.f)
#define STATE_LISTEN(s, f) (s->listen.f)

// helper functions
static ncclResult_t createListenSocket(struct ncclComm* comm, uint64_t magic, struct ncclSocket* socket,
                                       union ncclSocketAddress* addr, ncclSocketType type) {
  NCCLCHECK(ncclSocketInit(socket, &bootstrapNetIfAddr, magic, type, comm->abortFlag));
  NCCLCHECK(ncclSocketListen(socket));
  NCCLCHECK(ncclSocketGetAddr(socket, addr));
  return ncclSuccess;
}

static ncclResult_t bootstrapReshapeAcceptSendAllRingData(ncclComm_t comm, struct ncclSocket* sock) {
  ncclResult_t res = ncclSuccess;
  if (comm == NULL || comm->bootstrap == NULL || comm->activeRankMask == NULL) {
    WARN("bootstrapReshapeAcceptSendAllRingData: active mask is not ready");
    return ncclInternalError;
  }
  struct bootstrapState* bootstrap = (struct bootstrapState*)comm->bootstrap;
  union ringConnectInfo* ringAddresses = bootstrap->ringAddresses;
  union ncclSocketAddress* peerAddresses = bootstrap->peerP2pAddresses;
  union ncclSocketAddress* peerProxies = bootstrap->peerProxyAddresses;
  uint64_t* peerUDS = bootstrap->peerProxyAddressesUDS;
  struct rasRankInit* rasRanks = bootstrap->rasRanks;
  struct bootstrapRingDataPacked* ringData = NULL;
  int nRanks = comm->nRanks;

  NCCLCHECKGOTO(ncclCalloc(&ringData, nRanks), res, exit);

  // pack comm connection info
  NCCLCHECKGOTO(ringDataPackUnpack(true, ringData, nRanks, NULL, peerAddresses, peerProxies, peerUDS, rasRanks,
                                   ringAddresses),
                res, exit);
  // send comm connection info
  NCCLCHECKGOTO(ncclSocketSend(sock, ringData, sizeof(*ringData) * nRanks), res, exit);
  NCCLCHECKGOTO(ncclSocketSend(sock, comm->activeRankMask, sizeof(*comm->activeRankMask) * nRanks), res, exit);

exit:
  free(ringData);
  return res;
}

static ncclResult_t bootstrapReshapeConnectSendRingData(ncclComm_t comm, int rankHint, struct ncclSocket* sock) {
  ncclResult_t res = ncclSuccess;
  struct bootstrapState* bootstrap = comm ? (struct bootstrapState*)comm->bootstrap : NULL;
  union ringConnectInfo* ringAddresses = NULL;
  union ncclSocketAddress* peerAddresses = NULL;
  union ncclSocketAddress* peerProxies = NULL;
  uint64_t* peerUDS = NULL;
  struct rasRankInit* rasRanks = NULL;
  struct bootstrapRingDataPacked* ringData = NULL;
  int offset = rankHint;

  if (comm == NULL || bootstrap == NULL) {
    WARN("ReshapeConnect requires an existing communicator bootstrap state for ring-data exchange");
    return ncclInvalidUsage;
  }
  if (offset < 0 || offset >= bootstrap->nranks) {
    WARN("ReshapeConnect rank hint %d is invalid for bootstrap nranks %d", offset, bootstrap->nranks);
    return ncclInvalidUsage;
  }

  ringAddresses = bootstrap->ringAddresses;
  peerAddresses = bootstrap->peerP2pAddresses;
  peerProxies = bootstrap->peerProxyAddresses;
  peerUDS = bootstrap->peerProxyAddressesUDS;
  rasRanks = bootstrap->rasRanks;

  NCCLCHECKGOTO(ncclCalloc(&ringData, 1), res, exit);
  NCCLCHECKGOTO(ringDataPackUnpack(true, ringData, 1, &offset, peerAddresses, peerProxies, peerUDS, rasRanks,
                                   ringAddresses),
                res, exit);
  NCCLCHECKGOTO(ncclSocketSend(sock, ringData, sizeof(*ringData)), res, exit);

exit:
  free(ringData);
  return res;
}

static ncclResult_t bootstrapReshapeConnectRecvAllRingData(ncclComm_t comm, int nRanks, struct ncclSocket* sock) {
  ncclResult_t res = ncclSuccess;
  struct bootstrapState* bootstrap = comm ? (struct bootstrapState*)comm->bootstrap : NULL;
  union ringConnectInfo* ringAddresses = NULL;
  union ncclSocketAddress* peerAddresses = NULL;
  union ncclSocketAddress* peerProxies = NULL;
  uint64_t* peerUDS = NULL;
  struct rasRankInit* rasRanks = NULL;
  struct bootstrapRingDataPacked* ringData = NULL;
  if (comm == NULL || bootstrap == NULL) {
    WARN("ReshapeConnect requires an existing communicator bootstrap state to receive ring data");
    return ncclInvalidUsage;
  }
  if (nRanks <= 0 || nRanks != bootstrap->nranks) {
    WARN("ReshapeConnect ring-data count %d does not match bootstrap nranks %d", nRanks, bootstrap->nranks);
    return ncclInvalidUsage;
  }

  ringAddresses = bootstrap->ringAddresses;
  peerAddresses = bootstrap->peerP2pAddresses;
  peerProxies = bootstrap->peerProxyAddresses;
  peerUDS = bootstrap->peerProxyAddressesUDS;
  rasRanks = bootstrap->rasRanks;

  NCCLCHECKGOTO(ncclCalloc(&ringData, nRanks), res, exit);
  NCCLCHECKGOTO(ncclSocketRecv(sock, ringData, sizeof(*ringData) * nRanks), res, exit);
  NCCLCHECKGOTO(ringDataPackUnpack(false, ringData, nRanks, NULL, peerAddresses, peerProxies, peerUDS, rasRanks,
                                   ringAddresses),
                res, exit);

exit:
  free(ringData);
  return res;
}

static ncclResult_t getUDS(uint64_t* peerUDS) {
  uint64_t randId;
  NCCLCHECK(getRandomData(&randId, sizeof(randId)));
  *peerUDS = getPidHash() + randId;
  return ncclSuccess;
}

ncclResult_t bootstrapJoinInitLocalState(struct ncclComm* comm) {
  ncclResult_t result = ncclSuccess;
  int rank = comm ? comm->rank : -1;
  int nranks = comm ? comm->nRanks : 0;
  struct bootstrapState* state = NULL;
  struct ncclSocket* proxySocket = NULL;
  union ringConnectInfo ringInfo;

  if (comm == NULL || rank < 0 || rank >= nranks) return ncclInvalidArgument;
  memset(&ringInfo, 0, sizeof(ringInfo));

  NCCLCHECK(ncclCalloc(&state, 1));
  state->rank = rank;
  state->nranks = nranks;
  state->cudaDev = comm->cudaDev;
  state->abortFlag = comm->abortFlag;
  state->net = comm->ncclNet;
  state->activeRankMask = comm->activeRankMask;
  state->magic = comm->magic;
  comm->bootstrap = state;

  if (ncclParamBootstrapNetEnable()) {
    NCCLCHECKGOTO(netGetDevice(rank, comm, &STATE_LISTEN(state, net.dev)), result, fail);
    NCCLCHECKGOTO(state->net->listen(comm->netContext, STATE_LISTEN(state, net.dev), STATE_LISTEN(state, net.handle),
                                     &STATE_LISTEN(state, net.comm)),
                  result, fail);
    memcpy(ringInfo.handle, STATE_LISTEN(state, net.handle), NCCL_NET_HANDLE_MAXSIZE);
  } else {
    NCCLCHECKGOTO(createListenSocket(comm, comm->magic, &STATE_LISTEN(state, socket), &ringInfo.addr,
                                     ncclSocketTypeBootstrap),
                  result, fail);
  }

  NCCLCHECKGOTO(ncclCalloc(&state->ringAddresses, nranks), result, fail);
  memcpy(state->ringAddresses + rank, &ringInfo, sizeof(ringInfo));

  NCCLCHECKGOTO(ncclCalloc(&state->peerProxyAddresses, nranks), result, fail);
  NCCLCHECKGOTO(ncclCalloc(&proxySocket, 1), result, fail);
  NCCLCHECKGOTO(createListenSocket(comm, comm->magic, proxySocket, state->peerProxyAddresses + rank,
                                   ncclSocketTypeProxy),
                result, fail);

  NCCLCHECKGOTO(ncclCalloc(&state->peerProxyAddressesUDS, nranks), result, fail);
  NCCLCHECKGOTO(getUDS(state->peerProxyAddressesUDS + rank), result, fail);

  NCCLCHECKGOTO(ncclCalloc(&state->peerP2pAddresses, nranks), result, fail);
  NCCLCHECKGOTO(createListenSocket(comm, comm->magic, &STATE_LISTEN(state, peerSocket), state->peerP2pAddresses + rank,
                                   ncclSocketTypeBootstrap),
                result, fail);

  NCCLCHECKGOTO(ncclProxyInit(comm, proxySocket, state->peerProxyAddresses, state->peerProxyAddressesUDS), result,
                fail);
  proxySocket = NULL;

exit:
  if (proxySocket != NULL) {
    (void)ncclSocketClose(proxySocket);
    free(proxySocket);
  }
  return result;
fail:
  if (state != NULL) {
    if (comm->bootstrap == state) comm->bootstrap = NULL;
    free(state->peerProxyAddresses);
    state->peerProxyAddresses = NULL;
    free(state->peerProxyAddressesUDS);
    state->peerProxyAddressesUDS = NULL;
    (void)bootstrapClose(state);
  }
  goto exit;
}

#define MAX_OOB_DEVS 16
static ncclResult_t netGetDevice(int rank, struct ncclComm* comm, int* dev) {
  if (bootstrapNetDevOOB < 0) {
    std::lock_guard<std::mutex> lock(bootstrapNetMutex);
    if (bootstrapNetDevOOB < 0) {
      const char* userIfEnv = ncclGetEnv("NCCL_OOB_NET_IFNAME");
      if (userIfEnv && strlen(userIfEnv) > 0) {
        INFO(NCCL_BOOTSTRAP | NCCL_ENV, "NCCL_OOB_NET_IFNAME set to %s", userIfEnv);
        bool searchNot = userIfEnv && userIfEnv[0] == '^';
        if (searchNot) userIfEnv++;
        bool searchExact = userIfEnv && userIfEnv[0] == '=';
        if (searchExact) userIfEnv++;
        struct netIf userIfs[MAX_OOB_DEVS];
        int nUserIfs = parseStringList(userIfEnv, userIfs, MAX_OOB_DEVS);
        // loop over the device and return the first one matching
        int nDev = 0;
        NCCLCHECK(comm->ncclNet->devices(&nDev));
        int devId = 0;
        while (devId < nDev) {
          ncclNetProperties_t props;
          comm->ncclNet->getProperties(devId, &props);
          // check against user specified HCAs/ports
          if (matchIfList(props.name, props.port, userIfs, nUserIfs, searchExact) ^ searchNot) {
            // All plain physical devices have been initialized at this point
            bootstrapNetDevOOB = devId;
            break;
          }
          devId++;
        }
        if (bootstrapNetDevOOB == -1) {
          if (!searchNot) {
            WARN("no device found matching %s%s, verify NCCL_OOB_NET_IFNAME", searchExact ? "exactly " : "", userIfEnv);
          } else {
            WARN("no device found after excluding %s%s, verify NCCL_OOB_NET_IFNAME", searchExact ? "exactly " : "",
                 userIfEnv);
          }
          return ncclInvalidArgument;
        }
      } else {
        // default choice is device 0
        bootstrapNetDevOOB = 0;
      }
      // display info on the chosen device
      ncclNetProperties_t props;
      ncclResult_t res = comm->ncclNet->getProperties(bootstrapNetDevOOB, &props);
      bool hasProp = res == ncclSuccess;
      INFO(NCCL_BOOTSTRAP, "Bootstrap: Using %s:%d", (hasProp) ? props.name : "N/A", (hasProp) ? props.port : -1);
    }
  }
  *dev = bootstrapNetDevOOB;
  return ncclSuccess;
}

static ncclResult_t netRingConnect(void* ctx, ncclNet_t* net, struct bootstrapListen_t* listen,
                                   char peerHandle[NCCL_NET_HANDLE_MAXSIZE], void** sendComm,
                                   ncclNetDeviceHandle_t** sendDevHandle, void** recvComm,
                                   ncclNetDeviceHandle_t** recvDevHandle, volatile uint32_t* abortFlag) {
  int abortCounter = 0;
  do {
    NCCLCHECK(checkAbort(abortFlag, &abortCounter));
    if (!*sendComm) NCCLCHECK(net->connect(ctx, listen->net.dev, peerHandle, sendComm, sendDevHandle));
    if (!*recvComm) NCCLCHECK(net->accept(listen->net.comm, recvComm, recvDevHandle));
  } while (!*sendComm || !*recvComm);
  return ncclSuccess;
}
static ncclResult_t socketRingConnect(ncclSocketAddress* addr, struct ncclSocket* sendSocket,
                                      struct ncclSocket* listenSock, struct ncclSocket* recvSocket, uint64_t magic,
                                      volatile uint32_t* abortFlag) {
  NCCLCHECK(ncclSocketInit(sendSocket, addr, magic, ncclSocketTypeBootstrap, abortFlag));
  NCCLCHECK(ncclSocketConnect(sendSocket));
  NCCLCHECK(ncclSocketInit(recvSocket));
  NCCLCHECK(ncclSocketAccept(recvSocket, listenSock));
  return ncclSuccess;
}

static ncclResult_t bootstrapGetActiveRanks(struct bootstrapState* state, int** activeRanks, int* activeCount,
                                            int* activeIndex) {
  NCCLCHECK(ncclCalloc(activeRanks, state->nranks));
  *activeCount = 0;
  *activeIndex = -1;
  for (int rank = 0; rank < state->nranks; ++rank) {
    if (ncclRankMaskIsActive(state->activeRankMask, state->nranks, rank)) {
      if (rank == state->rank) *activeIndex = *activeCount;
      (*activeRanks)[(*activeCount)++] = rank;
    }
  }
  if (*activeIndex == -1) {
    WARN("bootstrapGetActiveRanks: local rank %d is inactive", state->rank);
    return ncclInvalidUsage;
  }
  return ncclSuccess;
}

static ncclResult_t bootstrapCloseRing(struct bootstrapState* state) {
  if (state == NULL || !state->ringConnected) return ncclSuccess;
  if (ncclParamBootstrapNetEnable()) {
    if (STATE_RING(state, net.sendComm) != NULL) {
      NCCLCHECK(state->net->closeSend(STATE_RING(state, net.sendComm)));
      STATE_RING(state, net.sendComm) = NULL;
      STATE_RING(state, net.sendDevHandle) = NULL;
    }
    if (STATE_RING(state, net.recvComm) != NULL) {
      NCCLCHECK(state->net->closeRecv(STATE_RING(state, net.recvComm)));
      STATE_RING(state, net.recvComm) = NULL;
      STATE_RING(state, net.recvDevHandle) = NULL;
    }
  } else {
    NCCLCHECK(ncclSocketClose(&STATE_RING(state, socket.send)));
    NCCLCHECK(ncclSocketClose(&STATE_RING(state, socket.recv)));
  }
  state->ringConnected = false;
  return ncclSuccess;
}

ncclResult_t bootstrapQuiesceLocalAddresses(struct ncclComm* comm) {
  struct bootstrapState* state = comm ? (struct bootstrapState*)comm->bootstrap : NULL;
  int activeCount = comm ? ncclRankMaskCountActive(comm->activeRankMask, comm->nRanks) : 0;

  if (comm == NULL || state == NULL) return ncclInvalidArgument;
  if (activeCount != 1 || !ncclRankMaskIsActive(comm->activeRankMask, comm->nRanks, comm->rank)) {
    WARN("bootstrapQuiesceLocalAddresses: comm must be masked to local rank only, active count %d rank %d",
         activeCount, comm->rank);
    return ncclInvalidUsage;
  }
  if (comm->proxyState != NULL || (comm->sharedRes != NULL && comm->sharedRes->proxyState != NULL)) {
    WARN("bootstrapQuiesceLocalAddresses: proxy state must be quiesced first");
    return ncclInvalidUsage;
  }
  NCCLCHECK(bootstrapCloseRing(state));
  if (ncclParamBootstrapNetEnable()) {
    if (STATE_LISTEN(state, net.comm) != NULL) {
      NCCLCHECK(state->net->closeListen(STATE_LISTEN(state, net.comm)));
      STATE_LISTEN(state, net.comm) = NULL;
    }
  } else {
    NCCLCHECK(ncclSocketClose(&STATE_LISTEN(state, socket)));
  }
  NCCLCHECK(ncclSocketClose(&STATE_LISTEN(state, peerSocket)));

  INFO(NCCL_INIT, "comm %p rank %d nRanks %d - BootstrapQuiesce COMPLETE", comm, comm->rank, comm->nRanks);
  return ncclSuccess;
}

ncclResult_t bootstrapRediscoverLocalAddresses(struct ncclComm* comm) {
  ncclResult_t result = ncclSuccess;
  struct bootstrapState* state = comm ? (struct bootstrapState*)comm->bootstrap : NULL;
  union ncclSocketAddress* newPeerProxyAddresses = NULL;
  uint64_t* newPeerProxyAddressesUDS = NULL;
  union ncclSocketAddress* oldPeerProxyAddresses = NULL;
  uint64_t* oldPeerProxyAddressesUDS = NULL;
  struct ncclSocket* proxySocket = NULL;
  union ringConnectInfo ringInfo;
  int rank = comm ? comm->rank : -1;
  int nranks = comm ? comm->nRanks : 0;
  int activeCount = comm ? ncclRankMaskCountActive(comm->activeRankMask, comm->nRanks) : 0;

  memset(&ringInfo, 0, sizeof(ringInfo));
  if (comm == NULL || state == NULL || rank < 0 || rank >= nranks) return ncclInvalidArgument;
  if (activeCount != 1 || !ncclRankMaskIsActive(comm->activeRankMask, comm->nRanks, rank)) {
    WARN("bootstrapRediscoverLocalAddresses: comm must be masked to local rank only, active count %d rank %d",
         activeCount, rank);
    return ncclInvalidUsage;
  }
  if (comm->sharedRes == NULL || comm->sharedRes->owner != comm || comm->proxyState != NULL ||
      comm->sharedRes->proxyState != NULL) {
    WARN("bootstrapRediscoverLocalAddresses: proxy state must be quiesced before rediscovery");
    return ncclInvalidUsage;
  }
  if (state->ringAddresses == NULL || state->peerP2pAddresses == NULL || state->peerProxyAddresses == NULL ||
      state->peerProxyAddressesUDS == NULL) {
    WARN("bootstrapRediscoverLocalAddresses: bootstrap address storage is incomplete");
    return ncclInternalError;
  }

  NCCLCHECKGOTO(bootstrapNetRediscover(), result, fail);
  NCCLCHECKGOTO(bootstrapCloseRing(state), result, fail);
  if (ncclParamBootstrapNetEnable()) {
    if (STATE_LISTEN(state, net.comm) != NULL) {
      NCCLCHECKGOTO(state->net->closeListen(STATE_LISTEN(state, net.comm)), result, fail);
      STATE_LISTEN(state, net.comm) = NULL;
    }
  }

  NCCLCHECKGOTO(ncclNetRediscover(comm), result, fail);

  state->net = comm->ncclNet;
  state->activeRankMask = comm->activeRankMask;
  state->abortFlag = comm->abortFlag;
  state->magic = comm->magic;

  if (ncclParamBootstrapNetEnable()) {
    NCCLCHECKGOTO(netGetDevice(rank, comm, &STATE_LISTEN(state, net.dev)), result, fail);
    NCCLCHECKGOTO(state->net->listen(comm->netContext, STATE_LISTEN(state, net.dev), STATE_LISTEN(state, net.handle),
                                     &STATE_LISTEN(state, net.comm)),
                  result, fail);
    memcpy(ringInfo.handle, STATE_LISTEN(state, net.handle), NCCL_NET_HANDLE_MAXSIZE);
  } else {
    NCCLCHECKGOTO(ncclSocketClose(&STATE_LISTEN(state, socket)), result, fail);
    NCCLCHECKGOTO(createListenSocket(comm, comm->magic, &STATE_LISTEN(state, socket), &ringInfo.addr,
                                     ncclSocketTypeBootstrap),
                  result, fail);
  }
  memcpy(state->ringAddresses + rank, &ringInfo, sizeof(ringInfo));

  NCCLCHECKGOTO(ncclSocketClose(&STATE_LISTEN(state, peerSocket)), result, fail);
  NCCLCHECKGOTO(createListenSocket(comm, comm->magic, &STATE_LISTEN(state, peerSocket), state->peerP2pAddresses + rank,
                                   ncclSocketTypeBootstrap),
                result, fail);

  NCCLCHECKGOTO(ncclCalloc(&newPeerProxyAddresses, nranks), result, fail);
  memcpy(newPeerProxyAddresses, state->peerProxyAddresses, sizeof(*newPeerProxyAddresses) * nranks);
  NCCLCHECKGOTO(ncclCalloc(&newPeerProxyAddressesUDS, nranks), result, fail);
  memcpy(newPeerProxyAddressesUDS, state->peerProxyAddressesUDS, sizeof(*newPeerProxyAddressesUDS) * nranks);
  NCCLCHECKGOTO(ncclCalloc(&proxySocket, 1), result, fail);
  NCCLCHECKGOTO(createListenSocket(comm, comm->magic, proxySocket, newPeerProxyAddresses + rank, ncclSocketTypeProxy),
                result, fail);
  NCCLCHECKGOTO(getUDS(newPeerProxyAddressesUDS + rank), result, fail);

  oldPeerProxyAddresses = state->peerProxyAddresses;
  oldPeerProxyAddressesUDS = state->peerProxyAddressesUDS;
  NCCLCHECKGOTO(ncclProxyInit(comm, proxySocket, newPeerProxyAddresses, newPeerProxyAddressesUDS), result, fail);
  state->peerProxyAddresses = newPeerProxyAddresses;
  state->peerProxyAddressesUDS = newPeerProxyAddressesUDS;
  proxySocket = NULL;
  newPeerProxyAddresses = NULL;
  newPeerProxyAddressesUDS = NULL;
  free(oldPeerProxyAddresses);
  oldPeerProxyAddresses = NULL;
  free(oldPeerProxyAddressesUDS);
  oldPeerProxyAddressesUDS = NULL;
  NCCLCHECKGOTO(ncclProxyCreate(comm), result, fail);

  if (state->rasRanks != NULL) {
    memcpy(&state->rasRanks[rank].addr, &bootstrapNetIfAddr, sizeof(state->rasRanks[rank].addr));
    state->rasRanks[rank].pid = ncclOsGetPid();
    state->rasRanks[rank].cudaDev = comm->cudaDev;
    state->rasRanks[rank].nvmlDev = comm->nvmlDev;
    state->rasRanks[rank].hostHash = getHostHash();
    state->rasRanks[rank].pidHash = getPidHash();
  }

exit:
  if (proxySocket != NULL) {
    (void)ncclSocketClose(proxySocket);
    free(proxySocket);
  }
  free(newPeerProxyAddresses);
  free(newPeerProxyAddressesUDS);
  return result;
fail:
  goto exit;
}

static ncclResult_t bootstrapReconnectRing(struct ncclComm* comm, struct bootstrapState* state,
                                           const ncclCommMaskValue_t* mask) {
  int nextRank = comm->rank;
  int activeCount = ncclRankMaskCountActive(mask, comm->nRanks);

  if (activeCount <= 1 || !ncclRankMaskIsActive(mask, comm->nRanks, comm->rank)) {
    NCCLCHECK(bootstrapCloseRing(state));
    return ncclSuccess;
  }
  do {
    nextRank = (nextRank + 1) % comm->nRanks;
  } while (!ncclRankMaskIsActive(mask, comm->nRanks, nextRank));

  NCCLCHECK(bootstrapCloseRing(state));
  if (ncclParamBootstrapNetEnable()) {
    NCCLCHECK(netRingConnect(comm->netContext, state->net, &state->listen, state->ringAddresses[nextRank].handle,
                             &STATE_RING(state, net.sendComm), &STATE_RING(state, net.sendDevHandle),
                             &STATE_RING(state, net.recvComm), &STATE_RING(state, net.recvDevHandle),
                             state->abortFlag));
  } else {
    NCCLCHECK(socketRingConnect(&state->ringAddresses[nextRank].addr, &STATE_RING(state, socket.send),
                                &STATE_LISTEN(state, socket), &STATE_RING(state, socket.recv), state->magic,
                                state->abortFlag));
  }
  state->ringConnected = true;
  return ncclSuccess;
}

static ncclResult_t bootstrapInstallRankMask(struct ncclComm* comm, struct bootstrapState* state,
                                             const ncclCommMaskValue_t* mask) {
  bool localActive = ncclRankMaskIsActive(mask, comm->nRanks, comm->rank);
  for (int rank = 0; rank < comm->nRanks; rank++) {
    if (rank != comm->rank && ncclCommIsRankActive(comm, rank) &&
        (!localActive || !ncclRankMaskIsActive(mask, comm->nRanks, rank))) {
      NCCLCHECK(ncclTransportClosePeer(comm, rank));
    }
  }
  memcpy(comm->activeRankMask, mask, sizeof(*comm->activeRankMask) * comm->nRanks);
  state->activeRankMask = comm->activeRankMask;
  return ncclSuccess;
}

ncclResult_t bootstrapReshapeJoinerComplete(ncclComm_t comm, struct ncclSocket** sockPtr) {
  struct bootstrapState* state = NULL;
  struct ncclSocket* sock = sockPtr ? *sockPtr : NULL;
  ncclCommMaskValue_t* newMask = NULL;
  ncclResult_t res = ncclSuccess;
  int isReady = 1;

  if (comm == NULL || comm->bootstrap == NULL || sock == NULL) {
    WARN("bootstrapReshapeJoinerComplete: missing communicator bootstrap or join socket");
    res = ncclInvalidUsage;
    goto fail;
  }
  state = (struct bootstrapState*)comm->bootstrap;

  NCCLCHECKGOTO(ncclSocketSend(sock, &isReady, sizeof(isReady)), res, fail);
  NCCLCHECKGOTO(bootstrapReshapeConnectRecvAllRingData(comm, comm->nRanks, sock), res, fail);

  NCCLCHECKGOTO(ncclCalloc(&newMask, comm->nRanks), res, fail);
  NCCLCHECKGOTO(ncclSocketRecv(sock, newMask, sizeof(*newMask) * comm->nRanks), res, fail);
  NCCLCHECKGOTO(ncclRankMaskValidate(newMask, comm->nRanks), res, fail);

  memcpy(comm->activeRankMask, newMask, sizeof(*newMask) * comm->nRanks);
  state->activeRankMask = comm->activeRankMask;
  NCCLCHECKGOTO(bootstrapReconnectRing(comm, state, newMask), res, fail);
  NCCLCHECKGOTO(ncclDevrJoinFinalizeGin(comm), res, fail);
  NCCLCHECKGOTO(ncclDevrJoinExchangeWindows(comm), res, fail);
  comm->joinDeferred = false;
  NCCLCHECKGOTO(ncclTransportCollectiveConnect(comm), res, fail);

exit:
  free(newMask);
  if (sock != NULL) {
    (void)ncclSocketClose(sock);
    free(sock);
    if (sockPtr != NULL) *sockPtr = NULL;
  }
  return res;
fail:
  goto exit;
}

ncclResult_t bootstrapReshapeSurvivorsCommit(ncclComm_t comm, struct ncclReshapeLeaderState* request, int leaderRank,
                                             const int* joinList, int joinListLen, const int* leaveList,
                                             int leaveListLen, int flags) {
  struct bootstrapState* state = NULL;
  struct bootstrapRingDataPacked* ringData = NULL;
  ncclCommMaskValue_t* newMask = NULL;
  ncclResult_t res = ncclSuccess;
  bool isLocal = flags == NCCL_COMM_RESHAPE_LOCAL_ONLY;
  bool isLeader = false;

  if (comm == NULL || comm->bootstrap == NULL) {
    WARN("bootstrapReshapeSurvivorsCommit: missing communicator bootstrap");
    res = ncclInternalError;
    goto fail;
  }
  state = (struct bootstrapState*)comm->bootstrap;
  isLeader = !isLocal && comm->rank == leaderRank;

  NCCLCHECKGOTO(ncclCalloc(&newMask, comm->nRanks), res, fail);
  if (isLocal) {
    memcpy(newMask, comm->activeRankMask, sizeof(*newMask) * comm->nRanks);
    for (int i = 0; i < leaveListLen; i++) {
      newMask[leaveList[i]] = ncclRankMaskInactive;
    }
  } else {
    NCCLCHECKGOTO(ncclCalloc(&ringData, comm->nRanks), res, fail);
  }
  if (isLeader) {
    if (joinListLen > 0 && request == NULL) {
      WARN("bootstrapReshapeSurvivorsCommit: leader has no reshape ID state");
      res = ncclInvalidUsage;
      goto fail;
    }
    memcpy(newMask, comm->activeRankMask, sizeof(*newMask) * comm->nRanks);
    for (int i = 0; i < leaveListLen; i++) {
      newMask[leaveList[i]] = ncclRankMaskInactive;
    }
    for (int i = 0; i < joinListLen; i++) {
      newMask[joinList[i]] = ncclRankMaskActive;
    }
    for (int i = 0; i < joinListLen; i++) {
      int rank = joinList[i];
      int offset = rank;
      struct ncclReshapeJoinerConn* conn = request->joiners + rank;
      // unpack from this peer into bootstrap state
      NCCLCHECKGOTO(ringDataPackUnpack(false, &conn->ringData, 1, &offset, state->peerP2pAddresses,
                                       state->peerProxyAddresses, state->peerProxyAddressesUDS, state->rasRanks,
                                       state->ringAddresses),
                    res, fail);
    }
    // pack from bootstrap state into ringData buffer for joiners
    NCCLCHECKGOTO(ringDataPackUnpack(true, ringData, comm->nRanks, NULL, state->peerP2pAddresses,
                                     state->peerProxyAddresses, state->peerProxyAddressesUDS, state->rasRanks,
                                     state->ringAddresses),
                  res, fail);
  }

  if (!isLocal) {
    NCCLCHECKGOTO(bootstrapBroadcast(comm->bootstrap, comm->rank, comm->nRanks, leaderRank, ringData,
                                     sizeof(*ringData) * comm->nRanks),
                  res, fail);
    NCCLCHECKGOTO(bootstrapBroadcast(comm->bootstrap, comm->rank, comm->nRanks, leaderRank, newMask,
                                     sizeof(ncclCommMaskValue_t) * comm->nRanks),
                  res, fail);
    if (!isLeader) {
      NCCLCHECKGOTO(ringDataPackUnpack(false, ringData, comm->nRanks, NULL, state->peerP2pAddresses,
                                       state->peerProxyAddresses, state->peerProxyAddressesUDS, state->rasRanks,
                                       state->ringAddresses),
                    res, fail);
    }
  }

  NCCLCHECKGOTO(bootstrapInstallRankMask(comm, state, newMask), res, fail);
  if (isLeader)
    NCCLCHECKGOTO(bootstrapReshapeLeaderSignalJoiners(request, joinList, joinListLen, ncclSuccess), res, fail);
  if (!ncclCommIsRankActive(comm, comm->rank)) {
    for (int rank = 0; rank < comm->nRanks; rank++) newMask[rank] = ncclRankMaskInactive;
    NCCLCHECKGOTO(bootstrapInstallRankMask(comm, state, newMask), res, fail);
    NCCLCHECKGOTO(bootstrapCloseRing(state), res, fail);
    if (isLeader)
      NCCLCHECKGOTO(bootstrapReshapeLeaderWaitJoiners(request, joinList, joinListLen, ncclSuccess), res, fail);
    goto exit;
  }
  NCCLCHECKGOTO(bootstrapReconnectRing(comm, state, comm->activeRankMask), res, fail);
  NCCLCHECKGOTO(ncclDevrJoinFinalizeGin(comm), res, fail);
  NCCLCHECKGOTO(ncclDevrJoinExchangeWindows(comm), res, fail);
  NCCLCHECKGOTO(ncclTransportCollectiveConnect(comm), res, fail);
  if (isLeader)
    NCCLCHECKGOTO(bootstrapReshapeLeaderWaitJoiners(request, joinList, joinListLen, ncclSuccess), res, fail);
  /* TODO: leader should add new ras ranks. */

exit:
  free(ringData);
  free(newMask);
  return res;
fail:
  if (isLeader) {
    (void)bootstrapReshapeLeaderSignalJoiners(request, joinList, joinListLen, res);
    (void)bootstrapReshapeLeaderWaitJoiners(request, joinList, joinListLen, res);
  }
  goto exit;
}

ncclResult_t bootstrapUpdateRankMask(void* commState) {
  if (commState == NULL) return ncclSuccess;
  struct bootstrapState* state = (struct bootstrapState*)commState;
  int activeCount = ncclRankMaskCountActive(state->activeRankMask, state->nranks);
  if (activeCount <= 0) {
    WARN("bootstrapUpdateRankMask: communicator has no active bootstrap ranks");
    return ncclInvalidUsage;
  }
  if (!ncclRankMaskIsActive(state->activeRankMask, state->nranks, state->rank)) {
    NCCLCHECK(bootstrapCloseRing(state));
    return ncclSuccess;
  }
  if (activeCount != state->nranks) NCCLCHECK(bootstrapCloseRing(state));
  return ncclSuccess;
}

static ncclResult_t ringAllInfo(struct ncclComm* comm, struct bootstrapState* state,
                                union ncclSocketAddress* peerAddresses, union ncclSocketAddress* peerProxy,
                                uint64_t* peerUDS, struct rasRankInit* rasRanks, union ringConnectInfo* ringAddresses) {
  ncclResult_t res = ncclSuccess;
  int rank = comm->rank;
  int nRanks = comm->nRanks;
  struct bootstrapRingDataPacked* ringData = NULL;

  NCCLCHECK(ncclCalloc(&ringData, nRanks));
  // pack our own data into the buffer at offset +rank.
  NCCLCHECKGOTO(ringDataPackUnpack(true, ringData + rank, 1, &rank, peerAddresses, peerProxy, peerUDS, rasRanks,
                                   ringAddresses),
                res, exit);

  // allgather
  NCCLCHECKGOTO(bootstrapAllGather(state, ringData, sizeof(struct bootstrapRingDataPacked)), res, exit);

  // unpack all data
  NCCLCHECKGOTO(ringDataPackUnpack(false, ringData, nRanks, NULL, peerAddresses, peerProxy, peerUDS, rasRanks,
                                   ringAddresses),
                res, exit);

exit:
  free(ringData);
  return res;
}

static ncclResult_t sendToRoot(struct ncclBootstrapHandle* handle, struct ncclComm* comm, struct extInfo* info) {
  ncclResult_t ret = ncclSuccess;
  struct ncclSocket sock;
  NCCLCHECK(ncclSocketInit(&sock, &handle->addr, handle->magic, ncclSocketTypeBootstrap, comm->abortFlag));
  NCCLCHECKGOTO(ncclSocketConnect(&sock), ret, fail);
  NCCLCHECKGOTO(socketSend(&sock, info, sizeof(struct extInfo)), ret, fail);
  NCCLCHECK(ncclSocketClose(&sock));
  return ret;
fail:
  (void)ncclSocketClose(&sock);
  return ret;
}

NCCL_PARAM(StaggerRate, "UID_STAGGER_RATE", 7000);
NCCL_PARAM(StaggerThreshold, "UID_STAGGER_THRESHOLD", 256);
extern int64_t ncclParamRasEnable();

ncclResult_t bootstrapInit(int nHandles, void* handles, struct ncclComm* comm, struct ncclComm* parent) {
  ncclResult_t result = ncclSuccess;
  int rank = comm->rank;
  int nranks = comm->nRanks;
  // char nextPeerHandle[NCCL_NET_HANDLE_MAXSIZE];
  struct bootstrapState* state;
  struct ncclSocket* proxySocket = NULL;
  struct ncclSocket sock, listenSockRoot;
  struct extInfo info = {0};
  union ringConnectInfo nextPeer;
  bool performRasAddRanks = true;
  struct rasRankInit* rasRanks = nullptr;

  uint64_t timers[BOOTSTRAP_INIT_TIME_N] = {0};

  NCCLCHECK(ncclCalloc(&state, 1));
  state->rank = rank;
  state->nranks = nranks;
  state->cudaDev = comm->cudaDev;
  state->abortFlag = comm->abortFlag;
  state->net = comm->ncclNet;
  state->activeRankMask = comm->activeRankMask;
  comm->bootstrap = state;

  // Set magic: for grow existing ranks, receive from coordinator; otherwise use handle magic.
  // This is consistent with the magic created in ncclCommGetUniqueId.
  if (handles != NULL) {
    // state and comm magic set to the first magic ID
    comm->magic = state->magic = BOOTSTRAP_HANDLE(handles, 0)->magic;
  } else if (parent != NULL) {
    comm->magic = state->magic = hashCombine(parent->magic, parent->childCount);
  } else {
    WARN("bootstrapInit: handles and parent are NULL");
    return ncclSystemError;
  }

  TRACE(NCCL_BOOTSTRAP, "rank %d nranks %d", rank, nranks);

  BOOTSTRAP_PROF_OPEN(timers[BOOTSTRAP_INIT_TIME_TOTAL]);
  // fill up the info
  info.nranks = nranks;
  info.nroots = nHandles;
  // get the ring connection info
  memset(&nextPeer, 0, sizeof(union ringConnectInfo));
  BOOTSTRAP_PROF_OPEN(timers[BOOTSTRAP_INIT_TIME_CREATE]);
  if (ncclParamBootstrapNetEnable()) {
    // Create net interface for other ranks to contact me (all gather)
    NCCLCHECK(netGetDevice(rank, comm, &STATE_LISTEN(state, net.dev)));
    NCCLCHECK(state->net->listen(comm->netContext, STATE_LISTEN(state, net.dev), STATE_LISTEN(state, net.handle),
                                 &STATE_LISTEN(state, net.comm)));
    memcpy(info.connectInfo.handle, STATE_LISTEN(state, net.handle), NCCL_NET_HANDLE_MAXSIZE);
  } else {
    // create socket for ring neighbor to contact me
    NCCLCHECK(createListenSocket(comm, comm->magic, &STATE_LISTEN(state, socket), &info.connectInfo.addr,
                                 ncclSocketTypeBootstrap));
  }
  // Create socket for root to contact me using the root's magic
  // For grow operations, offset is parent->nRanks - 1 (last existing rank joins the root)
  // For normal init, offset is 0
  int offset = 0;
  if (comm->isGrow) {
    if (parent != NULL) {
      offset = parent->nRanks - 1;
    } else {
      if (handles != NULL) {
        offset = BOOTSTRAP_HANDLE(handles, 0)->nRanks - 1;
      } else {
        WARN("bootstrapInit: handles and parent are NULL");
        return ncclSystemError;
      }
    }
  }
  int curr_root = rootIdFromRank(rank, nranks, nHandles, offset);
  if (curr_root >= 0) {
    NCCLCHECK(createListenSocket(comm, BOOTSTRAP_HANDLE(handles, curr_root)->magic, &listenSockRoot,
                                 &info.listenRootAddress, ncclSocketTypeBootstrap));
  }
  BOOTSTRAP_PROF_CLOSE(timers[BOOTSTRAP_INIT_TIME_CREATE]);

  // stagger connection times to avoid an overload of the root
  BOOTSTRAP_PROF_OPEN(timers[BOOTSTRAP_INIT_TIME_DELAY]);
  int nRankRoot = nRankFromRoot(curr_root, nranks, nHandles, offset);
  if (nRankRoot > ncclParamStaggerThreshold()) {
    // for socket the message rate in microsec
    double msg_rate = ncclParamStaggerRate() / 1.0e6;
    long musec = localIdFromRoot(rank, curr_root, nranks, nHandles, offset) / msg_rate;
    TRACE(NCCL_BOOTSTRAP, "rank %d delaying connection to root by %ld microsec", rank, musec);
    std::this_thread::sleep_for(std::chrono::microseconds(musec));
  }
  BOOTSTRAP_PROF_CLOSE(timers[BOOTSTRAP_INIT_TIME_DELAY]);

  // send info on my listening socket to root
  BOOTSTRAP_PROF_OPEN(timers[BOOTSTRAP_INIT_TIME_SEND]);
  // send contact info to my own root
  info.rank = rank;
  info.iroot = curr_root;
  info.offset = offset;
  if (curr_root >= 0) NCCLCHECK(sendToRoot(BOOTSTRAP_HANDLE(handles, curr_root), comm, &info));
  if (parent && comm->isGrow && rank != 0) {
    // Grow: Ranks 1 to N-1 use the parent bootstrap to send connection information to the previous rank
    NCCLCHECK(bootstrapSend(parent->bootstrap, rank - 1, 0, &info.connectInfo, sizeof(info.connectInfo)));
  }
  // if needed, send the connection info to the previous root
  // commGrow with more than = 1 rank in the parent comm is a special case of multiroot
  if (((comm->isGrow && parent && (parent->nRanks > 1)) || nHandles > 1) &&
      isFirstFromRoot(rank, curr_root, nranks, nHandles, offset)) {
    int prev_rank = BOOTSTRAP_PID(rank - 1, nranks);
    int prev_root = rootIdFromRank(prev_rank, nranks, nHandles, offset);
    info.rank = prev_rank + 1; // my rank as seen by the previous root
    info.iroot = prev_root;
    // only send if the root is valid, existing rank N-1 will use the bootstrapSend just above
    if (prev_root >= 0) NCCLCHECK(sendToRoot(BOOTSTRAP_HANDLE(handles, prev_root), comm, &info));
  }
  BOOTSTRAP_PROF_CLOSE(timers[BOOTSTRAP_INIT_TIME_SEND]);

  // get info on my "next" rank in the bootstrap ring from root
  BOOTSTRAP_PROF_OPEN(timers[BOOTSTRAP_INIT_TIME_RECV]);
  if (curr_root >= 0) {
    NCCLCHECK(ncclSocketInit(&sock));
    NCCLCHECK(ncclSocketAccept(&sock, &listenSockRoot));
    NCCLCHECK(socketRecv(&sock, &nextPeer, sizeof(nextPeer)));
    NCCLCHECK(ncclSocketClose(&sock));
    NCCLCHECK(ncclSocketClose(&listenSockRoot));
  }
  if (parent && comm->isGrow && rank != parent->nRanks - 1) {
    // Grow: Ranks 0 to N-2 use the parent bootstrap to recv connection information to the next rank.
    // This is consistent with the bootstrapSend above.
    NCCLCHECK(bootstrapRecv(parent->bootstrap, rank + 1, 0, &nextPeer, sizeof(nextPeer)));
  }
  BOOTSTRAP_PROF_CLOSE(timers[BOOTSTRAP_INIT_TIME_RECV]);

  // accept and connect the ring network
  if (ncclParamBootstrapNetEnable()) {
    NCCLCHECK(netRingConnect(comm->netContext, state->net, &state->listen, nextPeer.handle,
                             &STATE_RING(state, net.sendComm), &STATE_RING(state, net.sendDevHandle),
                             &STATE_RING(state, net.recvComm), &STATE_RING(state, net.recvDevHandle),
                             state->abortFlag));
  } else {
    NCCLCHECK(socketRingConnect(&nextPeer.addr, &STATE_RING(state, socket.send), &STATE_LISTEN(state, socket),
                                &STATE_RING(state, socket.recv), comm->magic, state->abortFlag));
  }
  state->ringConnected = true;

  // AllGather all listen handlers
  NCCLCHECKGOTO(ncclCalloc(&state->ringAddresses, nranks), result, fail);
  memcpy(state->ringAddresses + rank, &info.connectInfo, sizeof(info.connectInfo));

  // in case of failure, those resources will be free'd when calling bootstrapDestroy, so we can return immediatly
  NCCLCHECK(ncclCalloc(&state->peerProxyAddresses, nranks));
  NCCLCHECK(ncclCalloc(&proxySocket, 1));
  NCCLCHECKGOTO(createListenSocket(comm, comm->magic, proxySocket, state->peerProxyAddresses + rank,
                                   ncclSocketTypeProxy),
                result, fail);

  NCCLCHECKGOTO(ncclCalloc(&state->peerProxyAddressesUDS, nranks), result, fail);
  NCCLCHECKGOTO(getUDS(state->peerProxyAddressesUDS + rank), result, fail);

  // create a socket for others to reach out (P2P)
  union ncclSocketAddress peerSocketAddress;
  NCCLCHECKGOTO(createListenSocket(comm, comm->magic, &STATE_LISTEN(state, peerSocket), &peerSocketAddress,
                                   ncclSocketTypeBootstrap),
                result, fail);
  NCCLCHECKGOTO(ncclCalloc(&state->peerP2pAddresses, nranks), result, fail);
  memcpy(state->peerP2pAddresses + rank, &peerSocketAddress, sizeof(union ncclSocketAddress));

  // Initialize RAS
  if (ncclParamRasEnable() == 1) {
    // The RAS thread will take ownership after ncclRasAddRanks succeeds.
    NCCLCHECKGOTO(ncclCalloc(&rasRanks, nranks), result, fail);
    memcpy(&rasRanks[rank].addr, &bootstrapNetIfAddr, sizeof(rasRanks[rank].addr));
    rasRanks[rank].pid = ncclOsGetPid();
    rasRanks[rank].cudaDev = comm->cudaDev;
    rasRanks[rank].nvmlDev = comm->nvmlDev;
    rasRanks[rank].hostHash = getHostHash();
    rasRanks[rank].pidHash = getPidHash();
    if (ncclRasCommInit(comm, rasRanks + rank) != ncclSuccess) {
      INFO(NCCL_INIT | NCCL_RAS, "Continuing in spite of a RAS initialization error");
      // We should still participate in the ringAllInfo below as the peers will be waiting for us.
      // Just make sure that the address is clearly invalid...
      memset(rasRanks + rank, '\0', sizeof(*rasRanks));
      performRasAddRanks = false;
    }
  }

  BOOTSTRAP_PROF_OPEN(timers[BOOTSTRAP_INIT_TIME_RING]);
  state->rasRanks = rasRanks;
  NCCLCHECKGOTO(ringAllInfo(comm, state, state->peerP2pAddresses, state->peerProxyAddresses,
                            state->peerProxyAddressesUDS, state->rasRanks, state->ringAddresses),
                result, fail);
  BOOTSTRAP_PROF_CLOSE(timers[BOOTSTRAP_INIT_TIME_RING]);

  // Create the service proxy and get the UDS
  NCCLCHECKGOTO(ncclProxyInit(comm, proxySocket, state->peerProxyAddresses, state->peerProxyAddressesUDS), result,
                fail);

  if (ncclParamRasEnable() == 1 && performRasAddRanks) {
    struct rasRankInit* rasRanksForThread = NULL;
    NCCLCHECKGOTO(ncclCalloc(&rasRanksForThread, nranks), result, fail);
    memcpy(rasRanksForThread, state->rasRanks, sizeof(*rasRanksForThread) * nranks);
    if (ncclRasAddRanks(rasRanksForThread, nranks) != ncclSuccess) {
      INFO(NCCL_INIT | NCCL_RAS, "Continuing in spite of a RAS initialization error");
      free(rasRanksForThread);
    }
  }

  BOOTSTRAP_PROF_CLOSE(timers[BOOTSTRAP_INIT_TIME_TOTAL]);
  TRACE(NCCL_BOOTSTRAP, "rank %d nranks %d - DONE", rank, nranks);
  INFO(NCCL_BOOTSTRAP | NCCL_PROFILE, "Bootstrap timings total %f (create %f, send %f, recv %f, ring %f, delay %f)",
       timers[BOOTSTRAP_INIT_TIME_TOTAL] / 1e9, timers[BOOTSTRAP_INIT_TIME_CREATE] / 1e9,
       timers[BOOTSTRAP_INIT_TIME_SEND] / 1e9, timers[BOOTSTRAP_INIT_TIME_RECV] / 1e9,
       timers[BOOTSTRAP_INIT_TIME_RING] / 1e9, timers[BOOTSTRAP_INIT_TIME_DELAY] / 1e9);
exit:
  return result;
fail:
  free(proxySocket);
  goto exit;
}

ncclResult_t bootstrapSplit(uint64_t magic, struct ncclComm* comm, struct ncclComm* parent, int color, int key,
                            int* parentRanks) {
  ncclResult_t ret = ncclSuccess;
  int rank = comm->rank;
  int nranks = comm->nRanks;
  int prev, next;
  union ringConnectInfo info;
  union ringConnectInfo nextPeer;
  struct ncclSocket* proxySocket = NULL;
  struct bootstrapState* state;

  NCCLCHECKGOTO(ncclCalloc(&state, 1), ret, fail);
  state->rank = rank;
  state->nranks = nranks;
  state->cudaDev = comm->cudaDev;
  state->abortFlag = comm->abortFlag;
  state->net = comm->ncclNet;
  state->activeRankMask = comm->activeRankMask;
  comm->bootstrap = state;
  comm->magic = state->magic = magic;

  prev = parentRanks[(rank - 1 + nranks) % nranks];
  next = parentRanks[(rank + 1) % nranks];

  // create a handle for the others to reach out to me
  if (ncclParamBootstrapNetEnable()) {
    NCCLCHECKGOTO(netGetDevice(rank, comm, &STATE_LISTEN(state, net.dev)), ret, fail);
    NCCLCHECKGOTO(state->net->listen(comm->netContext, STATE_LISTEN(state, net.dev), STATE_LISTEN(state, net.handle),
                                     &STATE_LISTEN(state, net.comm)),
                  ret, fail);
    memcpy(info.handle, STATE_LISTEN(state, net.handle), NCCL_NET_HANDLE_MAXSIZE);
  } else {
    // create socket for ring neightbor to contact mee
    NCCLCHECK(createListenSocket(comm, comm->magic, &STATE_LISTEN(state, socket), &info.addr, ncclSocketTypeBootstrap));
  }
  NCCLCHECKGOTO(ncclCalloc(&state->ringAddresses, nranks), ret, fail);
  memcpy(state->ringAddresses + rank, &info, sizeof(info));
  // create a socket for others to reach out (P2P)
  union ncclSocketAddress peerSocketAddress;
  NCCLCHECK(createListenSocket(comm, comm->magic, &STATE_LISTEN(state, peerSocket), &peerSocketAddress,
                               ncclSocketTypeBootstrap));

  if (ncclParamRasEnable() == 1) {
    if (ncclRasCommInit(comm, nullptr) != ncclSuccess) {
      INFO(NCCL_INIT | NCCL_RAS, "Continuing in spite of a RAS initialization error");
    }
  }

  // Get addr from next rank using the parent's connections
  NCCLCHECKGOTO(bootstrapSend(parent->bootstrap, prev, BOOTSTRAP_TAG_COMMSPLIT, &info, sizeof(union ringConnectInfo)),
                ret, fail);
  NCCLCHECKGOTO(bootstrapRecv(parent->bootstrap, next, BOOTSTRAP_TAG_COMMSPLIT, &nextPeer,
                              sizeof(union ringConnectInfo)),
                ret, fail);
  if (ncclParamBootstrapNetEnable()) {
    NCCLCHECKGOTO(netRingConnect(comm->netContext, state->net, &state->listen, nextPeer.handle,
                                 &STATE_RING(state, net.sendComm), &STATE_RING(state, net.sendDevHandle),
                                 &STATE_RING(state, net.recvComm), &STATE_RING(state, net.recvDevHandle),
                                 state->abortFlag),
                  ret, fail);
  } else {
    NCCLCHECK(socketRingConnect(&nextPeer.addr, &STATE_RING(state, socket.send), &STATE_LISTEN(state, socket),
                                &STATE_RING(state, socket.recv), comm->magic, state->abortFlag));
  }
  state->ringConnected = true;

  NCCLCHECKGOTO(ncclCalloc(&state->peerP2pAddresses, nranks), ret, fail);
  memcpy(state->peerP2pAddresses + rank, &peerSocketAddress, sizeof(union ncclSocketAddress));
  if (parent->shareResources) {
    /* map local rank to top parent local rank. */
    for (int i = 0; i < nranks; ++i) {
      comm->topParentRanks[i] = parent->topParentRanks[parentRanks[i]];
    }
    NCCLCHECKGOTO(ringAllInfo(comm, state, state->peerP2pAddresses, NULL, NULL, NULL, state->ringAddresses), ret, fail);
  } else {
    NCCLCHECKGOTO(ncclCalloc(&state->peerProxyAddresses, nranks), ret, fail);
    NCCLCHECKGOTO(ncclCalloc(&state->peerProxyAddressesUDS, nranks), ret, fail);
    // Create the service proxy and get the UDS
    NCCLCHECKGOTO(ncclCalloc(&proxySocket, 1), ret, fail);
    NCCLCHECKGOTO(getUDS(state->peerProxyAddressesUDS + rank), ret, fail);
    NCCLCHECKGOTO(createListenSocket(comm, comm->magic, proxySocket, state->peerProxyAddresses + rank,
                                     ncclSocketTypeProxy),
                  ret, fail);
    NCCLCHECKGOTO(ringAllInfo(comm, state, state->peerP2pAddresses, state->peerProxyAddresses,
                              state->peerProxyAddressesUDS, NULL, state->ringAddresses),
                  ret, fail);
    NCCLCHECKGOTO(ncclProxyInit(comm, proxySocket, state->peerProxyAddresses, state->peerProxyAddressesUDS), ret, fail);
  }

  TRACE(NCCL_BOOTSTRAP, "bootstrapSplit: comm %p parent %p rank %d nranks %d color %d key %d prev %d next %d - DONE",
        comm, parent, rank, nranks, color, key, prev, next);

exit:
  return ret;
fail:
  free(proxySocket);
  goto exit;
}

struct socketAckInfo {
  int rank;
  int tag;
};
static ncclResult_t socketConnect(void* commState, int peer, int tag, struct ncclSocket* sock) {
  ncclResult_t ret = ncclSuccess;
  struct bootstrapState* state = (struct bootstrapState*)commState;

  struct socketAckInfo ack = (struct socketAckInfo){state->rank, tag};
  NCCLCHECKGOTO(ncclSocketInit(sock, state->peerP2pAddresses + peer, state->magic, ncclSocketTypeBootstrap,
                               state->abortFlag),
                ret, fail);
  NCCLCHECKGOTO(ncclSocketConnect(sock), ret, fail);
  NCCLCHECKGOTO(socketSend(sock, &ack, sizeof(struct socketAckInfo)), ret, fail);
  return ncclSuccess;
fail:
  (void)ncclSocketClose(sock);
  return ret;
}
ncclResult_t bootstrapSend(void* commState, int peer, int tag, void* data, int size) {
  ncclResult_t ret = ncclSuccess;
  struct ncclSocket sock;
  TRACE(NCCL_BOOTSTRAP, "Sending to peer=%d tag=%d size=%d", peer, tag, size);
  NCCLCHECK(socketConnect(commState, peer, tag, &sock));
  NCCLCHECKGOTO(socketSend(&sock, data, size), ret, fail);
  TRACE(NCCL_BOOTSTRAP, "Sent to peer=%d tag=%d size=%d", peer, tag, size);
  NCCLCHECK(ncclSocketClose(&sock));
  return ret;
fail:
  (void)ncclSocketClose(&sock);
  return ret;
}
// Bootstrap send/receive functions
static ncclResult_t unexpectedEnqueue(struct bootstrapState* state, int peer, int tag, struct ncclSocket* sock) {
  // New unex
  struct unexConn* unex;
  NCCLCHECK(ncclCalloc(&unex, 1));
  unex->peer = peer;
  unex->tag = tag;
  memcpy(&unex->sock, sock, sizeof(struct ncclSocket));

  // Enqueue
  struct unexConn* list = state->unexpectedConnections;
  if (list == NULL) {
    state->unexpectedConnections = unex;
    return ncclSuccess;
  }
  while (list->next) list = list->next;
  list->next = unex;
  return ncclSuccess;
}
static ncclResult_t unexpectedDequeue(struct bootstrapState* state, int peer, int tag, struct ncclSocket* sock,
                                      int* found) {
  struct unexConn* elem = state->unexpectedConnections;
  struct unexConn* prev = NULL;
  *found = 0;
  while (elem) {
    // peer < 0 means wildcard (accept from any peer)
    if ((peer < 0 || elem->peer == peer) && elem->tag == tag) {
      if (prev == NULL) {
        state->unexpectedConnections = elem->next;
      } else {
        prev->next = elem->next;
      }
      memcpy(sock, &elem->sock, sizeof(struct ncclSocket));
      free(elem);
      *found = 1;
      return ncclSuccess;
    }
    prev = elem;
    elem = elem->next;
  }
  return ncclSuccess;
}

static void unexpectedFree(struct bootstrapState* state) {
  struct unexConn* elem = state->unexpectedConnections;
  struct unexConn* prev = NULL;

  while (elem) {
    prev = elem;
    elem = elem->next;
    free(prev);
  }
  return;
}

// We can't know who we'll receive from, so we need to receive everything at once
static ncclResult_t socketAccept(void* commState, int peer, int tag, struct ncclSocket* sock) {
  ncclResult_t ret = ncclSuccess;
  struct bootstrapState* state = (struct bootstrapState*)commState;

  // Search unexpected connections first
  int found;
  NCCLCHECK(unexpectedDequeue(state, peer, tag, sock, &found));
  if (found) return ncclSuccess;

  // Then look for new connections
  while (1) {
    struct socketAckInfo ack = {0};
    NCCLCHECKGOTO(ncclSocketInit(sock), ret, fail);
    NCCLCHECKGOTO(ncclSocketAccept(sock, &STATE_LISTEN(state, peerSocket)), ret, fail);
    NCCLCHECKGOTO(socketRecv(sock, &ack, sizeof(struct socketAckInfo)), ret, fail);
    // Match: tag must match, and peer must match (peer < 0 means wildcard)
    if (ack.tag == tag && (peer < 0 || ack.rank == peer)) return ncclSuccess;
    // No match: queue for later and try next connection
    NCCLCHECKGOTO(unexpectedEnqueue(state, ack.rank, ack.tag, sock), ret, fail);
  }
  return ncclSuccess;
fail:
  (void)ncclSocketClose(sock);
  return ret;
}
// We can't know who we'll receive from, so we need to receive everything at once
ncclResult_t bootstrapRecv(void* commState, int peer, int tag, void* data, int size) {
  ncclResult_t ret;
  struct ncclSocket sock;
  NCCLCHECK(socketAccept(commState, peer, tag, &sock));
  TRACE(NCCL_BOOTSTRAP, "Receiving tag=%d peer=%d size=%d", tag, peer, size);
  NCCLCHECKGOTO(socketRecv(&sock, ((char*)data), size), ret, fail);
  NCCLCHECKGOTO(ncclSocketClose(&sock, /*wait*/ true), ret, fail);
  return ret;
fail:
  (void)ncclSocketClose(&sock);
  return ret;
}

static int bootstrapDataSlot(const int* slots, int rank) {
  return slots == NULL ? rank : slots[rank];
}

static ncclResult_t netRingAllGather(ncclNet_t* net, void* sendComm, void* recvComm, int rank, int nranks, char* data,
                                     int size, volatile uint32_t* abortFlag) {
  ncclResult_t res = ncclSuccess;
  uint64_t tFirst = 0, tRest = 0;
  void* sendDataHandle = NULL;
  void* recvDataHandle = NULL;

  NCCLCHECKGOTO(netReg(net, sendComm, data, nranks * size, &sendDataHandle), res, exit);
  NCCLCHECKGOTO(netReg(net, recvComm, data, nranks * size, &recvDataHandle), res, exit);
  /* Simple ring based AllGather
   * At each step i receive data from (rank-i-1) from prev
   * and send previous step's data from (rank-i) to next
   */
  TRACE(NCCL_BOOTSTRAP, "NetRingAllGather started");
  BOOTSTRAP_PROF_OPEN(tFirst);
  for (int i = 0; i < nranks - 1; i++) {
    int tag = i;
    size_t rslice = (rank - i - 1 + nranks) % nranks;
    size_t sslice = (rank - i + nranks) % nranks;
    void* recv_data = data + rslice * size;
    void* send_data = data + sslice * size;
    NCCLCHECKGOTO(netSendRecv(net, sendComm, send_data, size, sendDataHandle, recvComm, recv_data, size, recvDataHandle,
                              tag, abortFlag),
                  res, exit);
    if (i == 0) {
      BOOTSTRAP_PROF_CLOSE(tFirst);
      BOOTSTRAP_PROF_OPEN(tRest);
    }
  }
  BOOTSTRAP_PROF_CLOSE(tRest);
  TRACE(NCCL_BOOTSTRAP | NCCL_PROFILE, "netRingAllGather first message in %f (%f MB/sec), rest in %f (%f MB/sec)",
        tFirst / 1e9, (size / 1e6) / (tFirst / 1e9), tRest / 1e9, (nranks - 1) * (size / 1e6) / (tRest / 1e9));
exit:
  // do not fail in case of error, try to deregister as much as possible
  if (sendDataHandle) netDereg(net, sendComm, &sendDataHandle);
  if (recvDataHandle) netDereg(net, recvComm, &recvDataHandle);
  return res;
}
static ncclResult_t socketRingAllGather(struct ncclSocket* nextSock, struct ncclSocket* prevSock, int rank, int nranks,
                                        const int* slots, char* data, int size) {
  ncclResult_t res = ncclSuccess;
  uint64_t tFirst = 0, tRest = 0;
  /* Simple ring based AllGather
   * At each step i receive data from (rank-i-1) from prev
   * and send previous step's data from (rank-i) to next
   * Each rank's data is placed at data[slot[rank*size]].
   */
  TRACE(NCCL_BOOTSTRAP, "socketRingAllGather started: rank=%d nranks=%d", rank, nranks);
  int totalSteps = nranks / 2;
  TRACE(NCCL_BOOTSTRAP, "bidirectional bootstrap: totalSteps=%d", totalSteps);
  BOOTSTRAP_PROF_OPEN(tFirst);
  for (int step = 0; step < totalSteps; step++) {
    // N ranks requires (N-1)/2 steps for the double ring  algorithm.
    // If N is even, the last step is requires a single send/recv
    bool isFinalUnidirectional = (step == totalSteps - 1) && (nranks % 2 == 0);
    // Ring0: ring from previous to next
    int sendSliceRing0 = bootstrapDataSlot(slots, (rank - step + nranks) % nranks);      // Send this slice to next
    int recvSliceRing0 = bootstrapDataSlot(slots, (rank - step - 1 + nranks) % nranks); // Receive this slice from prev
    // Ring1: ring from next to previous
    int sendSliceRing1 = bootstrapDataSlot(slots, (rank + step) % nranks); // Send this slice to prev neighbor
    int recvSliceRing1 = bootstrapDataSlot(slots, (rank + step + 1) % nranks); // Receive this slice from next neighbor
    if (isFinalUnidirectional) {
      // last step, just a single sendrecv is needed:
      NCCLCHECKGOTO(socketSendRecv(nextSock, data + sendSliceRing0 * size, size, prevSock, data + recvSliceRing0 * size,
                                   size),
                    res, exit);
    } else {
      // clang-format off
      struct ncclSocketOp ops[4] = {
        {NCCL_SOCKET_SEND, nextSock, data + sendSliceRing0 * size, size, 0},  // Ring0: send to next
        {NCCL_SOCKET_RECV, prevSock, data + recvSliceRing0 * size, size, 0},  // Ring0: recv from prev
        {NCCL_SOCKET_SEND, prevSock, data + sendSliceRing1 * size, size, 0},  // Ring1: send to prev
        {NCCL_SOCKET_RECV, nextSock, data + recvSliceRing1 * size, size, 0}   // Ring1: recv from next
      };
      // clang-format on
      NCCLCHECKGOTO(socketDoubleSendRecv(ops), res, exit);
    }
    if (step == 0) {
      BOOTSTRAP_PROF_CLOSE(tFirst);
      BOOTSTRAP_PROF_OPEN(tRest);
    }
  }
  BOOTSTRAP_PROF_CLOSE(tRest);
  TRACE(NCCL_BOOTSTRAP | NCCL_PROFILE, "socketRingAllGather first message in %f (%f MB/sec), rest in %f (%f MB/sec)",
        tFirst / 1e9, (size / 1e6) / (tFirst / 1e9), tRest / 1e9, (nranks - 1) * (size / 1e6) / (tRest / 1e9));
exit:
  return res;
}

static ncclResult_t socketDynamicRingAllGather(struct bootstrapState* state, int* activeRanks, int activeCount,
                                               int activeIndex, char* data, int size) {
  ncclResult_t res = ncclSuccess;
  int prevRank = -1;
  int nextRank = -1;
  struct ncclSocket sendSocket;
  struct ncclSocket recvSocket;
  bool sendConnected = false;
  bool recvConnected = false;

  if (activeCount <= 1) goto exit;

  prevRank = activeRanks[(activeIndex - 1 + activeCount) % activeCount];
  nextRank = activeRanks[(activeIndex + 1) % activeCount];
  NCCLCHECKGOTO(socketConnect(state, nextRank, BOOTSTRAP_TAG_ACTIVE_ALLGATHER, &sendSocket), res, exit);
  sendConnected = true;
  NCCLCHECKGOTO(socketAccept(state, prevRank, BOOTSTRAP_TAG_ACTIVE_ALLGATHER, &recvSocket), res, exit);
  recvConnected = true;
  NCCLCHECKGOTO(socketRingAllGather(&sendSocket, &recvSocket, activeIndex, activeCount, activeRanks, data, size), res,
                exit);

exit:
  if (sendConnected) (void)ncclSocketClose(&sendSocket);
  if (recvConnected) (void)ncclSocketClose(&recvSocket);
  return res;
}
ncclResult_t bootstrapAllGather(void* commState, void* allData, int size) {
  ncclResult_t res = ncclSuccess;
  struct bootstrapState* state = (struct bootstrapState*)commState;
  int* activeRanks = NULL;
  int activeCount = 0;
  int activeIndex = -1;

  TRACE(NCCL_BOOTSTRAP, "rank %d nranks %d size %d - AllGather", state->rank, state->nranks, size);

  uint64_t time = 0;
  BOOTSTRAP_PROF_OPEN(time);
  NCCLCHECKGOTO(bootstrapGetActiveRanks(state, &activeRanks, &activeCount, &activeIndex), res, exit);
  if (activeCount <= 1) goto exit;

  if (activeCount == state->nranks && state->ringConnected && ncclParamBootstrapNetEnable()) {
    NCCLCHECKGOTO(netRingAllGather(state->net, STATE_RING(state, net.sendComm), STATE_RING(state, net.recvComm),
                                   state->rank, state->nranks, (char*)allData, size, state->abortFlag),
                  res, exit);
  } else if (activeCount == state->nranks && state->ringConnected) {
    NCCLCHECKGOTO(socketRingAllGather(&STATE_RING(state, socket.send), &STATE_RING(state, socket.recv), state->rank,
                                      state->nranks, NULL, (char*)allData, size),
                  res, exit);
  } else {
    NCCLCHECKGOTO(socketDynamicRingAllGather(state, activeRanks, activeCount, activeIndex, (char*)allData, size), res,
                  exit);
  }
exit:
  BOOTSTRAP_PROF_CLOSE(time);
  TRACE(NCCL_BOOTSTRAP | NCCL_PROFILE, "bootstrapAllGather for %d B done in %f sec: %f MB/sec", size, time / 1e9,
        (state->nranks * size / 1e6) / (time / 1e9));
  TRACE(NCCL_BOOTSTRAP, "rank %d nranks %d size %d - AllGather DONE", state->rank, state->nranks, size);
  free(activeRanks);
  return res;
}

static ncclResult_t bootstrapP2PBarrier(void* commState, int* ranks, int rank, int nranks, int tag) {
  ncclResult_t res = ncclSuccess;
  struct bootstrapState* state = (struct bootstrapState*)commState;
  bool allRanksActive = true;
  int* barrierData = NULL;
  int data[1] = {0};

  if (nranks == 1) return ncclSuccess;
  if (!ncclRankMaskIsActive(state->activeRankMask, state->nranks, ranks ? ranks[rank] : rank)) return ncclSuccess;
  for (int i = 0; i < nranks; i++) {
    int peerRank = ranks ? ranks[i] : i;
    if (!ncclRankMaskIsActive(state->activeRankMask, state->nranks, peerRank)) {
      allRanksActive = false;
      break;
    }
  }
  if (!allRanksActive) {
    int dataRanks = ranks ? nranks : state->nranks;
    NCCLCHECKGOTO(ncclCalloc(&barrierData, dataRanks), res, exit);
    barrierData[rank] = 1;
    if (ranks) {
      NCCLCHECKGOTO(bootstrapIntraNodeAllGather(commState, ranks, rank, nranks, barrierData, sizeof(int)), res, exit);
    } else {
      NCCLCHECKGOTO(bootstrapAllGather(commState, barrierData, sizeof(int)), res, exit);
    }
    free(barrierData);
    goto exit;
  }
  /* Simple [intra] process barrier
   *
   * Based on the dissemination algorithm by Debra Hensgen, Raphael Finkel, and Udi Manbet,
   * "Two Algorithms for Barrier Synchronization," International Journal of Parallel Programming, 17(1):1-17, 1988"
   */
  for (int mask = 1; mask < nranks; mask <<= 1) {
    int src = (rank - mask + nranks) % nranks;
    int dst = (rank + mask) % nranks;
    NCCLCHECK(bootstrapSend(commState, ranks ? ranks[dst] : dst, tag, data, sizeof(data)));
    NCCLCHECK(bootstrapRecv(commState, ranks ? ranks[src] : src, tag, data, sizeof(data)));
  }
exit:
  return res;
}

ncclResult_t bootstrapIntraNodeBarrier(void* commState, int* ranks, int rank, int nranks, int tag) {
  uint64_t time = 0;
  BOOTSTRAP_PROF_OPEN(time);
  NCCLCHECK(bootstrapP2PBarrier(commState, ranks, rank, nranks, tag));
  BOOTSTRAP_PROF_CLOSE(time);
  TRACE(NCCL_BOOTSTRAP | NCCL_PROFILE, "bootstrapIntraNodeBarrier done in %f sec", time / 1e9);
  return ncclSuccess;
}

ncclResult_t bootstrapBarrier(void* commState, int rank, int nranks, int tag) {
  uint64_t time = 0;
  BOOTSTRAP_PROF_OPEN(time);
  NCCLCHECK(bootstrapP2PBarrier(commState, NULL, rank, nranks, tag));
  BOOTSTRAP_PROF_CLOSE(time);
  TRACE(NCCL_BOOTSTRAP | NCCL_PROFILE, "bootstrapBarrier done in %f sec", time / 1e9);
  return ncclSuccess;
}

ncclResult_t bootstrapIntraNodeAllGather(void* commState, int* ranks, int rank, int nranks, void* allData, int size) {
  ncclResult_t res = ncclSuccess;
  struct bootstrapState* state = (struct bootstrapState*)commState;
  int* activeRanks = NULL;
  int* activeSlots = NULL;
  int activeCount = 0;
  int activeIndex = -1;
  int prevRank = -1;
  int nextRank = -1;
  struct ncclSocket recvSocket;
  struct ncclSocket sendSocket;
  bool recvConnected = false;
  bool sendConnected = false;

  TRACE(NCCL_INIT, "rank %d nranks %d size %d - ENTER", rank, nranks, size);

  NCCLCHECKGOTO(ncclCalloc(&activeRanks, nranks), res, exit);
  NCCLCHECKGOTO(ncclCalloc(&activeSlots, nranks), res, exit);
  for (int i = 0; i < nranks; i++) {
    int peerRank = ranks[i];
    if (ncclRankMaskIsActive(state->activeRankMask, state->nranks, peerRank)) {
      activeRanks[activeCount] = peerRank;
      activeSlots[activeCount] = i;
      if (i == rank) activeIndex = activeCount;
      activeCount++;
    }
  }
  if (activeCount <= 1 || activeIndex == -1) goto exit;

  prevRank = activeRanks[(activeIndex - 1 + activeCount) % activeCount];
  nextRank = activeRanks[(activeIndex + 1) % activeCount];
  // intraNode bootstrap is done defacto using the socket-based implementation
  NCCLCHECKGOTO(socketConnect(commState, nextRank, BOOTSTRAP_TAG_INTRANODE_ALLGATHER, &sendSocket), res, exit);
  sendConnected = true;
  NCCLCHECKGOTO(socketAccept(commState, prevRank, BOOTSTRAP_TAG_INTRANODE_ALLGATHER, &recvSocket), res, exit);
  recvConnected = true;

  NCCLCHECKGOTO(socketRingAllGather(&sendSocket, &recvSocket, activeIndex, activeCount, activeSlots, (char*)allData,
                                    size),
                res, exit);

exit:
  if (sendConnected) (void)ncclSocketClose(&sendSocket);
  if (recvConnected) (void)ncclSocketClose(&recvSocket);
  if (activeSlots) free(activeSlots);
  if (activeRanks) free(activeRanks);
  TRACE(NCCL_INIT, "rank %d nranks %d size %d - DONE", rank, nranks, size);
  return res;
}

// [IntraNode] in-place Broadcast
static ncclResult_t bootstrapP2PBroadcast(void* commState, int* ranks, int rank, int nranks, int root, void* bcastData,
                                          int size) {
  struct bootstrapState* state = (struct bootstrapState*)commState;
  if (nranks == 1) return ncclSuccess;
  int rootRank = ranks ? ranks[root] : root;
  int selfRank = ranks ? ranks[rank] : rank;
  if (!ncclRankMaskIsActive(state->activeRankMask, state->nranks, selfRank)) return ncclSuccess;
  if (!ncclRankMaskIsActive(state->activeRankMask, state->nranks, rootRank)) {
    WARN("bootstrapP2PBroadcast: root rank %d is inactive", rootRank);
    return ncclInvalidUsage;
  }
  if (rank == root) {
    for (int i = 0; i < nranks; i++) {
      int peerRank = ranks ? ranks[i] : i;
      if (i != root && ncclRankMaskIsActive(state->activeRankMask, state->nranks, peerRank)) {
        NCCLCHECK(bootstrapSend(commState, peerRank, /*tag=*/peerRank, bcastData, size));
      }
    }
  } else {
    NCCLCHECK(bootstrapRecv(commState, rootRank, /*tag=*/selfRank, bcastData, size));
  }
  return ncclSuccess;
}

ncclResult_t bootstrapIntraNodeBroadcast(void* commState, int* ranks, int rank, int nranks, int root, void* bcastData,
                                         int size) {
  uint64_t time = 0;
  BOOTSTRAP_PROF_OPEN(time);
  NCCLCHECK(bootstrapP2PBroadcast(commState, ranks, rank, nranks, root, bcastData, size));
  BOOTSTRAP_PROF_CLOSE(time);
  TRACE(NCCL_BOOTSTRAP | NCCL_PROFILE, "bootstrapIntraNodeBroadcast for %d B done in %f sec: %f MB/sec", size,
        time / 1e9, (nranks * size / 1e6) / (time / 1e9));
  return ncclSuccess;
}
ncclResult_t bootstrapBroadcast(void* commState, int rank, int nranks, int root, void* bcastData, int size) {
  uint64_t time = 0;
  BOOTSTRAP_PROF_OPEN(time);
  NCCLCHECK(bootstrapP2PBroadcast(commState, NULL, rank, nranks, root, bcastData, size));
  BOOTSTRAP_PROF_CLOSE(time);
  TRACE(NCCL_BOOTSTRAP | NCCL_PROFILE, "bootstrapBroadcast done in %f sec", time / 1e9);
  return ncclSuccess;
}

ncclResult_t bootstrapClose(void* commState) {
  if (commState == NULL) return ncclSuccess;
  struct bootstrapState* state = (struct bootstrapState*)commState;
  // close unexpected and return an error if we are not aborting and still operations in the pipe
  if (state->unexpectedConnections != NULL) {
    unexpectedFree(state);
    if (COMPILER_ATOMIC_LOAD(state->abortFlag, std::memory_order_acquire) == 0) {
      WARN("Unexpected connections are not empty");
      return ncclInternalError;
    }
  }
  NCCLCHECK(bootstrapCloseRing(state));
  if (ncclParamBootstrapNetEnable()) {
    if (STATE_LISTEN(state, net.comm) != NULL) NCCLCHECK(state->net->closeListen(STATE_LISTEN(state, net.comm)));
  } else {
    NCCLCHECK(ncclSocketClose(&STATE_LISTEN(state, socket)));
  }
  // close the p2p socket
  NCCLCHECK(ncclSocketClose(&STATE_LISTEN(state, peerSocket)));

  free(state->peerProxyAddresses);
  free(state->peerProxyAddressesUDS);
  free(state->ringAddresses);
  free(state->peerP2pAddresses);
  free(state->rasRanks);
  free(state);
  return ncclSuccess;
}

ncclResult_t bootstrapAbort(void* commState) {
  return bootstrapClose(commState);
}
