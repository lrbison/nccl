/*
 * shim_core.cc - checkpoint shim state.
 */

#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif

#include "shim_core.h"

thread_local int ncclDebugNoWarn = 0;

namespace nccl_checkpoint {

CommRegistry g_commRegistry;
static std::atomic<bool> g_checkpointPrepared{false};

ncclResult_t CommRegistry::track(ncclComm_t comm, int nranks, int rank, int cudaDev) {
  if (comm == nullptr) return ncclSuccess;

  std::lock_guard<std::mutex> lock(mtx_);
  for (CommRecord& record : comms_) {
    if (record.comm == comm) {
      if (nranks >= 0) record.nranks = nranks;
      if (rank >= 0) record.rank = rank;
      if (cudaDev >= 0) record.cudaDev = cudaDev;
      record.userState = comm_user_active;
      return ncclSuccess;
    }
  }

  CommRecord record;
  record.comm = comm;
  record.sequence = nextSequence_++;
  record.nranks = nranks;
  record.rank = rank;
  record.cudaDev = cudaDev;
  comms_.push_back(record);
  TRACE(NCCL_CHECKPOINT, "track comm %p sequence=%llu rank=%d nranks=%d cudaDev=%d", comm,
        (unsigned long long)record.sequence, record.rank, record.nranks, record.cudaDev);
  return ncclSuccess;
}

void CommRegistry::markState(ncclComm_t comm, CommUserState userState) {
  if (comm == nullptr) return;

  std::lock_guard<std::mutex> lock(mtx_);
  for (CommRecord& record : comms_) {
    if (record.comm == comm) {
      record.userState = userState;
      return;
    }
  }
}

void CommRegistry::remove(ncclComm_t comm) {
  if (comm == nullptr) return;

  std::lock_guard<std::mutex> lock(mtx_);
  for (auto it = comms_.begin(); it != comms_.end(); ++it) {
    if (it->comm == comm) {
      comms_.erase(it);
      return;
    }
  }
}

void CommRegistry::updateRuntime(const CommRecord& record) {
  if (record.comm == nullptr) return;

  std::lock_guard<std::mutex> lock(mtx_);
  for (CommRecord& stored : comms_) {
    if (stored.comm == record.comm) {
      if (record.nranks >= 0) stored.nranks = record.nranks;
      if (record.rank >= 0) stored.rank = record.rank;
      if (record.cudaDev >= 0) stored.cudaDev = record.cudaDev;
      if (record.commHash != 0) stored.commHash = record.commHash;
      stored.userState = record.userState;
      return;
    }
  }
}

std::vector<CommRecord> CommRegistry::snapshotActive() const {
  std::vector<CommRecord> snapshot;
  std::lock_guard<std::mutex> lock(mtx_);
  for (const CommRecord& record : comms_) {
    if (record.userState == comm_user_active && record.comm != nullptr) snapshot.push_back(record);
  }
  return snapshot;
}

bool isCheckpointPrepared() {
  return g_checkpointPrepared.load(std::memory_order_acquire);
}

void markCheckpointPrepared() {
  g_checkpointPrepared.store(true, std::memory_order_release);
}

void clearCheckpointPrepared() {
  g_checkpointPrepared.store(false, std::memory_order_release);
}

}  // namespace nccl_checkpoint
