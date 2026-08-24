/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * See LICENSE.txt for more license information
 *************************************************************************/

#ifndef NCCL_RANK_MASK_H_
#define NCCL_RANK_MASK_H_

#include "nccl.h"
#include "debug.h"

inline bool ncclRankMaskValueIsValid(ncclCommMaskValue_t value) {
  return value == ncclRankMaskInactive || value == ncclRankMaskActive;
}

inline ncclResult_t ncclRankMaskValidate(const ncclCommMaskValue_t* mask, int nRanks, int* invalidRank = nullptr) {
  if (invalidRank != nullptr) *invalidRank = -1;
  if (nRanks < 0) {
    WARN("ncclRankMaskValidate: invalid nRanks %d", nRanks);
    return ncclInvalidArgument;
  }
  if (mask == nullptr) return ncclSuccess;
  for (int rank = 0; rank < nRanks; ++rank) {
    if (!ncclRankMaskValueIsValid(mask[rank])) {
      if (invalidRank != nullptr) *invalidRank = rank;
      WARN("ncclRankMaskValidate: invalid mask value %d at rank %d", (int)mask[rank], rank);
      return ncclInvalidArgument;
    }
  }
  return ncclSuccess;
}

inline bool ncclRankMaskIsActive(const ncclCommMaskValue_t* mask, int nRanks, int rank) {
  if (rank < 0 || rank >= nRanks) return false;
  if (mask == nullptr) return true;
  return mask[rank] == ncclRankMaskActive;
}

inline ncclResult_t ncclRankMaskSetActive(ncclCommMaskValue_t* mask, int nRanks, int rank, bool active) {
  if (rank < 0 || rank >= nRanks) return ncclInvalidArgument;
  if (mask == nullptr) return active ? ncclSuccess : ncclInvalidArgument;
  mask[rank] = active ? ncclRankMaskActive : ncclRankMaskInactive;
  return ncclSuccess;
}

inline int ncclRankMaskCountActive(const ncclCommMaskValue_t* mask, int nRanks) {
  if (nRanks <= 0) return 0;
  if (mask == nullptr) return nRanks;
  int count = 0;
  for (int rank = 0; rank < nRanks; ++rank) {
    if (ncclRankMaskIsActive(mask, nRanks, rank)) count++;
  }
  return count;
}

inline bool ncclRankMaskIsFullyActive(const ncclCommMaskValue_t* mask, int nRanks) {
  if (nRanks < 0) return false;
  return ncclRankMaskCountActive(mask, nRanks) == nRanks;
}

#endif
