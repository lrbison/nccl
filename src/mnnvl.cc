/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2015-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * See LICENSE.txt for more license information
 *************************************************************************/

#include "mnnvl.h"
#include "transport.h"
#include <cuda.h>
#include "cudawrap.h"
#include <errno.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#if defined(NCCL_OS_LINUX)
#include <dirent.h>
#include <limits.h>
#include <sys/stat.h>
#include <sys/sysmacros.h>
#include <unistd.h>
#endif

static const char* ncclMnnvlCuErrorString(CUresult err) {
  const char* errStr = "unknown";
  if (CUPFN(cuGetErrorString) != NULL) (void)CUPFN(cuGetErrorString(err, &errStr));
  return errStr;
}

static const char* ncclMnnvlEnv(const char* name) {
  const char* value = getenv(name);
  return value ? value : "<unset>";
}

#if defined(NCCL_OS_LINUX)
static const char* ncclMnnvlFileType(const struct stat* st) {
  if (S_ISDIR(st->st_mode)) return "dir";
  if (S_ISCHR(st->st_mode)) return "char";
  if (S_ISREG(st->st_mode)) return "regular";
  if (S_ISLNK(st->st_mode)) return "symlink";
  return "other";
}

static void ncclMnnvlLogImexProcDevices(void) {
  FILE* f = fopen("/proc/devices", "r");
  if (f == NULL) {
    INFO(NCCL_INIT, "MNNVL IMEX proc-devices open failed errno %d (%s)", errno, strerror(errno));
    return;
  }

  char line[256];
  int found = 0;
  while (fgets(line, sizeof(line), f) != NULL) {
    if (strstr(line, "nvidia-caps-imex-channels") != NULL) {
      line[strcspn(line, "\n")] = '\0';
      INFO(NCCL_INIT, "MNNVL IMEX proc-devices entry: %s", line);
      found = 1;
    }
  }
  fclose(f);
  if (!found) INFO(NCCL_INIT, "MNNVL IMEX proc-devices entry missing for nvidia-caps-imex-channels");
}

static void ncclMnnvlLogImexChannelPath(const char* path, const char* name) {
  struct stat st;
  if (stat(path, &st) != 0) {
    INFO(NCCL_INIT, "MNNVL IMEX channel path %s stat failed errno %d (%s)", path, errno, strerror(errno));
    return;
  }

  int r = access(path, R_OK);
  int rErrno = r == 0 ? 0 : errno;
  int w = access(path, W_OK);
  int wErrno = w == 0 ? 0 : errno;
  if (S_ISCHR(st.st_mode)) {
    INFO(NCCL_INIT,
         "MNNVL IMEX channel %s type %s mode %04o uid %d gid %d major %u minor %u access R %d(%s) W %d(%s)",
         name, ncclMnnvlFileType(&st), (unsigned int)(st.st_mode & 07777), (int)st.st_uid, (int)st.st_gid,
         (unsigned int)major(st.st_rdev), (unsigned int)minor(st.st_rdev), r, rErrno ? strerror(rErrno) : "ok",
         w, wErrno ? strerror(wErrno) : "ok");
  } else {
    INFO(NCCL_INIT, "MNNVL IMEX channel %s type %s mode %04o uid %d gid %d access R %d(%s) W %d(%s)", name,
         ncclMnnvlFileType(&st), (unsigned int)(st.st_mode & 07777), (int)st.st_uid, (int)st.st_gid,
         r, rErrno ? strerror(rErrno) : "ok", w, wErrno ? strerror(wErrno) : "ok");
  }
}

static void ncclMnnvlLogImexChannels(void) {
  const char* dirPath = "/dev/nvidia-caps-imex-channels";
  uid_t uid = getuid();
  gid_t gid = getgid();
  uid_t euid = geteuid();
  gid_t egid = getegid();

  INFO(NCCL_INIT, "MNNVL IMEX identity uid %d euid %d gid %d egid %d", (int)uid, (int)euid, (int)gid, (int)egid);
  ncclMnnvlLogImexProcDevices();
  ncclMnnvlLogImexChannelPath(dirPath, dirPath);

  DIR* dir = opendir(dirPath);
  if (dir == NULL) {
    INFO(NCCL_INIT, "MNNVL IMEX opendir %s failed errno %d (%s)", dirPath, errno, strerror(errno));
    return;
  }

  int count = 0;
  struct dirent* entry;
  while ((entry = readdir(dir)) != NULL) {
    if (strcmp(entry->d_name, ".") == 0 || strcmp(entry->d_name, "..") == 0) continue;
    char path[PATH_MAX];
    int n = snprintf(path, sizeof(path), "%s/%s", dirPath, entry->d_name);
    if (n < 0 || n >= (int)sizeof(path)) {
      INFO(NCCL_INIT, "MNNVL IMEX channel path too long under %s: %s", dirPath, entry->d_name);
      continue;
    }
    ncclMnnvlLogImexChannelPath(path, entry->d_name);
    count++;
    if (count >= 16) {
      INFO(NCCL_INIT, "MNNVL IMEX channel listing truncated after %d entries", count);
      break;
    }
  }
  closedir(dir);
  INFO(NCCL_INIT, "MNNVL IMEX channel entries listed %d", count);
}
#else
static void ncclMnnvlLogImexChannels(void) {
}
#endif

static void ncclMnnvlLogCudaState(struct ncclComm* comm, const char* phase) {
  int cudaDev = -1;
  int cudaCount = -1;
  int driverVersion = -1;
  int runtimeVersion = -1;
  cudaError_t getDevErr = cudaGetDevice(&cudaDev);
  cudaError_t countErr = cudaGetDeviceCount(&cudaCount);
  cudaError_t driverErr = cudaDriverGetVersion(&driverVersion);
  cudaError_t runtimeErr = cudaRuntimeGetVersion(&runtimeVersion);
  CUcontext ctx = NULL;
  CUdevice ctxDev = -1;
  CUdevice cuDev = -1;
  int attrVmm = -1;
  int attrPosixFd = -1;
  int attrFabric = -1;
  int attrGdrVmm = -1;
  int attrMempoolHandles = -1;
  CUresult ctxErr = CUPFN(cuCtxGetCurrent(&ctx));
  CUresult ctxDevErr = ctx == NULL ? CUDA_ERROR_INVALID_CONTEXT : CUPFN(cuCtxGetDevice(&ctxDev));
  CUresult devErr = CUDA_ERROR_INVALID_DEVICE;

  if (getDevErr == cudaSuccess) devErr = CUPFN(cuDeviceGet(&cuDev, cudaDev));
  if (devErr == CUDA_SUCCESS) {
    (void)CUPFN(cuDeviceGetAttribute(&attrVmm, CU_DEVICE_ATTRIBUTE_VIRTUAL_MEMORY_MANAGEMENT_SUPPORTED, cuDev));
    (void)CUPFN(cuDeviceGetAttribute(&attrPosixFd, CU_DEVICE_ATTRIBUTE_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR_SUPPORTED,
                                     cuDev));
    (void)CUPFN(cuDeviceGetAttribute(&attrFabric, CU_DEVICE_ATTRIBUTE_HANDLE_TYPE_FABRIC_SUPPORTED, cuDev));
    (void)CUPFN(cuDeviceGetAttribute(&attrGdrVmm, CU_DEVICE_ATTRIBUTE_GPU_DIRECT_RDMA_WITH_CUDA_VMM_SUPPORTED,
                                     cuDev));
    (void)CUPFN(cuDeviceGetAttribute(&attrMempoolHandles, CU_DEVICE_ATTRIBUTE_MEMPOOL_SUPPORTED_HANDLE_TYPES, cuDev));
  }

  INFO(NCCL_INIT,
       "MNNVL diag %s comm %p rank %d nranks %d cudaDev %d(%s) commCudaDev %d cudaCount %d(%s) "
       "driver %d(%s) runtime %d(%s) cuDev %d(%d:%s) ctx %p(%d:%s) ctxDev %d(%d:%s) "
       "cuMemEnable %d globalCuMemHandleType 0x%x",
       phase, comm, comm->rank, comm->nRanks, cudaDev, cudaGetErrorString(getDevErr), comm->cudaDev, cudaCount,
       cudaGetErrorString(countErr), driverVersion, cudaGetErrorString(driverErr), runtimeVersion,
       cudaGetErrorString(runtimeErr), (int)cuDev, (int)devErr, ncclMnnvlCuErrorString(devErr), ctx, (int)ctxErr,
       ncclMnnvlCuErrorString(ctxErr), (int)ctxDev, (int)ctxDevErr, ncclMnnvlCuErrorString(ctxDevErr),
       ncclCuMemEnable(), (int)ncclCuMemHandleType);
  INFO(NCCL_INIT,
       "MNNVL diag %s CUDA attrs vmm %d posixFd %d fabric %d gdrVmm %d mempoolHandleTypes 0x%x",
       phase, attrVmm, attrPosixFd, attrFabric, attrGdrVmm, attrMempoolHandles);
  INFO(NCCL_INIT, "MNNVL diag %s env NCCL_MNNVL_ENABLE=%s CUDA_VISIBLE_DEVICES=%s NVIDIA_VISIBLE_DEVICES=%s",
       phase, ncclMnnvlEnv("NCCL_MNNVL_ENABLE"), ncclMnnvlEnv("CUDA_VISIBLE_DEVICES"),
       ncclMnnvlEnv("NVIDIA_VISIBLE_DEVICES"));
}

static void ncclMnnvlLogPeerFabric(struct ncclComm* comm, const char* phase) {
  for (int i = 0; i < comm->nRanks; i++) {
    nvmlGpuFabricInfoV_t* fabricInfo = &comm->peerInfo[i].fabricInfo;
    unsigned long uuid0 = 0;
    unsigned long uuid1 = 0;
    memcpy(&uuid0, fabricInfo->clusterUuid, sizeof(uuid0));
    memcpy(&uuid1, fabricInfo->clusterUuid + sizeof(uuid0), sizeof(uuid1));
    INFO(NCCL_INIT,
         "MNNVL diag %s peer rank %d cudaDev %d nvmlDev %d busId 0x%lx hostHash 0x%lx cuMemSupport %d "
         "fabricHandleSupport %d fabricState %d healthMask 0x%x uuid %lx.%lx cliqueId 0x%x",
         phase, comm->peerInfo[i].rank, comm->peerInfo[i].cudaDev, comm->peerInfo[i].nvmlDev,
         comm->peerInfo[i].busId, comm->peerInfo[i].hostHash, comm->peerInfo[i].cuMemSupport,
         comm->peerInfo[i].fabricHandleSupport, fabricInfo->state, fabricInfo->healthMask, uuid0, uuid1,
         fabricInfo->cliqueId);
  }
}

static void ncclMnnvlLogDiagnostics(struct ncclComm* comm, const char* phase) {
  ncclMnnvlLogCudaState(comm, phase);
  ncclMnnvlLogPeerFabric(comm, phase);
  ncclMnnvlLogImexChannels();
}

static void ncclMnnvlProbeFabricAllocation(struct ncclComm* comm, const char* phase) {
  size_t size = CUDA_IPC_MIN;
  size_t granularity = 0;
  CUdevice currentDev = -1;
  CUmemAllocationProp prop = {};
  CUmemAccessDesc accessDesc = {};
  CUmemGenericAllocationHandle handle = 0;
  CUmemGenericAllocationHandle importedHandle = 0;
  ncclCuDesc cuDesc;
  CUdeviceptr ptr = 0;
  int cudaDev = -1;
  int gdrFlag = 0;
  CUresult err;

  memset(&cuDesc, 0, sizeof(cuDesc));
  cudaError_t cudaErr = cudaGetDevice(&cudaDev);
  if (cudaErr != cudaSuccess) {
    INFO(NCCL_INIT, "MNNVL diag %s step cudaGetDevice failed %d(%s)", phase, cudaErr, cudaGetErrorString(cudaErr));
    return;
  }
  err = CUPFN(cuDeviceGet(&currentDev, cudaDev));
  INFO(NCCL_INIT, "MNNVL diag %s step cuDeviceGet cudaDev %d -> %d(%s) cuDev %d", phase, cudaDev, err,
       ncclMnnvlCuErrorString(err), (int)currentDev);
  if (err != CUDA_SUCCESS) return;

  prop.type = CU_MEM_ALLOCATION_TYPE_PINNED;
  prop.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
  prop.requestedHandleTypes = CU_MEM_HANDLE_TYPE_FABRIC;
  prop.location.id = currentDev;
  err = CUPFN(cuDeviceGetAttribute(&gdrFlag, CU_DEVICE_ATTRIBUTE_GPU_DIRECT_RDMA_WITH_CUDA_VMM_SUPPORTED, currentDev));
  INFO(NCCL_INIT, "MNNVL diag %s step gdrVmm attr -> %d(%s) value %d", phase, err, ncclMnnvlCuErrorString(err),
       gdrFlag);
  if (err == CUDA_SUCCESS && gdrFlag) prop.allocFlags.gpuDirectRDMACapable = 1;

  err = CUPFN(cuMemGetAllocationGranularity(&granularity, &prop, CU_MEM_ALLOC_GRANULARITY_MINIMUM));
  INFO(NCCL_INIT, "MNNVL diag %s step cuMemGetAllocationGranularity type FABRIC gdrCapable %u -> %d(%s) gran %zu",
       phase, prop.allocFlags.gpuDirectRDMACapable, err, ncclMnnvlCuErrorString(err), granularity);
  if (err != CUDA_SUCCESS) return;
  ALIGN_SIZE(size, granularity);

  err = CUPFN(cuMemCreate(&handle, size, &prop, 0));
  INFO(NCCL_INIT, "MNNVL diag %s step cuMemCreate size %zu type FABRIC dev %d gdrCapable %u -> %d(%s) handle 0x%llx",
       phase, size, (int)currentDev, prop.allocFlags.gpuDirectRDMACapable, err, ncclMnnvlCuErrorString(err), handle);
  if (err != CUDA_SUCCESS) return;

  err = CUPFN(cuMemAddressReserve(&ptr, size, granularity, 0, 0));
  INFO(NCCL_INIT, "MNNVL diag %s step cuMemAddressReserve size %zu gran %zu -> %d(%s) ptr %p", phase, size,
       granularity, err, ncclMnnvlCuErrorString(err), (void*)ptr);
  if (err != CUDA_SUCCESS) goto release_handle;

  err = CUPFN(cuMemMap(ptr, size, 0, handle, 0));
  INFO(NCCL_INIT, "MNNVL diag %s step cuMemMap ptr %p size %zu -> %d(%s)", phase, (void*)ptr, size, err,
       ncclMnnvlCuErrorString(err));
  if (err != CUDA_SUCCESS) goto free_addr;

  accessDesc.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
  accessDesc.location.id = currentDev;
  accessDesc.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
  err = CUPFN(cuMemSetAccess(ptr, size, &accessDesc, 1));
  INFO(NCCL_INIT, "MNNVL diag %s step cuMemSetAccess ptr %p dev %d -> %d(%s)", phase, (void*)ptr, (int)currentDev,
       err, ncclMnnvlCuErrorString(err));
  if (err != CUDA_SUCCESS) goto unmap;

  err = CUPFN(cuMemExportToShareableHandle(&cuDesc, handle, CU_MEM_HANDLE_TYPE_FABRIC, 0));
  INFO(NCCL_INIT, "MNNVL diag %s step cuMemExportToShareableHandle type FABRIC -> %d(%s)", phase, err,
       ncclMnnvlCuErrorString(err));
  if (err != CUDA_SUCCESS) goto unmap;

  err = CUPFN(cuMemImportFromShareableHandle(&importedHandle, &cuDesc, CU_MEM_HANDLE_TYPE_FABRIC));
  INFO(NCCL_INIT, "MNNVL diag %s step cuMemImportFromShareableHandle type FABRIC -> %d(%s) handle 0x%llx",
       phase, err, ncclMnnvlCuErrorString(err), importedHandle);
  if (err == CUDA_SUCCESS) {
    CUresult releaseErr = CUPFN(cuMemRelease(importedHandle));
    INFO(NCCL_INIT, "MNNVL diag %s step cuMemRelease imported handle -> %d(%s)", phase, releaseErr,
         ncclMnnvlCuErrorString(releaseErr));
  }

unmap:
  err = CUPFN(cuMemUnmap(ptr, size));
  INFO(NCCL_INIT, "MNNVL diag %s cleanup cuMemUnmap -> %d(%s)", phase, err, ncclMnnvlCuErrorString(err));
free_addr:
  err = CUPFN(cuMemAddressFree(ptr, size));
  INFO(NCCL_INIT, "MNNVL diag %s cleanup cuMemAddressFree -> %d(%s)", phase, err, ncclMnnvlCuErrorString(err));
release_handle:
  err = CUPFN(cuMemRelease(handle));
  INFO(NCCL_INIT, "MNNVL diag %s cleanup cuMemRelease original -> %d(%s)", phase, err, ncclMnnvlCuErrorString(err));
}

// Determine if MNNVL support is available
ncclResult_t ncclMnnvlCheck(struct ncclComm* comm) {
  // MNNVL requires cuMem to be enabled
  ncclMnnvlLogDiagnostics(comm, "entry");
  if (!ncclCuMemEnable()) {
    INFO(NCCL_INIT, "MNNVL disabled: cuMem is not enabled");
    return ncclSuccess;
  }

  // MNNVL also requires FABRIC handle support
  for (int i = 0; i < comm->nRanks; i++) {
    if (!comm->peerInfo[i].fabricHandleSupport) {
      INFO(NCCL_INIT, "MNNVL disabled: rank %d lacks CUDA fabric handle support", i);
      return ncclSuccess;
    }
  }
  // Check that all ranks have initialized the fabric fully
  for (int i = 0; i < comm->nRanks; i++) {
    if (comm->peerInfo[i].fabricInfo.state != NVML_GPU_FABRIC_STATE_COMPLETED) {
      INFO(NCCL_INIT, "MNNVL disabled: rank %d fabric state %d is not completed", i,
           comm->peerInfo[i].fabricInfo.state);
      return ncclSuccess;
    }
  }

  // Determine our MNNVL domain/clique and NVL domain size
  NCCLCHECK(ncclCalloc(&comm->clique.ranks, comm->nRanks));
  comm->clique.id = comm->peerInfo[comm->rank].fabricInfo.cliqueId;
  comm->nvlDomainSize = 0;
  for (int i = 0; i < comm->nRanks; i++) {
    nvmlGpuFabricInfoV_t* fabricInfo1 = &comm->peerInfo[comm->rank].fabricInfo;
    nvmlGpuFabricInfoV_t* fabricInfo2 = &comm->peerInfo[i].fabricInfo;
    // Check if the cluster UUID and cliqueId match
    // A zero UUID means we don't have MNNVL fabric info - disable MNNVL
    unsigned long uuid0 = 0;
    unsigned long uuid1 = 0;
    memcpy(&uuid0, fabricInfo2->clusterUuid, sizeof(uuid0));
    memcpy(&uuid1, fabricInfo2->clusterUuid + sizeof(uuid0), sizeof(uuid1));
    if ((uuid0 | uuid1) == 0) {
      INFO(NCCL_INIT, "MNNVL disabled: rank %d fabric UUID is zero", i);
      return ncclSuccess;
    }
    // Check if same NVL domain (clusterUuid match)
    if (memcmp(fabricInfo1->clusterUuid, fabricInfo2->clusterUuid, NVML_GPU_FABRIC_UUID_LEN) == 0) {
      comm->nvlDomainSize++;
      // Also check if same clique (cliqueId match)
      if (fabricInfo1->cliqueId == fabricInfo2->cliqueId) {
        if (i == comm->rank) {
          comm->cliqueRank = comm->clique.size;
        }
        comm->clique.ranks[comm->clique.size++] = i;
      }
    }
  }

  // ncclCommSplit: clique.size may be 1 while nvlDomainSize > 1; still enable MNNVL.
  INFO(NCCL_INIT, "MNNVL candidate cliqueId 0x%x cliqueSize %d cliqueRank %d nvlDomainSize %d", comm->clique.id,
       comm->clique.size, comm->cliqueRank, comm->nvlDomainSize);
  if (comm->clique.size <= 1 && comm->nvlDomainSize <= 1) {
    INFO(NCCL_INIT, "MNNVL disabled: cliqueSize %d and nvlDomainSize %d are not multi-node", comm->clique.size,
         comm->nvlDomainSize);
    return ncclSuccess;
  }

  // Check that FABRIC handles can be exported & imported by IMEX
  {
    void* ptr = NULL;
    CUmemGenericAllocationHandle handle;
    ncclCuDesc cuDesc;
    CUresult err;

    // Allocate FABRIC handle compatible memory
    ncclResult_t ret =
      ncclCuMemAlloc(&ptr, &handle, CU_MEM_HANDLE_TYPE_FABRIC, CUDA_IPC_MIN, comm->memManager, ncclMemOffload);
    if (ret != ncclSuccess) {
      ncclMnnvlLogDiagnostics(comm, "alloc-failed");
      ncclMnnvlProbeFabricAllocation(comm, "alloc-failed");
      // Return an error if this is a MNNVL capable system but FABRIC handles are not supported
      WARN("MNNVL (cliqueSize %d) is available but not working on this system. Check the IMEX channel configuration "
           "(/dev/nvidia-caps-imex-channels). Set NCCL_MNNVL_ENABLE=0 to ignore this issue.",
           comm->clique.size);
      return ncclSystemError;
    }
    err = CUPFN(cuMemExportToShareableHandle(&cuDesc, handle, CU_MEM_HANDLE_TYPE_FABRIC, 0));
    if (err != CUDA_SUCCESS ||
        (err = CUPFN(cuMemImportFromShareableHandle(&handle, &cuDesc, CU_MEM_HANDLE_TYPE_FABRIC))) != CUDA_SUCCESS) {
      const char* errStr;
      (void)pfn_cuGetErrorString(err, &errStr);
      INFO(NCCL_INIT, "MNNVL FABRIC export/import failed with CUDA error %d '%s'", err, errStr);
      ncclMnnvlLogDiagnostics(comm, "export-import-failed");
      ncclMnnvlProbeFabricAllocation(comm, "export-import-failed");
      NCCLCHECK(ncclCuMemFree(ptr, comm->memManager));
      // Return an error if this is a MNNVL capable system but it's not working
      WARN("MNNVL (cliqueSize %d) is available but not working on this system. Check the IMEX configuration "
           "(nvidia-imex-ctl -N). Set NCCL_MNNVL_ENABLE=0 to ignore this issue.",
           comm->clique.size);
      return ncclSystemError;
    }
    NCCLCHECK(ncclCuMemFree(ptr, comm->memManager));

    // Force the CUMEM handle type to be FABRIC for MNNVL
    ncclCuMemHandleType = CU_MEM_HANDLE_TYPE_FABRIC;
    comm->MNNVL = 1;
    INFO(NCCL_INIT, "MNNVL %d cliqueId %x cliqueSize %d cliqueRank %d nvlDomainSize %d", comm->MNNVL, comm->clique.id,
         comm->clique.size, comm->cliqueRank, comm->nvlDomainSize);
  }
  return ncclSuccess;
}
