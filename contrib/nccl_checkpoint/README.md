# NCCL Checkpoint Shim

This library enables a collective checkpoint of all processes that are part of
one or more NCCL communicators across multiple hosts.  During checkpoint
preparation, the shim masks every tracked communicator down to the local rank.
During restore, rank 0 of each communicator accepts the remaining ranks back
into the communicator with NCCL's mask/join APIs.  Application communicator
handles remain the original NCCL communicator pointers.

# Maintainers

| GitHub | Areas |
|--------|------|
| @lrbison | NCCL Checkpoint Core Features |

## Design

The application is launched with
`LD_PRELOAD=/path/to/libnccl-checkpoint-shim.so` in the environment.  This
allows the library to intercept communicator creation and destruction calls.

Checkpointing is a collective operation which involves all processes which share
one or more NCCL communicators.

Prior to checkpoint, the application invokes `ncclCheckpointPrepare()` and all
communicators will be masked so that only the local rank remains active.  No
communicator handles are replaced.

On checkpoint restore, the application invokes `ncclCheckpointRestore()` and the
library rejoins all ranks into each tracked communicator.  Rank 0 of each
communicator acts as the survivor and publishes a new join unique ID.  All other
ranks use their existing communicator as joiners.

Because it is useful to restore on different hardware, IP addresses may have
changed.  There is no convenient way to directly inform the NCCL Checkpoint
library of all peer addresses during the restore process, so the library
depends on a temporary Redis Key-Value store to be made available.  The shim
uses the Key-Value store to publish the join unique ID for each communicator
during restore.

## Usage Examples

### Python

```python
import os
import cuda.bindings.driver as drv
import nccl_checkpoint

# Complete pre-checkpoint work, then:
nccl_checkpoint.checkpoint_prepare()
drv.cuCheckpointProcessLock(os.getpid(), None)
drv.cuCheckpointProcessCheckpoint(os.getpid(), None)
# CRIU dump happens here.
drv.cuCheckpointProcessRestore(os.getpid(), None)
drv.cuCheckpointProcessUnlock(os.getpid(), None)
nccl_checkpoint.checkpoint_restore()
# Continue post-checkpoint work.
```

Launch the application with the shim preloaded:

```bash
LD_PRELOAD=build/lib/libnccl-checkpoint-shim.so \
NCCL_CHECKPOINT_KVS_PATH=/shared/path/kvs.txt \
python3 application.py
```

### C

```c
#include <cuda.h>
#include <nccl_checkpoint.h>
#include <stdio.h>
#include <unistd.h>

int main(void) {
  ncclCheckpointApi checkpoint;
  /* resolve ncclCheckpointPrepare and ncclCheckpointRestore from LD_PRELOAD
     and store them as pointers in checkpoint.prepare() and
     checkpoint.restore() respectively. */
  if (ncclCheckpointApiLoad(&checkpoint) != 0) {
    fprintf(stderr, "NCCL checkpoint shim is not loaded; set LD_PRELOAD\n");
    return 1;
  }

  /* Complete NCCL communicator creation and pre-checkpoint work, then: */
  checkpoint.prepare();
  cuCheckpointProcessLock(getpid(), NULL);
  cuCheckpointProcessCheckpoint(getpid(), NULL);
  /* CRIU dump happens here. */
  cuCheckpointProcessRestore(getpid(), NULL);
  cuCheckpointProcessUnlock(getpid(), NULL);
  checkpoint.restore();
  /* Continue post-checkpoint work. */

  return 0;
}
```

Launch the application with the shim preloaded:

```bash
LD_PRELOAD=build/lib/libnccl-checkpoint-shim.so \
NCCL_CHECKPOINT_KVS_PATH=/shared/path/kvs.txt \
./application
```

Because the NCCL Checkpointing library depends on LD_PRELOAD, C applications
which intend to call ncclCheckpointPrepare() functions need not link directly
against libnccl-checkpoint-shim.so.  Instead a header-only library is provided
to resolve the ncclCheckpointPrepare and ncclCheckpointRestore symbols using
dlsym at runtime.  If LD_PRELOAD has been properly set, ncclCheckpointApiLoad
will return 0 to indicate success, and .prepare() and .restore() functions will
be callable.

The variable `NCCL_CHECKPOINT_KVS_PATH` is discussed in the **Environment**
section.

## Current POC Assumptions

The NCCL Checkpointing library continues to improve, but at this time the
following assumptions exist.  These assumptions will be addressed in future
development.

- Before `ncclCheckpointPrepare()`, tracked communicators must have a full
  active-rank mask.
- After `ncclCheckpointRestore()`, tracked communicators are expected to be
  fully active.
- Rank 0 of each communicator is the restore survivor.  All other ranks rejoin
  using their existing communicator.
- Communicator creation and destruction must not race with prepare or restore.
- `ncclCommShrink` and `ncclCommGrow` are not checkpoint-managed in this first
  implementation.


# Installation

## Compilation

Invoke `make` from within this directory (nccl/contrib/nccl_checkpoint):

```bash
make
```

By default, the build reads NCCL sources from `../../src`, generates the NCCL
public header from `NCCL_SRC/nccl.h.in`, and writes checkpoint build artifacts
under `./build`.

The install target copies the checkpoint shared library and C header into
`PREFIX`, which defaults to `/usr/local`.

```bash
make install PREFIX=/path/to/install
```

### Dependencies and Build Configuration

The NCCL Checkpoint library has build-dependencies on CUDA, hiredis, and NCCL
itself.  The checkpointing library depends on both public and internal NCCL
headers from `NCCL_SRC`.  It is recommended the version of NCCL found in the
`NCCL_SRC` path precisely matches the version of the NCCL library used at
runtime.

| Variable | Purpose | Default Value |
| --- | --- | --- |
| `NCCL_SRC` | NCCL source tree. Used for both internal and public NCCL headers. | `../../src` |
| `BUILDDIR` | NCCL Checkpoint build directory for generated shim code, objects, libraries, and test binaries. | `./build` |
| `PREFIX` | Install prefix used by `make install`; the shim installs under `$PREFIX/lib`. | `/usr/local` |
| `CUDA_HOME` | CUDA Toolkit root inherited from the NCCL common Makefile. | `/usr/local/cuda` |
| `HIREDIS_HOME` | Hiredis install prefix. Requires >= v1.0 | Determined using `pkg-config` |

On Debian and Ubuntu, system Hiredis development files are provided by
`libhiredis-dev`.

## Python Build (Optional)

The Python bindings are found in `python/` and use `ctypes` to expose the
function provided by the shared library.

Editable install:

```bash
python3 -m pip install -e python
```

Wheel build:

```bash
python3 -m build python
```

The Python package itself has no runtime dependency on CUDA Python, although to
run the tests, additional dependencies are required.  See the [Python test
Readme](python/tests/README.md) for more details.


## Environment and Runtime Configuration

Two environment variables are required to be set: The application must be
launched with `LD_PRELOAD` pointing at libnccl-checkpoint-shim.so, and
`NCCL_CHECKPOINT_KVS_PATH` must be set prior to checkpointing.

```bash
LD_PRELOAD=build/lib/libnccl-checkpoint-shim.so \
NCCL_CHECKPOINT_KVS_PATH=/shared/path/kvs.txt \
program.py ...
```

`LD_PRELOAD` is a Linux dynamic linker environment variable that loads the named
shared object before the application's normal shared libraries. See the
[`ld.so(8)` manual page](https://man7.org/linux/man-pages/man8/ld.so.8.html) for
the full `LD_PRELOAD` behavior and restrictions.  This must be set before the
application is launched, because it has no effect if changed mid-execution.

`NCCL_CHECKPOINT_KVS_PATH` is a path to a file which will be used during restore
time.  During checkpoint preparation, this environment variable is not used.
However, it will be captured as part of the checkpoint and cannot be easily
modified during restore. Prior to restore, a redis server should be launched,
and the IP and port of that redis server should be provided as
`<address>:<port>` in the file named by NCCL_CHECKPOINT_KVS_PATH.  All processes
will read this file and exchange information using the KVS server.

The redis server is only required to bootstrap the restore process.  Once
checkpoint_restore() has returned on all processes, the redis server is not
needed, and can be stopped or destroyed (unless it is shared with other
services).

`NCCL_CHECKPOINT_KVS_TIMEOUT` controls how long the restore path waits for the
redis server to accept connections and for restore keys to appear. The value is
in seconds and defaults to 300.

## Additional Documentation

- [Python bindings](python/README.md)
- [Python tests](python/tests/README.md)
