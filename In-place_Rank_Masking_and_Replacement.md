# In-place Rank Masking and Replacement
<!-- use this template to share the details of your feature. Non-commented sections are strongly encouraged -->

## Abstract

In-place rank masking and reshape lets NCCL temporarily remove failed or unavailable
rank slots from a communicator without replacing the communicator object and
without invalidating graph-captured communicator state. A new
`ncclCommReshape` operation supports collective rank-slot changes with leave and
join lists, plus a local-only remove mode for failure or checkpoint quiescence.
It updates a communicator-owned active mask while preserving rank numbering,
maximum-rank layout, local CUDA allocations, symmetric-memory reservations,
device communicator storage, and other graph-visible pointers. Reshape either
commits the requested membership change or returns a non-success result without
committing a new communicator membership state. Replacement or restored ranks
enter reshape through normal communicator initialization using a reusable reshape
unique ID.

The primary goal is dynamic membership repair for graph-captured applications.
The immediate success conditions are:

1. graph-captured applications function after NCCL checkpoint/restore; and
2. graph-captured applications can execute a mask/reshape sequence to replace
   non-functional ranks without recapturing survivor graphs.

<!-- ============================================================================================-->
<details>
<summary><h2>Motivation and requirements</h2></summary>
<!-- ============================================================================================-->

### Feature context and focus

Current `ncclCommShrink` and `ncclCommGrow` create new communicators. That is a
good fit for conventional elastic execution, but it is not sufficient for
graph-captured applications because captured CUDA graphs embed NCCL kernel
arguments and device pointers by value. Replacing the communicator, rebuilding
graph-visible allocations at different addresses, or renumbering surviving
ranks can make existing graphs unusable.

Rank masking adds a different dynamic-membership primitive:

- `nRanks` remains the original communicator rank count and graph ABI.
- Rank IDs remain stable. Removing rank `f` leaves slot `f` inactive instead of
  creating dense survivor ranks.
- A communicator-owned active mask records which rank slots currently have live
  peer state.
- Masking releases remote liveness, transport connections, proxy state, and
  imported peer mappings for affected inactive ranks.
- Masking must not release local graph-visible storage, local symmetric-memory
  reservations, local window objects, or device communicator storage.
- Reshape refills inactive or to-be-replaced slots in-place after replacement
  ranks indicate they are compatible with the original communicator shape.

This is not intended to replace regular shrink/grow. Shrink/grow remain the
APIs for running with a new dense communicator. Rank masking is for cases where
the communicator's graph-visible layout must be preserved.

### User Experience

#### CRIU-checkpointed applications

Phase 1 of NCCL checkpoint work recreated communicators by replaying
initialization work. NCCL hid the fact that these are new resources by
using indirection, but stream-captured applications still observed the
differences and failed checkpoint compatibility.

With masked communicators, NCCL can use the mask to track low-level resource
teardown without removing the entire communicator or moving scratch buffers:

1. At checkpoint time, each rank uses local-only reshape/remove mode to mask all
   other ranks as deactivated. This closes network connections and unmaps fabric
   handles, allowing the process to be checkpointed.
2. After all tracked communicators are masked to self-only, the checkpoint shim
   calls `ncclNetQuiesce()` to release process-wide NET provider resources.
3. At restore time, each communicator calls `ncclCommRediscoverDevices()` while
   still masked to self-only so NCCL can rediscover local addresses/devices and
   clear stale discovery state.
4. At restore time, rank 0 acts as the survivor/founder and creates or reuses a
   reshape unique ID through `ncclCommGetUniqueId_v2` with a reshape flag. All
   other ranks act as joiners and call `ncclCommInit*` with that reshape ID.
5. Joiners perform application-specific warmup and window registration, then
   call `ncclCommReshape` on their pending communicators to signal readiness.
6. Survivor-side `ncclCommReshape` completes by sharing new fabric handles,
   mapping them into existing buffers, and remaking network connections using
   existing scratch buffers and FIFOs. If the joiners are not ready, reshape
   returns early without modifying the survivor communicator.

#### Fault-reactive resizing

A faulted rank can already be excluded by creating a new, smaller communicator
using shrink, and it can be replaced by growing the smaller communicator back to
the original size. Those are all new resources, which both strain resource
limits and are not graph-capture compatible.

With masked communicators, existing functional connections are retained and only
the non-functional parts are removed. Replacement ranks can join in place, and
survivor collectives do not need to be recaptured because the same buffers and
graph-visible communicator state can be reused.

#### In-situ masking within a kernel

This use case is under active development and is not the primary motivation for
this work. The active-mask design remains compatible with inference frameworks
that detect failures in-kernel.

A specially designed fault-tolerant collective, such as FTAllReduce, can have
immediate direct connections to all peers. This algorithm can learn of failures
during kernel execution and locally mask out peers to continue. This can result
in out-of-sync masks, which are expected to be synchronized later by the
application or by a future consensus path.

### Alternatives considered

Several alternative designs were considered:

- A design that applies a layer of abstraction between communicators and NCCL
  kernel arguments: this would likely have performance impacts, and would only
  help checkpointing, not fault-tolerant replacement.

- A design that applies a layer of abstraction to CUDA graphs, allowing NCCL
  to swap one communicator for another given a graph template. For
  checkpointing, this would require the ability to edit all user graphs.

- A design that lets NCCL insert a host-side check to reconnect dynamically
  added peers before the graph executes. This is not possible in CUDA today
  because graphs do not support modification while in flight. Furthermore,
  host functions within a graph are not allowed to invoke other CUDA APIs,
  which makes this kind of on-demand detection very limiting in what it could
  do.

### Proposed API shape

Public API additions:

```c
typedef enum {
  ncclRankMaskInactive = 0,
  ncclRankMaskActive = 1,
} ncclCommMaskValue_t;

#define NCCL_UNIQUE_ID_DEFAULT 0
#define NCCL_UNIQUE_ID_RESHAPE 1

#define NCCL_COMM_RESHAPE_DEFAULT 0 /* no optional reshape behavior */
#define NCCL_COMM_RESHAPE_LOCAL_ONLY 1 /* local remove/mask; no survivor consensus */

#define NCCL_COMM_INIT_DEFAULT 0
#define NCCL_COMM_INIT_REUSE_EXISTING 1 /* ncclComm_t* is in/out for reshape-ID init */

/* ncclConfig_t gains an append-only field: int commInitFlags; */
/* ncclResult_t gains ncclResourceNotReady for retryable not-ready reshape. */

ncclResult_t ncclCommGetUniqueId_v2(
    ncclComm_t comm,
    ncclUniqueId* uniqueId,
    int flags);

/* Reshape membership operation. Uses reshape ID state selected by leaderRank. */
ncclResult_t ncclCommReshape(
    ncclComm_t comm,
    int leaderRank,
    const int* joinList,
    int joinListLen,
    const int* leaveList,
    int leaveListLen,
    int flags);

/* local operation */
ncclResult_t ncclCommMaskGet(
    ncclComm_t comm,
    ncclCommMaskValue_t* mask);

ncclResult_t ncclCommMaskCountActive(
    ncclComm_t comm,
    int* count);

/* local operation: refresh local IP/NIC discovery after restore */
ncclResult_t ncclCommRediscoverDevices(ncclComm_t comm);

/* process-wide operation: release NET provider resources after all comms are quiesced. */
ncclResult_t ncclNetQuiesce(void);

/* internal struct: mask is owned by comm and updates happen through reshape only */
typedef struct ncclRankMaskState_t {
  int nRanks;
  /* don't give application direct access to this mask, only let them update via API */
  ncclCommMaskValue_t* activeRankMask;
} ncclRankMask_t;
```

### API semantics

`ncclCommReshape(comm, leaderRank, joinList, joinListLen, leaveList,
leaveListLen, flags)` is the high-level API for changing the active rank slots
of `comm` in place. `joinList` and `leaveList` are arrays of integer rank slots
in the original communicator shape, and each has an explicit length.

With `NCCL_COMM_RESHAPE_DEFAULT`, survivor ranks perform a collective and
consistent reshape. This mode is used for graceful swaps and joins: every
currently active rank that remains active after the reshape participates, and
any live rank named in `leaveList` participates before leaving.

With `NCCL_COMM_RESHAPE_LOCAL_ONLY`, a rank can perform a local remove/mask
transition without requiring failed or unreachable ranks to participate and
without requiring all surviving ranks to have identical masks immediately. This
mode is intended for failure handling and checkpoint prepare. It can only remove
rank slots from the local active mask; reactivating or replacing ranks requires
a subsequent collective reshape with ready joiners. When this flag is set,
`ncclCommReshape` does no collective communication, but continued successful
communication, including future non-local `ncclCommReshape` operations, may
require all ranks to have the same active-mask view.

Reshape is atomic with respect to committed communicator membership: it either
commits the requested membership transition or returns without installing a new
active mask. NCCL may create temporary or pending rendezvous resources while
checking readiness and compatibility, but those resources remain owned by
pending reshape state or are cleaned before return. A retryable failure must not
partially commit graph-visible communicator state, transport ownership, window
state, or enqueue eligibility.

The leader rank owns a reusable reshape ID scoped to `(comm, leaderRank)` and
created by `ncclCommGetUniqueId_v2(comm, &uniqueId,
NCCL_UNIQUE_ID_RESHAPE)`. Each rank owns at most one reshape ID per
communicator, but different leader ranks can own distinct reshape IDs for the
same communicator. NCCL stores enough state on `comm` to map each leader rank's
ID to a reshape rendezvous and to that leader's pool of staged joiners. Because
the ID is selected by `leaderRank`, `ncclCommReshape` only needs `comm` and
`leaderRank`; it does not take the ID as an argument. Repeated
`ncclCommGetUniqueId_v2(..., NCCL_UNIQUE_ID_RESHAPE)` calls by the same rank on
the same communicator return that rank's existing reshape ID.

Joiners first call `ncclCommInitRank` or `ncclCommInitRankConfig` with the
reshape ID and the destination full-rank slot as the `rank` argument. NCCL
determines from the ID that this is a reshape joiner rather than a normal init.
The joiner is added to the joiner pool associated with that reshape ID under
the target rank identifier used in init. For fault replacement, init creates or
specializes a full-shape communicator for the destination slot. For
checkpoint/restore, `NCCL_COMM_INIT_REUSE_EXISTING` in `ncclConfig_t` tells NCCL
to reuse the existing communicator pointed to by `*comm` and refresh it in
place.

After initial communicator setup, the joiner performs application-specific
warmup, including required window registrations and any graph-capture
preparation. Once ready, the joiner calls `ncclCommReshape` on its pending
communicator. On joiner-side calls, the joiner's target rank and leader are
derived from reshape-ID init state. `joinList` and `leaveList` must be `NULL`,
their lengths must be 0, and `flags` must be compatible with the pending init
state. The call signals readiness to the leader and then waits until the leader
commits a reshape that includes the joiner's target rank.

Joiner-side reshape follows the communicator's existing blocking mode. On a
blocking communicator, the call can block until the leader commits a matching
reshape or reports a failure for the pending joiner. On a nonblocking
communicator, reshape progresses through NCCL's existing async completion
mechanism and the application observes completion through the communicator's
normal nonblocking status checks. Applications cancel a pending reshape by
aborting the communicator with `ncclCommAbort`.

A committed reshape can close or reconnect remote transport connections,
detach or attach rank-scoped proxy state, and unmap or remap imported peer
symmetric memory for affected ranks. It must preserve communicator object
identity, local allocations, local registrations, device communicator storage,
work FIFO storage, and local window table entries.

`ncclCommCount(comm, count)` continues to report the original `nRanks`.
`ncclCommMaskCountActive(comm, count)` reports the number of currently active
rank slots. A communicator is fully populated when `activeCount == nRanks`.
The active count is derived by counting the active mask; it is not a separate
stored source of truth.

If a reshape includes join slots whose joiners are not ready, `ncclCommReshape`
returns `ncclResourceNotReady` and leaves committed membership unchanged. This
result is retryable: it must not poison `comm->asyncResult`, revoke the
communicator, or prevent a later identical reshape attempt. The exact public
result spelling is still subject to API review, but its semantics are
"not ready; safe to retry."

`ncclNetQuiesce()` is a process-wide recovery helper. It is called after all
communicators in the process have been quiesced, for example after checkpoint
prepare masks every tracked communicator to self-only. It releases NET provider
resources and is not implicitly triggered by one communicator's reshape.

`ncclCommRediscoverDevices(comm)` is a per-communicator restore helper. It is
called while the communicator is masked to self-only so NCCL can rediscover
local bootstrap/network addresses and clear stale discovery state before reshape
reconnects the communicator.

A partially masked communicator is management-only for normal collectives:
full-communicator collectives fail to enqueue until the active mask is full or a
specific mask-aware algorithm is selected.

The host-side reshape transition runs between collectives. Host reshape does
not update the mask with work in flight. In-kernel mask-out updates remain
future work for kernels that are first to detect a failure.

Initial collective reshape uses local readiness checks and requires
survivor-side argument consistency. Local-only remove mode is explicitly allowed
to be inconsistent across ranks so failure handling and checkpoint prepare do not
depend on failed peers. Stronger consensus modes for agreeing on masks across
active ranks remain future work and will be exposed through reshape flags.

For checkpoint restore, a joiner calls `ncclCommInitRankConfig` with a reshape
ID and `NCCL_COMM_INIT_REUSE_EXISTING` in `ncclConfig_t`. Because the
checkpointed communicator was already configured with shared resources prior to
checkpoint, the joiner does not need new registrations unless compatibility
validation discovers that the restored process state no longer matches the
preserved communicator shape. NCCL must not infer reuse from a non-`NULL` value
in `*comm` because existing init APIs treat `ncclComm_t*` as an output parameter
and callers can pass uninitialized storage.

Rank slots can appear in both `joinList` and `leaveList` to express atomic
same-slot replacement. For example, rank slot `X` can be removed and replaced
by a warmed spare that initialized as target rank `X` and is waiting for commit.
In this graceful same-slot replacement case, the old rank `X` is still a live
participant in the collective reshape before leaving. For real failures, the
application first removes failed ranks with local-only reshape/remove, then later
performs a collective reshape to refill the inactive slots.

Because a rank can be deactivated using a local-only `ncclCommReshape` with a
`leaveList`, there is no guarantee that all messages to or from the leaver were
delivered, nor that the leaver completed all existing collectives. After a
replacement rank's reactivation in `ncclCommReshape`, NCCL guarantees that all
messages and collective participation for that rank slot come only from the
replacement rank, and cannot be the result of previous actions from the leaver.

All survivor ranks participating in a reshape must pass consistent `joinList`,
`joinListLen`, `leaveList`, `leaveListLen`, `leaderRank`, and `flags` values.

### Assumptions, constraints, and dependencies

- `nRanks` remains the graph-captured communicator size. It is not decremented
  by masking.
- Rank IDs are stable across mask and reshape.
- The communicator object address and graph-visible device allocations must
  remain stable.
- Local memory reservations remain intact. Imported mappings for masked ranks
  can be unmapped and later remapped.
- Captured graphs that require inactive ranks are not replayable until reshape
  restores a compatible active mask.
- Fault-replacement ranks capture graphs after reshape-ID
  `ncclCommInit*` specializes them to their destination ranks and derives a
  topology.
- Checkpoint/restore joiners are different: they already have the graph-visible
  communicator shape from before checkpoint, so explicit init reuse must refresh
  those objects in place rather than requiring replacement-side graph recapture.
- Dev comms and GIN resources must continue to exist in place after reshape.
  They are management-visible but unavailable for operations that require
  inactive ranks while the active mask is incomplete.
- Existing revoke/finalize/suspend machinery can be reused for quiescence, but
  masking must not destroy or replace the communicator.

### Use Cases

1. **CRIU checkpoint and restore**

   Each rank masks remote rank slots, closing network connections and unmapping
   fabric handles before checkpoint. After restore, rank 0 acts as founder and
   other ranks initialize with a reshape ID using their existing communicators.

2. **Fault repair with spare rank**

   A rank fails in a graph-captured training or inference job. Survivors mask
   the failed rank slot and enter a repair state. A spare process initializes
   with the reusable reshape ID into the inactive or to-be-replaced slot.
   Survivors resume without graph recapture after reshape commits.

3. **Maintenance window for transport replacement**

   A rank or network path can be deliberately masked, its connections and
   imported handles released, then reshaped back with refreshed transport state.

4. **Future executable masked collectives**

   Future specialized algorithms can support execution on a partially active
   mask. Sparse execution is opt-in per algorithm or per operation.

### Non-goals for the first implementation

- Do not make ring/tree/NVLS algorithms automatically replan for arbitrary
  active subsets.
- Do not renumber surviving ranks.
- Do not promise that N-rank captured graphs can run while rank slots are
  inactive.
- Do not replace regular `ncclCommShrink` or `ncclCommGrow`.
- Do not release local graph-visible allocations, work FIFO storage, dev comm
  storage, symmetric-memory reservations, or local window objects during
  masking.

### Platform Requirements

- CUDA graph-capable NCCL configurations.
- Existing fault-tolerance quiescence support, including nonblocking
  communicators and revoke-like cancellation.
- Transport, proxy, symmetric-memory, device API, and GIN paths must expose
  enough teardown/reconnect hooks to close remote state while preserving local
  reservations.

</details>

<!-- ============================================================================================-->
<details>
<summary><h2>Design</h2></summary>
<!-- ============================================================================================-->

### Proposed Design

#### State model

Add in-place membership state to the communicator:

```c
struct ncclComm {
  ...
  int nRanks;                            // existing graph ABI and maximum rank slots
  ncclCommMaskValue_t* activeRankMask;   // one value per rank slot
  struct ncclReshapeLeaderState* reshapeLeader; // local reshape ID/listener state, if this rank is a leader
  ...
};
```

The active mask is the only stored membership state. Counting the active mask
equal to `nRanks` means the communicator is fully populated. A count less than
`nRanks` means normal full-communicator collectives are not allowed until reshape
restores the missing slots or a future mask-aware algorithm explicitly supports
sparse execution.

The reshape implementation avoids epoch-tracking. If stale
operation detection eventually needs a generation, keep it internal and do not
expose it in the public ABI.

The active mask has a device-visible representation for device APIs,
fault-tolerant kernels, and GIN code that need to check peer liveness. This is a
mirror or generation of the host active mask, not a separate communicator
membership enum.

Reshape IDs and joiner pools are scoped to `(comm, leaderRank)`. Each rank can
create at most one reusable reshape ID for a communicator, and each reshape ID
has its own pool of preliminary joiner connections keyed by the rank identifier
the joiner passed to reshape-ID init. A leader pool contains at most one pending
joiner per target rank. A second joiner for the same target rank either replaces
a failed or canceled pending joiner or returns an explicit in-use error.
`ncclCommReshape(comm, leaderRank, ...)` selects the leader state to use for a
reshape attempt.

The API shape can allow distinct leaders to own distinct reshape IDs for the
same communicator, which may help at scale by keeping joiner staging local to
the chosen leader. The initial implementation does not need to prove concurrent
multi-leader reshape attempts. Reshape follows normal NCCL collective ordering:
applications issue only one collective operation on a communicator at a time, so
committed reshape operations are ordered by the same rules as other collective
operations on that communicator.

#### Reshape operation

`ncclCommReshape` performs a cooperative in-place transition:

1. Validate the list pointers, list lengths, flags, leader rank, selected
   reshape ID state, and current communicator state.
2. Determine whether the call is a local-only remove/mask or a collective
   reshape. Local-only remove mode allows survivor masks to diverge temporarily
   and cannot add ranks. Collective reshape requires consistent survivor-side
   arguments.
3. Validate that `joinList` and `leaveList` entries are valid stable rank slots
   in the original `nRanks` shape. A rank slot can appear in both lists to
   express atomic same-slot replacement.
4. Compute the target active mask from the current active mask, the leave slots,
   and the join slots without changing the committed communicator membership.
5. For every rank in `joinList`, including same-slot replacement where the rank
   is also in `leaveList`, check that matching joiner init state is already
   staged through the reshape ID associated with `leaderRank`. If not, return
   `ncclResourceNotReady` without committing a new membership state.
6. Validate CUDA device properties, device API requirements, symmetric-window
   registration sequence, graph compatibility, GIN requirements, and transport
   compatibility for every staged joiner.
7. Stage all transport, proxy, peer-info, symmetric-memory, dev comm, and GIN
   updates needed for the target active mask.
8. Commit in one transition: close or reconnect rank-scoped resources for
   affected slots, install replacement state for join slots, refresh
   device-visible state, and update the host active mask.
9. Stop new non-management operations from being enqueued if the committed
   active mask is partial.
10. If the committed active mask is full, allow normal collective enqueue and
   compatible captured graph replay.

Consensus policies can be added later after the failure-detection and
survivor-agreement requirements are clearer. The first API shape uses reshape
flags instead of a separate configuration object.

#### Reshape protocol

Reshape is a role-specific protocol exposed through `ncclCommReshape` and
reshape-ID init. Survivors keep the original graph-visible communicator object.
Joiners initialize into inactive or to-be-replaced full-shape rank slots,
prepare any application-specific state, then use `ncclCommReshape` as their
ready-and-wait call.

1. The leader calls `ncclCommGetUniqueId_v2` with a reshape flag to create a
   reusable reshape ID scoped to `(comm, leaderRank)`, then shares it out of
   band with replacement or restored ranks.
2. A joiner calls `ncclCommInitRank` or `ncclCommInitRankConfig` with the
   reshape ID and its destination full-communicator rank. NCCL creates or
   specializes a pending full-shape communicator for that rank and records it in
   the joiner pool associated with that reshape ID.
3. The joiner performs application-specific warmup, including window
   registration and graph-capture preparation if needed.
4. The joiner calls `ncclCommReshape` on its pending communicator. This signals
   readiness to the leader and waits for a leader-side reshape commit naming the
   same rank slot. The target rank and leader are derived from reshape-ID init
   state; the list arguments must be `NULL` with length 0 on this joiner-side
   call.
5. Survivors collectively call `ncclCommReshape(comm, leaderRank, joinList,
   joinListLen, leaveList, leaveListLen, flags)` over the active survivor set.
   The leader consults the selected joiner pool for each rank in `joinList`.
6. If any named joiner is absent or not ready, survivor-side reshape returns a
   `ncclResourceNotReady` and committed membership remains unchanged.
7. If all required joiners are ready, reshape validates compatibility, stages
   resource updates, and commits the active-mask transition. Joiner-side reshape
   returns after the leader either commits a reshape containing that rank or
   fails the pending joiner with a clear error.

#### Replacement preparation

A warm spare can start generic and become specialized only when a failed rank
slot is known. The application assigns the spare to one inactive slot by passing
that slot as the `rank` argument to reshape-ID init. Reshape-ID init imports the
destination rank number, full `nRanks`, rank-to-node metadata, channel shape,
symmetric-memory layout, dev comm requirements, and GIN resource counts from the
masked communicator. A fault-replacement rank captures graphs after this
specialization. A graph captured against a generic staging communicator is
compatible only if it does not embed the staging rank identity or staging
communicator layout.

A checkpoint/restore joiner uses explicit init reuse mode, so its
pre-checkpoint graph-visible objects are refreshed in place rather than
replaced.

#### Compatibility contract

Before allowing a fully populated communicator to resume graph-captured work,
NCCL validates:

- same `nRanks`
- same local `rank` for explicit init reuse checkpoint restore
- same graph-visible communicator pointer for survivor and checkpoint-restore
  paths
- same channel count and algorithm plan shape
- same dev comm ABI/version and requirements
- same symmetric-window count, order, size, and alignment
- same local user buffer addresses for local windows when required by captured
  graphs
- imported windows remapped into expected reservation slots
- compatible GIN connection mode and resource counts
- compatible rank-to-node and local-rank metadata when algorithms depend on it

This produces a clear NCCL error rather than letting a captured graph fail
later in a harder-to-debug way.

#### Dev comm and team persistence

`ncclDevComm` storage and registered window handles are part of the
graph-visible contract. Captured kernels can have the device communicator
pointer and window references fixed in their launch arguments. Team views such
as world, LSA, rails, and GIN state are derived from the device communicator at
execution time, but reshape still has to preserve their logical shape for
captured kernels that expect the original communicator layout. Mask/reshape
therefore cannot replace graph-visible device communicator or window storage on
survivor or checkpoint-restore ranks.

The required behavior is:

- The device communicator pointer stays stable on survivors across mask and
  reshape.
- World team size remains `nRanks`; inactive ranks are represented by the active
  mask rather than by shrinking the team.
- LSA/rail/team descriptors remain valid only if the replacement rank is
  compatible with the original team membership. If a replacement changes
  local-rank, NVLink, rail, or GIN backend topology for a captured team,
  reshape fails.
- A device-visible active mask is updated in place. Device APIs and GIN paths
  reject operations targeting inactive ranks while the communicator is
  incomplete.
- On reshape, GIN handles and symmetric-memory peer mappings are refreshed in
  existing slots. The count and ordering of handles visible to captured device
  code must not change.

#### Relationship to existing APIs

- `ncclCommShrink`: creates a new communicator with surviving ranks renumbered.
  It remains the path for actually executing at a smaller dense size.
- `ncclCommGrow`: creates a new larger communicator. It remains the path for
  conventional elastic growth from `NULL` or an existing smaller communicator.
- `ncclCommReshape`: mutates active rank-slot membership in place on the
  existing communicator. It is the graph-capture-compatible path for masking and
  refilling stable rank slots without renumbering survivors.
- `ncclCommGetUniqueId_v2`: creates IDs whose flags identify normal versus
  reshape rendezvous semantics. Reshape IDs are reusable and can be handed to
  many joiners over multiple reshape iterations.
- `ncclCommInitRank` / `ncclCommInitRankConfig`: normal IDs preserve existing
  init behavior. Reshape IDs route joiners into reshape rendezvous. Reusing an
  existing communicator during checkpoint restore requires an explicit init
  extension so NCCL knows `ncclComm_t*` is in/out.
- `ncclCommRevoke`: quiesces in-flight work. Mask can reuse its cancellation and
  stream synchronization ideas, but revoke currently leaves the communicator
  permanently rejecting collectives. Mask needs a recoverable management state.
- `ncclCommSuspend/Resume`: releases and restores selected resources. Mask is
  complementary: it releases remote liveness and imported mappings while
  preserving local graph-visible storage.

### Implementation areas

The prototype already has substantial dynamic-membership machinery. This PLC
tracks the code areas that need to change as the public shape settles around
per-leader reshape IDs, init-based joiners, and atomic reshape commit.

#### Public API and init routing

- Add `ncclCommReshape`, `ncclCommMaskGet`, and
  `ncclCommMaskCountActive` declarations.
- Add `ncclCommGetUniqueId_v2(comm, uniqueId, flags)` with reshape IDs scoped
  to `(comm, leaderRank)`. Each rank owns at most one reusable reshape ID per
  communicator.
- Teach `ncclCommInitRank` and `ncclCommInitRankConfig` to detect reshape IDs
  and route joiners into the reshape rendezvous instead of normal clique init.
- Add `commInitFlags` to `ncclConfig_t`. `NCCL_COMM_INIT_REUSE_EXISTING` marks
  `ncclComm_t*` as in/out for checkpoint/restore reshape-ID init. Do not infer
  reuse by reading `*comm` from the existing output-only init signatures.
- Add `ncclResourceNotReady` for "joiners not ready; committed membership unchanged."
- Keep `NCCL_COMM_RESHAPE_DEFAULT` as the zero value for no optional reshape
  behavior.
- Add a local-only reshape flag for failure and checkpoint prepare paths that
  remove ranks without requiring identical survivor masks.

Candidate files:

- `src/nccl.h.in`
- `src/init.cc`
- `src/group.cc`
- `test/apitest/CMakeLists.txt`

#### Communicator membership state

- Add host active-mask fields to `struct ncclComm`.
- Initialize all rank slots active in `commAlloc`.
- Store per-leader reshape IDs and joiner pools keyed by target rank.
- Rely on normal NCCL collective ordering when multiple leaders have staged
  independent reshape IDs for the same communicator.
- Add internal helpers:
  - `ncclCommIsRankActive(comm, rank)`
  - `ncclCommIsFullyActive(comm)`
  - `ncclCommSetRankActive(comm, rank, active)`
  - active-mask allgather/debug formatting
- Add an enqueue guard so normal collectives fail fast when the active mask is
  incomplete, except for explicitly allowed algorithms.
- Avoid epoch-tracking unless an internal generation is needed for stale
  operation detection.

Candidate files:

- `src/include/comm.h`
- `src/init.cc`
- `src/enqueue.cc`
- `src/group.cc`

#### Reshape commit path

- Validate `joinList`, `joinListLen`, `leaveList`, `leaveListLen`, and reshape
  flags without survivor renumbering.
- Support leave-only, join-only, and combined leave/join reshapes.
- Support same-slot replacement when a rank appears in both `joinList` and
  `leaveList`.
- Support local-only leave/remove mode for failed peers and checkpoint prepare.
- On survivor leader calls, consult the selected leader's joiner pool and
  return `ncclResourceNotReady` without committing a new membership state when
  any named joiner is absent or not ready.
- On joiner calls, signal readiness after application warmup and wait for a
  leader-side commit or a clear failure.
- Stage resource changes before mutating committed communicator-visible state.
- Mark rank slots active or inactive only as part of the atomic reshape commit.

Candidate files:

- `src/init.cc`
- `src/bootstrap.cc`
- `src/include/bootstrap.h`
- `src/graph/connect.cc`

#### Transport, proxy, and peer state

- Identify all per-rank connector storage reachable from `channels[c].peers`,
  `channels[c].devPeers`, `connectSend`, and `connectRecv`.
- Add batched teardown/reconnect helpers for ranks changing membership without
  freeing the owning channel arrays.
- Clear or rebuild pending preconnect state for affected ranks.
- Add proxy detach/attach logic for affected top-parent ranks, taking care with
  shared proxy state and split/shared resources.
- Keep the teardown/reconnect path internal to NCCL transport/proxy ownership.
  It does not add a net plugin ABI entry; plugins continue to observe existing
  close/deregister operations when NCCL tears down selected connectors.
- Preserve array sizes based on `nRanks`/`tpNRanks`.

Candidate files:

- `src/transport.cc`
- `src/include/transport.h`
- `src/transport/net.cc`
- `src/transport/shm.cc`
- `src/transport/p2p.cc`
- `src/proxy.cc`
- `src/group.cc`

#### Symmetric memory and windows

- Split symmetric-memory cleanup into:
  - unmap imported peer mappings for inactive ranks
  - keep local reservations and local handles
  - keep window objects and window table entries
- Track per-rank mapping validity for each `ncclDevrMemory`.
- Teach window/device views to tolerate inactive peer mappings while the active
  mask is incomplete.
- Re-register or re-import affected rank handles during reshape.

Candidate files:

- `src/dev_runtime.cc`
- `src/dev_runtime_internal.h`
- `src/dev_runtime_segments.cc`
- `src/mem_manager.cc`
- `src/include/nccl_device/comm.h`

#### Dev comm, GIN, and graph compatibility

- Add a device-visible active mask to dev comm state for device APIs,
  fault-tolerant kernels, and GIN.
- Refresh dev comm fields in place after reshape rather than allocating a new
  graph-visible dev comm pointer.
- Validate that captured world/LSA/rail teams remain compatible with the
  replacement rank's specialized topology.
- Preserve GIN resource arrays and update handles for refilled slots.
- Add GIN-side checks so operations against inactive ranks fail with a clear
  error when the active mask is incomplete.
- Record graph-compatible communicator shape at capture or first graph use, and
  compare shape after reshape before allowing graph replay.
- Return clear errors for incompatible replacement rank, window sequence,
  channel count, GIN mode, or rank topology.

Candidate files:

- `src/dev_runtime.cc`
- `src/include/dev_runtime.h`
- `src/include/nccl_device/comm.h`
- `src/include/nccl_device/gin*.h`
- `src/gin/gin_host.cc`
- `src/gin/gin_host_proxy.cc`
- `src/transport/net_ib/gin.cc`
- `src/nccl_device/gin_barrier.cc`
- `src/nccl_device/gin_scratch.cc`
- `src/enqueue.cc`
- `src/debug.cc` or existing debug/logging helpers

#### Executable masked algorithms

- Add opt-in algorithms that can operate on a partial active mask.
- Start with simple singleton/local-only cases or collectives that can prove
  they do not require inactive peers.
- Keep default ring/tree behavior unsupported when the mask is incomplete.

This area is not required for the initial checkpoint/restore or fault-repair
stories.

</details>
