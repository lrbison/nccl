.. _api_flags:

************************
NCCL API Supported Flags
************************

The following show all flags which are supported by NCCL APIs.

.. _win_flags:

Window Registration Flags
-------------------------

.. c:macro:: NCCL_WIN_DEFAULT

 Register buffer into NCCL window with default behavior. The default behavior allows users to
 pass any offset to the buffer head address as the input of NCCL collective operations. However,
 this behavior can cause suboptimal performance in NCCL due to the asymmetric buffer usage.

.. c:macro:: NCCL_WIN_COLL_SYMMETRIC

 Register buffer into NCCL window, and users need to guarantee the offset to the buffer head address
 from all ranks must be equal when calling NCCL collective operations. It allows NCCL to operate
 buffer in a symmetric way and provide the best performance.

.. c:macro:: NCCL_WIN_STRICT_ORDERING

  Register buffer into NCCL window while ensuring strict ordering for window operations using the IB Verbs transport.
  This flag is mostly intended for buffers used for GIN VA Signals (see :ref:`devapi_signals`).

.. _cta_policy_flags:

NCCL Communicator CTA Policy Flags
----------------------------------

.. c:macro:: NCCL_CTA_POLICY_DEFAULT

  Use the default CTA policy for NCCL communicator. In this policy, NCCL will automatically adjust resource usage and achieve
  maximal performance. This policy is suitable for most applications.

.. c:macro:: NCCL_CTA_POLICY_EFFICIENCY

  Use the CTA efficiency policy for NCCL communicator. In this policy, NCCL will optimize CTA usage and use minimal
  number of CTAs to achieve the decent performance when possible. This policy is suitable for applications which require
  better compute and communication overlap.

.. c:macro:: NCCL_CTA_POLICY_ZERO

  Use the Zero-CTA policy for NCCL communicator. In this policy, NCCL will use zero CTA whenever it can, even when that choice
  may sacrifice some performance. Select this mode when your application must preserve the maximum number of CTAs for compute kernels.

.. _comm_shrink_flags:

Communicator Shrink Flags
--------------------------

These flags modify the behavior of the ``ncclCommShrink`` operation.

.. c:macro:: NCCL_SHRINK_DEFAULT

   Default behavior. Shrink the parent communicator without affecting ongoing operations.
   Value: ``0x00``.

.. c:macro:: NCCL_SHRINK_ABORT

   First, terminate ongoing parent communicator operations, and then proceed with shrinking the communicator.
   This is used for error recovery scenarios where the parent communicator might be in a hung state.
   Resources of parent comm are still not freed, users should decide whether to call ncclCommAbort on the parent communicator after shrink.
   Value: ``0x01``.

.. _unique_id_flags:

Communicator Unique ID Flags
----------------------------

These flags modify the behavior of :c:func:`ncclCommGetUniqueId_v2`.

.. c:macro:: NCCL_UNIQUE_ID_DEFAULT

   Generate a communicator-scoped unique ID with the same behavior as
   :c:func:`ncclCommGetUniqueId`.
   Value: ``0``.

.. c:macro:: NCCL_UNIQUE_ID_RESHAPE

   Create or return the reshape unique ID for the calling rank on a communicator.
   Applications distribute this ID to replacement or restored ranks before those
   ranks initialize with :c:func:`ncclCommInitRank` or
   :c:func:`ncclCommInitRankConfig`.
   Value: ``1``.

.. _comm_reshape_flags:

Communicator Reshape Flags
--------------------------

These flags modify the behavior of :c:func:`ncclCommReshape`.

.. c:macro:: NCCL_COMM_RESHAPE_DEFAULT

   Perform a collective in-place reshape, or signal joiner readiness when called
   on a communicator initialized with a reshape unique ID.
   Value: ``0``.

.. c:macro:: NCCL_COMM_RESHAPE_LOCAL_ONLY

   Locally remove rank slots from the communicator's active mask without
   collective communication. This mode cannot add or reactivate rank slots and
   cannot remove the local rank.
   Value: ``1``.

.. _comm_init_flags:

Communicator Initialization Flags
---------------------------------

These flags can be set in :c:macro:`commInitFlags` in :c:type:`ncclConfig_t`.

.. c:macro:: NCCL_COMM_INIT_DEFAULT

   Default communicator initialization behavior.
   Value: ``0``.

.. c:macro:: NCCL_COMM_INIT_REUSE_EXISTING

   Reuse and refresh the existing communicator pointed to by the ``ncclComm_t*``
   argument to :c:func:`ncclCommInitRankConfig`. This flag is only valid when
   initializing with a reshape unique ID.
   Value: ``1``.
