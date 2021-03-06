.. default-domain:: chpl

===============
GPUAPI
===============

MID-level API Reference
########################

.. class:: GPUArray

   .. method:: proc init(ref arr)

      Allocates memory on the device. The allocation size is automatically computed by this module -i.e., ``(arr.size: size_t) * c_sizeof(arr.eltType)``.

      :arg arr: The reference of the non-distributed Chapel Array that will be mapped onto the device.

      .. code-block:: chapel
         :emphasize-lines: 6,21

         // Example 1: Non-distributed array
         var A: [1..n] int;

         proc GPUCallBack(lo: int, hi: int, N: int) {
           // n * sizeof(int) will be allocated onto the device
           var dA = new GPUArray(A);
           ...
         }

         // GPUIterator
         forall i in GPU(1..n, GPUCallBack) { A(i) = ...; }

         // Example 2: Distributed array
         use BlockDist;
         var D: domain(1) dmapped Block(boundingBox = {1..n}) = {1..n};
         var A: [D] int;
         proc GPUCallBack(lo: int, hi: int, n: int) {
           // get the local portion of the distributed array
           var localA = A.localSlice(lo...hi);
           // n * sizeof(int) will be allocated onto the device
           var dA = new GPUArray(localA);
           ...
         }

         // GPUIterator
         forall i in GPU(D, GPUCallBack) { A(i) = ...; }

      .. note:: The allocated memory resides on the `current device`. With the ``GPUIterator``, the current device is automatically set by it. Without it, it is the user's responsibilities to set the current device (e.g., by calling the ``SetDevice`` API below). Otherwise, the default device (usually the first GPU) will be used.

      .. note:: With distributed arrays, it is required to use Chapel array's `localSlice API <https://chapel-lang.org/docs/builtins/ChapelArray.html#ChapelArray.localSlice>`_ to get the local portion of the distributed array. With the ``GPUIterator``, the local portion is already computed and given as the first two arguments (``lo`` and ``hi``).

   .. method:: toDevice()

      Transfers the contents of the Chapel array to the device.

      .. code-block:: chapel
         :emphasize-lines: 3

         proc GPUCallBack(lo: int, hi: int, n:int) {
           var dA = GPUArray(A);
           dA.toDevice();
         }

   .. method:: fromDevice()

      Transfers back the contents of the device array to the Chapel array.

      .. code-block:: chapel
         :emphasize-lines: 3

         proc GPUCallBack(lo: int, hi: int, n:int) {
           var dA = GPUArray(A);
           dA.fromDevice();
         }

   .. method:: free()

      Frees memory on the device.

      .. code-block:: chapel
         :emphasize-lines: 3

         proc GPUCallBack(lo: int, hi: int, n:int) {
           var dA = GPUArray(A);
           dA.free();
         }

   .. method:: dPtr(): c_void_ptr

      Returns a pointer to the allocated device memory.

      :returns: pointer to the allocated device memory
      :rtype: `c_void_ptr`

   .. method:: hPtr(): c_void_ptr

      Returns a pointer to the head of the Chapel array.

      :returns: pointer to the head of the Chapel array
      :rtype: `c_void_ptr`


.. method:: toDevice(args: GPUArray ...?n)

   Utility function that takes a variable number of ``GPUArray`` and performs the ``toDevice`` operation for each.

.. method:: fromDevice(args: GPUArray ...?n)

   Utility function that takes a variable number of ``GPUArray`` and performs the ``fromDevice`` operation for each.

.. method:: free(args: GPUArray ...?n)

   Utility function that takes a variable number of ``GPUArray`` and performs the ``free`` operation for each.

.. code-block:: chapel

   var dA = GPUArray(A);
   var dB = GPUArray(B);
   var dC = GPUArray(C);

   toDevice(A, B)
   ..
   fromDevice(C);
   free(A, B, C);


LOW-MID-level API Reference
############################

.. method:: Malloc(ref devPtr: c_void_ptr, size: size_t)

   Allocates memory on the device.

   :arg devPtr: Pointer to the allocated device array
   :type devPtr: `c_voidPtr`

   :arg size: Allocation size in bytes
   :type size: `size_t`

   .. code-block:: chapel
      :emphasize-lines: 6,21

      // Example 1: Non-distributed array
      var A: [1..n] int;

      proc GPUCallBack(lo: int, hi: int, N: int) {
        var dA: c_void_ptr;
        Malloc(dA, (A.size: size_t) * c_sizeof(A.eltType));
        ...
      }

      // GPUIterator
      forall i in GPU(1..n, GPUCallBack) { A(i) = ...; }

      // Example 2: Distributed array
      use BlockDist;
      var D: domain(1) dmapped Block(boundingBox = {1..n}) = {1..n};
      var A: [D] int;
      proc GPUCallBack(lo: int, hi: int, n: int) {
        var dA: c_void_ptr;
        // get the local portion of the distributed array
        var localA = A.localSlice(lo...hi);
        Malloc(dA, (localA.size: size_t) * c_sizeof(localA.eltType));
        ...
      }

      // GPUIterator
      forall i in GPU(D, GPUCallBack) { A(i) = ...; }

   .. note:: ``c_sizeofo(A.eltType)`` returns the size in bytes of the element of the Chapel array ``A``. For more details, please refer to `this <https://chapel-lang.org/docs/builtins/CPtr.html#CPtr.c_sizeof>`_.


.. method:: Memcpy(dst: c_void_ptr, src: c_void_ptr, count: size_t, kind: int)

   Transfers data between the host and the device

   :arg dst: the desination address
   :type dst: `c_void_ptr`

   :arg src: the source address
   :type src: `c_void_ptr`

   :arg count: size in bytes to be transferred
   :type count: `size_t`

   :arg kind: type of transfer (``0``: host-to-device, ``1``: device-to-host)
   :type kind: `int`

   .. code-block:: chapel
      :emphasize-lines: 7-10

      // Non-distributed array
      var A: [1..n] int;

      proc GPUCallBack(lo: int, hi: int, N: int) {
        var dA: c_void_ptr;
        Malloc(dA, (A.size: size_t) * c_sizeof(A.eltType));
        // host-to-device
        Memcpy(dA, c_ptrTo(A), size, 0);
        // device-to-host
        Memcpy(c_ptrTo(A), dA, size, 1));
      }

   .. note:: ``c_ptrTo(A)`` returns a pointer to the Chapel rectangular array ``A``. For more details, see `this document <https://chapel-lang.org/docs/builtins/CPtr.html#CPtr.c_ptrTo>`_.


.. method:: Free(devPtr: c_void_ptr)

   Frees memory on the device

   :arg devPtr: Device pointer to memory to be freed.
   :type devPtr: `c_void_ptr`

.. method:: GetDeviceCount(ref count: int(32))

   Returns the number of GPU devices on the current locale.

   :arg count: the number of GPU devices
   :type count: `int(32)`

   .. code-block:: chapel

      var nGPUs: int(32);
      GetDeviceCount(nGPUs);
      writeln(nGPUs);

.. method:: GetDevice(ref id: int(32))

   Returns the device ID currently being used.

   :arg id: the device ID current being used
   :type id: `int(32)`

.. method:: SetDevice(device: int(32))

   Sets the device ID to be used.

   :arg id: the device ID to be used. ``id`` must be 1) greater than or equal to zero, and 2) less than the number of GPU devices.
   :type id: `int(32)`

.. method:: ProfilerStart()

   **NVIDIA GPUs Only** Start profiling with ``nvprof``

.. method:: ProfilerStop()

   **NVIDIA GPUs Only** Stop profiling with ``nvprof``

   .. code-block:: chapel

      proc GPUCallBack(lo: int, hi: int, N: int) {
        ProfilerStart();
        ...
        ProfilerStop();
      }

.. method:: DeviceSynchronize()

   Waits for the device to finish.
