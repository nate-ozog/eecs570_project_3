#ifndef CUDA_ERROR_CHECK_CUH
#define CUDA_ERROR_CHECK_CUH

void cuda_error_check(cudaError_t e) {
  if (e != cudaSuccess) {
    std::cerr << "CUDA FAILURE: " << cudaGetErrorString(e) << std::endl;
    exit(0);
  }
}

static cudaError_t convertToCudartError(CUresult error) {
	switch (error) {
	case CUDA_SUCCESS:
		return cudaSuccess;
	case CUDA_ERROR_INVALID_VALUE:
		return cudaErrorInvalidValue;
	case CUDA_ERROR_OUT_OF_MEMORY:
		return cudaErrorMemoryAllocation;
	case CUDA_ERROR_NOT_INITIALIZED:
		return cudaErrorInitializationError;
	case CUDA_ERROR_DEINITIALIZED:
		return cudaErrorCudartUnloading;
	case CUDA_ERROR_PROFILER_DISABLED:
		return cudaErrorProfilerDisabled;
	case CUDA_ERROR_PROFILER_NOT_INITIALIZED:
		return cudaErrorProfilerNotInitialized;
	case CUDA_ERROR_PROFILER_ALREADY_STARTED:
		return cudaErrorProfilerAlreadyStarted;
	case CUDA_ERROR_PROFILER_ALREADY_STOPPED:
		return cudaErrorProfilerAlreadyStopped;
	case CUDA_ERROR_NO_DEVICE:
		return cudaErrorNoDevice;
	case CUDA_ERROR_INVALID_DEVICE:
		return cudaErrorInvalidDevice;
	case CUDA_ERROR_INVALID_IMAGE:
		return cudaErrorInvalidKernelImage;
	case CUDA_ERROR_INVALID_CONTEXT:
		return cudaErrorIncompatibleDriverContext;
	case CUDA_ERROR_MAP_FAILED:
		return cudaErrorMapBufferObjectFailed;
	case CUDA_ERROR_UNMAP_FAILED:
		return cudaErrorUnmapBufferObjectFailed;
	case CUDA_ERROR_NO_BINARY_FOR_GPU:
		return cudaErrorNoKernelImageForDevice;
	case CUDA_ERROR_ECC_UNCORRECTABLE:
		return cudaErrorECCUncorrectable;
	case CUDA_ERROR_UNSUPPORTED_LIMIT:
		return cudaErrorUnsupportedLimit;
	case CUDA_ERROR_PEER_ACCESS_UNSUPPORTED:
		return cudaErrorPeerAccessUnsupported;
	case CUDA_ERROR_INVALID_PTX:
		return cudaErrorInvalidPtx;
	case CUDA_ERROR_INVALID_GRAPHICS_CONTEXT:
		return cudaErrorInvalidGraphicsContext;
	case CUDA_ERROR_SHARED_OBJECT_SYMBOL_NOT_FOUND:
		return cudaErrorSharedObjectSymbolNotFound;
	case CUDA_ERROR_SHARED_OBJECT_INIT_FAILED:
		return cudaErrorSharedObjectInitFailed;
	case CUDA_ERROR_OPERATING_SYSTEM:
		return cudaErrorOperatingSystem;
	case CUDA_ERROR_INVALID_HANDLE:
		return cudaErrorInvalidResourceHandle;
	case CUDA_ERROR_NOT_FOUND:
		return cudaErrorInvalidSymbol;
	case CUDA_ERROR_NOT_READY:
		return cudaErrorNotReady;
	case CUDA_ERROR_ILLEGAL_ADDRESS:
		return cudaErrorIllegalAddress;
	case CUDA_ERROR_LAUNCH_OUT_OF_RESOURCES:
		return cudaErrorLaunchOutOfResources;
	case CUDA_ERROR_LAUNCH_TIMEOUT:
		return cudaErrorLaunchTimeout;
	case CUDA_ERROR_PEER_ACCESS_ALREADY_ENABLED:
		return cudaErrorPeerAccessAlreadyEnabled;
	case CUDA_ERROR_PEER_ACCESS_NOT_ENABLED:
		return cudaErrorPeerAccessNotEnabled;
	case CUDA_ERROR_CONTEXT_IS_DESTROYED:
		return cudaErrorIncompatibleDriverContext;
	case CUDA_ERROR_ASSERT:
		return cudaErrorAssert;
	case CUDA_ERROR_TOO_MANY_PEERS:
		return cudaErrorTooManyPeers;
	case CUDA_ERROR_HOST_MEMORY_ALREADY_REGISTERED:
		return cudaErrorHostMemoryAlreadyRegistered;
	case CUDA_ERROR_HOST_MEMORY_NOT_REGISTERED:
		return cudaErrorHostMemoryNotRegistered;
	case CUDA_ERROR_HARDWARE_STACK_ERROR:
		return cudaErrorHardwareStackError;
	case CUDA_ERROR_ILLEGAL_INSTRUCTION:
		return cudaErrorIllegalInstruction;
	case CUDA_ERROR_MISALIGNED_ADDRESS:
		return cudaErrorMisalignedAddress;
	case CUDA_ERROR_INVALID_ADDRESS_SPACE:
		return cudaErrorInvalidAddressSpace;
	case CUDA_ERROR_INVALID_PC:
		return cudaErrorInvalidPc;
	case CUDA_ERROR_LAUNCH_FAILED:
		return cudaErrorLaunchFailure;
	case CUDA_ERROR_NOT_PERMITTED:
		return cudaErrorNotPermitted;
	case CUDA_ERROR_NOT_SUPPORTED:
		return cudaErrorNotSupported;
	case CUDA_ERROR_UNKNOWN:
		return cudaErrorUnknown;
	default:
		return cudaErrorUnknown;
	}
}

#endif