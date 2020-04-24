package com.nvidia.grcuda.gpu.computation;

/**
 * GPUs with pre-Pascal architecture or older, with compute capability < 6.0,
 * require to exclusively associate a managed memory array to a single stream to provide
 * asynchronous host access to managed memory while a kernel is running.
 * This interface wraps and executes the array association function specified in {@link GrCUDAComputationalElement}
 */
public class PrePascalArrayStreamAssociation implements ArrayStreamArchitecturePolicy {

    @Override
    public void execute(Runnable runnable) {
        runnable.run();
    }
}
