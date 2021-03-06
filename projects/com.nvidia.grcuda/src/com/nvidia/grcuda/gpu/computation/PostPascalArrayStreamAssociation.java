package com.nvidia.grcuda.gpu.computation;

import com.nvidia.grcuda.gpu.stream.CUDAStream;

import java.util.Optional;
import java.util.concurrent.Callable;

/**
 * GPUs with Pascal architecture or newer (e.g. Tesla P100), with compute capability >= 6.0,
 * do not require to exclusively associate a managed memory array to a single stream to provide
 * asynchronous host access to managed memory while a kernel is running.
 * As such, no stream association is performed;
 */
public class PostPascalArrayStreamAssociation implements ArrayStreamArchitecturePolicy {

    @Override
    public void execute(Runnable callable) {

    }

    @Override
    public Optional<CUDAStream> execute(Callable<Optional<CUDAStream>> callable) {
        return Optional.empty();
    }
}
