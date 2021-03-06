package com.nvidia.grcuda.gpu.stream;

import com.nvidia.grcuda.gpu.executioncontext.ExecutionDAG;

/**
 * This abstract class defines how a {@link GrCUDAStreamManager}
 * will assign a {@link CUDAStream} to a {@link com.nvidia.grcuda.gpu.computation.GrCUDAComputationalElement}
 * that has at least one parent active computation.
 * For example, it use the same stream of the parent, or understand that a different stream can be used,
 * to have multiple children computation run in parallel.
 */
public abstract class RetrieveParentStream {
    abstract CUDAStream retrieve(ExecutionDAG.DAGVertex vertex);
}
