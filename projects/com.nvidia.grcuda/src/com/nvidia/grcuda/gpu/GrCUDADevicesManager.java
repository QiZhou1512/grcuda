package com.nvidia.grcuda.gpu;

import java.util.ArrayList;
/**
 * Manager of multiple GPUs, keeps track of various GPUs, the validity of the arrays that are stored in each of them
 *
 * */
public class GrCUDADevicesManager {
    private final CUDARuntime runtime;
    private final ArrayList<Device> deviceArrayList = new ArrayList<>();
    private final Integer numberOfGPUs;
    public GrCUDADevicesManager(CUDARuntime runtime){
        this.runtime = runtime;
        this.numberOfGPUs = runtime.cudaGetDeviceCount();

        for(int i = 0; i<numberOfGPUs;i++) {
            deviceArrayList.add(new Device(i, runtime));
        }


    }



    private void initDevices(){
        runtime.cudaGetDeviceCount();
    }
}
