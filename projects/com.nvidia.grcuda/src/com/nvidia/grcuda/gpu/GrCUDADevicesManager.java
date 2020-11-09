package com.nvidia.grcuda.gpu;

import com.nvidia.grcuda.gpu.stream.GrCUDAStreamManager;

import java.util.ArrayList;
import java.util.HashMap;

/**
 * Manager of multiple GPUs, keeps track of various GPUs, the validity of the arrays that are stored in each of them
 *
 * */
public class GrCUDADevicesManager {
    private final CUDARuntime runtime;
    private final HashMap<Device, GrCUDAStreamManager> deviceStreamManagerHashMap = new HashMap<>();
    private final Integer numberOfGPUs;
    private Integer currentDeviceId;
    public GrCUDADevicesManager(CUDARuntime runtime){
        this.runtime = runtime;
        this.numberOfGPUs = runtime.cudaGetDeviceCount();
        this.currentDeviceId = runtime.cudaGetDevice();
        initDevices();
    }

    private void initDevices(){
        for(int i = 0; i<numberOfGPUs;i++) {
            deviceStreamManagerHashMap.put(new Device(i, runtime),new GrCUDAStreamManager(runtime));
        }
    }

    public int getCurrentDeviceId(){
        return this.currentDeviceId;
    }

    public int getNumberOfGPUs(){
        return this.numberOfGPUs;
    }

    public void setDevice(int id){
        runtime.cudaSetDevice(id);
        this.currentDeviceId = id;
    }


}
