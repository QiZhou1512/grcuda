package com.nvidia.grcuda.test.gpu;

import com.nvidia.grcuda.GrCUDAContext;
import com.nvidia.grcuda.gpu.CUDARuntime;
import com.nvidia.grcuda.gpu.Device;
import com.oracle.truffle.api.TruffleLanguage;
import com.oracle.truffle.api.instrumentation.TruffleInstrument;
import org.graalvm.polyglot.Context;
import org.graalvm.polyglot.Value;
import org.junit.runners.Parameterized;
import org.junit.Test;
import org.junit.runner.RunWith;

import java.util.Arrays;
import java.util.Collection;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;

@RunWith(Parameterized.class)

public class DevicePropertiesTest {
    /**
     * Tests are executed for each of the {@link com.nvidia.grcuda.gpu.executioncontext.GrCUDAExecutionContext} values;
     * @return the current stream policy
     */
    @Parameterized.Parameters
    public static Collection<Object[]> data() {
        return Arrays.asList(new Object[][]{
                {"sync"},
                {"default"}
        });
    }

    private final String policy;

    public DevicePropertiesTest(String policy) {
        this.policy = policy;
    }

    @Test
    public void deviceProperties(){
        System.out.println(this.policy);
        try (Context context = Context.newBuilder().option("grcuda.ExecutionPolicy", this.policy).allowAllAccess(true).build()) {
            Value devices = context.eval("grcuda", "getdevices()");
            for (int i = 0; i < devices.getArraySize(); ++i) {
                Value device = devices.getArrayElement(i);
                Value prop = device.getMember("properties");
                // Sanity tests on some of the properties
                // device name is a non-zero string
                assertTrue(prop.getMember("deviceName").asString().length() > 0);
                System.out.println(prop.getMember("deviceName").asString());

                // compute capability is at least compute Kepler (3.0)
                assertTrue(prop.getMember("computeCapabilityMajor").asInt() >= 3);
                System.out.println(prop.getMember("computeCapabilityMajor").asInt());
                // there is at least one multiprocessors
                assertTrue(prop.getMember("multiProcessorCount").asInt() > 0);
                System.out.println(prop.getMember("multiProcessorCount").asInt());
                // there is some device memory
                assertTrue(prop.getMember("totalDeviceMemory").asLong() > 0L);
                System.out.println(prop.getMember("totalDeviceMemory").asLong());
            }
            //System.out.println(devices.as());
        }


    }
}
