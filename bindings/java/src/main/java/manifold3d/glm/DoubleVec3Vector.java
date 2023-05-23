package manifold3d.glm;

import java.nio.DoubleBuffer;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.DoubleBuffer;

import java.util.Arrays;

import manifold3d.BufferUtils;
import manifold3d.glm.DoubleVec3;
import org.bytedeco.javacpp.*;
import org.bytedeco.javacpp.annotation.*;

@Platform(include = {"<vector>", "glm/glm.hpp"})
@Name("std::vector<glm::vec3>")
public class DoubleVec3Vector extends Pointer {
    static { Loader.load(); }

    @Override
    public String toString() {
        return "value=" + Arrays.deepToString(this.toArrays()) + '}';
    }

    public DoubleVec3Vector() { allocate(); }
    private native void allocate();

    public native @Cast("size_t") long size();
    public native void resize(@Cast("size_t") long n);

    @Name("operator[]") public native @ByRef DoubleVec3 get(@Cast("size_t") long i);
    @Name("push_back") public native void pushBack(@ByRef DoubleVec3 value);

    public double[][] toArrays() {
        int length = (int) this.size();

        double[][] values = new double[length][3];

        for (int i = 0; i < this.size(); i++) {
            DoubleVec3 vec3 = this.get(i);
            double[] v = {vec3.get(0), vec3.get(1), vec3.get(2)};
            values[i] = v;
        }

        return values;
    }

    public static DoubleVec3Vector FromBuffer(DoubleBuffer buff) {
        DoublePointer ptr = new DoublePointer(buff);
        return BufferUtils.createDoubleVec3Vector(ptr, buff.capacity());
    }

    public static DoubleVec3Vector FromArray(double[] data) {
        ByteBuffer buff = ByteBuffer.allocateDirect(data.length * Double.BYTES).order(ByteOrder.nativeOrder());
        DoubleBuffer doubleBuffer = buff.asDoubleBuffer();
        doubleBuffer.put(data);
        doubleBuffer.flip();

        return FromBuffer(doubleBuffer);
    }
}
