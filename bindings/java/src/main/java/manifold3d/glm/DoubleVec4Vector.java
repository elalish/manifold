package manifold3d.glm;

import java.nio.DoubleBuffer;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.DoubleBuffer;

import manifold3d.BufferUtils;
import manifold3d.glm.DoubleVec4;
import org.bytedeco.javacpp.*;
import org.bytedeco.javacpp.annotation.*;

import java.util.Iterator;
import java.lang.Iterable;
import java.util.NoSuchElementException;

@Platform(include = {"<vector>", "glm/glm.hpp"})
@Name("std::vector<glm::vec4>")
public class DoubleVec4Vector extends Pointer  implements Iterable<DoubleVec4>{
    static { Loader.load(); }

    @Override
    public Iterator<DoubleVec4> iterator() {
        return new Iterator<DoubleVec4>() {

            private long index = 0;

            @Override
            public boolean hasNext() {
                return index < size();
            }

            @Override
            public DoubleVec4 next() {
                if (!hasNext()) {
                    throw new NoSuchElementException();
                }
                return get(index++);
            }
        };
    }

    public DoubleVec4Vector() { allocate(); }
    private native void allocate();

    public native @Cast("size_t") long size();
    public native void resize(@Cast("size_t") long n);

    @Name("operator[]") public native @ByRef DoubleVec4 get(@Cast("size_t") long i);

    public void set(@Cast("size_t") long i, DoubleVec4 value) {
        get(i).put(value);
    }

    @Name("push_back") public native void pushBack(@ByRef DoubleVec4 value);

    public static DoubleVec4Vector FromBuffer(DoubleBuffer buff) {
        DoublePointer ptr = new DoublePointer(buff);
        return BufferUtils.createDoubleVec4Vector(ptr, buff.capacity());
    }

    public static DoubleVec4Vector FromArray(double[] data) {
        ByteBuffer buff = ByteBuffer.allocateDirect(data.length * Double.BYTES).order(ByteOrder.nativeOrder());
        DoubleBuffer doubleBuffer = buff.asDoubleBuffer();
        doubleBuffer.put(data);
        doubleBuffer.flip();

        return FromBuffer(doubleBuffer);
    }
}
