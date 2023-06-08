package manifold3d.glm;

import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.IntBuffer;

import manifold3d.BufferUtils;
import manifold3d.glm.IntegerVec4;
import org.bytedeco.javacpp.*;
import org.bytedeco.javacpp.annotation.*;

import java.util.Iterator;
import java.lang.Iterable;
import java.util.NoSuchElementException;

@Platform(compiler = "cpp17", include = {"<vector>", "glm/glm.hpp"})
@Name("std::vector<glm::ivec4>")
public class IntegerVec4Vector extends Pointer implements Iterable<IntegerVec4> {
    static { Loader.load(); }

    @Override
    public Iterator<IntegerVec4> iterator() {
        return new Iterator<IntegerVec4>() {

            private long index = 0;

            @Override
            public boolean hasNext() {
                return index < size();
            }

            @Override
            public IntegerVec4 next() {
                if (!hasNext()) {
                    throw new NoSuchElementException();
                }
                return get(index++);
            }
        };
    }

    public IntegerVec4Vector() { allocate(); }
    private native void allocate();

    public native @Cast("size_t") long size();
    public native void resize(@Cast("size_t") long n);

    @Name("operator[]") public native @ByRef IntegerVec4 get(@Cast("size_t") long i);
    @Name("push_back") public native void pushBack(@ByRef IntegerVec4 value);

    public static IntegerVec4Vector FromBuffer(IntBuffer buff) {
        IntPointer ptr = new IntPointer(buff);
        return BufferUtils.createIntegerVec4Vector(ptr, buff.capacity());
    }

    public static IntegerVec4Vector FromArray(int[] data) {
        ByteBuffer buff = ByteBuffer.allocateDirect(data.length * Integer.BYTES).order(ByteOrder.nativeOrder());
        IntBuffer intBuffer = buff.asIntBuffer();
        intBuffer.put(data);
        intBuffer.flip();

        return FromBuffer(intBuffer);
    }

}
