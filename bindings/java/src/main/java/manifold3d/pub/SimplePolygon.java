package manifold3d.pub;

import java.nio.DoubleBuffer;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.DoubleBuffer;

import manifold3d.BufferUtils;
import manifold3d.glm.DoubleVec2;
import org.bytedeco.javacpp.*;
import org.bytedeco.javacpp.annotation.*;

import java.util.Iterator;
import java.lang.Iterable;
import java.util.NoSuchElementException;

@Platform(compiler = "cpp17", include = {"manifold.h"})
@Namespace("manifold")
@Name("SimplePolygon")
public class SimplePolygon extends Pointer implements Iterable<DoubleVec2> {
    static { Loader.load(); }

    @Override
    public Iterator<DoubleVec2> iterator() {
        return new Iterator<DoubleVec2>() {

            private long index = 0;

            @Override
            public boolean hasNext() {
                return index < size();
            }

            @Override
            public DoubleVec2 next() {
                if (!hasNext()) {
                    throw new NoSuchElementException();
                }
                return get(index++);
            }
        };
    }

    public SimplePolygon() { allocate(); }
    private native void allocate();

    public native @Cast("size_t") long size();
    public native void resize(@Cast("size_t") long n);

    @Name("operator[]") public native @ByRef DoubleVec2 get(@Cast("size_t") long i);
    @Name("push_back") public native void pushBack(@ByRef DoubleVec2 value);

    public static SimplePolygon FromBuffer(DoubleBuffer buff) {
        DoublePointer ptr = new DoublePointer(buff);
        return BufferUtils.createDoubleVec2Vector(ptr, buff.capacity());
    }

    public static SimplePolygon FromArray(double[] data) {
        ByteBuffer buff = ByteBuffer.allocateDirect(data.length * Double.BYTES).order(ByteOrder.nativeOrder());
        DoubleBuffer doubleBuffer = buff.asDoubleBuffer();
        doubleBuffer.put(data);
        doubleBuffer.flip();

        return FromBuffer(doubleBuffer);
    }

}
