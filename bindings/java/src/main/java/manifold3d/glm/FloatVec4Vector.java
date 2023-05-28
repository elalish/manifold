package manifold3d.glm;

import manifold3d.glm.FloatVec4;
import org.bytedeco.javacpp.*;
import org.bytedeco.javacpp.annotation.*;

import java.util.Iterator;
import java.lang.Iterable;
import java.util.NoSuchElementException;

@Platform(include = {"<vector>", "glm/glm.hpp"})
@Name("std::vector<glm::vec4>")
public class FloatVec4Vector extends Pointer implements Iterable<FloatVec4> {
    static { Loader.load(); }

    @Override
    public Iterator<FloatVec4> iterator() {
        return new Iterator<FloatVec4>() {

            private long index = 0;

            @Override
            public boolean hasNext() {
                return index < size();
            }

            @Override
            public FloatVec4 next() {
                if (!hasNext()) {
                    throw new NoSuchElementException();
                }
                return get(index++);
            }
        };
    }

    public FloatVec4Vector() { allocate(); }
    private native void allocate();

    public native @Cast("size_t") long size();
    public native void resize(@Cast("size_t") long n);

    @Name("operator[]") public native @ByRef FloatVec4 get(@Cast("size_t") long i);
    @Name("push_back") public native void pushBack(@ByRef FloatVec4 value);

    public void set(@Cast("size_t") long i, FloatVec4 value) {
        get(i).put(value);
    }
}
