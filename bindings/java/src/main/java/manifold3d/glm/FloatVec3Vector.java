package manifold3d.glm;

import manifold3d.glm.FloatVec3;
import org.bytedeco.javacpp.*;
import org.bytedeco.javacpp.annotation.*;

import java.util.Iterator;
import java.lang.Iterable;
import java.util.NoSuchElementException;

@Platform(compiler = "cpp17", include = {"<vector>", "glm/glm.hpp"})
@Name("std::vector<glm::vec3>")
public class FloatVec3Vector extends Pointer implements Iterable<FloatVec3> {
    static { Loader.load(); }

    @Override
    public Iterator<FloatVec3> iterator() {
        return new Iterator<FloatVec3>() {

            private long index = 0;

            @Override
            public boolean hasNext() {
                return index < size();
            }

            @Override
            public FloatVec3 next() {
                if (!hasNext()) {
                    throw new NoSuchElementException();
                }
                return get(index++);
            }
        };
    }

    public FloatVec3Vector() { allocate(); }
    private native void allocate();

    public native @Cast("size_t") long size();
    public native void resize(@Cast("size_t") long n);

    @Name("operator[]") public native @ByRef FloatVec3 get(@Cast("size_t") long i);
    @Name("push_back") public native void pushBack(@ByRef FloatVec3 value);
}
