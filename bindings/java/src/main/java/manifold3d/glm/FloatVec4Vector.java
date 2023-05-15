package manifold3d.glm;

import manifold3d.glm.FloatVec4;
import org.bytedeco.javacpp.*;
import org.bytedeco.javacpp.annotation.*;

@Platform(include = {"<vector>", "glm/glm.hpp"})
@Name("std::vector<glm::vec4>")
public class FloatVec4Vector extends Pointer {
    static { Loader.load(); }

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
