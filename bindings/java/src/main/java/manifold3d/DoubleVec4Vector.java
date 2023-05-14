package manifold3d;

import manifold3d.Glm.DoubleVec4;
import org.bytedeco.javacpp.*;
import org.bytedeco.javacpp.annotation.*;

@Platform(include = {"<vector>", "glm/glm.hpp"})
@Name("std::vector<glm::vec4>")
public class DoubleVec4Vector extends Pointer {
    static { Loader.load(); }

    public DoubleVec4Vector() { allocate(); }
    private native void allocate();

    public native @Cast("size_t") long size();
    public native void resize(@Cast("size_t") long n);

    @Name("operator[]") public native @ByRef DoubleVec4 get(@Cast("size_t") long i);

    public void set(@Cast("size_t") long i, DoubleVec4 value) {
        get(i).put(value);
    }

    @Name("push_back") public native void pushBack(@ByRef DoubleVec4 value);
}
