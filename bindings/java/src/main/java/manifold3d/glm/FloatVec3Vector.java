package manifold3d.glm;

import manifold3d.glm.FloatVec3;
import org.bytedeco.javacpp.*;
import org.bytedeco.javacpp.annotation.*;

@Platform(include = {"<vector>", "glm/glm.hpp"})
@Name("std::vector<glm::vec3>")
public class FloatVec3Vector extends Pointer {
    static { Loader.load(); }

    public FloatVec3Vector() { allocate(); }
    private native void allocate();

    public native @Cast("size_t") long size();
    public native void resize(@Cast("size_t") long n);

    @Name("operator[]") public native @ByRef FloatVec3 get(@Cast("size_t") long i);
    @Name("push_back") public native void pushBack(@ByRef FloatVec3 value);
}
