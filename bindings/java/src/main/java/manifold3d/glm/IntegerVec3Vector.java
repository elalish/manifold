package manifold3d.glm;

import manifold3d.glm.IntegerVec3;
import org.bytedeco.javacpp.*;
import org.bytedeco.javacpp.annotation.*;

@Platform(include = {"<vector>", "glm/glm.hpp"})
@Name("std::vector<glm::ivec3>")
public class IntegerVec3Vector extends Pointer {
    static { Loader.load(); }

    public IntegerVec3Vector() { allocate(); }
    private native void allocate();

    public native @Cast("size_t") long size();
    public native void resize(@Cast("size_t") long n);

    @Name("operator[]") public native @ByRef IntegerVec3 get(@Cast("size_t") long i);
    @Name("push_back") public native void pushBack(@ByRef IntegerVec3 value);
}
