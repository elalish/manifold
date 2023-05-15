package manifold3d.pub;

import manifold3d.glm.DoubleVec2;
import org.bytedeco.javacpp.*;
import org.bytedeco.javacpp.annotation.*;

@Platform(include = {"manifold.h"})
@Namespace("manifold")
@Name("SimplePolygon")
public class SimplePolygon extends Pointer {
    static { Loader.load(); }

    public SimplePolygon() { allocate(); }
    private native void allocate();

    public native @Cast("size_t") long size();
    public native void resize(@Cast("size_t") long n);

    @Name("operator[]") public native @ByRef DoubleVec2 get(@Cast("size_t") long i);
    @Name("push_back") public native void pushBack(@ByRef DoubleVec2 value);
}
