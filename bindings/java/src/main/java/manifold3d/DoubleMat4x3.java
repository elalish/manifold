package manifold3d;

import manifold3d.DoubleVec3;

import org.bytedeco.javacpp.*;
import org.bytedeco.javacpp.annotation.*;

@Platform(include = "glm/glm.hpp")
@Namespace("glm")
@Name("mat4x3")
public class DoubleMat4x3 extends DoublePointer {
    static { Loader.load(); }

    public DoubleMat4x3() { allocate(); }
    private native void allocate();

    public DoubleMat4x3(double v) { allocate(v); }
    private native void allocate(double v);

    @Name("operator[]") public native @ByRef DoubleVec3 getColumn(int i);

    public native @Name("operator=") @ByRef DoubleMat4x3 put(@ByRef DoubleMat4x3 rhs);
}
