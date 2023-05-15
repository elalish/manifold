package manifold3d.glm;

import manifold3d.glm.DoubleVec2;

import org.bytedeco.javacpp.*;
import org.bytedeco.javacpp.annotation.*;

@Platform(include = "glm/glm.hpp")
@Namespace("glm")
@Name("mat3x2")
public class DoubleMat3x2 extends DoublePointer {
    static { Loader.load(); }

    public DoubleMat3x2() { allocate(); }
    private native void allocate();

    public DoubleMat3x2(double v) { allocate(v); }
    private native void allocate(double v);

    @Name("operator[]") public native @ByRef DoubleVec2 getColumn(int i);

    public native @Name("operator=") @ByRef DoubleMat3x2 put(@ByRef DoubleMat3x2 rhs);
}
