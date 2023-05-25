package manifold3d.glm;

import manifold3d.glm.DoubleVec3;
import manifold3d.glm.DoubleVec4;

import org.bytedeco.javacpp.*;
import org.bytedeco.javacpp.annotation.*;

@Platform(include = {"glm/glm.hpp"})
@Namespace("glm")
@Name("mat4x3")
public class DoubleMat4x3 extends DoublePointer {
    static { Loader.load(); }

    public DoubleMat4x3() { allocate(); }
    private native void allocate();

    private native void allocate(double v);
    public DoubleMat4x3(double v) { allocate(v); }

    public DoubleMat4x3(@ByRef DoubleVec3 col1, @ByRef DoubleVec3 col2,
                        @ByRef DoubleVec3 col3, @ByRef DoubleVec3 col4) {
        allocate(col1, col2, col3, col4);
    }
    public native void allocate(@ByRef DoubleVec3 col1, @ByRef DoubleVec3 col2,
                                @ByRef DoubleVec3 col3, @ByRef DoubleVec3 col4);

    public DoubleMat4x3(double c0, double c1, double c2, double c3,
                        double c4, double c5, double c6, double c7,
                        double c8, double c9, double c10, double c11) {
        allocate(c0, c1, c2, c3, c4, c5, c6, c7, c8, c9, c10, c11);
    }
    public native void allocate(double c0, double c1, double c2, double c3,
                                double c4, double c5, double c6, double c7,
                                double c8, double c9, double c10, double c11);

    @Name("operator[]") public native @ByRef DoubleVec3 getColumn(int i);

    public native @Name("operator=") @ByRef DoubleMat4x3 put(@ByRef DoubleMat4x3 rhs);
}
