package manifold3d.glm;

import org.bytedeco.javacpp.*;
import org.bytedeco.javacpp.annotation.*;

@Platform(compiler = "cpp17", include = "glm/glm.hpp")
@Namespace("glm")
@Name("vec4")
public class DoubleVec4 extends DoublePointer {
    static { Loader.load(); }

    public DoubleVec4() { allocate(); }
    private native void allocate();

    public DoubleVec4(double x, double y, double z, double w) { allocate(x, y, z, w); }
    private native void allocate(double x, double y, double z, double w);

    @Name("operator[]")
    public native double get(int i);

    public native @MemberGetter double x();
    public native @MemberGetter double y();
    public native @MemberGetter double z();
    public native @MemberGetter double w();

    public native @Name("operator=") @ByRef DoubleVec4 put(@ByRef DoubleVec4 rhs);
}
