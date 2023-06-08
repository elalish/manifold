package manifold3d.glm;

import org.bytedeco.javacpp.*;
import org.bytedeco.javacpp.annotation.*;

@Platform(compiler = "cpp17", include = "glm/glm.hpp")
@Namespace("glm")
@Name("vec3")
public class DoubleVec3 extends DoublePointer {
    static { Loader.load(); }

    public DoubleVec3() { allocate(); }
    private native void allocate();

    public DoubleVec3(double x, double y, double z) { allocate(x, y, z); }
    private native void allocate(double x, double y, double z);

    @Name("operator []")
    public native double get(int i);

    public native @MemberGetter double x();
    public native @MemberGetter double y();
    public native @MemberGetter double z();

    public native @Name("operator=") @ByRef DoubleVec3 put(@ByRef DoubleVec3 rhs);
}
