package manifold3d.glm;

import org.bytedeco.javacpp.*;
import org.bytedeco.javacpp.annotation.*;

@Platform(include = "glm/glm.hpp")
@Namespace("glm")
@Name("vec2")
public class DoubleVec2 extends DoublePointer {
    static { Loader.load(); }

    public DoubleVec2() { allocate(); }
    private native void allocate();

    public DoubleVec2(double x, double y) { allocate(x, y); }
    private native void allocate(double x, double y);

    @Name("operator []")
    public native double get(int i);

    public native @MemberGetter double x();
    public native @MemberGetter double y();

    public native @Name("operator=") @ByRef DoubleVec2 put(@ByRef DoubleVec2 rhs);
}
