package manifold3d;

import org.bytedeco.javacpp.*;
import org.bytedeco.javacpp.annotation.*;

@Platform(include = "glm/glm.hpp")
@Namespace("glm")
@Name("vec4")
public class FloatVec4 extends FloatPointer {
    static { Loader.load(); }

    public FloatVec4() { allocate(); }
    private native void allocate();

    public FloatVec4(float x, float y, float z, float w) { allocate(x, y, z, w); }
    private native void allocate(float x, float y, float z, float w);

    @Name("operator[]")
    public native float get(int i);

    public native @MemberGetter float x();
    public native @MemberGetter float y();
    public native @MemberGetter float z();
    public native @MemberGetter float w();

    public native @Name("operator=") @ByRef FloatVec4 put(@ByRef FloatVec4 rhs);
}
