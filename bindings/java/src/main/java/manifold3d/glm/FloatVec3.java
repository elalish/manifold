package manifold3d.glm;

import org.bytedeco.javacpp.*;
import org.bytedeco.javacpp.annotation.*;

@Platform(include = "glm/glm.hpp")
@Namespace("glm")
@Name("vec3")
public class FloatVec3 extends FloatPointer {
    static { Loader.load(); }

    public FloatVec3() { allocate(); }
    private native void allocate();

    public FloatVec3(float x, float y, float z) { allocate(x, y, z); }
    private native void allocate(float x, float y, float z);

    @Name("operator []")
    public native float get(int i);

    public native @MemberGetter float x();
    public native @MemberGetter float y();
    public native @MemberGetter float z();

    //public native void put(int i, float value);
    //public native void set(int component, float value);

    public native @Name("operator=") @ByRef FloatVec3 put(@ByRef FloatVec3 rhs);
}
