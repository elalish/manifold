package manifold3d.glm;

import org.bytedeco.javacpp.*;
import org.bytedeco.javacpp.annotation.*;

@Platform(compiler = "cpp17", include = "glm/glm.hpp")
@Namespace("glm")
@Name("ivec3")
public class IntegerVec3 extends IntPointer {
    static { Loader.load(); }

    public IntegerVec3() { allocate(); }
    private native void allocate();

    public IntegerVec3(int x, int y, int z) { allocate(x, y, z); }
    private native void allocate(int x, int y, int z);

    @Name("operator []")
    public native int get(int i);

    public native @MemberGetter int x();
    public native @MemberGetter int y();
    public native @MemberGetter int z();

    //public native void put(int i, int value);
    //public native void set(int component, int value);

    public native @Name("operator=") @ByRef IntegerVec3 put(@ByRef IntegerVec3 rhs);
}
