package manifold3d;

import org.bytedeco.javacpp.*;
import org.bytedeco.javacpp.annotation.*;

@Platform(include = "glm/glm.hpp")
@Namespace("glm")
@Name("ivec4")
public class IntegerVec4 extends IntPointer {
    static { Loader.load(); }

    public IntegerVec4() { allocate(); }
    private native void allocate();

    public IntegerVec4(int x, int y, int z, int w) { allocate(x, y, z, w); }
    private native void allocate(int x, int y, int z, int w);

    @Name("operator[]")
    public native int get(int i);

    public native @MemberGetter int x();
    public native @MemberGetter int y();
    public native @MemberGetter int z();
    public native @MemberGetter int w();

    public native @Name("operator=") @ByRef IntegerVec4 put(@ByRef IntegerVec4 rhs);
}
