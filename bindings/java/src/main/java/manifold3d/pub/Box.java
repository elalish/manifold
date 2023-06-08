package manifold3d.pub;

import manifold3d.glm.DoubleVec3;
import manifold3d.glm.DoubleMat4x3;

import org.bytedeco.javacpp.*;
import org.bytedeco.javacpp.annotation.*;

@Platform(compiler = "cpp17", include = "public.h")
@Namespace("manifold")
public class Box extends Pointer {
    static { Loader.load(); }

    public Box() { allocate(); }
    private native void allocate();

    public Box(@ByRef DoubleVec3 p1, @ByRef DoubleVec3 p2) { allocate(p1, p2); }
    private native void allocate(@ByRef DoubleVec3 p1, @ByRef DoubleVec3 p2);

    public native @ByVal DoubleVec3 Size();
    public native @ByVal DoubleVec3 Center();
    public native float Scale();
    public native boolean Contains(@ByRef DoubleVec3 p);

    public native boolean Contains(@ByRef Box box);
    public native void Union(@ByRef DoubleVec3 p);
    public native @ByVal Box Union(@ByRef Box box);
    public native @ByVal Box Transform(@ByRef DoubleMat4x3 transform);

    @Name("operator+")
    public native @ByVal Box add(@ByRef DoubleVec3 shift);
    @Name("operator+=")
    public native @ByRef Box addPut(@ByRef DoubleVec3 shift);

    @Name("operator*")
    public native @ByVal Box multiply(@ByRef DoubleVec3 scale);
    @Name("operator*=")
    public native @ByRef Box multiplyPut(@ByRef DoubleVec3 scale);

    public native boolean DoesOverlap(@ByRef Box box);
    public native boolean DoesOverlap(@ByRef DoubleVec3 p);
    public native boolean IsFinite();
}
