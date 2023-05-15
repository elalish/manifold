package manifold3d.manifold;

import org.bytedeco.javacpp.*;
import org.bytedeco.javacpp.annotation.*;

import manifold3d.glm.DoubleVec2;
import manifold3d.glm.DoubleMat3x2;

@Platform(include = "cross_section.h")
@Namespace("manifold")
@Opaque
class CrossSection extends Pointer {
    // Empty class for forward declaration
}

@Platform(include = "cross_section.h")
@Namespace("manifold")
public class Rect extends Pointer {
    static { Loader.load(); }

    public Rect() { allocate(); }
    private native void allocate();

    @Name("operator=")
    public native @ByRef Rect put(@ByRef Rect other);

    public Rect(@ByVal DoubleVec2 a, @ByVal DoubleVec2 b) { allocate(a, b); }
    private native void allocate(@ByVal DoubleVec2 a, @ByVal DoubleVec2 b);

    public native @ByVal DoubleVec2 Size();
    public native float Scale();
    public native @ByVal DoubleVec2 Center();
    public native boolean Contains(@ByVal DoubleVec2 pt);
    public native boolean Contains(@ByRef Rect other);
    public native boolean DoesOverlap(@ByRef Rect other);
    public native boolean IsEmpty();
    public native boolean IsFinite();

    public native void Union(@ByVal DoubleVec2 p);
    public native @ByVal Rect Union(@ByRef Rect other);

    @Name("operator+")
    public native @ByVal Rect add(@ByVal DoubleVec2 shift);
    @Name("operator+=")
    public native @ByVal Rect addPut(@ByVal DoubleVec2 shift);
    @Name("operator*")
    public native @ByVal Rect multiply(@ByVal DoubleVec2 scale);
    @Name("operator*=")
    public native @ByVal Rect multiplyPut(@ByVal DoubleVec2 scale);

    public native @ByVal Rect Transform(@ByVal DoubleMat3x2 m);
    public native @ByVal CrossSection AsCrossSection();
}
