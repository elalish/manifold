import org.bytedeco.javacpp.*;
import org.bytedeco.javacpp.annotation.*;

import manifold3d.glm.DoubleVec2;
import manifold3d.glm.DoubleMat3x2;
import manifold3d.manifold.Rect;

import manifold3d.pub.SimplePolygon;
import manifold3d.pub.Polygons;

@Platform(include = "cross_section.h")
@Namespace("manifold") // Replace with your actual namespace
public class CrossSection extends Pointer {
    static { Loader.load(); }

    public CrossSection() { allocate(); }
    private native void allocate();

    @Name("operator=")
    public native @ByRef CrossSection put(@ByRef CrossSection other);

    public enum FillRule {
        EvenOdd,
        NonZero,
        Positive,
        Negative
    };

    public enum JoinType {
        Square,
        Round,
        Miter
    };

    public CrossSection(@ByRef SimplePolygon contour, @Cast("FillRule") int fillrule) { allocate(contour, fillrule); }
    private native void allocate(@ByRef SimplePolygon contour, @Cast("FillRule") int fillrule);

    public CrossSection(@ByRef Polygons contours, @Cast("FillRule") int fillrule) { allocate(contours, fillrule); }
    private native void allocate(@ByRef Polygons contours, @Cast("FillRule") int fillrule);

    //// Other methods
    public native @ByVal DoubleVec2 Area();
    public native int NumVert();
    public native int NumContour();
    public native boolean IsEmpty();
    public native @ByVal Rect Bounds();

    public native @ByVal CrossSection Translate(@ByVal DoubleVec2 v);
    public native @ByVal CrossSection Rotate(float degrees);
    public native @ByVal CrossSection Scale(@ByVal DoubleVec2 s);
    public native @ByVal CrossSection Mirror(@ByVal DoubleVec2 ax);
    public native @ByVal CrossSection Transform(@ByVal DoubleMat3x2 m);
    // Warp method is omitted because of the std::function parameter
    public native @ByVal CrossSection Simplify(double epsilon);

    public native @ByVal CrossSection Offset(double delta, @Cast("JoinType") int jt, double miter_limit, double arc_tolerance);

    public native @ByVal CrossSection Boolean(@ByRef CrossSection second, @Cast("OpType") int op);
    // BatchBoolean method is omitted because of the std::vector parameter

    @Name("operator+")
    public native @ByVal CrossSection add(@ByRef CrossSection rhs);
    @Name("operator+=")
    public native @ByVal CrossSection addPut(@ByRef CrossSection rhs);
    @Name("operator-")
    public native @ByVal CrossSection subtract(@ByRef CrossSection rhs);
    @Name("operator-=")
    public native @ByRef CrossSection subtractPut(@ByRef CrossSection rhs);
    @Name("operator^")
    public native @ByVal CrossSection bitwiseXor(@ByRef CrossSection rhs);
    @Name("operator^=")
    public native @ByVal CrossSection bitwiseXorPut(@ByRef CrossSection rhs);

    public native @ByVal CrossSection RectClip(@ByVal Rect rect);
    // Compose and Decompose methods are omitted because of the std::vector parameter

    public native @ByVal Polygons ToPolygons();
}
