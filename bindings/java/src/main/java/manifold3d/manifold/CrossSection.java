package manifold3d.manifold;

import org.bytedeco.javacpp.*;
import org.bytedeco.javacpp.annotation.*;

import manifold3d.glm.DoubleVec2;
import manifold3d.glm.DoubleMat3x2;
import manifold3d.manifold.Rect;

import manifold3d.pub.SimplePolygon;
import manifold3d.pub.Polygons;

@Platform(include = "cross_section.h", link = {"manifold"})
@Namespace("manifold")
public class CrossSection extends Pointer {
    static { Loader.load(); }

    public CrossSection() { allocate(); }
    private native void allocate();

    @Name("operator=")
    public native @ByRef CrossSection put(@ByRef CrossSection other);

    public enum FillRule {
        NonZero,
        Positive,
        Negative
    };

    public enum JoinType {
        Square,
        Round,
        Miter
    };

    public CrossSection(@ByRef SimplePolygon contour, @Cast("manifold::CrossSection::FillRule") int fillrule) { allocate(contour, fillrule); }
    private native void allocate(@ByRef SimplePolygon contour, @Cast("manifold::CrossSection::FillRule") int fillrule);

    public CrossSection(@ByRef Polygons contours, @Cast("manifold::CrossSection::FillRule") int fillrule) { allocate(contours, fillrule); }
    private native void allocate(@ByRef Polygons contours, @Cast("manifold::CrossSection::FillRule") int fillrule);

    //// Other methods
    public native @ByVal DoubleVec2 Area();
    public native int NumVert();
    public native int NumContour();
    public native boolean IsEmpty();
    public native @ByVal Rect Bounds();

    public native @ByVal CrossSection Translate(@ByVal DoubleVec2 v);
    public CrossSection translateX(double x) {
        return this.Translate(new DoubleVec2(x, 0));
    }
    public CrossSection translateY(double y) {
        return this.Translate(new DoubleVec2(0, y));
    }

    public native @ByVal CrossSection Rotate(float degrees);
    public native @ByVal CrossSection Scale(@ByVal DoubleVec2 s);
    public native @ByVal CrossSection Mirror(@ByVal DoubleVec2 ax);
    public native @ByVal CrossSection Transform(@ByVal DoubleMat3x2 m);
    // Warp method is omitted because of the std::function parameter
    public native @ByVal CrossSection Simplify(double epsilon);

    public native @ByVal CrossSection Offset(double delta, @Cast("manifold::CrossSection::JoinType") int jt, double miter_limit, double arc_tolerance);

    public native @ByVal CrossSection Boolean(@ByRef CrossSection second, @Cast("manifold::OpType") int op);

    @Name("ConvexHull") public native @ByVal CrossSection convexHull();
    @Name("ConvexHull") public native @ByVal CrossSection convexHull(@ByRef CrossSection other);

    @Name("operator+") public native @ByVal CrossSection add(@ByRef CrossSection rhs);
    @Name("operator+=") public native @ByVal CrossSection addPut(@ByRef CrossSection rhs);
    @Name("operator-") public native @ByVal CrossSection subtract(@ByRef CrossSection rhs);
    @Name("operator-=") public native @ByRef CrossSection subtractPut(@ByRef CrossSection rhs);
    @Name("operator^") public native @ByVal CrossSection intersect(@ByRef CrossSection rhs);
    @Name("operator^=") public native @ByVal CrossSection intersectPut(@ByRef CrossSection rhs);

    public native @ByVal CrossSection RectClip(@ByVal Rect rect);
    // Compose and Decompose methods are omitted because of the std::vector parameter

    public static native @ByVal CrossSection Circle(float radius, int circularSegments);
    public static native @ByVal CrossSection Square(@ByRef DoubleVec2 size, boolean center);

    public native @ByVal Polygons ToPolygons();
}
