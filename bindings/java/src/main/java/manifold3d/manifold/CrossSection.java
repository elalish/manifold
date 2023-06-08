package manifold3d.manifold;

import org.bytedeco.javacpp.*;
import org.bytedeco.javacpp.annotation.*;

import manifold3d.LibraryPaths;
import manifold3d.ConvexHull;
import manifold3d.glm.DoubleVec2;
import manifold3d.glm.DoubleMat3x2;
import manifold3d.manifold.Rect;
import manifold3d.manifold.CrossSectionVector;

import manifold3d.pub.SimplePolygon;
import manifold3d.pub.Polygons;

@Platform(compiler = "cpp17", include = "cross_section.h", linkpath = { LibraryPaths.MANIFOLD_LIB_DIR }, link = {"manifold"})
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

    @Name("Area") public native double area();
    @Name("NumVert") public native int numVert();
    @Name("NumContour") public native int numContour();
    @Name("IsEmpty") public native boolean isEmpty();
    @Name("Bounds") public native @ByVal Rect bounds();

    @Name("Translate") public native @ByVal CrossSection translate(@ByVal DoubleVec2 v);
    public CrossSection translate(double x, double y) {
        return this.translate(new DoubleVec2(x, y));
    }
    public CrossSection translateX(double x) {
        return this.translate(new DoubleVec2(x, 0));
    }
    public CrossSection translateY(double y) {
        return this.translate(new DoubleVec2(0, y));
    }

    @Name("Rotate") public native @ByVal CrossSection rotate(float degrees);
    @Name("Scale") public native @ByVal CrossSection scale(@ByVal DoubleVec2 s);
    @Name("Mirror") public native @ByVal CrossSection mirror(@ByVal DoubleVec2 ax);
    @Name("Transform") public native @ByVal CrossSection transform(@ByVal DoubleMat3x2 m);
    @Name("Simplify") public native @ByVal CrossSection simplify(double epsilon);

    @Name("Offset") public native @ByVal CrossSection offset(double delta, @Cast("manifold::CrossSection::JoinType") int joinType, double miterLimit, int arcTolerance);

    @Name("Boolean") public native @ByVal CrossSection booleanOp(@ByRef CrossSection second, @Cast("manifold::OpType") int op);
    public static native @ByVal CrossSection BatchBoolean(@ByRef CrossSectionVector sections, @Cast("manifold::OpType") int op);

    public @ByVal CrossSection convexHull() {
        return ConvexHull.ConvexHull(this);
    }

    public @ByVal CrossSection convexHull(float precision) {
        return ConvexHull.ConvexHull(this, precision);
    }

    public @ByVal CrossSection convexHull(@Const @ByRef CrossSection other) {
       return ConvexHull.ConvexHull(this, other);
    }

    public @ByVal CrossSection convexHull(@Const @ByRef CrossSection other, float precision) {
       return ConvexHull.ConvexHull(this, other, precision);
    }

    @Name("operator+") public native @ByVal CrossSection add(@ByRef CrossSection rhs);
    @Name("operator+=") public native @ByVal CrossSection addPut(@ByRef CrossSection rhs);
    @Name("operator-") public native @ByVal CrossSection subtract(@ByRef CrossSection rhs);
    @Name("operator-=") public native @ByRef CrossSection subtractPut(@ByRef CrossSection rhs);
    @Name("operator^") public native @ByVal CrossSection intersect(@ByRef CrossSection rhs);
    @Name("operator^=") public native @ByVal CrossSection intersectPut(@ByRef CrossSection rhs);

    @Name("RectClip") public native @ByVal CrossSection rectClip(@ByVal Rect rect);
    public static native @ByVal CrossSection Compose(@ByRef CrossSectionVector crossSection);
    @Name("Decompose") public native @ByVal CrossSectionVector decompose();

    public static native @ByVal CrossSection Circle(float radius, int circularSegments);
    public static native @ByVal CrossSection Square(@ByRef DoubleVec2 size, boolean center);

    @Name("ToPolygons") public native @ByVal Polygons toPolygons();
}
