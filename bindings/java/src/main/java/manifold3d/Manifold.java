package manifold3d;

import java.net.URL;
import java.security.CodeSource;
import java.security.ProtectionDomain;

import manifold3d.pub.DoubleMesh;

import manifold3d.UIntVector;
import manifold3d.FloatVector;

import java.io.IOException;
import java.io.File;

import manifold3d.ManifoldPair;
import manifold3d.ManifoldVector;
import manifold3d.manifold.MeshGL;
import manifold3d.manifold.ExportOptions;
import manifold3d.manifold.CrossSection;

import manifold3d.pub.DoubleMesh;
import manifold3d.pub.Box;
import manifold3d.pub.Properties;
import manifold3d.pub.Curvature;
import manifold3d.pub.SmoothnessVector;
import manifold3d.pub.OpType;

import manifold3d.glm.DoubleVec3Vector;
import manifold3d.glm.DoubleMat4x3;
import manifold3d.glm.DoubleVec2;
import manifold3d.glm.DoubleVec3;

import org.bytedeco.javacpp.*;
import org.bytedeco.javacpp.annotation.*;

@Platform(include = {"manifold.h", "meshIO.h"}, link = {"manifold"})
@Namespace("manifold")
public class Manifold extends Pointer {
    static {

        String osName = System.getProperty("os.name").toLowerCase();
        if (osName.contains("linux")) {
            try {
                System.load(Loader.extractResource("/libmanifold.so", null, "libmanifold", ".so").getAbsolutePath());
                System.load(Loader.extractResource("/libmeshIO.so", null, "libmeshIO", ".so").getAbsolutePath());
            } catch (IOException e) {
                throw new RuntimeException(e);
            }
        }

        Loader.load();
    }

    // Constructors and destructor
    public Manifold() { allocate(); }
    private native void allocate();

    public Manifold(@ByRef Manifold other) { allocate(other); }
    private native void allocate(@ByRef Manifold other);

    public native @ByRef @Name("operator=") Manifold put(@ByRef Manifold other);

    public Manifold(@ByRef MeshGL mesh, @ByRef FloatVector propertyTolerance) { allocate(mesh, propertyTolerance); }
    private native void allocate(@ByRef MeshGL mesh, @ByRef FloatVector propertyTolerance);

    public Manifold(@ByRef DoubleMesh mesh) { allocate(mesh); }
    private native void allocate(@ByRef DoubleMesh mesh);

    // Methods
    public native @ByVal DoubleMesh GetMesh();
    public native @ByVal MeshGL GetMeshGL(@ByRef DoubleVec3 normalIdx);
    public native boolean IsEmpty();
    public native @Cast("manifold::Manifold::Error") int Status();
    public native int NumVert();
    public native int NumEdge();
    public native int NumTri();
    public native int NumProp();
    public native int NumPropVert();
    @Name("BoundingBox") public native @ByVal Box boundingBox();
    public native float Precision();
    public native int Genus();
    public native @ByVal Properties GetProperties();
    public native @ByVal Curvature GetCurvature();
    public native int OriginalID();
    public native @ByVal Manifold AsOriginal();

    public native @ByVal Manifold ConvexHull(@ByRef Manifold other);

    //// Modifiers
    public native @ByVal Manifold Translate(@ByRef DoubleVec3 translation);
    public Manifold Translate(Manifold manifold, double x, double y, double z) {
        return manifold.Translate(new DoubleVec3(x, y, z));
    }
    public Manifold TranslateX(double x) {
        return this.Translate(new DoubleVec3(x, 0, 0));
    }
    public Manifold TranslateY(double y) {
        return this.Translate(new DoubleVec3(0, y, 0));
    }
    public Manifold TranslateZ(double z) {
        return this.Translate(new DoubleVec3(0, 0, z));
    }

    public native @ByVal Manifold Scale(@ByRef DoubleVec3 scale);
    public Manifold Scale(double x, double y, double z) {
        return this.Scale(new DoubleVec3(x, y, z));
    }

    public native @ByVal Manifold Rotate(float xDegrees, float yDegrees, float zDegrees);
    public native @ByVal Manifold Transform(@ByRef DoubleMat4x3 transform);
    public native @ByVal Manifold Mirror(@ByRef DoubleVec3 mirrorAxis);
    public native @ByVal Manifold Refine(int refineValue);

    // CSG operators
    public native @ByVal Manifold Boolean(@ByRef Manifold second, @Cast("manifold::OpType") int op);
    @Name("operator+") public native @ByVal Manifold add(@ByRef Manifold manifold);
    @Name("operator+=") public native @ByRef Manifold addPut(@ByRef Manifold manifold);
    @Name("operator-") public native @ByVal Manifold subtract(@ByRef Manifold manifold);
    @Name("operator-=") public native @ByRef Manifold subtractPut(@ByRef Manifold manifold);
    @Name("operator^") public native @ByVal Manifold intersect(@ByRef Manifold manifold);
    @Name("operator^=") public native @ByRef Manifold intersectPut(@ByRef Manifold manifold);

    @Name("SplitByPlane")
    public native @ByVal ManifoldPair splitByPlane(@ByRef DoubleVec3 normal, float originOffset);

    @Name("Split")
    public native @ByVal ManifoldPair split(@ByRef Manifold otehr);

    @Name("TrimByPlane")
    public native @ByVal Manifold trimByPlane(@ByRef DoubleVec3 normal, float originOffset);

    @Name("Decompose")
    public native @ByVal ManifoldVector decompose();

    @Name("BatchBoolean")
    public static native @ByVal Manifold batchBoolean(@ByRef ManifoldVector manifolds, @Cast("manifold::OpType") int op);

    //// Static methods
    public static native @ByVal Manifold Smooth(@ByRef MeshGL mesh, @ByRef SmoothnessVector sharpenedEdges);
    public static native @ByVal Manifold Smooth(@ByRef DoubleMesh mesh, @ByRef SmoothnessVector sharpenedEdges);
    public static native @ByVal Manifold Tetrahedron();
    public static native @ByVal Manifold Cube(@ByRef DoubleVec3 size, boolean center);
    public static native @ByVal Manifold Cylinder(float height, float radiusLow, float radiusHigh, int circularSegments, boolean center);
    public static native @ByVal Manifold Sphere(float radius, int circularSegments);
    public static native @ByVal Manifold Extrude(@ByRef CrossSection crossSection, float height, int nDivisions, float twistDegrees, @ByRef DoubleVec2 scaleTop);
    public static native @ByVal Manifold Revolve(@ByRef CrossSection crossSection, int circularSegments);
    public static native @ByVal Manifold Revolve(@ByRef CrossSection crossSection, int circularSegments, float revolveDegrees);
    public static native @ByVal Manifold Compose(@ByRef ManifoldVector manifolds);
}
