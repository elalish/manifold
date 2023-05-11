package manifold3d;

import manifold3d.FloatVec4;
import manifold3d.FloatVec4Vector;
import manifold3d.IntegerVec3;
import manifold3d.IntegerVec4;

import manifold3d.Public.DoubleMesh;

import manifold3d.StdVector.UIntVector;
import manifold3d.StdVector.FloatVector;

import org.bytedeco.javacpp.*;
import org.bytedeco.javacpp.annotation.*;

@Platform(include = "meshIO.h")
@Namespace("manifold")
public class Manifold {
    static { Loader.load(); }

    public class Material extends Pointer {
        public Material() { allocate(); }
        public native void allocate();

        public native @MemberGetter float roughness();
        public native @MemberSetter void roughness(float roughness);

        public native @MemberGetter float metalness();
        public native @MemberSetter void metalness(float metalness);

        public native @MemberGetter @ByRef FloatVec4 color();
        public native @MemberSetter void color(@ByRef FloatVec4 color);

        public native @MemberGetter @ByRef FloatVec4Vector vertColor();
        public native @MemberSetter void vertColor(@ByRef FloatVec4Vector vertColor);

        public native @MemberGetter @ByRef IntegerVec3 normalChannels();
        public native @MemberSetter void normalChannels(@ByRef IntegerVec3 color);

        public native @MemberGetter @ByRef IntegerVec4 colorChannels();
        public native @MemberSetter void colorChannels(@ByRef IntegerVec4 color);
    }

    public class ExportOptions extends Pointer {
        public ExportOptions() { allocate(); }
        public native void allocate();

        public native @MemberGetter boolean faceted();
        public native @MemberSetter void faceted(boolean faceted);
    }

    public native static @ByVal DoubleMesh ImportMesh(@StdString BytePointer filename, @Cast("bool") boolean forceCleanup);

    public native static void ExportMesh(@StdString BytePointer filename, @ByRef DoubleMesh mesh, @ByRef ExportOptions options);

    public static class MeshGL extends Pointer {

        public native @Cast("uint32_t") int NumVert();
        public native @Cast("uint32_t") int NumTri();

        public native @Cast("uint32_t") int numProp();
        public native MeshGL numProp(@Cast("uint32_t") int numProp);

        public native @ByRef FloatVector vertProperties();
        public native MeshGL vertProperties(@ByRef FloatVector vertProperties);

        public native @ByRef UIntVector triVerts();
        public native MeshGL triVerts(@ByRef UIntVector triVerts);

        public native @ByRef UIntVector mergeFromVert();
        public native MeshGL mergeFromVert(@ByRef UIntVector mergeFromVert);

        public native @ByRef UIntVector mergeToVert();
        public native MeshGL mergeToVert(@ByRef UIntVector mergeToVert);

        public native @ByRef UIntVector runIndex();
        public native MeshGL runIndex(@ByRef UIntVector runIndex);

        public native @ByRef UIntVector runOriginalID();
        public native MeshGL runOriginalID(@ByRef UIntVector runOriginalID);

        public native @ByRef FloatVector runTransform();
        public native MeshGL runTransform(@ByRef FloatVector runTransform);

        public native @ByRef UIntVector faceID();
        public native MeshGL faceID(@ByRef UIntVector faceID);

        public native @ByRef FloatVector halfedgeTangent();
        public native MeshGL halfedgeTangent(@ByRef FloatVector halfedgeTangent);

        public native float precision();
        public native MeshGL precision(float precision);

        //// MeshGL constructor and other methods
        public native boolean Merge();
    }
}
