package manifold3d;

import org.bytedeco.javacpp.*;
import org.bytedeco.javacpp.annotation.*;

import manifold3d.LibraryPaths;
import manifold3d.manifold.CrossSectionVector;
import manifold3d.manifold.CrossSection;
import manifold3d.glm.DoubleVec3Vector;
import manifold3d.glm.DoubleMat4x3Vector;
import manifold3d.glm.DoubleMat4x3Vector;
import manifold3d.UIntVecVector;

import manifold3d.Manifold;
import java.nio.DoubleBuffer;
import java.nio.IntBuffer;

@Platform(compiler = "cpp17", include = {"mesh_utils.hpp", "buffer_utils.hpp"}, linkpath = { LibraryPaths.MANIFOLD_LIB_DIR, LibraryPaths.MANIFOLD_LIB_DIR_WINDOWS }, link = {"manifold"})
public class MeshUtils extends Pointer {
    static { Loader.load(); }

    public MeshUtils() { }

    public static native @ByVal Manifold Polyhedron(DoublePointer vertices, @Cast("std::size_t") long nVertices, IntPointer faceBuf, IntPointer faceLengths, @Cast("std::size_t") long nFaces);
    public static Manifold PolyhedronFromBuffers(DoubleBuffer vertices, long nVertices, IntBuffer faceBuf, IntBuffer faceLengths, long nFaces) {

        DoublePointer verticesPtr = new DoublePointer(vertices);
        IntPointer faceBufPtr = new IntPointer(faceBuf);
        IntPointer lengthsPtr = new IntPointer(faceLengths);

        return Polyhedron(verticesPtr, nVertices, faceBufPtr, lengthsPtr, nFaces);
    }

    public static native @ByVal Manifold Loft(@ByRef CrossSectionVector sections, @ByRef DoubleMat4x3Vector transforms);
    public static native @ByVal Manifold Loft(@ByRef CrossSection section, @ByRef DoubleMat4x3Vector transforms);

    public static native @ByVal Manifold Revolve(@ByRef CrossSection crossSection, int circularSegments);
    public static native @ByVal Manifold Revolve(@ByRef CrossSection crossSection, int circularSegments, float revolveDegrees);
}
