package manifold3d.manifold;

import manifold3d.pub.DoubleMesh;
import manifold3d.manifold.ExportOptions;

import org.bytedeco.javacpp.*;
import org.bytedeco.javacpp.annotation.*;


@Platform(include = "meshIO.h")
@Namespace("manifold")
public class MeshIO {
    public native static @ByVal DoubleMesh ImportMesh(@StdString BytePointer filename, @Cast("bool") boolean forceCleanup);

    public native static void ExportMesh(@StdString BytePointer filename, @ByRef DoubleMesh mesh, @ByRef ExportOptions options);
}
