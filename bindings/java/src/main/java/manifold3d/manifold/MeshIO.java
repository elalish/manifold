package manifold3d.manifold;

import manifold3d.pub.DoubleMesh;
import manifold3d.manifold.ExportOptions;

import org.bytedeco.javacpp.*;
import org.bytedeco.javacpp.annotation.*;


@Platform(include = {"meshIO.h"}, link = {"meshIO"})
@Namespace("manifold")
public class MeshIO {
    static { Loader.load(); }

    public native static @ByVal DoubleMesh ImportMesh(@StdString String filename, @Cast("bool") boolean forceCleanup);

    public native static void ExportMesh(@StdString String filename, @Const @ByRef DoubleMesh mesh, @Const @ByRef ExportOptions options);
}
