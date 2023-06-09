package manifold3d.manifold;

import manifold3d.pub.DoubleMesh;
import manifold3d.manifold.ExportOptions;

import manifold3d.LibraryPaths;
import org.bytedeco.javacpp.*;
import org.bytedeco.javacpp.annotation.*;


@Platform(compiler = "cpp17", include = {"meshIO.h"}, linkpath = { LibraryPaths.MESHIO_LIB_DIR, LibraryPaths.MESHIO_LIB_DIR_WINDOWS }, link = {"meshIO"})
@Namespace("manifold")
public class MeshIO {
    static { Loader.load(); }

    public native static @ByVal DoubleMesh ImportMesh(@StdString String filename, @Cast("bool") boolean forceCleanup);

    public native static void ExportMesh(@StdString String filename, @Const @ByRef DoubleMesh mesh, @Const @ByRef ExportOptions options);
}
