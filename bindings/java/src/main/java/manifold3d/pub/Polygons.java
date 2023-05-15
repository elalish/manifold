package manifold3d.pub;

import manifold3d.pub.SimplePolygon;
import org.bytedeco.javacpp.*;
import org.bytedeco.javacpp.annotation.*;

@Platform(include = "manifold.h")
@Namespace("manifold")
@Name("Polygons")
public class Polygons extends Pointer {
    static { Loader.load(); }

    public Polygons() { allocate(); }
    private native void allocate();

    public native @Cast("size_t") long size();
    public native void resize(@Cast("size_t") long n);

    @Name("operator[]") public native @ByRef SimplePolygon get(@Cast("size_t") long i);
    @Name("push_back") public native void pushBack(@ByRef SimplePolygon value);
}
