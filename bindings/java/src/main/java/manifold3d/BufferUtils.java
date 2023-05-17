package manifold3d;

import org.bytedeco.javacpp.*;
import org.bytedeco.javacpp.annotation.*;

import manifold3d.pub.SimplePolygon;
import manifold3d.glm.DoubleVec3Vector;
import manifold3d.glm.DoubleVec4Vector;
import manifold3d.glm.IntegerVec3Vector;

@Platform(include= {"buffer_utils.hpp", "manifold.h"})
public class BufferUtils extends Pointer {
    static { Loader.load(); }

    public BufferUtils() { }

    public static native @ByVal SimplePolygon createDoubleVec2Vector(DoublePointer values, @Cast("std::size_t") long count);
    public static native @ByVal DoubleVec3Vector createDoubleVec3Vector(DoublePointer values, @Cast("std::size_t") long count);
    public static native @ByVal IntegerVec3Vector createIntegerVec3Vector(IntPointer values, @Cast("std::size_t") long count);
    public static native @ByVal DoubleVec4Vector createDoubleVec4Vector(DoublePointer values, @Cast("std::size_t") long count);
}
