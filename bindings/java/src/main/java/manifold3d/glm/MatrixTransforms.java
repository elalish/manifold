package manifold3d.glm;

import manifold3d.glm.DoubleMat4x3;
import manifold3d.glm.DoubleVec3;

import org.bytedeco.javacpp.*;
import org.bytedeco.javacpp.annotation.*;

@Platform(include = {"matrix_transforms.hpp"})
public class MatrixTransforms extends Pointer {
    static { Loader.load(); }

    public static native @ByVal DoubleMat4x3 yaw(@ByRef DoubleMat4x3 mat, float angle);
    public static native @ByVal DoubleMat4x3 pitch(@ByRef DoubleMat4x3 mat, float angle);
    public static native @ByVal DoubleMat4x3 roll(@ByRef DoubleMat4x3 mat, float angle);

    public static native @ByVal DoubleMat4x3 rotate(@ByRef DoubleMat4x3 mat, @ByRef DoubleVec3 angles);
    public static native @ByVal DoubleMat4x3 rotate(@ByRef DoubleMat4x3 mat, @ByRef DoubleVec3 axis, float angle);
    public static native @ByVal DoubleMat4x3 translate(@ByRef DoubleMat4x3 mat, @ByRef DoubleVec3 vec);

    public static native @ByVal DoubleMat4x3 transform(@ByRef DoubleMat4x3 mat1, @ByRef DoubleMat4x3 mat2);
}
