package manifold3d.glm;

import manifold3d.glm.DoubleMat4x3;
import manifold3d.glm.DoubleVec3;

import org.bytedeco.javacpp.*;
import org.bytedeco.javacpp.annotation.*;

@Platform(include = {"matrix_transforms.hpp"})
public class MatrixTransforms extends Pointer {
    static { Loader.load(); }

    public static native @ByVal DoubleMat4x3 Yaw(@ByRef DoubleMat4x3 mat, float angle);
    public static native @ByVal DoubleMat4x3 Pitch(@ByRef DoubleMat4x3 mat, float angle);
    public static native @ByVal DoubleMat4x3 Roll(@ByRef DoubleMat4x3 mat, float angle);

    public static native @ByVal DoubleMat4x3 Rotate(@ByRef DoubleMat4x3 mat, @ByRef DoubleVec3 angles);
    public static native @ByVal DoubleMat4x3 Rotate(@ByRef DoubleMat4x3 mat, @ByRef DoubleVec3 axis, float angle);
    public static native @ByVal DoubleMat4x3 Translate(@ByRef DoubleMat4x3 mat, @ByRef DoubleVec3 vec);

    public static native @ByVal DoubleMat4x3 Transform(@ByRef DoubleMat4x3 mat1, @ByRef DoubleMat4x3 mat2);
    public static native @ByVal DoubleMat4x3 InvertTransform(@ByRef DoubleMat4x3 mat);
}
