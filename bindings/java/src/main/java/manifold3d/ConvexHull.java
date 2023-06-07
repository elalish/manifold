package manifold3d;

import org.bytedeco.javacpp.*;
import org.bytedeco.javacpp.annotation.*;

import manifold3d.Manifold;
import manifold3d.glm.DoubleVec3Vector;
import manifold3d.glm.DoubleMat4x3Vector;
import manifold3d.glm.DoubleMat4x3Vector;
import manifold3d.UIntVecVector;

import manifold3d.Manifold;
import java.nio.DoubleBuffer;
import java.nio.IntBuffer;

@Platform(include = {"convex_hull.hpp"}, link = {"manifold"})
public class ConvexHull extends Pointer {
    static { Loader.load(); }

    public ConvexHull() { }

    public static native @ByVal Manifold ConvexHull(@Const @ByRef Manifold manifold);
    public static native @ByVal Manifold ConvexHull(@Const @ByRef Manifold manifold, @Const @ByRef Manifold other);
}
