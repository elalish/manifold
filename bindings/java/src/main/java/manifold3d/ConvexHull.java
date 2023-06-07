package manifold3d;

import org.bytedeco.javacpp.*;
import org.bytedeco.javacpp.annotation.*;

import manifold3d.Manifold;
import manifold3d.manifold.CrossSection;
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
    public static native @ByVal Manifold ConvexHull(@Const @ByRef Manifold manifold, @Const float precision);
    public static native @ByVal Manifold ConvexHull(@Const @ByRef Manifold manifold, @Const @ByRef Manifold other);
    public static native @ByVal Manifold ConvexHull(@Const @ByRef Manifold manifold, @Const @ByRef Manifold other, float precision);

    public static native @ByVal CrossSection ConvexHull(@Const @ByRef CrossSection crossSection);
    public static native @ByVal CrossSection ConvexHull(@Const @ByRef CrossSection crossSection, @Const float precision);
    public static native @ByVal CrossSection ConvexHull(@Const @ByRef CrossSection crossSection, @Const @ByRef CrossSection other);
    public static native @ByVal CrossSection ConvexHull(@Const @ByRef CrossSection crossSection, @Const @ByRef CrossSection other, float precision);
}
