package manifold3d.pub;

import org.bytedeco.javacpp.*;
import org.bytedeco.javacpp.annotation.*;

@Platform(include = "public.h")
@Namespace("manifold")
public class Curvature extends Pointer {
    static { Loader.load(); }

    public Curvature() { allocate(); }
    public native void allocate();

    public native @MemberGetter float maxMeanCurvature();
    public native @MemberSetter void maxMeanCurvature(float maxMeanCurvature);

    public native @MemberGetter float minMeanCurvature();
    public native @MemberSetter void minMeanCurvature(float minMeanCurvature);

    public native @MemberGetter float maxGaussianCurvature();
    public native @MemberSetter void maxGaussianCurvature(float maxGaussianCurvature);

    public native @MemberGetter float minGaussianCurvature();
    public native @MemberSetter void minGaussianCurvature(float minGaussianCurvature);

}
