package manifold3d;

import org.bytedeco.javacpp.*;
import org.bytedeco.javacpp.annotation.*;

@Platform(compiler = "cpp17", include = "public.h")
@Namespace("manifold")
public class Quality extends Pointer {
    //private native static int circularSegments_();
    //private native static float circularAngle_();
    //private native static float circularEdgeLength_();

    public static native void SetMinCircularAngle(float angle);
    public static native void SetMinCircularEdgeLength(float length);
    public static native void SetCircularSegments(int number);
    public static native int GetCircularSegments(float radius);
}
