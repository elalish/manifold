package manifold3d.pub;

import org.bytedeco.javacpp.*;
import org.bytedeco.javacpp.annotation.*;

@Platform(include = "public.h")
@Namespace("manifold")
public class Smoothness extends Pointer {
    static { Loader.load(); }

    public Smoothness() { allocate(); }
    public native void allocate();

    public native @MemberGetter int halfedge();
    public native @MemberSetter void halfedge(float halfedge);

    public native @MemberGetter float smoothness();
    public native @MemberSetter void smoothness(float smoothness);
}
