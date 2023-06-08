package manifold3d.pub;

import org.bytedeco.javacpp.*;
import org.bytedeco.javacpp.annotation.*;

@Platform(compiler = "cpp17", include = "public.h")
@Namespace("manifold")
public class Properties extends Pointer {
    static { Loader.load(); }

    public Properties() { allocate(); }
    public native void allocate();

    public native @MemberGetter float surfaceArea();
    public native @MemberSetter void surfaceArea(float surfaceArea);

    public native @MemberGetter float volume();
    public native @MemberSetter void volume(float volume);
}
