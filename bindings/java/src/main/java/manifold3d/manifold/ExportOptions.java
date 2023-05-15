package manifold3d.manifold;

import org.bytedeco.javacpp.*;
import org.bytedeco.javacpp.annotation.*;

@Platform(include = "meshIO.h")
@Namespace("manifold")
public class ExportOptions extends Pointer {
    static { Loader.load(); }

    public ExportOptions() { allocate(); }
    public native void allocate();

    public native @MemberGetter boolean faceted();
    public native @MemberSetter void faceted(boolean faceted);
}
