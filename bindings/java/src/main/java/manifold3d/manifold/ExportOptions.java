package manifold3d.manifold;

import manifold3d.manifold.Material;

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

    @Name("mat") public native @MemberGetter @ByRef Material material();
    @Name("mat") public native @MemberSetter void material(@ByRef Material material);
}
