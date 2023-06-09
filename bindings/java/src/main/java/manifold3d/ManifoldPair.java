package manifold3d;

import org.bytedeco.javacpp.*;
import org.bytedeco.javacpp.annotation.*;

import manifold3d.Manifold;

@Platform(compiler = "cpp17", include = {"manifold.h"}, linkpath = { LibraryPaths.MANIFOLD_LIB_DIR, LibraryPaths.MANIFOLD_LIB_DIR_WINDOWS }, link = { "manifold" })
@Name("std::pair<manifold::Manifold, manifold::Manifold>")
public class ManifoldPair extends Pointer {
    static { Loader.load(); }

    public ManifoldPair() { allocate(); }
    private native void allocate();

    public native @Name("operator=") @ByRef ManifoldPair put(@ByRef ManifoldPair x);

    @NoOffset public native @MemberGetter @ByRef Manifold first();
    @MemberSetter @Index public native @ByRef ManifoldPair first(@ByRef Manifold manifold);

    @NoOffset public native @MemberGetter @ByRef Manifold second();
    @MemberSetter @Index public native ManifoldPair second(@ByRef Manifold second);
}
