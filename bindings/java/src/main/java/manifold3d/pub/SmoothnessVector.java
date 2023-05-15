package manifold3d.pub;

import manifold3d.pub.Smoothness;
import org.bytedeco.javacpp.*;
import org.bytedeco.javacpp.annotation.*;

@Platform(include = {"<vector>", "public.h"})
@Name("std::vector<manifold::Smoothness>")
public class SmoothnessVector extends Pointer {
    static { Loader.load(); }

    public SmoothnessVector() { allocate(); }
    private native void allocate();

    public native @Cast("size_t") long size();
    public native void resize(@Cast("size_t") long n);

    @Name("operator[]") public native @ByRef Smoothness get(@Cast("size_t") long i);

    public void set(@Cast("size_t") long i, Smoothness value) {
        get(i).put(value);
    }

    @Name("push_back") public native void pushBack(@ByRef Smoothness value);
}
