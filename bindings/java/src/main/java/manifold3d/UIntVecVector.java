package manifold3d;

import org.bytedeco.javacpp.*;
import org.bytedeco.javacpp.annotation.*;


@Platform(include = { "<vector>" })
@Name("std::vector<std::vector<uint32_t>>")
public class UIntVecVector extends Pointer {
    static { Loader.load(); }

    public UIntVecVector() { allocate(); }
    public native void allocate();

    public native @Cast("size_t") long size();
    public native void resize(@Cast("size_t") long n);

    @Name("operator[]") public native @ByRef UIntVector get(@Cast("size_t") long i);
    @Name("push_back") public native void pushBack(@ByRef UIntVector value);
}
