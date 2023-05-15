package manifold3d;

import org.bytedeco.javacpp.*;
import org.bytedeco.javacpp.annotation.*;

@Platform(include = "<vector>")
@Name("std::vector<uint32_t>")
public class UIntVector extends Pointer {
    static { Loader.load(); }

    public UIntVector(Pointer p) { super(p); }
    public UIntVector() { allocate(); }
    private native void allocate();

    public native @Cast("size_t") long size();
    public native @Cast("bool") boolean empty();
    public native void resize(@Cast("size_t") long n);
    public native void reserve(@Cast("size_t") long n);
    public native @Name("operator[]") long get(@Cast("size_t") long n);
    public native @Name("push_back") void pushBack(@Cast("uint32_t") int value);
}
