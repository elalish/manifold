package manifold3d;

import org.bytedeco.javacpp.*;
import org.bytedeco.javacpp.annotation.*;

@Platform(include = "<vector>")
@Name("std::vector<float>")
public class FloatVector extends Pointer {
    static { Loader.load(); }
    /** Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}. */
    public FloatVector(FloatPointer p) { super(p); }
    public FloatVector() { allocate(); }
    private native void allocate();
    public native @Cast("size_t") long size();
    public native @Cast("bool") boolean empty();
    public native void resize(@Cast("size_t") long n);
    public native void reserve(@Cast("size_t") long n);
    public native @Name("operator[]") float get(@Cast("size_t") long n);
    public native @Name("push_back") void pushBack(float value);
}
