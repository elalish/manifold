package manifold3d.pub;

import org.bytedeco.javacpp.*;
import org.bytedeco.javacpp.annotation.*;

@Platform(include = "public.h")
@Namespace("manifold")
public class ExecutionParams extends Pointer {
    static { Loader.load(); }

    public ExecutionParams() { allocate(); }
    private native void allocate();

    public native @Cast("bool") boolean intermediateChecks();
    public native ExecutionParams intermediateChecks(boolean intermediateChecks);

    public native @Cast("bool") boolean verbose();
    public native ExecutionParams verbose(boolean verbose);

    public native @Cast("bool") boolean processOverlaps();
    public native ExecutionParams processOverlaps(boolean processOverlaps);

    public native @Cast("bool") boolean suppressErrors();
    public native ExecutionParams suppressErrors(boolean suppressErrors);
}
