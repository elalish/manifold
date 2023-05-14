package manifold3d;

import org.bytedeco.javacpp.*;
import org.bytedeco.javacpp.annotation.*;

@Platform(include = "glm/glm.hpp")
@Namespace("glm")
public class Glm {
    static { Loader.load(); }

    @Name("vec3")
    public static class FloatVec3 extends FloatPointer {

        public FloatVec3() { allocate(); }
        private native void allocate();

        public FloatVec3(float x, float y, float z) { allocate(x, y, z); }
        private native void allocate(float x, float y, float z);

        @Name("operator []")
        public native float get(int i);

        public native @MemberGetter float x();
        public native @MemberGetter float y();
        public native @MemberGetter float z();

        public native @Name("operator=") @ByRef FloatVec3 put(@ByRef FloatVec3 rhs);
    }

    @Name("vec3")
    public static class DoubleVec3 extends DoublePointer {

        public DoubleVec3() { allocate(); }
        private native void allocate();

        public DoubleVec3(double x, double y, double z) { allocate(x, y, z); }
        private native void allocate(double x, double y, double z);

        @Name("operator []")
        public native double get(int i);

        public native @MemberGetter double x();
        public native @MemberGetter double y();
        public native @MemberGetter double z();

        public native @Name("operator=") @ByRef DoubleVec3 put(@ByRef DoubleVec3 rhs);
    }

    @Name("vec4")
    public static class FloatVec4 extends FloatPointer {

        public FloatVec4() { allocate(); }
        private native void allocate();

        public FloatVec4(float x, float y, float z, float w) { allocate(x, y, z, w); }
        private native void allocate(float x, float y, float z, float w);

        @Name("operator[]")
        public native float get(int i);

        public native @MemberGetter float x();
        public native @MemberGetter float y();
        public native @MemberGetter float z();
        public native @MemberGetter float w();

        public native @Name("operator=") @ByRef FloatVec4 put(@ByRef FloatVec4 rhs);
    }

    @Name("vec4")
    public static class DoubleVec4 extends DoublePointer {

        public DoubleVec4() { allocate(); }
        private native void allocate();

        public DoubleVec4(double x, double y, double z, double w) { allocate(x, y, z, w); }
        private native void allocate(double x, double y, double z, double w);

        @Name("operator[]")
        public native double get(int i);

        public native @MemberGetter double x();
        public native @MemberGetter double y();
        public native @MemberGetter double z();
        public native @MemberGetter double w();

        public native @Name("operator=") @ByRef DoubleVec4 put(@ByRef DoubleVec4 rhs);
    }

    @Name("ivec3")
    public class IntegerVec3 extends IntPointer {

        public IntegerVec3() { allocate(); }
        private native void allocate();

        public IntegerVec3(int x, int y, int z) { allocate(x, y, z); }
        private native void allocate(int x, int y, int z);

        @Name("operator []")
        public native int get(int i);

        public native @MemberGetter int x();
        public native @MemberGetter int y();
        public native @MemberGetter int z();

        public native @Name("operator=") @ByRef IntegerVec3 put(@ByRef IntegerVec3 rhs);
    }

    @Name("ivec4")
    public class IntegerVec4 extends IntPointer {

        public IntegerVec4() { allocate(); }
        private native void allocate();

        public IntegerVec4(int x, int y, int z, int w) { allocate(x, y, z, w); }
        private native void allocate(int x, int y, int z, int w);

        @Name("operator[]")
        public native int get(int i);

        public native @MemberGetter int x();
        public native @MemberGetter int y();
        public native @MemberGetter int z();
        public native @MemberGetter int w();

        public native @Name("operator=") @ByRef IntegerVec4 put(@ByRef IntegerVec4 rhs);
    }

    @Name("mat4x3")
    public class DoubleMat4x3 extends DoublePointer {

        public DoubleMat4x3() { allocate(); }
        private native void allocate();

        public DoubleMat4x3(double v) { allocate(v); }
        private native void allocate(double v);

        @Name("operator[]") public native @ByRef DoubleVec3 getColumn(int i);

        public native @Name("operator=") @ByRef DoubleMat4x3 put(@ByRef DoubleMat4x3 rhs);
    }
}
