package manifold3d;

import manifold3d.FloatVec4;
import manifold3d.FloatVec4Vector;
import manifold3d.IntegerVec3;
import manifold3d.IntegerVec4;

import org.bytedeco.javacpp.*;
import org.bytedeco.javacpp.annotation.*;

@Platform(include = "meshIO.h")
@Namespace("manifold")
public class Manifold {
    static { Loader.load(); }

    public class Material extends Pointer {
        public Material() { allocate(); }
        public native void allocate();

        public native @MemberGetter float roughness();
        public native @MemberSetter void roughness(float roughness);

        public native @MemberGetter float metalness();
        public native @MemberSetter void metalness(float metalness);

        public native @MemberGetter @ByRef FloatVec4 color();
        public native @MemberSetter void color(@ByRef FloatVec4 color);

        public native @MemberGetter @ByRef FloatVec4Vector vertColor();
        public native @MemberSetter void vertColor(@ByRef FloatVec4Vector vertColor);

        public native @MemberGetter @ByRef IntegerVec3 normalChannels();
        public native @MemberSetter void normalChannels(@ByRef IntegerVec3 color);

        public native @MemberGetter @ByRef IntegerVec4 colorChannels();
        public native @MemberSetter void colorChannels(@ByRef IntegerVec4 color);
    }

    public class ExportOptions extends Pointer {
        public ExportOptions() { allocate(); }
        public native void allocate();

        public native @MemberGetter boolean faceted();
        public native @MemberSetter void faceted(boolean faceted);
    }
}
