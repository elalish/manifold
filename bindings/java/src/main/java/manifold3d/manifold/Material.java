package manifold3d.manifold;

import org.bytedeco.javacpp.*;
import org.bytedeco.javacpp.annotation.*;

import manifold3d.glm.DoubleVec4;
import manifold3d.glm.DoubleVec4Vector;
import manifold3d.glm.IntegerVec3;
import manifold3d.glm.IntegerVec4;

@Platform(include = "meshIO.h")
@Namespace("manifold")
public class Material extends Pointer {
    static { Loader.load(); }

    public Material() { allocate(); }
    public native void allocate();

    public native @MemberGetter float roughness();
    public native @MemberSetter void roughness(float roughness);

    public native @MemberGetter float metalness();
    public native @MemberSetter void metalness(float metalness);

    public native @MemberGetter @ByRef DoubleVec4 color();
    public native @MemberSetter void color(@ByRef DoubleVec4 color);

    public native @MemberGetter @ByRef DoubleVec4Vector vertColor();
    public native @MemberSetter void vertColor(@ByRef DoubleVec4Vector vertColor);

    public native @MemberGetter @ByRef IntegerVec3 normalChannels();
    public native @MemberSetter void normalChannels(@ByRef IntegerVec3 color);

    public native @MemberGetter @ByRef IntegerVec4 colorChannels();
    public native @MemberSetter void colorChannels(@ByRef IntegerVec4 color);
}
