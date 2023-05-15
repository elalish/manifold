package manifold3d.pub;

import manifold3d.glm.DoubleVec3Vector;
import manifold3d.glm.IntegerVec3Vector;
import manifold3d.glm.DoubleVec4Vector;

import org.bytedeco.javacpp.*;
import org.bytedeco.javacpp.annotation.*;

@Platform(include = "public.h")
@Namespace("manifold")
@Name("Mesh")
public class DoubleMesh extends Pointer {
    static { Loader.load(); }

    public DoubleMesh() { allocate(); }
    public native void allocate();

    public native @MemberGetter @ByRef DoubleVec3Vector vertPos();
    public native @MemberSetter void vertPos(@ByRef DoubleVec3Vector vertPos);

    public native @MemberGetter @ByRef IntegerVec3Vector triVerts();
    public native @MemberSetter void triVerts(@ByRef IntegerVec3Vector triVerts);

    public native @MemberGetter @ByRef DoubleVec3Vector vertNormal();
    public native @MemberSetter void vertNormal(@ByRef DoubleVec3Vector vertNormal);

    public native @MemberGetter @ByRef DoubleVec4Vector halfedgeTangent();
    public native @MemberSetter void halfedgeTangent(@ByRef DoubleVec4Vector halfedgeTangent);

    public native @MemberGetter float precision();
    public native @MemberSetter void precision(float precision);
}
