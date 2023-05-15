package manifold3d.pub;

import manifold3d.glm.FloatVec3Vector;
import manifold3d.glm.IntegerVec3Vector;
import manifold3d.glm.FloatVec4Vector;

import org.bytedeco.javacpp.*;
import org.bytedeco.javacpp.annotation.*;

@Platform(include = "public.h")
@Namespace("manifold")
@Name("Mesh")
public class FloatMesh extends Pointer {
    static { Loader.load(); }

    public FloatMesh() { allocate(); }
    public native void allocate();

    public native @MemberGetter @ByRef FloatVec3Vector vertPos();
    public native @MemberSetter void vertPos(@ByRef FloatVec3Vector vertPos);

    public native @MemberGetter @ByRef IntegerVec3Vector triVerts();
    public native @MemberSetter void triVerts(@ByRef IntegerVec3Vector triVerts);

    public native @MemberGetter @ByRef FloatVec3Vector vertNormal();
    public native @MemberSetter void vertNormal(@ByRef FloatVec3Vector vertNormal);

    public native @MemberGetter @ByRef FloatVec4Vector halfedgeTangent();
    public native @MemberSetter void halfedgeTangent(@ByRef FloatVec4Vector halfedgeTangent);

    public native @MemberGetter float precision();
    public native @MemberSetter void precision(float precision);
}
