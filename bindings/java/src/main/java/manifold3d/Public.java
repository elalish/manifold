package manifold3d;

import manifold3d.FloatVec4;
import manifold3d.FloatVec3Vector;
import manifold3d.FloatVec4Vector;
import manifold3d.DoubleVec3Vector;
import manifold3d.IntegerVec3Vector;
import manifold3d.FloatVec4;
import manifold3d.IntegerVec3;
import manifold3d.IntegerVec4;

import org.bytedeco.javacpp.*;
import org.bytedeco.javacpp.annotation.*;

@Platform(include = "public.h")
@Namespace("manifold")
public class Public {
    static { Loader.load(); }

    public class Mesh extends Pointer {
        public Mesh() { allocate(); }
        public native void allocate();

        @Name("vertPos") public native @MemberGetter @ByRef FloatVec3Vector vertPosFloat();
        @Name("vertPos") public native @MemberSetter void vertPosFloat(@ByRef FloatVec3Vector vertPosDouble);
        @Name("vertPos") public native @MemberGetter @ByRef DoubleVec3Vector vertPosDouble();
        @Name("vertPos") public native @MemberSetter void vertPosDouble(@ByRef DoubleVec3Vector vertPosDouble);

        public native @MemberGetter @ByRef IntegerVec3Vector triVerts();
        public native @MemberSetter void triVerts(@ByRef IntegerVec3Vector triVerts);

        @Name("vertNormal") public native @MemberGetter @ByRef FloatVec3Vector vertNormalFloat();
        @Name("vertNormal") public native @MemberSetter void vertNormalFloat(@ByRef FloatVec3Vector vertNormalFloat);
        @Name("vertNormal") public native @MemberGetter @ByRef DoubleVec3Vector vertNormalDouble();
        @Name("vertNormal") public native @MemberSetter void vertNormalFloat(@ByRef DoubleVec3Vector vertNormalDouble);

        @Name("halfedgeTangent") public native @MemberGetter @ByRef FloatVec4Vector halfedgeTangentFloat();
        @Name("halfedgeTangent") public native @MemberSetter void halfedgeTangentFloat(@ByRef FloatVec4Vector halfedgeTangentFloat);
        @Name("halfedgeTangent") public native @MemberGetter @ByRef DoubleVec4Vector halfedgeTangentDouble();
        @Name("halfedgeTangent") public native @MemberSetter void halfedgeTangentDouble(@ByRef DoubleVec4Vector halfedgeTangentDouble);

        public native @MemberGetter float precision();
        public native @MemberSetter void precision(float precision);

    }
}
