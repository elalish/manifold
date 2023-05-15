package manifold3d;

import manifold3d.glm.FloatVec3Vector;
import manifold3d.glm.FloatVec4Vector;
import manifold3d.glm.DoubleVec3Vector;
import manifold3d.glm.IntegerVec3Vector;
import manifold3d.glm.FloatVec4;
import manifold3d.glm.DoubleVec3;
import manifold3d.glm.IntegerVec3;
import manifold3d.glm.IntegerVec4;
import manifold3d.glm.DoubleMat4x3;

import org.bytedeco.javacpp.*;
import org.bytedeco.javacpp.annotation.*;

@Platform(include = "public.h")
@Namespace("manifold")
public class Public {
    static { Loader.load(); }

    //@Name("Mesh")
    //public static class DoubleMesh extends Pointer {
    //    public DoubleMesh() { allocate(); }
    //    public native void allocate();

    //    public native @MemberGetter @ByRef DoubleVec3Vector vertPos();
    //    public native @MemberSetter void vertPos(@ByRef DoubleVec3Vector vertPos);

    //    public native @MemberGetter @ByRef IntegerVec3Vector triVerts();
    //    public native @MemberSetter void triVerts(@ByRef IntegerVec3Vector triVerts);

    //    public native @MemberGetter @ByRef DoubleVec3Vector vertNormal();
    //    public native @MemberSetter void vertNormal(@ByRef DoubleVec3Vector vertNormal);

    //    public native @MemberGetter @ByRef DoubleVec4Vector halfedgeTangent();
    //    public native @MemberSetter void halfedgeTangent(@ByRef DoubleVec4Vector halfedgeTangent);

    //    public native @MemberGetter float precision();
    //    public native @MemberSetter void precision(float precision);
    //}

    @Name("Mesh")
    public class FloatMesh extends Pointer {
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

    public class Smoothness extends Pointer {
        public Smoothness() { allocate(); }
        public native void allocate();

        public native @MemberGetter int halfedge();
        public native @MemberSetter void halfedge(float halfedge);

        public native @MemberGetter float smoothness();
        public native @MemberSetter void smoothness(float smoothness);
    }

    public class Properties extends Pointer {
        public Properties() { allocate(); }
        public native void allocate();

        public native @MemberGetter float surfaceArea();
        public native @MemberSetter void surfaceArea(float surfaceArea);

        public native @MemberGetter float volume();
        public native @MemberSetter void volume(float volume);
    }


    public class Curvature extends Pointer {
        public Curvature() { allocate(); }
        public native void allocate();

        public native @MemberGetter float maxMeanCurvature();
        public native @MemberSetter void maxMeanCurvature(float maxMeanCurvature);

        public native @MemberGetter float minMeanCurvature();
        public native @MemberSetter void minMeanCurvature(float minMeanCurvature);

        public native @MemberGetter float maxGaussianCurvature();
        public native @MemberSetter void maxGaussianCurvature(float maxGaussianCurvature);

        public native @MemberGetter float minGaussianCurvature();
        public native @MemberSetter void minGaussianCurvature(float minGaussianCurvature);

    }

    public class Box extends Pointer {
        public Box() { allocate(); }
        private native void allocate();

        public Box(@ByRef DoubleVec3 p1, @ByRef DoubleVec3 p2) { allocate(p1, p2); }
        private native void allocate(@ByRef DoubleVec3 p1, @ByRef DoubleVec3 p2);

        public native @ByVal DoubleVec3 Size();
        public native @ByVal DoubleVec3 Center();
        public native float Scale();
        public native boolean Contains(@ByRef DoubleVec3 p);

        public native boolean Contains(@ByRef Box box);
        public native void Union(@ByRef DoubleVec3 p);
        public native @ByVal Box Union(@ByRef Box box);
        public native @ByVal Box Transform(@ByRef DoubleMat4x3 transform);

        @Name("operator+")
        public native @ByVal Box add(@ByRef DoubleVec3 shift);
        @Name("operator+=")
        public native @ByRef Box addPut(@ByRef DoubleVec3 shift);

        @Name("operator*")
        public native @ByVal Box multiply(@ByRef DoubleVec3 scale);
        @Name("operator*=")
        public native @ByRef Box multiplyPut(@ByRef DoubleVec3 scale);

        public native boolean DoesOverlap(@ByRef Box box);
        public native boolean DoesOverlap(@ByRef DoubleVec3 p);
        public native boolean IsFinite();
    }

    public class OpType {
        public static final int
            Add = 0,
            Subtract = 1,
            Intersect = 2;
    }

    public class ExecutionParams extends Pointer {

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
}
