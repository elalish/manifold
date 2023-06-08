package manifold3d.glm;

import manifold3d.glm.DoubleVec3;
import manifold3d.glm.DoubleVec4;
import manifold3d.glm.MatrixTransforms;

import org.bytedeco.javacpp.*;
import org.bytedeco.javacpp.annotation.*;

import java.util.Iterator;
import java.lang.Iterable;
import java.util.NoSuchElementException;

@Platform(compiler = "cpp17", include = {"glm/glm.hpp"})
@Namespace("glm")
@Name("mat4x3")
public class DoubleMat4x3 extends DoublePointer implements Iterable<DoubleVec3> {
    static { Loader.load(); }

    @Override
    public Iterator<DoubleVec3> iterator() {
        return new Iterator<DoubleVec3>() {

            private int index = 0;

            @Override
            public boolean hasNext() {
                return index < 3;
            }

            @Override
            public DoubleVec3 next() {
                if (!hasNext()) {
                    throw new NoSuchElementException();
                }
                return getColumn(index++);
            }
        };
    }

    public DoubleMat4x3() { allocate(); }
    private native void allocate();

    private native void allocate(double x);
    public DoubleMat4x3(double x) { allocate(x); }

    public DoubleMat4x3(@ByRef DoubleVec3 col1, @ByRef DoubleVec3 col2,
                        @ByRef DoubleVec3 col3, @ByRef DoubleVec3 col4) {
        allocate(col1, col2, col3, col4);
    }
    public native void allocate(@ByRef DoubleVec3 col1, @ByRef DoubleVec3 col2,
                                @ByRef DoubleVec3 col3, @ByRef DoubleVec3 col4);

    public DoubleMat4x3(double c0, double c1, double c2, double c3,
                        double c4, double c5, double c6, double c7,
                        double c8, double c9, double c10, double c11) {
        allocate(c0, c1, c2, c3, c4, c5, c6, c7, c8, c9, c10, c11);
    }
    public native void allocate(double c0, double c1, double c2, double c3,
                                double c4, double c5, double c6, double c7,
                                double c8, double c9, double c10, double c11);

    @Name("operator[]") public native @ByRef DoubleVec3 getColumn(int i);

    public native @Name("operator=") @ByRef DoubleMat4x3 put(@ByRef DoubleMat4x3 rhs);

    public DoubleMat4x3 transform(@ByRef DoubleMat4x3 other) {
        return MatrixTransforms.Transform(this, other);
    }

    public DoubleMat4x3 rotate(@ByRef DoubleVec3 angles) {
        return MatrixTransforms.Rotate(this, angles);
    }

    public DoubleMat4x3 rotate(@ByRef DoubleVec3 axis, float angle) {
        return MatrixTransforms.Rotate(this, axis, angle);
    }

    public DoubleMat4x3 translate(@ByRef DoubleVec3 vec) {
        return MatrixTransforms.Translate(this, vec);
    }
}
