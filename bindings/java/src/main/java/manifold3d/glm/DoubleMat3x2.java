package manifold3d.glm;

import manifold3d.glm.DoubleVec2;

import org.bytedeco.javacpp.*;
import org.bytedeco.javacpp.annotation.*;

import java.util.Iterator;
import java.lang.Iterable;
import java.util.NoSuchElementException;

@Platform(include = "glm/glm.hpp")
@Namespace("glm")
@Name("mat3x2")
public class DoubleMat3x2 extends DoublePointer implements Iterable<DoubleVec2> {
    static { Loader.load(); }

    @Override
    public Iterator<DoubleVec2> iterator() {
        return new Iterator<DoubleVec2>() {

            private int index = 0;

            @Override
            public boolean hasNext() {
                return index < 3;
            }

            @Override
            public DoubleVec2 next() {
                if (!hasNext()) {
                    throw new NoSuchElementException();
                }
                return getColumn(index++);
            }
        };
    }

    public DoubleMat3x2() { allocate(); }
    private native void allocate();

    public DoubleMat3x2(double x) { allocate(x); }
    private native void allocate(double x);

    public DoubleMat3x2(@ByRef DoubleVec2 col1, @ByRef DoubleVec2 col2, @ByRef DoubleVec2 col3) {
        allocate(col1, col2, col3);
    }
    public native void allocate(@ByRef DoubleVec2 col1, @ByRef DoubleVec2 col2, @ByRef DoubleVec2 col3);

    public DoubleMat3x2(double c0, double c1, double c2,
                        double c3, double c4, double c5) {
        allocate(c0, c1, c2, c3, c4, c5);
    }
    public native void allocate(double c1, double c2, double c3, double c4, double c5, double c6);

    @Name("operator[]") public native @ByRef DoubleVec2 getColumn(int i);

    public native @Name("operator=") @ByRef DoubleMat3x2 put(@ByRef DoubleMat3x2 rhs);
}
