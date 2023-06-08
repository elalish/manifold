package manifold3d;

import org.bytedeco.javacpp.*;
import org.bytedeco.javacpp.annotation.*;

import manifold3d.LibraryPaths;
import java.util.ArrayList;
import manifold3d.Manifold;

import java.util.Iterator;
import java.lang.Iterable;
import java.util.NoSuchElementException;

@Platform(compiler = "cpp17", include = {"manifold.h", "<vector>"}, linkpath = { LibraryPaths.MANIFOLD_LIB_DIR }, link = { "manifold" })
@Name("std::vector<manifold::Manifold>")
public class ManifoldVector extends Pointer implements Iterable<Manifold>  {
    static { Loader.load(); }

    private int current = 0;

    public ManifoldVector() { allocate(); }
    public native void allocate();

    @Override
    public Iterator<Manifold> iterator() {
        return new Iterator<Manifold>() {

            private long index = 0;

            @Override
            public boolean hasNext() {
                return index < size();
            }

            @Override
            public Manifold next() {
                if (!hasNext()) {
                    throw new NoSuchElementException();
                }
                return get(index++);
            }
        };
    }

    public ManifoldVector(@Cast("size_t") long size) { allocate(size); }
    public native void allocate(@Cast("size_t") long size);

    public ManifoldVector(ArrayList<Manifold> manifolds) {
        allocate();
        for (Manifold manifold: manifolds) {
            this.pushBack(manifold);
        }
    }
    public ManifoldVector(Manifold[] manifolds) {
        allocate();
        for (Manifold manifold: manifolds) {
            this.pushBack(manifold);
        }
    }

    public native @Cast("size_t") long size();
    public native void resize(@Cast("size_t") long n);

    @Name("operator[]") public native @ByRef Manifold get(@Cast("size_t") long i);
    @Name("push_back") public native void pushBack(@ByRef Manifold value);
}
