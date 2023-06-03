package manifold3d.pub;

import manifold3d.pub.SimplePolygon;
import org.bytedeco.javacpp.*;
import org.bytedeco.javacpp.annotation.*;

import java.util.Iterator;
import java.lang.Iterable;
import java.util.NoSuchElementException;

@Platform(include = "manifold.h")
@Namespace("manifold")
@Name("Polygons")
public class Polygons extends Pointer implements Iterable<SimplePolygon> {
    static { Loader.load(); }

    @Override
    public Iterator<SimplePolygon> iterator() {
        return new Iterator<SimplePolygon>() {

            private long index = 0;

            @Override
            public boolean hasNext() {
                return index < size();
            }

            @Override
            public SimplePolygon next() {
                if (!hasNext()) {
                    throw new NoSuchElementException();
                }
                return get(index++);
            }
        };
    }

    public Polygons() { allocate(); }
    private native void allocate();

    public native @Cast("size_t") long size();
    public native void resize(@Cast("size_t") long n);

    @Name("operator[]") public native @ByRef SimplePolygon get(@Cast("size_t") long i);
    @Name("push_back") public native void pushBack(@ByRef SimplePolygon value);
}
