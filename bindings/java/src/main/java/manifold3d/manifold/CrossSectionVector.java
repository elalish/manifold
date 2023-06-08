package manifold3d.manifold;

import org.bytedeco.javacpp.*;
import org.bytedeco.javacpp.annotation.*;

import manifold3d.LibraryPaths;
import java.util.ArrayList;
import manifold3d.manifold.CrossSection;

import java.util.Iterator;
import java.lang.Iterable;
import java.util.NoSuchElementException;

@Platform(compiler = "cpp17", include = {"manifold.h", "<vector>"}, linkpath = { LibraryPaths.MANIFOLD_LIB_DIR }, link = { "manifold" })
@Name("std::vector<manifold::CrossSection>")
public class CrossSectionVector extends Pointer implements Iterable<CrossSection> {
    static { Loader.load(); }

    @Override
    public Iterator<CrossSection> iterator() {
        return new Iterator<CrossSection>() {

            private long index = 0;

            @Override
            public boolean hasNext() {
                return index < size();
            }

            @Override
            public CrossSection next() {
                if (!hasNext()) {
                    throw new NoSuchElementException();
                }
                return get(index++);
            }
        };
    }

    public CrossSectionVector() { allocate(); }
    public native void allocate();

    public CrossSectionVector(@Cast("size_t") long size) { allocate(size); }
    public native void allocate(@Cast("size_t") long size);

    public CrossSectionVector(ArrayList<CrossSection> crossSections) {
        allocate();
        for (CrossSection section: crossSections) {
            this.pushBack(section);
        }
    }
    public CrossSectionVector(CrossSection[] crossSections) {
        allocate();
        for (CrossSection section: crossSections) {
            this.pushBack(section);
        }
    }

    public native @Cast("size_t") long size();
    public native void resize(@Cast("size_t") long n);

    @Name("operator[]") public native @ByRef CrossSection get(@Cast("size_t") long i);
    @Name("push_back") public native void pushBack(@ByRef CrossSection value);
}
