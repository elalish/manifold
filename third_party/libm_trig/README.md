Deterministic Trig Kernel Sources

This project uses reduced-range sine/cosine kernel polynomials adapted from
musl libc:

- `__sin.c`: https://git.musl-libc.org/cgit/musl/plain/src/math/__sin.c
- `__cos.c`: https://git.musl-libc.org/cgit/musl/plain/src/math/__cos.c

These files note their origin as FreeBSD msun (`k_sin.c`, `k_cos.c`) with the
Sun Microsystems permissive notice.

Licensing

- musl libc project license: MIT
  - https://git.musl-libc.org/cgit/musl/tree/COPYRIGHT
- Kernel file origin notice (Sun Microsystems): 
Copyright (C) 1993 by Sun Microsystems, Inc. All rights reserved.

Developed at SunPro/SunSoft, a Sun Microsystems, Inc. business.
Permission to use, copy, modify, and distribute this software is freely
granted, provided that this notice is preserved.

