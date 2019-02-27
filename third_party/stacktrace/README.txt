This is the "stacktrace" project.
Copyright (c) 2009, Ethan Tira-Thompson, Fredrik Orderud.

License: New BSD License (BSD) and LGPL.
         http://www.opensource.org/licenses/bsd-license.php
         http://www.opensource.org/licenses/lgpl-2.1.php

Webpage: http://stacktrace.sourceforge.net/

== Documentation ==
Stacktrace provides a convenient and platform neutral interface for retrieving stack traces from within C++ programs. In addition, it provides a reference implementation for C++ exceptions with stack trace metadata. 

Supported platforms:
Stacktrace support both Microsoft Windows and UNIX-like platforms with a GCC compiler, such as e.g. Linux and Mac OS. Windows support is implemented through the StackWalker project, while the GCC support is implemented through the GLIBC backtrace call.

== Content ==
* existing   : Previous stacktrace code (not yet integrated)
* stacktrace : Main stacktrace code
* test       : Miscellaneous test code
* externals  : External dependencies for platform implementations.
* www        : Webpage content
