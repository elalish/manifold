/* Copyright (c) 2009, Fredrik Orderud
   License: BSD licence (http://www.opensource.org/licenses/bsd-license.php)
   Based on: http://stupefydeveloper.blogspot.com/2008/10/cc-call-stack.html */

/* Liux/gcc implementation of the call_stack class. */
#ifdef __GNUC__

#include <stdio.h>
#include <execinfo.h>
#include <cxxabi.h>
#include <dlfcn.h>
#include <stdlib.h>
#include "call_stack.hpp"

#define MAX_DEPTH 32

namespace stacktrace {

call_stack::call_stack (const size_t num_discard /*= 0*/) {
    using namespace abi;

    // retrieve call-stack
    void * trace[MAX_DEPTH];
    int stack_depth = backtrace(trace, MAX_DEPTH);

    for (int i = num_discard+1; i < stack_depth; i++) {
        Dl_info dlinfo;
        if(!dladdr(trace[i], &dlinfo))
            break;

        const char * symname = dlinfo.dli_sname;

        int    status;
        char * demangled = abi::__cxa_demangle(symname, NULL, 0, &status);
        if(status == 0 && demangled)
            symname = demangled;

        //printf("entry: %s, %s\n", dlinfo.dli_fname, symname);

        // store entry to stack
        if (dlinfo.dli_fname && symname) {
            entry e;
            e.file     = dlinfo.dli_fname;
            e.line     = 0; // unsupported
            e.function = symname;
            stack.push_back(e);
        } else {
            break; // skip last entries below main
        }

        if (demangled)
            free(demangled);
    }
}

call_stack::~call_stack () throw() {
    // automatic cleanup
}

} // namespace stacktrace

#endif // __GNUC__
