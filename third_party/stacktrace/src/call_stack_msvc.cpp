/* Copyright (c) 2009, Fredrik Orderud
   License: BSD licence (http://www.opensource.org/licenses/bsd-license.php) */

/* Windows (Microsoft visual studio) implementation of the call_stack class. */
#ifdef _WIN32 // also defined in 64bit

#include "call_stack.hpp"
#include "../externals/StackWalker/StackWalker.h"


/** Adapter class to interfaces with the StackWalker project.
 *  Source: http://stackwalker.codeplex.com/ */
class StackWalkerAdapter : public StackWalker {
public:
    StackWalkerAdapter (const size_t num_discard) : 
        StackWalker(StackWalker::RetrieveVerbose | StackWalker::SymBuildPath), // do not use public Microsoft-Symbol-Server
        discard_idx(static_cast<int>(num_discard)+2) {
    }
    virtual ~StackWalkerAdapter () {}

protected:
    virtual void OnCallstackEntry (CallstackEntryType /*eType*/, CallstackEntry &entry) {
        if (entry.lineFileName[0] == 0)
            discard_idx = -1; // skip all entries from now on

        // discard first N stack entries
        if (discard_idx > 0) {
            discard_idx--;
        } else if (discard_idx == 0) {
            stacktrace::entry e;
            e.file = entry.lineFileName;
            e.line = entry.lineNumber;
            e.function = entry.name;
            stack.push_back(e);
        }
    }
    virtual void OnOutput (LPCSTR /*szText*/) {
        // discard (should never be called)
    }
    virtual void OnSymInit (LPCSTR /*szSearchPath*/, DWORD /*symOptions*/, LPCSTR /*szUserName*/) {
        // discard
    }
    virtual void OnLoadModule (LPCSTR /*img*/, LPCSTR /*mod*/, DWORD64 /*baseAddr*/, DWORD /*size*/, DWORD /*result*/, LPCSTR /*symType*/, LPCSTR /*pdbName*/, ULONGLONG /*fileVersion*/) {
        // discard
    }
    virtual void OnDbgHelpErr (LPCSTR /*szFuncName*/, DWORD /*gle*/, DWORD64 /*addr*/) {
        // discard
    }

public:
    std::vector<stacktrace::entry> stack;       ///< populated stack trace
private:
    int                            discard_idx; ///< the number of stack entries to discard
};


namespace stacktrace {

// windows 32 & 64 bit impl.
call_stack::call_stack (const size_t num_discard /*= 0*/) {
    StackWalkerAdapter sw(num_discard);
    sw.ShowCallstack();
    stack = sw.stack;
}

call_stack::~call_stack () throw() {
    // automatic cleanup
}

} // namespace stacktrace

#endif // _WIN32
