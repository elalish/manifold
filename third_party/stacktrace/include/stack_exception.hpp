/* Copyright (c) 2009, Fredrik Orderud
   License: BSD licence (http://www.opensource.org/licenses/bsd-license.php) */

#pragma once
#include <stdexcept>
#include <string>
#include "call_stack.hpp"

namespace stacktrace {

/** Abstract base-class for all stack-augmented exception classes.
 *  Enables catching of all stack-augmented exception classes. */
class stack_exception_base : public call_stack {
 public:
  stack_exception_base(const bool _show_stack)
      : call_stack(2), show_stack(_show_stack) {}
  virtual ~stack_exception_base() throw() {}

  virtual const char* what() const throw() = 0;

  /// flag to indicate if stack-trace is included in what() messages
  bool show_stack;
};

/** Template for stack-augmented exception classes. */
template <class T>
class stack_exception : public T, public stack_exception_base {
 public:
  stack_exception(const std::string& msg)
      : T(msg), stack_exception_base(true) {}
  virtual ~stack_exception() throw() {}

  virtual const char* what() const throw() {
    if (show_stack) {
      // concatenate message with stack trace
      buffer =
          "[" + std::string(T::what()) + "]\n" + stack_exception::to_string();
      return buffer.c_str();
    } else {
      return T::what();
    }
  }

 private:
  mutable std::string buffer;
};

/** Stack-augmented exception classes for all std::exception classes. */
typedef stack_exception<std::runtime_error> stack_runtime_error;
typedef stack_exception<std::range_error> stack_range_error;
typedef stack_exception<std::overflow_error> stack_overflow_error;
typedef stack_exception<std::underflow_error> stack_underflow_error;
typedef stack_exception<std::logic_error> stack_logic_error;
typedef stack_exception<std::domain_error> stack_domain_error;
typedef stack_exception<std::invalid_argument> stack_invalid_argument;
typedef stack_exception<std::length_error> stack_length_error;
typedef stack_exception<std::out_of_range> stack_out_of_range;

}  // namespace stacktrace
