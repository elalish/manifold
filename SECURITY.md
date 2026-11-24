# Security Policy

Manifold is a geometry library. It doesn't run as a service, handle
credentials, or cross a trust boundary on its own, so we treat memory-safety
bugs such as out-of-bounds reads or crashes on malformed input like any other
bug. Manifold isn't hardened against hostile input, so we don't recommend
running it on untrusted meshes or files in a security-sensitive setting, such
as a server request path, without isolating it.

## Reporting a bug

Please open a regular [issue](https://github.com/elalish/manifold/issues) with
a reproducer. That includes crashes, sanitizer findings and bugs found by
fuzzing or automated analysis.

If you can show how an existing application that uses Manifold could be
exploited through one of these bugs, please say so in an issue, but leave out
the reproducer and exploit details, and we'll set up a private channel.
