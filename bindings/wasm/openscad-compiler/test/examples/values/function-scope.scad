function duplicate_let() = let(x=42) let(x=33) x;
echo(duplicate_let=duplicate_let());

function defaults(b, x=42) = b ? x : defaults(true);
echo(defaults=defaults(false, 33));

function scope_leak_config(b=false) = b ? $x : let($x=33) let($x=42) scope_leak_config(true);
echo(scope_leak_config=scope_leak_config());