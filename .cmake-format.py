# -----------------------------
# Options effecting formatting.
# -----------------------------
with section("format"):

    # How wide to allow formatted cmake files
    line_width = 80

    # How many spaces to tab for indent
    tab_size = 2

    # If true, separate flow control names from their parentheses with a space
    separate_ctrl_name_with_space = False

    # If true, separate function names from parentheses with a space
    separate_fn_name_with_space = False

    # If a statement is wrapped to more than one line, than dangle the closing
    # parenthesis on its own line.
    dangle_parens = False

with section("markup"):
    enable_markup = False

# parse kwargs for nanobind_add_stub, to not interpret keywords as args
with section("parse"):
    additional_commands = {
        "nanobind_add_stub": {
            "kwargs": {
                "MODULE": "*",
                "OUTPUT": "*",
                "PYTHON_PATH": "*",
                "DEPENDS": "*",
                "PATTERN_FILE": "*",
            }
        }
    }
