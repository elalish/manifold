import re, sys;

with open(sys.argv[1], 'r') as f:
  content = f.read()

content = re.sub(r"Annotated\[ArrayLike, dict\(dtype='([^']*)'.*\)\]", r"np.ndarray[Any, np.dtype[np.\1]]", content)

# Remove ArrayLike and Annotated import
content = re.sub(r"\s*from numpy.typing import ArrayLike\s", r"", content)
content = re.sub(r"from typing import Annotated, overload", r"from typing import overload", content)

with open(sys.argv[1], 'w') as f:
  f.write(content)