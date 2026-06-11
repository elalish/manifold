# OpenSCAD to Manifold.js Prototype Compiler


## Setup and Running Locally

To get the OpenSCAD compiler running on your local machine, follow these steps:

### 1. Install Dependencies
Ensure you have Node.js installed, then install the package dependencies:
```bash
npm install
```

### 2. Configure Libraries Search Path (Optional)
If you want to use external libraries (such as BOSL2), configure the `OPENSCADPATH` environment variable in your terminal:
- **PowerShell**:
  ```powershell
  $env:OPENSCADPATH = "C:\Users\<you>\Documents\OpenSCAD\libraries"
  ```
- **Bash**:
  ```bash
  export OPENSCADPATH="$HOME/Documents/OpenSCAD/libraries"
  ```

### 3. Build the CLI Tool
Build the compiler script into its final CommonJS bundle:
```bash
npm run build
```

## Special Variables

The compiler supports OpenSCAD special variables:

- **`$fn`, `$fa`, `$fs`**: Control mesh resolution (default: `$fn=0`, `$fa=12`, `$fs=2`)
- **`$vpr`, `$vpt`, `$vpd`**: Viewport settings (default: `[0,0,0]`, `[0,0,0]`, `500`)
- **`$t`**: Animation time variable (default: `0`, range: `0` to `1`)
- **`$preview`**: Preview mode flag (default: `false`)
- **`$parent_modules`**: Parent module count (default: `0`)
- **`$children`**: Number of child elements in a module

To animate compiled models, modify the `$t` variable in the generated TypeScript before calling `result`.

## Available Commands
