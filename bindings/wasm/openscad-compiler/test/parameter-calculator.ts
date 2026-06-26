import { importManifold } from 'manifold-3d/manifoldCAD';
import { exec } from "child_process";
import fs from "fs";
import path from "path";
import { promisify } from 'util';

const execAsync = promisify(exec);


// get the file path passed as argument
const openScadFile = process.argv[2];
if (!openScadFile) {
  console.log("Please provide a file path");
  process.exit(1);
}

const filename = openScadFile as string;
if (filename.endsWith(".scad")) {
  await addPropertiesToFile(filename);
}

async function addPropertiesToFile(filename: string) {
  try {
    await execAsync(`openscad -o output.3mf --backend=manifold ${filename}`);

    const manifold = await importManifold("./output.3mf", {
      mimetype: 'model/3mf',
    });


    const volume = manifold.volume();
    const surfaceArea = manifold.surfaceArea();

    console.log("Volume:", volume);
    console.log("Surface Area:", surfaceArea);

    const propertyString = `// Volume: ${volume}, SurfaceArea: ${surfaceArea}`;

    // open the .scad file and write the propertyString at top
    const content = fs.readFileSync(`${filename}`, "utf-8");

    // check if the first line already contains the propertyString (if yes then replace otherwise insert)
    const firstLine = content.trim().split("\n")[0];
    if (!firstLine) {
      console.log("File is empty", filename);
      return;
    }
    else if (firstLine.startsWith("// Volume:")) {
      const firstNewlineIndex = content.indexOf("\n");
      const rest = firstNewlineIndex === -1 ? "" : content.slice(firstNewlineIndex + 1);
      fs.writeFileSync(`${filename}`, `${propertyString}\n${rest}`);
    } else {
      fs.writeFileSync(`${filename}`, `${propertyString}\n${content}`);
    }

    console.log("Successfully processed", filename);
  } catch (error) {
    console.log(error);
  }
}
