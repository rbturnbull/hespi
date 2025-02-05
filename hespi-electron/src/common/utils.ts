
import fs from 'fs';
import cp from "child_process";
import util from "util";
import path from 'path';


export const findPython = (version = "3.10", envName = "hespi-env", hespiDir = "hespi") => {
  const prodBasePath = path.join(process.resourcesPath, "python");
  const devBasePath = path.join(__dirname, "../../", "python");
  const possibilities = [
    path.join(prodBasePath, `python${version}`), // In packaged app
    path.join(devBasePath, `python${version}`), // In development
  ];
  for (const p of [prodBasePath, devBasePath]) {
    var execPath = path.join(p, envName, "bin", `python${version}`);
    if (fs.existsSync(execPath)) {
      console.log(`Found python${version} at: ${execPath}`);
      return { pythonPath: p, execPath: execPath, hespiPath: path.join(p, hespiDir) };
    }
  }
  console.log(`Could not find python${version}, checked`, possibilities);
  return null
}

export const runHespi = async (pythonExecPath, hespiPath, pythonDirPath = null) => {
  const execFile = util.promisify(cp.execFile);
  try {
    const result = await execFile(pythonExecPath, ["-m", "typer", path.join(hespiPath, "main.py"), "--help"]);
    console.log("Python HESPI Result:", result.stdout);
    return result.stdout;
  } catch (error) {
    console.error("Python HESPI Error:", error);
  }
}

export const runTest = async (pythonExecPath) => {
  const execFile = util.promisify(cp.execFile);
  try {
    const result = await execFile(pythonExecPath, ["-m", "random"]);
    console.log("Python TEST Result:", result.stdout);
    return result.stdout;
  } catch (error) {
    console.error("Python TEST Error:", error);
  }
}