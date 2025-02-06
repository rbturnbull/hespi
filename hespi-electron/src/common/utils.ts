import { app } from 'electron';
import fs, { write } from 'fs';
import cp from "child_process";
import util from "util";
import path from 'path';

const USER_DATA_PATH = path.join(app.getPath("userData"), 'pythonData.json');

const dependencies = ["rich", "typer", "torch", "torchapp", "pytesseract", "transformers", "appdirs", "jinja2", "ultralytics", "langchain", "gradio", "drawyolo", "llmloader", "libsass", "orjson", "rcssmin", "pytest", "ipykernel", "coverage", "autopep8", "sphinx", "nbsphinx", "sphinx-rtd-theme", "sphinx-autobuild", "myst-parser", "pre-commit", "sphinx-copybutton", "sphinx-click", "black"]

export function readUserData() {
  try {
    const data = fs.readFileSync(USER_DATA_PATH, 'utf-8');
    return JSON.parse(data);
  } catch (error) {
    console.log('Error retrieving user data', error);
    // you may want to propagate the error, up to you
    return null;
  }
}

export function writeUserData(data) {
  fs.writeFileSync(USER_DATA_PATH, JSON.stringify(data));
}

var userData = readUserData();

export const installPythonDependencies = async (envName='hespi-gui') => {
  const { pythonPath, pythonExecPath, hespiPath, dependenciesInstalled } = getPython();
  const envPath = `${pythonPath}/${envName}`
  if (!dependenciesInstalled || !fs.existsSync(envPath)) {
    const execFile = util.promisify(cp.execFile);
    const envFile = `${envName}-specs.yaml`
    try {
      console.log(`Installing Python dependencies from ${envFile}...`);
      const result = await execFile(`${pythonPath}/micromamba`, ['create', '-f', `${pythonPath}/${envFile}`, '-y', '--prefix', envPath]);
      userData.dependenciesInstalled = true;
      writeUserData(userData);
      console.log(`Python dependencies installed. Environment path: ${envPath}`);
      return result.stdout;
    } catch (error) {
      console.error("Python TEST Error:", error);
    }
  }

}

export const getPython = (version = "3.10", envName = "hespi-env", hespiDir = "hespi") => {
  if (userData){
    try{
      const hasAllKeys = ['pythonPath', 'pythonExecPath', 'hespiPath'].every((key) => userData.hasOwnProperty(key));
      if(hasAllKeys){
        return userData;
      } else {
        console.log("No user python data found")
      }
    } catch (error) {
      console.error("Error reading user python data: ", error);
    }
  }
  const prodBasePath = path.join(process.resourcesPath, "python");
  const devBasePath = path.join(__dirname, "../../", "python");
  const possibilities = [
    path.join(prodBasePath, `python${version}`), // In packaged app
    path.join(devBasePath, `python${version}`), // In development
  ];
  for (const p of [prodBasePath, devBasePath]) {
    var pythonExecPath = path.join(p, envName, "bin", `python${version}`);
    if (fs.existsSync(pythonExecPath)) {
      console.log(`Found python${version} at: ${pythonExecPath}`);
      userData = { pythonPath: p, pythonExecPath: pythonExecPath, hespiPath: path.join(p, hespiDir), dependenciesInstalled: false };
      writeUserData(userData);
      return userData;
    }
  }
  console.log(`Could not find python${version}, checked`, possibilities);
  return null
}

export const runHespi = async () => {
  const execFile = util.promisify(cp.execFile);
  const { pythonPath, pythonExecPath, hespiPath } = getPython();
  try {
    const result = await execFile(pythonExecPath, ["-m", "typer", path.join(hespiPath, "main.py"), "--help"]);
    console.log("Python HESPI Result:", result.stdout);
    return result.stdout;
  } catch (error) {
    console.error("Python HESPI Error:", error);
  }
}

export const runTest = async () => {
  const execFile = util.promisify(cp.execFile);
  const { pythonPath, pythonExecPath, hespiPath } = getPython();
  try {
    const result = await execFile(pythonExecPath, ["-m", "random"]);
    console.log("Python TEST Result:", result.stdout);
    return result.stdout;
  } catch (error) {
    console.error("Python TEST Error:", error);
  }
}