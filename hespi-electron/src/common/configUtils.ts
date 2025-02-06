import { app } from 'electron';
import fs, { write } from 'fs';
import path from 'path';


export const USER_DATA_PATH = app.getPath("userData");
export const APP_DATA_PATH = path.join(app.getPath("appData"), "hespi-cache");
if (!fs.existsSync(APP_DATA_PATH)) {
  fs.mkdirSync(APP_DATA_PATH);
}
const CONFIG_FILE = path.join(USER_DATA_PATH, 'pythonData.json');
const DEFAULT_CONFIG = {
  python: {
    useBundledPython: false,
    dependencies: ["rich", "typer", "torch", "torchapp", "pytesseract", "transformers", "appdirs", "jinja2", "ultralytics", "langchain", "gradio", "drawyolo", "llmloader", "libsass", "orjson", "rcssmin", "pytest", "ipykernel", "coverage", "autopep8", "sphinx", "nbsphinx", "sphinx-rtd-theme", "sphinx-autobuild", "myst-parser", "pre-commit", "sphinx-copybutton", "sphinx-click", "black"],
    dependenciesInstalled: false,
    rootPath: null,
    exec: null,
  },
  hespi: {
    srcPath: null  
  }
}

const readConfig = () => {
  try {
    const data = fs.readFileSync(CONFIG_FILE, 'utf-8');
    return JSON.parse(data);
  } catch (error) {
    console.log('Error retrieving user data', error);
    return DEFAULT_CONFIG;
  }
}

export const CONFIG = readConfig();
console.log(`Loaded config from ${CONFIG_FILE}`, CONFIG);

export const updateConfig = () => {
  fs.writeFileSync(CONFIG_FILE, JSON.stringify(CONFIG));
}

