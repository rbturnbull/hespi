import { app } from 'electron';
import fs, { write } from 'fs';
import cp from "child_process";
import util from "util";
import path from 'path';
import { CONFIG, updateConfig, APP_DATA_PATH } from './configUtils';

const dependencies = ["rich", "typer", "torch", "torchapp", "pytesseract", "transformers", "appdirs", "jinja2", "ultralytics", "langchain", "gradio", "drawyolo", "llmloader", "libsass", "orjson", "rcssmin", "pytest", "ipykernel", "coverage", "autopep8", "sphinx", "nbsphinx", "sphinx-rtd-theme", "sphinx-autobuild", "myst-parser", "pre-commit", "sphinx-copybutton", "sphinx-click", "black"]
const reqFile = "hespi-gui-req.txt"
const envFile = "hespi-env-specs.yaml"

var isInstalling = false;
const execFile = util.promisify(cp.execFile);
const execCmd = util.promisify(cp.exec);

export const getPython = async (event = null, version = "3.10", envName = "hespi-env", hespiDir = "hespi") => {
  if (!CONFIG.python.exec) {
    const basePaths = [path.join(process.resourcesPath, "python"), path.join(__dirname, "../../", "python")]
    for (const p of basePaths) {
      if (fs.existsSync(p)) {
        CONFIG.python.rootPath = p;
        CONFIG.python.dependenciesInstalled = false
        CONFIG.hespi.srcPath = path.join(p, hespiDir)
        if (CONFIG.python.useBundledPython) {
          const pythonExec = path.join(p, envName, "bin", `python${version}`);
          if (fs.existsSync(pythonExec)) {
            const out = `Found python${version} at: ${pythonExec}`
            console.log(out);
            event?.sender?.send('python:install:update', out, false);
            CONFIG.python.exec = pythonExec;
          }
        } else {
          event?.sender?.send('python:install:update', "Using system python", false);
          CONFIG.python.exec = 'python'; // Use system python
        }
        updateConfig();
        break;
      }
    }
    if (!CONFIG.python.exec) {
      const out = `Could not find python${version}, checked: ${basePaths}`
      console.log(out);
      event?.sender?.send('python:install:update', out, false);
      return null;
    }
  }
  await checkPythonDependencies(envName, event);
  return CONFIG.python;
}


const installAllDependencies = (libsDir, cmdOpts = {}, event = null) => {
  isInstalling = true;
  const args = ['install', '-r', reqFile, '--target', libsDir, '--upgrade'];
  const cmd = cp.spawn('pip', args, cmdOpts);
  console.log(`Installing Python dependencies: '${args.join(' ')}'...`);

  cmd.stdout.on('data', (data) => {
    console.log(`stdout: ${data}`);
  });

  cmd.stderr.on('data', (data) => {
    console.error(`stderr: ${data}`);
  });

  cmd.on('close', (code) => {
    console.log(`child process exited with code ${code}`);
    isInstalling = false;
    if (code === 0) {
      const out = `Python dependencies installed successfully! Libs Path: ${libsDir}`;
      console.log(out);
      CONFIG.python.libsDir = libsDir;
      CONFIG.python.dependenciesInstalled = true;
      event?.sender?.send('python:install:update', out, true);
    } else {
      CONFIG.python.dependenciesInstalled = false;
      const out = `Error installing dependencies into library directory '${libsDir}' python env`
      console.error(out);
      event?.sender?.send('python:install:update', out, false);
    }
    updateConfig();
  });
}

const batchInstallDependencies = (libsDir, cmdOpts = {}, event = null) => {
  isInstalling = true;
  const packages = fs.readFileSync(path.join(CONFIG.python.rootPath, reqFile), 'utf-8').split('\n');
  var installed = 0;
  const controller = new AbortController();
  const { signal } = controller;
  cmdOpts.signal = signal;
  var aborting = false;
  process.on('SIGINT', () => { controller.abort(); aborting = true; });  // CTRL+C
  process.on('SIGQUIT', () => { controller.abort(); aborting = true; }); // Keyboard quit
  process.on('SIGTERM', () => { controller.abort(); aborting = true; }); // `kill` command
  packages.forEach((pkg, idx) => {
    if (aborting) return;
    try {
      const child = cp.execSync(`pip install ${pkg} --target "${libsDir}" --upgrade`, cmdOpts);
      console.log(`${++installed}/${packages.length} ${pkg} Installed!`);
    } catch (error) {
      console.error(`Error installing package: ${pkg}`);
    }

    // child.on('exit', (code) => {
    //   if (code === 0) {
    //     console.log(`${++installed}/${packages.length} ${pkg} Installed!`);
    //   } else {
    //   }
    // });

    // child.stdout?.on('data', (data) => {
    //   console.log(data);
    // })

    // child.stderr?.on('data', (data) => {
    //   console.error(data);
    // })
  });
  isInstalling = false;
}

export const checkPythonDependencies = async (envName = 'hespi-gui', event = null, batchInstall = false) => {
  if (isInstalling) return;
  if (CONFIG.python.useBundledPython) {
    isInstalling = true;
    const envPath = `${CONFIG.python.rootPath}/${envName}`
    if (!fs.existsSync(envPath)) {
      const envFile = `${envName}-specs.yaml`
      try {
        console.log(`Installing Python dependencies from ${envFile}...`);
        const result = await execFile('micromamba', ['create', '-f', envFile, '-y', '--prefix', envName], { cwd: CONFIG.python.rootPath });
        const out = `Python dependencies installed. Environment path: ${envPath}`
        console.log(out, result.stdout);
        CONFIG.python.dependenciesInstalled = true;
        event?.sender?.send('python:install:update', out, true);
      } catch (error) {
        CONFIG.python.dependenciesInstalled = false;
        const out = "Error installing dependencies into bundled python env: " + error
        event?.sender?.send('python:install:update', out, false);
        console.error(out);
      }
    }
    isInstalling = false;
    updateConfig();
  }
  else {
    const libsDir = APP_DATA_PATH + '/hespi-libs';

    if (fs.existsSync(libsDir)) {
      console.log(`Python lib directory '${libsDir}' already exists...`);
      CONFIG.python.libsDir = libsDir;
      CONFIG.python.dependenciesInstalled = true;
      isInstalling = false;
      event?.sender?.send('python:install:update', "Python already installed", true);
    } else {
      const cmdOpts = { cwd: CONFIG.python.rootPath };
      if (batchInstall) {
        batchInstallDependencies(libsDir, cmdOpts, event);
      } else {
        installAllDependencies(libsDir, cmdOpts, event);
      }
    }
  }
}

const execPython = async (args, options = null) => {
  options = options || {};
  if (!options.hasOwnProperty('cwd')) {
    options['cwd'] = CONFIG.python.rootPath;
  }
  if (CONFIG.python.useBundledPython) {
    return execFile(CONFIG.python.exec, args, options);
  } else {
    return execCmd("python " + args.join(' '), options);
  }
}

export const runHespi = async (event, hespiArgs) => {
  if (isInstalling) {
    const msg = "Python dependencies are still installing...";
    console.log(msg);
    return msg;
  }

  console.log("Running HESPI...", hespiArgs);
  var imgPaths = hespiArgs[0].join(" ");
  // imgPaths += TEST_IMAGES.join(" ")
  const cliArgs = ["run_hespi.py"];
  cliArgs.push("-l", "'" + CONFIG.python.libsDir + "'");
  cliArgs.push("-l", "'" + CONFIG.hespi.srcPath + "'");
  cliArgs.push(imgPaths);
  try {
    const result = await execPython(cliArgs);
    console.log("Python HESPI Result:", result.stdout);
    fs.writeFile(`${APP_DATA_PATH}/hespi_output.txt`, result.stdout.toString(), err => {
      if (err) console.error(err);
    });
    return result.stdout.toString();
  } catch (error) {
    console.error("Python HESPI Error:", error);
  }
}

export const runTest = async () => {
  if (isInstalling) {
    const msg = "Python dependencies are still installing...";
    console.log(msg);
    return msg;
  }
  try {
    const result = await execPython(["-m", "random"]);
    console.log("Python TEST Result:", result.stdout);
    fs.writeFile(`${APP_DATA_PATH}/test_output.txt`, result.stdout.toString(), err => {
      if (err) console.error(err);
    });
    return result.stdout.toString();
  } catch (error) {
    console.error("Python TEST Error:", error);
  }
}