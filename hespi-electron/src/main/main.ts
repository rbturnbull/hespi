/* eslint global-require: off, no-console: off, promise/always-return: off */

/**
 * This module executes inside of electron's main process. You can start
 * electron renderer process from here and communicate with the other processes
 * through IPC.
 *
 * When running `npm run build` or `npm run build:main`, this file is compiled to
 * `./src/main.js` using webpack. This gives us some performance wins.
 */
import path from 'path';
import { app, protocol, net, BrowserWindow, shell, ipcMain,  } from 'electron';
import { autoUpdater } from 'electron-updater';
import log from 'electron-log';
import MenuBuilder from './menu';
import { resolveHtmlPath } from './util';
import { findPython, runHespi, runTest } from '../common/utils';


class AppUpdater {
  constructor() {
    log.transports.file.level = 'info';
    autoUpdater.logger = log;
    autoUpdater.checkForUpdatesAndNotify();
  }
}


let mainWindow: BrowserWindow | null = null;

ipcMain.on('ipc-example', async (event, arg) => {
  const msgTemplate = (pingPong: string) => `IPC test: ${pingPong}`;
  console.log(msgTemplate(arg));
  event.reply('ipc-example', msgTemplate('pong'));
});

if (process.env.NODE_ENV === 'production') {
  const sourceMapSupport = require('source-map-support');
  sourceMapSupport.install();
}

const isDebug =
  process.env.NODE_ENV === 'development' || process.env.DEBUG_PROD === 'true';

if (isDebug) {
  require('electron-debug')();
}

const installExtensions = async () => {
  const installer = require('electron-devtools-installer');
  const forceDownload = !!process.env.UPGRADE_EXTENSIONS;
  const extensions = ['REACT_DEVELOPER_TOOLS'];

  return installer
    .default(
      extensions.map((name) => installer[name]),
      forceDownload,
    )
    .catch(console.log);
};

const createWindow = async () => {
  if (isDebug) {
    await installExtensions();
  }

  const RESOURCES_PATH = app.isPackaged
    ? path.join(process.resourcesPath, 'assets')
    : path.join(__dirname, '../../assets');

  const getAssetPath = (...paths: string[]): string => {
    return path.join(RESOURCES_PATH, ...paths);
  };

  mainWindow = new BrowserWindow({
    show: false,
    width: 1024,
    height: 728,
    icon: getAssetPath('icon.png'),
    webPreferences: {
      preload: app.isPackaged
        ? path.join(__dirname, 'preload.js')
        : path.join(__dirname, '../../.erb/dll/preload.js'),
      nodeIntegration: true,
    },
  });


  const navigationHistory = mainWindow.webContents
  ipcMain.handle('nav:back', () =>
    navigationHistory.goBack()
  )

  ipcMain.handle('nav:forward', () => {
    navigationHistory.goForward()
  })

  ipcMain.handle('nav:canGoBack', () => navigationHistory.canGoBack())
  ipcMain.handle('nav:canGoForward', () => navigationHistory.canGoForward())
  ipcMain.handle('nav:loadURL', (_, url) =>
    mainWindow?.webContents.loadURL(url)
  )
  ipcMain.handle('nav:getCurrentURL', () => mainWindow?.webContents.getURL())
  ipcMain.handle('nav:getHistory', () => {
    return navigationHistory?.getAllEntries()
  })


  const pathMap = findPython();
  // const { pythonPath, execPath, hespiPath } = pathMap;

  ipcMain.handle('python:hespi', async () => {
    if(!pathMap) return null;
    const { execPath, hespiPath } = pathMap;
    return runHespi(execPath, hespiPath);    
  })
  ipcMain.handle('python:test', async () => {
    if (!pathMap) return null;
    const { execPath } = pathMap;
    return runTest(execPath);
  })




  mainWindow.webContents.on('did-navigate', () => {
    mainWindow?.webContents.send('nav:updated')
  })

  mainWindow.webContents.on('did-navigate-in-page', () => {
    mainWindow?.webContents.send('nav:updated')
  })



  
  mainWindow.loadURL(resolveHtmlPath('index.html'));

  mainWindow.on('ready-to-show', () => {
    if (!mainWindow) {
      throw new Error('"mainWindow" is not defined');
    }
    if (process.env.START_MINIMIZED) {
      mainWindow.minimize();
    } else {
      mainWindow.show();
    }
  });

  mainWindow.on('closed', () => {
    mainWindow = null;
  });

  const menuBuilder = new MenuBuilder(mainWindow);
  menuBuilder.buildMenu();

  // Open urls in the user's browser
  mainWindow.webContents.setWindowOpenHandler((edata) => {
    shell.openExternal(edata.url);
    return { action: 'deny' };
  });

  // Remove this if your app does not use auto updates
  // eslint-disable-next-line
  new AppUpdater();
};

/**
 * Add event listeners...
 */

app.on('window-all-closed', () => {
  // Respect the OSX convention of having the application in memory even
  // after all windows have been closed
  if (process.platform !== 'darwin') {
    app.quit();
  }
});

protocol.registerSchemesAsPrivileged([
  {
    scheme: 'file-loader',
    privileges: {
      bypassCSP: true,
      // standard: true, // Setting it as standard would not parse absolute paths properly (it seemed to remove the initial '/' and lowercase the first path's letter)
      secure: true,
      supportFetchAPI: true,
    }
  }
]);



app
  .whenReady()
  .then(() => {
    protocol.handle('file-loader', (request) => {
      var fileUrl = 'file://' + request.url.replace('file-loader://', '')
      console.log('Fetching with file-loader: ' + fileUrl);
      return net.fetch(fileUrl)
    })
    createWindow();
    app.on('activate', () => {
      // On macOS it's common to re-create a window in the app when the
      // dock icon is clicked and there are no other windows open.
      if (mainWindow === null) createWindow();
    });
  })
  .catch(console.log);
