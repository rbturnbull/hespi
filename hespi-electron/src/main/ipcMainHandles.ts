import { app, protocol, net, BrowserWindow, shell, ipcMain, } from 'electron';
import { getPython, installPythonDependencies, runHespi, runTest } from '../common/utils';

export const registerIpcMainHandles = (mainWindow: BrowserWindow) => {
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

  ipcMain.on('ipc-example', async (event, arg) => {
    const msgTemplate = (pingPong: string) => `IPC test: ${pingPong}`;
    console.log(msgTemplate(arg));
    event.reply('ipc-example', msgTemplate('pong'));
  });

  ipcMain.handle('python:install', async () => installPythonDependencies())
  ipcMain.handle('python:hespi', async () => runHespi())
  ipcMain.handle('python:test', async () => runTest())

}
