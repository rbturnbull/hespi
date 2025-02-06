// Disable no-unused-vars, broken for spread args
/* eslint no-unused-vars: off */
import { contextBridge, ipcRenderer, IpcRendererEvent } from 'electron';
import { installPythonDependencies } from '../common/utils';

export type Channels = 'ipc-example';

const electronHandler = {
  ipcRenderer: {
    sendMessage(channel: Channels, ...args: unknown[]) {
      ipcRenderer.send(channel, ...args);
    },
    on(channel: Channels, func: (...args: unknown[]) => void) {
      const subscription = (_event: IpcRendererEvent, ...args: unknown[]) =>
        func(...args);
      ipcRenderer.on(channel, subscription);

      return () => {
        ipcRenderer.removeListener(channel, subscription);
      };
    },
    once(channel: Channels, func: (...args: unknown[]) => void) {
      ipcRenderer.once(channel, (_event, ...args) => func(...args));
    },
  },
};

contextBridge.exposeInMainWorld('electron', electronHandler);

contextBridge.exposeInMainWorld('electronAPI', {
  runHespi: (imagesList, llmTemperature) => ipcRenderer.invoke('python:hespi', imagesList, llmTemperature),
  runTest: () => ipcRenderer.invoke('python:test'),
  installPython: () => ipcRenderer.invoke('python:install'),
  goBack: () => ipcRenderer.invoke('nav:back'),
  goForward: () => ipcRenderer.invoke('nav:forward'),
  canGoBack: () => ipcRenderer.invoke('nav:canGoBack'),
  canGoForward: () => ipcRenderer.invoke('nav:canGoForward'),
  loadURL: (url) => ipcRenderer.invoke('nav:loadURL', url),
  getCurrentURL: () => ipcRenderer.invoke('nav:getCurrentURL'),
  getHistory: () => ipcRenderer.invoke('nav:getHistory'),
  onNavigationUpdate: (callback) => ipcRenderer.on('nav:updated', callback)
})

export type ElectronHandler = typeof electronHandler;
