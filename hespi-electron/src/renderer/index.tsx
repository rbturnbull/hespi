import { createRoot } from 'react-dom/client';
import App from './App';
import "assets/css/bootstrap.hespi.min.css"
import "assets/css/hespi.min.css"
import "assets/css/hespi-react.css"
import "assets/js/bootstrap.min.js"

const rootEl = document.getElementById('hespi-root') as HTMLElement;
const root = createRoot(rootEl);
root.render(<App />);


// calling IPC exposed from preload script
window.electron.ipcRenderer.once('ipc-example', (arg) => {
  // eslint-disable-next-line no-console
  console.log(arg);
});
// window.electron.ipcRenderer.sendMessage('ipc-example', ['ping']);
