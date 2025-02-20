import { useEffect, useState } from 'react';
import Button from 'react-bootstrap/Button';
import Modal from 'react-bootstrap/Modal';

const TEST_IMAGES = ["/Users/gabrielem/GitHub/hespi-gui/tests/testdata/test-558d5d8ab1da205b9cfc9754513a9882.jpg"]
export default function PythonInstallerModal(props) {
  const [pythonInstalled, setPythonInstalled] = useState(false);

  const testPython = () => {
    console.log(`Testing Python (Renderer)`)
    window.electronAPI.runTest().then((stdout) => {
      console.log(`Python Test (Renderer) output`, stdout)
    }).catch((err) => {
      console.error(`Testing Python (Renderer): ${err}`)
    });
  }

  const testHespi = () => {
    console.log(`Running HESPI (Renderer)...`)
    window.electronAPI.runHespi(TEST_IMAGES).then((stdout) => {
      console.log(`HESPI (Renderer) output`, stdout)
    }).catch((err) => {
      console.error(`Testing HESPI (Renderer): ${err}`)
    });
  }

  useEffect(() => {
    console.log(`Calling python install (Renderer)`)
    window.electronAPI.installPython().then((res) => {
      console.log(`Python install output: `, res)
    }).catch((err) => {
      console.error(`Error installing python (Renderer): ${err}`)
    });
    window.electron.ipcRenderer.on("python:install:update", function (res, isInstalled) {
      console.log(`Python install update (Renderer): `, res);
      setPythonInstalled(isInstalled);
    })
  }, []);

  return (
    <Modal
      {...props}
      size="lg"
      aria-labelledby="contained-modal-title-vcenter"
      centered
    >
      <Modal.Header closeButton>
        <Modal.Title id="contained-modal-title-vcenter">
          Modal heading
        </Modal.Title>
      </Modal.Header>
      <Modal.Body>
        <h4>Centered Modal</h4>
        <p>
          Cras mattis consectetur purus sit amet fermentum. Cras justo odio,
          dapibus ac facilisis in, egestas eget quam. Morbi leo risus, porta ac
          consectetur ac, vestibulum at eros.
        </p>
      </Modal.Body>
      <Modal.Footer>
        <Button onClick={testPython} disabled={!pythonInstalled}>Test Python</Button>
        <Button onClick={testHespi} disabled={!pythonInstalled}>Test Hespi</Button>
        <Button onClick={props.onHide}>Close</Button>
      </Modal.Footer>
    </Modal>
  );
}