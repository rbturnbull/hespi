import { MemoryRouter as Router, Routes, Route } from 'react-router-dom';
import React, {useState, useEffect} from 'react';
import './App.css';
import Hello from './Hello';
import HespiReport from './HespiReport';
import HespiGUI from './HespiGUI';
import { ArrowLeft, ArrowRight } from 'react-bootstrap-icons';
import PythonInstallerModal from './PythonInstallerModal';


export default function App() {
  const [modalShow, setModalShow] = useState(true);


  const updateButtons = async () => {
    const canGoBack = await window.electronAPI.canGoBack()
    const canGoForward = await window.electronAPI.canGoForward()
    console.log(`canGoBack: ${canGoBack}\tcanGoForward: ${canGoForward}`)
  }
  useEffect(() => {
    // window.electronAPI.runTest()
    window.electronAPI.onNavigationUpdate(() => {
      updateButtons()
      // updateURL()
    })  
    updateButtons()
  }, [])

  return (
    <Router>
      <div>
          <PythonInstallerModal
            show={modalShow}
            onHide={() => setModalShow(false)}
          />
        
        <div className='navigation-container'>
          <div className='navigation-buttons'>
            <div className='navigation-button back'><ArrowLeft /></div>
            <div className='navigation-button forward' style={{display: 'none'}}><ArrowRight /></div>
          </div>
        </div>
        <Routes>
          {/* <Route path="/Hello" element={<Hello />} /> */}
          <Route path="/report" element={<HespiReport />} />
          <Route path="/" element={<HespiGUI />} />
        </Routes>
      </div>
    </Router>
  );
}
