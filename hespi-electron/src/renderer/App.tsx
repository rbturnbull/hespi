import { MemoryRouter as Router, Routes, Route } from 'react-router-dom';
import './App.css';
import Hello from './Hello';
import HespiReport from './HespiReport';
import HespiGUI from './HespiGUI';

export default function App() {
  return (
    <Router>
      <Routes>
        <Route path="/Hello" element={<Hello />} />
        <Route path="/" element={<HespiGUI />} />
      </Routes>
    </Router>
  );
}
