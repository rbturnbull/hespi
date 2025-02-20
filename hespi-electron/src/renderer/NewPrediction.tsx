import React, { useState, useEffect} from 'react'
import { useCallback } from 'react'
import FilesDropzone from './FilesDropzone';

import icon from '../../assets/icon.svg';
import { ReactComponent as HespiLogo } from 'assets/img/hespi-logo2.svg';
import { ReactComponent as HespiBanner } from 'assets/img/hespi-banner.svg';

export default function NewPrediction({ onImagesUploaded, ...params }) {
  const [filesUploaded, setFilesUploaded] = useState([])

  const predict = () => {

  }
  return (
    <>
      <div className='prediction-section' {...params}>
        <h2>New Prediction</h2>
        <FilesDropzone onFile={(acceptedFiles) => {
          if (acceptedFiles.length > 0){
            onImagesUploaded(acceptedFiles)
            setFilesUploaded(acceptedFiles)
          }
        }} />
        <button className='btn btn-primary' disabled={filesUploaded.length <= 0} onClick={() => {
          console.log("Calling hespi electron api ", filesUploaded)
          window.electronAPI.runHespi(filesUploaded.map(f => "'" + f.path + "'"), 0.2)
        }}>Predict</button>
      </div>
    </>
  );
}