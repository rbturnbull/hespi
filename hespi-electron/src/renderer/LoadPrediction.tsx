import React, { useCallback, useState, useEffect } from 'react'
import FilesDropzone from './FilesDropzone';
import Alert from 'react-bootstrap/Alert';

import icon from '../../assets/icon.svg';
import { ReactComponent as HespiLogo } from 'assets/img/hespi-logo2.svg';
import { ReactComponent as HespiBanner } from 'assets/img/hespi-banner.svg';
import { TOASTS } from './ToastCommunications';




export default function LoadPrediction({ onJsonLoaded }) {
  const [filesUploaded, setFilesUploaded] = useState([])
  const readPredictionFiles = (files: File[]) => {
    var jsonResult = {}
    files.forEach((file) => {
      console.log(`File: ${file.path}`)
      const reader = new FileReader()
      reader.onabort = () => console.log('file reading was aborted')
      reader.onerror = () => console.log('file reading has failed')
      reader.onload = () => {
        try {
          // Do whatever you want with the file contents
          const text = reader.result
          jsonResult = JSON.parse(text as string)
          console.log(jsonResult)
          onJsonLoaded(jsonResult)
        } catch (e) {
          TOASTS.PREDICTION_LOAD_FAILED.show({ file });
          console.error(`Error reading file ${file.path}: ${e}`)
        }
      }
      reader.readAsText(file)
    })
  }

  return (
    <>
      <div className='prediction-section'>
        <h2>Load Prediction</h2>

        <FilesDropzone onFile={(acceptedFiles) => {
          if (acceptedFiles.length > 0) {
            setFilesUploaded(acceptedFiles)
          }
        }} />
      </div>
      <button className='btn btn-primary' disabled={filesUploaded.length <= 0} onClick={() => {
        readPredictionFiles(filesUploaded)
      }}>Load Prediction</button>
    </>
  );
}