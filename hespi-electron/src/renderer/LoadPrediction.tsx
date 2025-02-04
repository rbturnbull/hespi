import React from 'react'
import { useCallback } from 'react'
import FilesDropzone from './FilesDropzone';

import icon from '../../assets/icon.svg';
import { ReactComponent as HespiLogo } from 'assets/img/hespi-logo2.svg';
import { ReactComponent as HespiBanner } from 'assets/img/hespi-banner.svg';


export default function LoadPrediction({onJsonLoaded}) {
  const onPredictionFile = (files: File[]) => {
    var jsonResult = {}
    files.forEach((file) => {
      console.log(`File: ${file.path}`)
      const reader = new FileReader()
      reader.onabort = () => console.log('file reading was aborted')
      reader.onerror = () => console.log('file reading has failed')
      reader.onload = () => {
        // Do whatever you want with the file contents
        const text = reader.result
        jsonResult = JSON.parse(text as string)
        console.log(jsonResult)
        onJsonLoaded(jsonResult)
      }
      reader.readAsText(file)
    })
  }

  return (
    <>
      <div className='prediction-section'>
        <h2>Load Prediction</h2>
        <FilesDropzone onFile={(files) => onPredictionFile(files)}/>
      </div>
    </>
  );
}