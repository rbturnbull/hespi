import React from 'react'
import { useCallback } from 'react'
import FilesDropzone from './FilesDropzone';

import icon from '../../assets/icon.svg';
import { ReactComponent as HespiLogo } from 'assets/img/hespi-logo2.svg';
import { ReactComponent as HespiBanner } from 'assets/img/hespi-banner.svg';


const onAcceptedFiles = (acceptedFiles: File[]) => {
  acceptedFiles.forEach((file) => {
    console.log(`File: ${file.path}`)
    const reader = new FileReader()
    reader.onabort = () => console.log('file reading was aborted')
    reader.onerror = () => console.log('file reading has failed')
    reader.onload = () => {
      // Do whatever you want with the file contents
      const binaryStr = reader.result
      console.log(binaryStr)
    }
    reader.readAsArrayBuffer(file)
  })
}


export default function HespiGUI() {
  return (
    <>
      <HespiBanner style={{width:'100%', maxWidth: '700px'}}/>
      <FilesDropzone />
    </>
  );
}