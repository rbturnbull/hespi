import React, {useState, useCallback, useEffect} from 'react'
import Dropzone from 'react-dropzone'
import { useDropzone } from 'react-dropzone'
import { Upload } from 'react-bootstrap-icons';

import icon from '../../assets/icon.svg';

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


export default function FilesDropzone({onFile}) {
  const [files, setFiles] = React.useState([])
  const onDrop = useCallback(acceptedFiles => {
    setFiles(acceptedFiles)
    onFile(acceptedFiles)    
  }, [])
  return (
    <Dropzone onDrop={onDrop}>
      {({ getRootProps, getInputProps }) => {
        return <section className='files-dropzone-container'>
          {files.map(file => <p>{file.name}</p>)}
          <div {...getRootProps()} className='files-dropzone'>
            <Upload size={30}/>
            <div style={{textAlign: 'center'}}>
              <p>Drop File Here</p>
              <p style={{color: '#AAAAAA'}}>- or -</p>
              <input {...getInputProps()} />
              <p>Click to Upload</p>
            </div>
          </div>
        </section>
      }}
    </Dropzone>
  );
}