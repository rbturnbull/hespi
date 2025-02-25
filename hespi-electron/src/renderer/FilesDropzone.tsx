import React, {useState, useCallback, useEffect} from 'react'
import Dropzone from 'react-dropzone'
import { useDropzone } from 'react-dropzone'
import { Upload, X } from 'react-bootstrap-icons';

import icon from '../../assets/icon.svg';

function humanFileSize(size) {
  var i = size == 0 ? 0 : Math.floor(Math.log(size) / Math.log(1024));
  return +((size / Math.pow(1024, i)).toFixed(2)) * 1 + ' ' + ['B', 'kB', 'MB', 'GB', 'TB'][i];
}

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
  const removeUploadedFile = (file) => setFiles((prevFiles) => prevFiles.filter((f) => f !== file))

  function UploadedFile({ file }) {
    return <div className='uploaded-file'>
      <span className='file-name'>{file.name}</span>
      <span className='file-info'>
        <span className='file-size'>{humanFileSize(file.size)}</span>
        <span className='remove-file-btn' onClick={() => removeUploadedFile(file)}><X /></span>
      </span>
    </div>
  }

  return (
    <Dropzone onDrop={onDrop}>
      {({ getRootProps, getInputProps }) => {
        return <section className='files-dropzone-container'>
          <div className='uploaded-files'>
            {files.map((file: File) => <UploadedFile file={file} />)}
          </div>
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