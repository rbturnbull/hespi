import React from 'react'
import { useCallback } from 'react'
import FilesDropzone from './FilesDropzone';

import icon from '../../assets/icon.svg';
import {ReactComponent as HespiLogo} from 'assets/img/hespi-logo2.svg';


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


export default function HespiReport() {
  return (
    <>
      <header className="navbar navbar-dark sticky-top bg-light flex-md-nowrap p-0 shadow">
        <button className="navbar-toggler position-absolute d-md-none collapsed" type="button" data-bs-toggle="collapse"
          data-bs-target="#sidebarMenu" aria-controls="sidebarMenu" aria-expanded="false" aria-label="Toggle navigation">
          <span className="navbar-toggler-icon"></span>
        </button>
      </header>

      <div className="container-fluid">
        <div className="row">
          <nav id="sidebarMenu" className="col-md-3 col-lg-2 d-md-block sidebar collapse vh-100 overflow-auto">
            <div id="brand">
              <a href="https://rbturnbull.github.io/hespi/" target="_blank"><HespiLogo /></a>
            </div>

            <div className="position-sticky">
              <ul className="nav flex-column pt-2">
                <li className="nav-item">
                  <a className="nav-link {% if loop.first %}active{% endif %}" aria-current="page" href="#" id="{{ stub }}-tab" data-bs-toggle="tab" data-bs-target="#{{ stub }}" type="button" role="tab" aria-controls="{{ stub }}" aria-selected="true">
                    <img src='{{ stub }}/{{ stub }}.thumbnail.jpg' />
                  </a>
                </li>
              </ul>
            </div>
          </nav>

        </div>
      </div>
    <FilesDropzone />
    </>
  );
}