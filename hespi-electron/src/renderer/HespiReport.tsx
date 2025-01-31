import React from 'react'
import { useCallback } from 'react'
import FilesDropzone from './FilesDropzone';

import icon from '../../assets/icon.svg';
import {ReactComponent as HespiLogo} from 'assets/img/hespi-logo2.svg';
import componentFiles from "public/hespi-output/hespi-results.json";
import Stub from './Stub';
import { convertOutputPath } from './utils';




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

function NavItem({stub, index}) {
  const baseImg = "/hespi-output/"+stub.id + "/" + stub.id;
  return (
    <li className="nav-item">
      <a className={"nav-link" + (index <= 0 ? " active" : "")} aria-current="page" href="#" id={stub.id + "-tab"} data-bs-toggle="tab" data-bs-target={`#${stub.id}`} type="button" role="tab" aria-controls={stub.id} aria-selected="true">
        <img src={`${baseImg}.thumbnail.jpg`}/>
        {stub.id}
      </a>
    </li>
  )
}


export default function HespiReport() {

  const stubs = componentFiles.map((stub, i) => <Stub stub={stub} key={`stub-${i}`} index={i} />)
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
                {componentFiles.map((stub, i) => <NavItem stub={stub} index={i} key={`stub-${i}`} />)}
              </ul>
            </div>
          </nav>

          <main className="col-md-9 ms-sm-auto col-lg-10 px-md-4">
            <div className="tab-content" id="myTabContent">
              {stubs}
            </div>
          </main>

        </div>
      </div>
    </>
  );
}