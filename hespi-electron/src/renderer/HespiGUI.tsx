import React, {useState, useCallback, useEffect} from 'react'
import FilesDropzone from './FilesDropzone';

import icon from '../../assets/icon.svg';
import { ReactComponent as HespiLogo } from 'assets/img/hespi-logo2.svg';
import { ReactComponent as HespiBanner } from 'assets/img/hespi-banner.svg';
import NewPrediction from './NewPrediction';
import LoadPrediction from './LoadPrediction';
import HespiReport from './HespiReport';
import { Route } from 'react-router-dom';

const ROUTE = {
  NONE: 'none',
  LOAD: 'load',
  NEW: 'new'
}

export default function HespiGUI() {
  const [route, setRoute] = useState(ROUTE.NONE);
  const [jsonData, setJsonData] = useState(null);

  return (
    <>
    {route === ROUTE.NONE &&
      <>
        <HespiBanner style={{width:'100%', maxWidth: '500px'}}/>
        <NewPrediction style={{ marginBottom: '5rem' }} onImagesUploaded={(imageFiles) => {
          console.log("On images uploaded", imageFiles)
          // setRoute(() => ROUTE.NEW)
        }}/>
        <LoadPrediction onJsonLoaded={(json) => {
          console.log("On json prediction loaded", json)
          setJsonData(json)
          setRoute(() => ROUTE.LOAD)
        }} />
      </>
    }
    {route === ROUTE.LOAD &&
      <>
        <HespiReport jsonData={jsonData} />
      </>
    }
    </>
  );
}