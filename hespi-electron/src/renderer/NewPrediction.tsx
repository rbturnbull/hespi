import React from 'react'
import { useCallback } from 'react'
import FilesDropzone from './FilesDropzone';

import icon from '../../assets/icon.svg';
import { ReactComponent as HespiLogo } from 'assets/img/hespi-logo2.svg';
import { ReactComponent as HespiBanner } from 'assets/img/hespi-banner.svg';

export default function NewPrediction({ onImagesUploaded, ...params }) {
  return (
    <>
      <div className='prediction-section' {...params}>
        <h2>New Prediction</h2>
        <FilesDropzone onFile={onImagesUploaded} />
      </div>
    </>
  );
}