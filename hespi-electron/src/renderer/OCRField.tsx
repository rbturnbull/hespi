import React from 'react'
import { useCallback, useState, useEffect } from 'react'
import icon from '../../assets/icon.svg';
import { ReactComponent as HespiLogo } from 'assets/img/hespi-logo2.svg';
import useImage from '../hooks/useImage';
import path from 'path';
import { convertOutputPath } from './utils';
import { PencilSquare, FloppyFill, TrashFill } from 'react-bootstrap-icons';



const OCR_TYPES = ["Tesseract", "TrOCR", "LLM"]
const OCR_SCORES = {
  original: "original",
  adjusted: "adjusted",
  score: "match_score"
}



const getFieldImages = (data, field) => {
  const base = `${field}_image`;
  var images: string[] = [];
  if (!data.hasOwnProperty(base)) {
    return images;
  }
  images.push(data[base]);
  while (true) {
    const next = `${base}_${images.length}`;
    if (!data.hasOwnProperty(next)) {
      break;
    }
    images.push(data[next]);
  }
  images = images.filter(x => x != null).map((imagePath) => convertOutputPath(imagePath));
  // console.log(images);
  return images
}

// def ocr_result_str(row, field_name: str, ocr_type: str) -> str:
//     def get_list(row, key):
// result = row.get(key, "")
// if not isinstance(result, list):
// result = [result]

// return result

// original_list = get_list(row, f"{field_name}_{ocr_type}_original")
// adjusted_list = get_list(row, f"{field_name}_{ocr_type}_adjusted")
// match_score_list = get_list(row, f"{field_name}_{ocr_type}_match_score")

//     assert len(original_list) == len(adjusted_list) == len(match_score_list)

// texts = []
// for original, adjusted, match_score in zip(original_list, adjusted_list, match_score_list):
//   text = original
// if adjusted and adjusted != original:
// text += f" → {adjusted}"

// if match_score:
//   text += f" ({match_score})"

// texts.append(text)

// return " | ".join(texts)

function EditableOCRField({data, field, ocr}) {
  const [editMode, setEditMode] = useState(false)
  const iconSize = 15;
  const iconColor = "#999999";
  return <div className="editable-ocr-field">
    <input readOnly={!editMode} className="textfield--input recognized-text" value={data[field]} />
    {/* <p className="recognized-text">{data[field]}</p> */}
    {!editMode ?
      <PencilSquare color={iconColor} size={iconSize} onClick={() => {
        console.log("Toggling editing mode!")
        setEditMode(prevEditMode => !prevEditMode)
      }} onBlur={() => setEditMode(false)} />
      :
      <div style={{display: "flex", flexDirection: "row", gap: "0.5rem"}}>
        <FloppyFill color={iconColor} size={iconSize} onClick={() => {
          console.log("Toggling editing mode!")
          setEditMode(prevEditMode => !prevEditMode)
        }} />
        <TrashFill color={iconColor} size={iconSize} onClick={() => {
          console.log("Toggling editing mode!")
          setEditMode(prevEditMode => !prevEditMode)
        }} />
      </div>
  }
  </div>
}

function OCRResults({data, field, ocr}) {
  var results = [];
  OCR_TYPES.forEach((ocrType, i) => {
    const base = `${field}_${ocrType}_`;
    const original = data[`${base}${OCR_SCORES.original}`];
    if (original) {
      const adjusted = data[`${base}${OCR_SCORES.adjusted}`];
      const score = data[`${base}${OCR_SCORES.score}`];
      results.push(<p className="alternative-text">{ocrType}: {original} → {adjusted} ({score})</p>);
    }
  })
  return results;
}

export default function OCRField({ data, field, index }) {
  return (
    <tr>
      <th scope="row">{field}</th>
      <td>
        {getFieldImages(data, field).map((image, i) => 
          <a href={ image }>
            <img src={ image } className="hespi-image" />
          </a>
        )}
      </td>
      <td>
        <EditableOCRField data={data} field={field} />
        <OCRResults data={data} field={field} />
      </td>
    </tr>
  );
}