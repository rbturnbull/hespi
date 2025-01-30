import React from 'react'
import { useCallback } from 'react'
import icon from '../../assets/icon.svg';
import { ReactComponent as HespiLogo } from 'assets/img/hespi-logo2.svg';
import OCRField from './OCRField';
import useImage from '../hooks/useImage';
import { convertOutputPath } from './utils';

import path from 'path';

interface StubData {
  id?: string;
}

const LABEL_FIELDS = [
  "family",
  "genus",
  "species",
  "infrasp_taxon",
  "authority",
  "collector_number",
  "collector",
  "locality",
  "geolocation",
  "year",
  "month",
  "day",
]


export default function Stub({ stub, index }) {
  console.log("Stub: ", stub.id)
  const isFirst = index === 0;
  const showActive = isFirst ? "show active" : "";
  const elId = `${stub.id}-institional-label-${index}`
  const tabId = `${stub.id}-institional-label-tab-${index}`
  const imgPath = convertOutputPath(stub.predictions as string);
  const baseImg = stub.id + "/" + stub.id;
  // const customImgPath = "/hespi-output/MELUD109283a_sp62022822692917164542_original/MELUD109283a_sp62022822692917164542_original..institutional_label/MELUD109283a_sp62022822692917164542_original..institutional_label.all.jpg"
  // console.log(`Custom Image Path: ${customImgPath}`)
  // console.log(`Image Path: ${imgPath}`)
  
  // const { loading, error, image } = useImage("assets/img/hespi-output/MELUD109283a_sp62022822692917164542_original./MELUD109283a_sp62022822692917164542_original..all.jpg")
  // console.log(stub.predictions as string)
  // console.log(`Stub: ${stub}\nFiles: ${files}`)
  return (
    <div className={`tab-pane fade ` + showActive} id={stub.id} role="tabpanel" aria-labelledby={ stub.id+'-tab'}>
      <h1>{ stub.id }</h1>

      {/* {% set my_ocr_df = ocr_df[ocr_df["id"] == stub] %} */}
      <ul className="nav nav-tabs" id={stub.id+"-tabs"} role="tablist">
        {/* {% for _, row in my_ocr_df.iterrows() %} */}
        <li className="nav-item" role="presentation">
          <button className={"nav-link " + showActive} id={tabId}
            data-bs-toggle="tab" data-bs-target={'#' + elId} type="button" role="tab" aria-controls={elId} aria-selected="true">Institutional Label {isFirst ? "" : index} </button>
        </li>
        {/* {% endfor %} */}
        <li className="nav-item" role="presentation">
          {/* <button className="nav-link {% if len(my_ocr_df) == 0 %}show active{% endif %}" id="{{ stub }}-sheet-components-tab" data-bs-toggle="tab" data-bs-target="#{{ stub }}-sheet-components" type="button" role="tab" aria-controls="{{ stub }}-sheet-components" aria-selected="{% if len(my_ocr_df) == 0 %}true{% else%}false{% endif %}">Sheet Components</button> */}
        </li>
      </ul>
      <div className="tab-content" id={ stub.id+"-myTabContent"}>
        {/* {% for _, row in my_ocr_df.iterrows() %} */}
        <div className={"tab-pane fade " + showActive} id={elId} role="tabpanel" aria-labelledby={tabId}>
          <center>
            <a href={imgPath}><img src={imgPath} className="hespi-image" /></a>
            <p>Classification: {stub.label_classification}</p>
          </center>

          <table className="table table-hover table-striped ">
            <thead>
              <tr>
                <th scope="col">Field</th>
                <th scope="col">Image</th>
                <th scope="col">Text</th>
              </tr>
            </thead>
            <tbody>
              {LABEL_FIELDS.map((field, i) => <OCRField data={stub} field={field} index={i} key={`ocr-${field}-${i}`}/>)}
              {/* {% for field in label_fields %}
                        {% if field in row  %} */}
              
              {/* {% endif %}
                        {% endfor %} */}
            </tbody>
          </table>
        </div>
        {/* {% endfor %} */}
        <div className="tab-pane fade show active" id={ stub.id+"-sheet-components"} role="tabpanel" aria-labelledby={stub.id +"-sheet-components-tab"}>
          <center>
            <a href={baseImg + ".all.jpg"}><img src={baseImg+".medium.jpg"} className="hespi-image" /></a>
          </center>
          <h2>Sheet Components</h2>

          <table className="table table-hover table-striped ">
            <thead>
              <tr>
                <th scope="col">Component</th>
                <th scope="col">Image</th>
              </tr>
            </thead>
            <tbody>
              {/* {% for file in files %} */}
              <tr>
                <th scope="row">{/* get_classification(file) */}</th>
                <td>
                  <a href='{{ relative_to_output(file) }}'>
                    <img src='{{ relative_to_output(file) }}' className="hespi-image" />
                  </a>
                </td>
              </tr>
              {/* {% endfor %} */}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
}