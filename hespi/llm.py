import re
from pathlib import Path
import base64
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain.prompts import ChatPromptTemplate
from langchain.schema.output_parser import StrOutputParser
from langchain.chat_models.base import BaseChatModel
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic


from .util import label_fields

def encode_image(path:Path|str) -> str:
    path = Path(path)
    with path.open('rb') as image_file:
        image_data = image_file.read()

    # Encode the image data to base64
    base64_encoded_image = base64.b64encode(image_data).decode('utf-8')
    return base64_encoded_image


def build_template(primary_specimen_label_image:Path, detection_results:dict) -> ChatPromptTemplate:
    base64_encoded_image = encode_image(primary_specimen_label_image)
    value_dict = {key: value.replace('\n', ' ') for key, value in detection_results.items() if key in label_fields}
    values_string = "\n".join([f"{field}: {value}" for field, value in value_dict.items()])

    # Build the OCR results string
    ocr_results_strings = []
    for field in label_fields:
        ocr_results_key = f"{field}_ocr_results"
        if ocr_results_key in detection_results:
            for ocr_result in detection_results[ocr_results_key]:
                engine = ocr_result['ocr']
                original_text_detected = ocr_result['original_text_detected'].replace('\n', ' ')
                adjusted_text = ocr_result['adjusted_text'].replace('\n', ' ')
                
                ocr_results_string = f"The {engine} model thought the {field} was '{original_text_detected}'"
            
                if adjusted_text and adjusted_text != original_text_detected:
                    ocr_results_string += f" and it was adjusted to '{adjusted_text}'"

                ocr_results_strings.append(ocr_results_string)

    ocr_results = "\n".join(ocr_results_strings)

    system_message = SystemMessage("You are an expert curator of a herbarium with vast knowledge of plant species.")
    main_prompt = f"""
        We have a pipeline for automatically reading the primary specimen labels and extracting the following fields:\n{', '.join(label_fields)}.
        
        You need to inspect an image and see if the fields have been extracted correctly. 
        If there are errors, then print out the field name with a colon and then the correct value. Each correction is on a new line.
        If the values provided are correct, then don't output anything for that field.
        When you are finished, print out 5 hyphens '-----' to indicate the end of the text.

        For example, if the 'genus' field and the 'species' field were extracted incorrectly, then you would print:
        genus: Abies
        species: alba
        -----

        Here are the following fields that we have extracted from the primary specimen label:
        {values_string}

        {ocr_results}

        Here is the image of the primary specimen label:
    """
    main_prompt = re.sub(r'[\t ]+', ' ', main_prompt).strip()
    human_message = HumanMessage(
        content=[
            {
                "type": "text", 
                "text": main_prompt 
            },
            {
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{base64_encoded_image}"},
            },
        ]
    )
    ai_message = AIMessage("Certainly, here are the corrections:")
    return ChatPromptTemplate.from_messages(messages=[system_message, human_message, ai_message])


def output_parser(text:str) -> dict[str, str]:
    lines = text.split("\n")
    result = {}
    for line in lines:
        if not line.strip():
            continue

        # check end condition
        if "----" in line:
            break

        # split at first colon
        colon = line.find(":")
        if colon == -1:
            continue

        field = line[:colon].strip().lower()
        value = line[colon+1:].strip()

        if field not in label_fields:
            continue

        result[field] = value

    return result


def get_llm(
    model_id:str="gpt-4o",
    api_key:str="",
    temperature:float=0.0,
) -> BaseChatModel:
    if model_id.startswith('gpt'):
        return ChatOpenAI(
            openai_api_key=api_key, 
            temperature=temperature,
            model_name=model_id,
        )
    
    if model_id.startswith('claude'):
        return ChatAnthropic(
            model=model_id,
            temperature=temperature,
            max_tokens=1024,
            timeout=None,
            max_retries=2,
            api_key=api_key,
        )
    raise ValueError(f"Model {model_id} not recognized.")


def llm_correct_detection_results(llm:BaseChatModel, primary_specimen_label_image:Path, detection_results:dict) -> None:
    template = build_template(primary_specimen_label_image, detection_results)
    
    chain = template | llm | StrOutputParser() | output_parser

    result = chain.invoke({})
    detection_results.update(result)
    for field_name, value in result.items():
        ocr_results_key = f"{field_name}_ocr_results"
        if ocr_results_key not in detection_results:
            detection_results[ocr_results_key] = []

        detection_results[ocr_results_key].append(dict(
            ocr="LLM",
            original_text_detected=value,
            adjusted_text="",
            match_score=0,
        ))

