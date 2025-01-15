import gradio as gr
from pathlib import Path
from .hespi import Hespi

def process_images(image_list: list[str], llm_model: str, llm_temperature: float):
    output_dir = Path().cwd() / "hespi-output"
    hespi = Hespi(llm_model=llm_model, llm_temperature=llm_temperature)
    report_path = hespi.detect(image_list, output_dir)
    return gr.update(value=str(report_path), visible=True)


def build_interface():
    with gr.Blocks() as interface:
        banner = "https://raw.githubusercontent.com/rbturnbull/hespi/main/docs/images/hespi-banner.svg"
        gr.HTML(f"<div style='text-align:center;'><center><img src='{banner}' alt='Hespi Banner' style='width:100%; max-width: 700px;'></center></div>")
        with gr.Row():
            image_input = gr.Files(label="Upload specimen sheet images", file_types=["image"])
        with gr.Accordion("Advanced Options", open=False):
            with gr.Row():
                llm_model = gr.Dropdown(
                    choices=["gpt-4o", "claude-3-5-sonnet-20241022", "gpt-4o-mini", "claude-3-5-haiku-20241022"],
                    label="LLM Model"
                )
                llm_temperature = gr.Slider(
                    minimum=0.0,
                    maximum=2.0,
                    value=0.0,
                    step=0.1,
                    label="LLM Temperature",
                )

        process_button = gr.Button("Run Pipeline")
        output = gr.Text(label="Report", visible=False)
        process_button.click(process_images, inputs=[image_input,llm_model,llm_temperature], outputs=output)

    return interface
