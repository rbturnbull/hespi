import gradio as gr
from pathlib import Path
from .hespi import Hespi
from .util import Generator

def process_images(image_list: list[str], llm_model: str, llm_temperature: float, progress=gr.Progress()):
    output_dir = Path().cwd() / "hespi-output"
    progress(0.0, desc="Starting")
    hespi = Hespi(llm_model=llm_model, llm_temperature=llm_temperature)
    gen = Generator(hespi.detect(image_list, output_dir, progress=progress))
    for ocr_data in gen:
        yield f"{ocr_data['id']}"
    yield f"HTML report: {str(gen.value)}"


def compile_sass(assets_dir, sass_in_dir="sass", css_out_dir="__css__"):
    print(f"Compiling sass: {assets_dir / sass_in_dir}, {assets_dir / css_out_dir}")
    import sass
    sass.compile(dirname=(assets_dir / sass_in_dir, assets_dir / css_out_dir), output_style="nested")


def build_blocks():
    compile_sass(Path(__file__).parent / "templates" / "assets")
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
        btn = gr.Button("Run Pipeline")
        pbar = gr.Text(label="Progress", elem_id="pbar", visible=True)
        progress_log = gr.TextArea(elem_id="p_log", value="", show_label=False, visible=True)
        process_inputs = [image_input, llm_model, llm_temperature]
        # Check out: https://www.gradio.app/guides/blocks-and-event-listeners#running-events-consecutively
        # and a good progress bar example: https://github.com/gradio-app/gradio/issues/8895
        # Also this: https://www.gradio.app/guides/dynamic-apps-with-render-decorator
        # btn.click(start, inputs=[], outputs=pbar, show_progress="full")
        btn.click(process_images, inputs=process_inputs, outputs=pbar, show_progress="full")

        def on_detect(p_log, progress):
            return f"{p_log} ✅ {progress}\n"

        pbar.change(on_detect, inputs=[progress_log, pbar], outputs=progress_log)

    return interface.queue()
