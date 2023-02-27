import jinja2
from pathlib import Path

from .util import institutional_label_fields


def write_report(output, component_files, ocr_data, ocr_df):
    template_dir = Path(__file__).parent/"templates"
    loader = jinja2.FileSystemLoader(template_dir)

    def relative_to_output(path):
        return Path(path).relative_to(output.parent)

    def get_classification(path):
        return Path(path).name.split(".")[-2].replace("_", " ").title()

    def truncate(string):
        if len(string) < 30:
            return string
        return f"{string[:30]}..."

    env = jinja2.Environment(
        loader=loader,
        autoescape=jinja2.select_autoescape()
    )
    env.globals['relative_to_output'] = relative_to_output
    env.globals['get_classification'] = get_classification
    env.globals['len'] = len
    env.globals['truncate'] = truncate
    template = env.get_template("report-template.html")

    try:        
        result = template.render(
            component_files=component_files,
            institutional_label_fields=institutional_label_fields,
            ocr_df=ocr_df,
        )
        with open(str(output), 'w') as f:
            print(f"Writing result to {output}")
            f.write(result)                
    except Exception as err:
        print(f"failed to render {err}")
    
