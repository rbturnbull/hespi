#!/bin/bash
tar -xzvf "cpython-3.10.16+20250205-aarch64-apple-darwin-install_only.tar.gz"
mv python hespi-gui-env
chmod -R 777 hespi-gui-env
chown -R $USER hespi-gui-env
# ./hespi-gui-env/bin/python -m pip install rich typer torch torchapp pytesseract transformers appdirs jinja2 ultralytics langchain gradio drawyolo llmloader libsass orjson rcssmin pytest ipykernel coverage autopep8 sphinx nbsphinx sphinx-rtd-theme sphinx-autobuild myst-parser pre-commit sphinx-copybutton sphinx-click black
./hespi-gui-env/bin/python -m pip install -r hespi-gui-req.txt