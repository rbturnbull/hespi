================================================================
hespi
================================================================

.. image:: https://raw.githubusercontent.com/rbturnbull/hespi/main/docs/images/hespi-banner.svg

.. start-badges

|testing badge| |coverage badge| |docs badge| |black badge|

.. |testing badge| image:: https://github.com/rbturnbull/hespi/actions/workflows/testing.yml/badge.svg
    :target: https://github.com/rbturnbull/hespi/actions

.. |docs badge| image:: https://github.com/rbturnbull/hespi/actions/workflows/docs.yml/badge.svg
    :target: https://rbturnbull.github.io/hespi
    
.. |black badge| image:: https://img.shields.io/badge/code%20style-black-000000.svg
    :target: https://github.com/psf/black
    
.. |coverage badge| image:: https://img.shields.io/endpoint?url=https://gist.githubusercontent.com/rbturnbull/f31036b00473b6d0af3a160ea681903b/raw/coverage-badge.json
    :target: https://rbturnbull.github.io/hespi/coverage/
    
.. end-badges

HErbarium Specimen sheet PIpeline

.. start-quickstart

Hespi takes images of specimen sheets from herbaria and first detects the various components of the sheet. These components include:

- small database label
- handwritten data
- stamp
- annotation label
- scale
- swing tag
- full database label
- database label
- swatch
- institutional label
- number

Then it takes any `institutional label` and detects the following fields from it:

- family,
- genus,
- species,
- infrasp_taxon,
- authority,
- collector_number,
- collector,
- locality,
- geolocation,
- year,
- month,
- day,

These text fields are then run through the OCR program Tesseract.

Installation
==================================

Install hespi using pip:

.. code-block:: bash

    pip install hespi

The first time it runs, it will download the required model weights from the internet.

It is recommended that you also install `Tesseract <https://tesseract-ocr.github.io/tessdoc/Home.html>`_ so that this can be used in the text recognition part of the pipeline.

Usage
==================================

To run the pipeline, use the executable ``hespi`` and give it any number of images:

.. code-block:: bash

    hespi image1.jpg image2.jpg

This will prompt you to specify an output directory. You can set the output directory with the command with the ``--output-dir`` argument:

.. code-block:: bash

    hespi images/*.tif --output-dir ./hespi-output

The detected components and text fields will be cropped and stored in the output directory. There will also be a CSV file with the text recognition results for any institutional labels found.

.. end-quickstart

Credits
==================================

.. start-credits

Robert Turnbull, Karen Thompson, Emily Fitzgerald, Jo Birch.

Publication and citation details to follow.

This pipeline depends on `YOLOv5 <https://github.com/ultralytics/yolov5>`_, 
`torchapp <https://github.com/rbturnbull/torchapp>`_,
Microsoft's `TrOCR <https://www.microsoft.com/en-us/research/publication/trocr-transformer-based-optical-character-recognition-with-pre-trained-models/>`_.

Logo derived from artwork by `ka reemov <https://thenounproject.com/icon/plant-1386076/>`_.

.. end-credits
