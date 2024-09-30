================================================================
hespi
================================================================

.. image:: https://raw.githubusercontent.com/rbturnbull/hespi/main/docs/images/hespi-banner.svg

.. start-badges

|pypi badge| |testing badge| |coverage badge| |docs badge| |black badge|

.. |pypi badge| image:: https://img.shields.io/pypi/v/hespi
    :target: https://pypi.org/project/hespi/

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

Hespi takes images of specimen sheets from herbaria and first detects the various components of the sheet. 


.. image:: https://raw.githubusercontent.com/rbturnbull/hespi/main/docs/images/HespiDiagram.jpg
    :alt: Hespi pipeline
    :align: center


Hespi first takes a specimen sheet and detects the various components of it using the Sheet-Component Model. 
Then any full database label detected is cropped and this is given to the Label-Field Model 
which detects different textual fields written on the label. 
A Label Classifier is also used to determine the type of text written on the label. 
If it is printed or typewritten, then the text of each field is given to an Optical Character Recognition (OCR) engine 
and if there is handwriting, then each field is given to the Handwritten Text Recognition (HTR) engine. 
The recognized text is then corrected using a multimodal Large Language Model (LLM).
Finally, the result of the fields is post-processed before being written into 
an HTML report, a CSV file and text files. 

The stages of the pipeline are explained in the `documentation for the pipeline <https://rbturnbull.github.io/hespi/pipeline.html>`_.


Installation
==================================

Install hespi using pip:

.. code-block:: bash

    pip install hespi

The first time it runs, it will download the required model weights from the internet.

It is recommended that you also install `Tesseract <https://tesseract-ocr.github.io/tessdoc/Home.html>`_ so that this can be used in the text recognition part of the pipeline.

To install the development version, see the `documentation for contributing <https://rbturnbull.github.io/hespi/contributing.html>`_.


Usage
==================================

To run the pipeline, use the executable ``hespi`` and give it any number of images:

.. code-block:: bash

    hespi image1.jpg image2.jpg

By default the output will go to a directory called ``hespi-output``. 
You can set the output directory with the command with the ``--output-dir`` argument:

.. code-block:: bash

    hespi images/*.tif --output-dir ./hespi-output

The detected components and text fields will be cropped and stored in the output directory. 
There will also be a CSV file with the filename ``hespi-results.csv`` in the output directory with the text recognition results for any institutional labels found.

By default ``hespi`` will use OpenAI's ``gpt-4o`` large language model (LLM) in the pipeline to produce the final results.
If you wish to use a different model from OpenAI or Anthropic, add it on the command-line like this: ``--llm MODEL_NAME``
You will need to include an API key for the LLM. This can be ``OPENAI_API_KEY`` for an OpenAI LLM or ``ANTHROPIC_API_KEY`` for Anthropic.
You can also pass the API key to hespi with the ``--llm-api-key API_KEY`` argument.

More information on the command line arguments can be found in the `Command Line Reference <https://rbturnbull.github.io/hespi/cli.html>`_ in the documentation.

There is another command line utility called ``hespi-tools`` which provides additional functionality.
See the `documentation <https://rbturnbull.github.io/hespi/cli.html#hespi-tools>`_ for more information.

Training with custom data
==================================

To train the model with custom data, see the documention.

.. end-quickstart

Credits
==================================

.. start-credits

Robert Turnbull, Emily Fitzgerald, Karen Thompson and Jo Birch from the University of Melbourne.

This research was supported by The University of Melbourne’s Research Computing Services and the Petascale Campus Initiative. 
The authors thank collaborators Niels Klazenga, Heroen Verbruggen, Nunzio Knerr, Noel Faux, Simon Mutch, Babak Shaban, Andrew Drinnan, Michael Bayly and Hannah Turnbull.

Plant refererence data obtained from the `Australian National Species List (auNSL) <https://biodiversity.org.au/nsl>`_, as of March 2024, using the:

- Australian Plant Name Index (APNI)
- Australian Bryophyte Name Index (AusMoss)
- Australian Fungi Name Index (AFNI) 
- Australian Lichen Name Index (ALNI) 
- Australian Algae Name Index (AANI)

and the `World Flora Online Taxonomic Backbone v.2023.12 <https://www.worldfloraonline.org/downloadData>`_, accessed 13 June 2024.

This pipeline depends on `YOLOv8 <https://github.com/ultralytics/ultralytics>`_, 
`torchapp <https://github.com/rbturnbull/torchapp>`_,
Microsoft's `TrOCR <https://www.microsoft.com/en-us/research/publication/trocr-transformer-based-optical-character-recognition-with-pre-trained-models/>`_.

Logo derived from artwork by `ka reemov <https://thenounproject.com/icon/plant-1386076/>`_.

.. end-credits

See the documentation for more information for references in BibTeX format or use the command:

.. code-block:: bash

    hespi-tools bibtex
