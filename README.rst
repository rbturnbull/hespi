================================================================
hespi
================================================================

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

- 'genus',
- 'species',
- 'year',
- 'month',
- 'day',
- 'family',
- 'collector',
- 'authority',
- 'locality',
- 'geolocation',
- 'collector_number',
- 'infrasp taxon'

These text fields are then run through the OCR program Tesseract.

Installation
==================================

Install phytest using pip:

.. code-block:: bash

    pip install git+https://github.com/rbturnbull/hespi.git

At the moment it requires that the model weight files are in the working directory:

- sheet-component-weights.pt
- institutional-label-fields.pt

.. note ::

    Soon these will download automatically.

Usage
==================================

To run the pipeline, use the executable ``hespi`` and give it any number of images:

.. code-block:: bash

    hespi image1.jpg image2.jpg

This will prompt you to specify an output directory. You can set the output directory with the command with the ``--output-dir`` argument:

.. code-block:: bash

    hespi images/*.tif --output-dir ./hespi-output

The detected components and text fields will be 

.. end-quickstart

Credits
==================================

* Robert Turnbull, Karen Thompson, Emily Fitzgerald, Jo Birch

