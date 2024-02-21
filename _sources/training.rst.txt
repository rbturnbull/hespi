=========================
Training with custom data
=========================

``hespi`` comes with pretrained models, trained on the dataset provided by the authors. 
However, you can train your own model using your own dataset.


Annotate your images
=====================

Now you need to annotate your images. We recommend using `CVAT <https://www.cvat.ai/>`_ which is a free and open-source annotation tool and exports to YOLO format.
Other annotation tools can also be used.
Use the same class names as the ones used in the pretrained models.

Split the dataset so that has training images and validation images.

The dataset should have a YAML file which looks like this for the Sheet-Component model:

.. code-block:: yaml

    train: train.txt
    val: valid.txt
    nc: 11
    names: ['small database label', 'handwritten data', 'stamp', 'annotation label', 'scale', 'swing tag', 'full database label', 'database label', 'swatch', 'institutional label', 'number']

Or like this for the Label-Field model:

.. code-block:: yaml

    train: train.txt
    val: valid.txt
    nc: 12
    names: ['genus','species','year','month','day','family','collector','authority','locality','geolocation','collector_number','infrasp taxon']


Locate pretrained model
========================

You need to get the location of the pretrained models.

To get the location of the ``Sheet-Component`` model, you can use the following command:

.. code-block:: bash

    hespi-tools sheet-component-location

To get the location of the ``Label-Field`` model, you can use the following command:

.. code-block:: bash

    hespi-tools label-field-location    


Train the model
================

Now you can fine-tune the model using `YOLOv8 <https://github.com/ultralytics/ultralytics>`_ on your using the following command:

.. code-block:: bash
    
    yolo detect train data=${DATA} model=${MODEL} epochs=50 imgsz=640 batch=4

Repleace the ``${DATA}`` with the path to the YAML file and ``${MODEL}`` with the path to the pretrained model.

Save the location of the best weights for the trained model.

Using your model
================

Now that you've trained your model, you can use it when you use ``hespi``. 
Just set the ``HESPI_SHEET_COMPONENT_WEIGHTS`` environment variable to the location of the new Sheet-Component model weights file.
Or set the ``HESPI_LABEL_FIELD_WEIGHTS`` environment variable to the location of the Label-Field weights file.

You can also use the ``--sheet-component-weights`` or ``--label-field-weights`` command line options to specify the location of the weights file when you use ``hespi`` on the command line.
