============
Development
============

To develop hespi, you will need to install poetry and clone the repository.

Poetry
======

Poetry is the package manager used to manage the dependencies of the project. You can install it with this command on Linux and Mac:

.. code-block:: bash

    curl -sSL https://install.python-poetry.org | python3 -

Or with this command on Windows:

.. code-block:: bash

    (Invoke-WebRequest -Uri https://install.python-poetry.org -UseBasicParsing).Content | py -

More information for installing poetry can be found at https://python-poetry.org/docs/#installation.


Clone the repository
====================

You can clone the repository with the following command:

.. code-block:: bash

    git clone https://github.com/rbturnbull/hespi.git
    cd hespi


Then install the dependencies and activate the virtual environment:

.. code-block:: bash

    poetry install
    poetry shell

Testing
=======

Test the code with pytest:

.. code-block:: bash

    pytest

To check the coverage of the tests and build the HTML report, run the following commands:

.. code-block:: bash

    coverage run -m pytest
    coverage html
    coverage report
