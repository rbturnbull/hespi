=======================
Contributing
=======================

These practices are subject to change based on the decisions of the team.

- Use clear and explicit variable names.
- Python code should be formatted using black with the settings in pyproject.toml. The maximum line length is 120 characters.
- Contributions should be commited to a new branch and will be merged with main only after tests and documentation are complete.

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


- All tests must be passing before merging with the ``main`` branch.
- The code coverage must be at 100% before merging with the ``main`` branch.
- Tests are automatically included in the CI/CD pipeline using Github actions.

Git Commits
===========

We use the `git3moji <https://robinpokorny.github.io/git3moji/>`_ standard for expressive git commit messages. 
Use one of the following five short emojis at the start of your of your git commit messages:

- ``:zap:`` ⚡️ – Features and primary concerns
- ``:bug:`` 🐛 – Bugs and fixes
- ``:tv:``  📺 – CI, tooling, and configuration
- ``:cop:`` 👮 – Tests and linting
- ``:abc:`` 🔤 – Documentation

As far as possible, please keep your git commits granular and focussed on one thing at a time. 
Please cite an the number of a Github issue if it relates to your commit.

Documentation
==================

- Docstrings for Python functions should use the Google docstring convention (https://google.github.io/styleguide/pyguide.html#38-comments-and-docstrings)
- Documentation generated using sphinx and automatically deployed as part of the CI/CD pipeline.
- Docs should be written in reStructuredText.

Files need to start with a heading for the section. The convention used here is to use the equals sign above and below the heading::

    ===============
    Section Heading
    ===============

Subsections also use an equals sign but just below the heading::

    Subsection Heading
    ==================

Subsubsections have a single dash below the heading::

    Subsubsection Heading
    ---------------------

Try not to have any other sections within this but if it is necessary, use tildas below the heading::

    Further Subsection Headings
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~

Other information for using reStructuredText in Sphinx can be found here: https://www.sphinx-doc.org/en/master/usage/restructuredtext/basics.html#rst-primer and https://thomas-cokelaer.info/tutorials/sphinx/rest_syntax.html.
