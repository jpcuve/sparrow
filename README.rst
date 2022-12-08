Sparrow
=======

Sparrow is ...blablabla.

.. _Flask: https://palletsprojects.com/p/flask/
.. _SQLAlchemy: https://www.sqlalchemy.org


Installing
----------

Install and update using `pip`_:

.. code-block:: text

  $ pip install -U Sparrow

.. _pip: https://pip.pypa.io/en/stable/getting-started/


A Simple Example
----------------

.. code-block:: python

    from pathlib import Path
    from sparrow.client import Client

    sparrow_client = Client('localhost:5000', 'vicky-api-key')
    sparrow_client.upload('/my/path/to/images_and_prompts_folder')


Contributing
------------

For guidance on setting up a development environment and how to make a
contribution to Sparrow, see the `contributing guidelines`_.

.. _contributing guidelines: https://github.com/jpcuve/sparrow/blob/main/CONTRIBUTING.rst


Donate
------

I need a lot of money to support my lifestyle, so `please donate today`_.

.. _please donate today: https://www.sparrow-python.com/donate


Links
-----

-   Documentation: https://www.sparrow-python.com/
-   Changes: https://www.sparrow-python.com/changes/
-   PyPI Releases: https://pypi.org/project/sparrow-python/
-   Source Code: https://github.com/jpcuve/sparrow/
-   Issue Tracker: https://github.com/jpcuve/sparrow/issues/
-   Website: https://www.sparrow-python.com/
-   Twitter: https://twitter.com/sparrow-python
-   Chat: https://discord.gg/sparrow-python
