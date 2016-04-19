Installation
============

For installation of xtool we fully rely on the
`anaconda <https://www.continuum.io/downloads>`_ installation environment.

The anaconda environment already brings a number of packages - for a barebone
installation please refer to
`miniconda <http://conda.pydata.org/miniconda.html>`_.

Currently, xtool is designed to work with Python 2.7. We have provided an
environment that contains all the necessary dependencies for xtool::

    curl -O https://raw.githubusercontent.com/eso/xtool/master/xtool_env.yml
    conda env create -n <yourname_for_the_environment> --file xtool_env.yml
    source activate <yourname_for_the_environment>

Most of the analysis is performed in notebooks (one can think of them as cookbooks)::

    curl -O https://raw.githubusercontent.com/eso/xtool/master/docs/notebooks/cookbook_reading.ipynb
    jupyter notebook cookbook_reading.ipynb