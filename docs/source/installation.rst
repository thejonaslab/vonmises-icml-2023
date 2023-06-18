.. _installation:

Installation
============

Installing from Source
----------------------

vonmises-icml-2023 can be installed from source as follows:

1. :code:`git clone https://github.com/thejonaslab/vonmises-icml-2023.git`
2. :code:`cd vonmises-icml-2023`
3. :code:`conda env create -n $NAME -f environment.yml`, where $NAME is the desired name of the environment. Comment out the line containing `cudatoolkit` if installing on a machine without a GPU.
4. :code:`conda activate $NAME`
5. :code:`pip install -e .`
