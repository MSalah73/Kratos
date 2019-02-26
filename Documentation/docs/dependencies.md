# Kratos Dependencies

As Kratos was developed by a team working in parallel, yet independently, certain models have separate requirements. To get around this, the use of the Conda package and environment manager was implemented. Each model, and the framework as a whole has a yaml file associated with it for a Conda environment setup.

## Installing Conda

The Conda environment manager can be installed with either the full Anaconda or Miniconda packages.

Instructions for the installation of Conda [can be found here](https://conda.io/projects/conda/en/latest/user-guide/install/index.html#regular-installation).

## Setting up a Conda environment

Included in the repository files are `enviroment.yml` files. To setup a Conda environment using one of these files, use the following command line sequence:

    conda env create -f <environment.yml>

Your Conda environment is now setup. Setup will not need to be run again. To activate your new Conda environment, use the following command:

    conda activate <environment>

>[TODO](TODO) - We will have to create yml files with environment names identical to the file name.

Your Conda environment is now installed and activated. To deactivate the environment, use the following command:

    conda deactivate

