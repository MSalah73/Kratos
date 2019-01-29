# Install Instructions

## Clone the repository

```bash
git clone https://github.com/ZackSalah/Kratos.git
git checkout Adam
```

## Ensure conda and python are installed

```bash
which conda
which python
```

If either of these didn't return a value then install [miniconda](https://docs.conda.io/en/latest/miniconda.html)

## Download required python libraries

```bash
conda env create -f environment.yml
source activate Adam
```

## Get the dataset

Download the [deep fashion dataset](http://mmlab.ie.cuhk.edu.hk/projects/DeepFashion.html)
Place it in a folder named `deep-fashion`

## Train your model

```bash
python model.py
```
