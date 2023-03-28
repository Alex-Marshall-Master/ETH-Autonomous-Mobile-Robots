# Exercise 5 - Image Filtering

## Python code

You are initially provided with two (incomplete) questions `ex_5_smoothing.py` and `ex_5_pointFeatures.py`. Completed answers can be found in `ex_5_smoothing_solution.py` and `ex_5_pointFeatures_solution.py` once released.

## Requirements
The requirements are found in the `requirements.txt` file, and tested with Python 3.9. A virtual evironment is highly recommended, you are welcome to use the standard python virtual environments or a package management system like conda.


## Setting up a virtual environment

### Virtualenvwrapper:
This is a virtualenv helper that can be very convenient. [Install it](https://virtualenvwrapper.readthedocs.io/en/latest/#)

```
mkvirtualenv --python=$( which python3.9 ) AMR
workon AMR
cd /path/to/ex5_image_filtering/python
pip install -r requirements.txt
python3 ex_5_smoothing_solution.py
```

### Conda:
Install [conda](https://docs.conda.io/projects/conda/en/latest/user-guide/install/linux.html).
```
conda create -n AMR python=3.8
conda activate AMR
conda install --file requirements.txt
```

