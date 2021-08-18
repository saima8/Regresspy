"""A setuptools based setup module.
See:
https://packaging.python.org/guides/distributing-packages-using-setuptools/
"""

# Always prefer setuptools over distutils
from setuptools import setup, find_packages
from os import path
# io.open is needed for projects that support Python 2.7
# It ensures open() defaults to text mode with universal newlines,
# and accepts an argument to specify the text encoding
# Python 3 only projects can skip this import
from io import open

here = path.abspath(path.dirname(__file__))

# Get the long description from the README file
with open(path.join(here, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

# Arguments marked as "Required" below must be included for upload to PyPI.
# Fields marked as "Optional" may be commented out.

setup(
    name='regresspy',  
    version='0.1.0',  
    description='Regresspy for simple regressions',
    long_description=long_description,  # Optional
    long_description_content_type='text/markdown',  # Optional 
    url='https://github.com/saima8/Saima---Regressepy',  # Optional
    author='Saima Khan',  # Optional
    author_email='tithi888khan@gmail.com',  # Optional
    classifiers=[  # Optional
        'Development Status :: 3 - Alpha',
    ],
    keywords=['gradient descent', 'SGDRegressor', 'linear-regression', 'loss function','machine-learning'],  # Optional

    packages=find_packages(),  # Required
    python_requires='>=3',

    install_requires = [
        'numpy',
        'sklearn',    
    ],

    package_data={  # Optional
    },
    project_urls={  # Optional
        'Source': 'https://github.com/saima8/Saima---Regressepy',
    },
)