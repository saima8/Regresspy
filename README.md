![Regresspy logo](https://github.com/saima8/Saima---Regressepy/blob/main/logo/2.png?raw=true)
Regresspy is a python module for carrying out simple regressions using gradient descent algorithm.
This python module is for performing robust linear regression on (X,Y) data points.

## Installation
The Code is written in Python 3.9.5. If you don't have Python installed you can find it [here](https://www.python.org/downloads/). If you are using a lower version of Python you can upgrade using the pip package, ensuring you have the latest version of pip.

Install Regresspy using ```pip```:

```bash
  pip install regresspy
```
If that does not work, you can install it using the setup.py script:
```bash
  python setup.py install
```
Alternatively, if you plan to modify the source then install the package with a symlink, so that changes to the source files will be immediately available:
```bash
  python setup.py develop
```
Besides, to install the required packages and libraries, run this command in the project directory after [cloning](https://www.howtogeek.com/451360/how-to-clone-a-github-repository/) the repository:
```bash
pip install -r requirements.txt
```
## Usage

### Overview

This project is a great package for calculating gradient descent, loss function as well as regression.The gradient descent is calculated by using *Forward Propagation* and *Back Propagation* technique. This will help you both in Machine learning and Deep learning problems. You won't have to hard code everything after this. You only have to choose dataset and necessary things. After that declare parameters and evaluate your loss function results.

### Test the regresspy_package

You can check the linear regression trained rmse value from scikitlearn and also can observe loss values according to the epochs. You can also check your train rmse according to my code. To check the model.py, open powershell and run this code:

```bash
python model.py
```
You can also test different kind of loss such as  mae, sse, mse, rmse. For this just run the below code: 
```bash
python test_loss.py
```




  
## Directory Tree

See the Directory tree of this project

```bash
├── regresspy 
│   ├── __init__.py
│   ├── gradient_descent.py
│   ├── loss.py
│   ├── regression.py
│   └── model.py
├── test
│   ├── __init__.py
│   └── test_loss.py
├── logo
│   ├── 1.jpg
│   └── 2.png
├── requirements.txt
├── setup.py
├── LICENSE
└── README.md
```

  
## Tech Stack

![](https://forthebadge.com/images/badges/made-with-python.svg)


  
## License

[![GPLv3 License](https://img.shields.io/badge/License-GPL%20v3-yellow.svg)](https://opensource.org/licenses/)

You may not use this file except in compliance with the License.

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions and limitations under the License.
## Authors

[Saima Khan](https://www.github.com/saima8)
  
  Email: tithi888khan@gmail.com