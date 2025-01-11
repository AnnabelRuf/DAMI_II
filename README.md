# Usage
## Dependecies
All needed dependecies are stored in ````requirements.txt````. To install them, first create a python virtual environment, activate it and install the dependencies from the file.
#### Linux:
````cmd
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
````
In one-line
````cmd
python3 -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt
````

#### Windows
Same code, but activate the venv via Powershell or CMD

CMD:
````cmd
.\.venv\Scripts\activate.bat
````

Powershell:
````Powershell
.\.venv\Scripts\activate.ps1
````

In total for powershell:
````Powershell
python3 -m venv .venv
.\.venv\Scripts\activate.ps1
pip install -r requirements.txt
````
## Execution
Each model can be executed separately with its python file. For the execution, the cell line of the dataset has to be added as an argument. So for example, for XGBoost:
````Python
source .venv/bin/activate
python3 XGBoost.py HMEC
````
To execute all models, we have created the shell-skript main.sh. Again, the cell line needs to be added as an argument:
````Python
source .venv/bin/activate
./main.sh HMEC
````
For some models (e.g. SVM or NN), the SHAP evaluation is not finished fully. However, the shap values are saved in a .npy file
