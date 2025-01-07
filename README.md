# Usage
## Dependecies
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
Same code but activate the venv via Powershell or CMD

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
