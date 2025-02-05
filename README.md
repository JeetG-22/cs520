### Create Python Virtual Environment

In this project, we utilized Python virtual enviroments to keep pip packages consistent across all machines, while avoiding externally managed environment errors.

To create it on MacOS, we use:

```shell
source create_venv.sh
```

which implicitly runs:

```shell
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

Unfortunately, this only runs on MacOS (bash/zsh). If you are running Windows command prompt you have to type:
```shell
python3 -m venv venv
.\venv\Scripts\activate.bat
```
If running Windows PowerShell type:
```shell
python3 -m venv venv
.\venv\Scripts\Activate.ps1
```

If we want to update the requirements.txt file with the packages that we've installed with pip thus far, we can run:

```shell
pip freeze > requirements.txt
```

and then if we need to update the requirements again later:

```shell
pip install -r requirements.txt
```
