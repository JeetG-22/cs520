# Running the Program
The program should be run from space.py. Answers to questions given in the writeup are contained in this file.

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

# P3 Explanation
We want to localize the target cell (figure out the exact position of the "bot") in a maze-like 2-D numpy matrix (the ship) in the least number of moves possible. We are not allowed to traverse the grid to find out where the target cell is, but rather are given some set of candidate open cells and have to figure out which one the bot is in, using the fact that we can attempt to move in a given direction, and if the movement is blocked, the bot will remain in the same cell. The bot has access to a map of the ship, is able to move in each of the cardinal directions, and starts with a set of possible locations.  
