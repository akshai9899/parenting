## PARENTING : Safe Reinforcement Learning from Human Input

#### Dependencies and running the code
1. Set up the dependencies - [AI Safety Grid Worlds](https://github.com/deepmind/ai-safety-gridworlds), [Pycolab](https://github.com/deepmind/pycolab), [Abseil-py](https://github.com/abseil/abseil-py)
2. Set up the path for AI Safety Grid Worlds

#### Setting up on Linux
1. Install the Abseil package
```bash
pip install absl-py
```

2. Install Pycolab from source
```bash
git clone (pycolab git directory)
python setup.py install
```

3. Get the AI Safety Grid Worlds and set up python path to it
```bash
export PYTHONPATH=/example/path/to/folder/conaining/ai-safety/gridwordls/
```

#### Setting up on Windows
1. Install the Abseil package
```cmd
pip install absl-py
```

2. Install Pycolab from source
```cmd
python setup.py install
```

You should be able to see the environment now
```cmd
python path\ai_safety_gridworlds\environments\absent_supervisor.py
```

3. Get the AI Safety Grid Worlds and set up python path to it
```cmd
set PYTHONPATH=C:\Users\Example\path\to\folder\containing\ai-safety\gridwordls\
```

You can now run the run.py file from this repository to compute the Q-values



**Issue**
>No module named '__curses'
```cmd
pip install windows-curses
