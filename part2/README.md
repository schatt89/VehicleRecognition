# This is the second part of the solution

This part is implemented using PyTorch.

## Logs

`EXPERIMENTS_LOG.md` contains the log of experiments. 

## Environment

We used `conda` virtual environments. If you want to replicate the experimentation, you may try to install the same environment that we used for this part:

```
conda env create -f environment.yml
```

## How to run

For one-staged training (best, if want to run two-staged replace `main.py` with `main_two_stage.py`)
```
# make sure to activate conda environment (for example `conda activate openimg`)
python ./part2/main.py
```
(sorry, you have to replace the paths in `main.py`) and if you want to log the output, try to run
```
(python ./part2/main.py 2>&1) | tee /path/where/you/want/to/save/log/$(date +'%y%m%d%H%M%S').txt
```