# CoTracker3 for Tumor Tracking 

Run from the root of this repository:
```
git submodule init
git submodule update

export TORCH_HOME=${PWD}/cotracker-algorithm/torch
cd cotracker-algorithm
pixi run python download_model.py
```