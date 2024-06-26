# Reachability-based Trajectory Design via Exact Formulation of Implicit Neural Signed Distance Functions
[Project Page](https://roahmlab.github.io/redefined/) | [Paper](https://arxiv.org/pdf/2403.12280.pdf)

## Introduction
We propose a novel real-time, receding-horizon motion planning algorithm named Reachability-based trajectory Design via Exact Formulation of Implicit NEural signed Distance functions (REDEFINED). REDEFINED first applies offline reachability analysis to com- pute zonotope-based reachable sets that overapproximate the motion of the ego vehicle. During online planning, REDEFINED leverages zonotope arithmetic to construct a neural implicit representation that computes the exact signed distance between a parameterized swept volume of the ego vehicle and obstacle vehicles. REDEFINED then implements a novel, real-time opti- mization framework that utilizes the neural network to construct a collision avoidance constraint. REDEFINED is compared to a variety of state-of-the-art techniques and is demonstrated to successfully enable the vehicle to safely navigate through complex environments. 

## Dependencies 
We implement our simultor in MATLAB and our optimization problem in python. Dependencies on both sources are therefore required.
To collect the dependencies:

### Python
- Python packages can be installed by `conda env create -n redefined --file=environment.yml`.
- [Cyipopt](https://cyipopt.readthedocs.io/en/stable/index.html) is used to solve the trajectory optimization problems. While [cyipopt](https://cyipopt.readthedocs.io/en/stable/index.html) is included in `environment.yml`, linear solvers such as MA57 are required to be obtained following the tutorials from [IPOPT](https://coin-or.github.io/Ipopt/INSTALL.html) (See Section: **HSL (Harwell Subroutines Library)**).

### MATLAB
- [MATLAB](https://matlab.mathworks.com)(R2022b) is used to simulate the highway driving scenarios. Required toolboxes include `ROS_Toolbox`, `Mapping_Toolbox`, `Optimization_Toolbox`, `Phased_Array_System_Toolbox`, `DSP_System_Toolbox`, and `Signal_Processing_Toolbox`.
- MATLAB dependencies can be collected by running `./download-dependecies.sh` under `./REDEFINED-main` directory.
Collected dependencies include [CORA](https://tumcps.github.io/CORA/)(2018) for reachability analysis and [RTD](https://github.com/skvaskov/RTD) for reachability based trajectory design.

## Reproducing Results
### Prerequisites
1. Follow the procedures in [Dependencies](#dependencies) Section to collect all the dependencies.
2. Download the following into the top-level [`util`](https://github.com/roahmlab/REFINE/tree/main/util) directory: `lane_change_Ay_info.mat`, `dir_change_Ay_info.mat`, `car_frs.mat`, `car_frs.txt` from the data folder [here](https://drive.google.com/drive/folders/1WZbFFhCyhYQlMJxuV4caIzNoa-Q9VZkW?usp=share_link).
3. Open MATLAB to run `pyenv(ExecutionMode="OutOfProcess", Version="~/anaconda3/envs/redefined/bin/python")` so that the user can call python function from MATLAB. 
4. In MATLAB, from `./REDEFINED-main` run `install.m` to add MATLAB dependecies to MATLAB path.

### Highway Simulations
In MATLAB, from `./REDEFINED-main/simulator` run `highway_simulation.m` to run highway simulation. Note that first several planning iterations may take longer than usual - this is because pytorch [compile](https://pytorch.org/tutorials/intermediate/torch_compile_tutorial.html) is used to compile computation modules and it needs several iterations to warm up. The user can run serveral trials first for warmup and re-run the experiment without clearing the variables in MATLAB.

### Limited-Time Highway Simulations
In MATLAB, from `./REDEFINED-main/simulator` run `limited_time_highway_simulation.m` to run highway simulation under limited planning time.

### Single-Iteration Planning 
In MATLAB, from `./REDEFINED-main/simulator` run `single_planning_highway_simulation.m` to run single-iteration planning on random initial conditions of highway simulations. 

## Citation
If you use REDEFINED in an academic work, please cite using the following BibTex entry:
```
@misc{michaux2024reachabilitybased,
      title={Reachability-based Trajectory Design via Exact Formulation of Implicit Neural Signed Distance Functions}, 
      author={Jonathan Michaux and Qingyi Chen and Challen Enninful Adu and Jinsun Liu and Ram Vasudevan},
      year={2024},
      eprint={2403.12280},
      archivePrefix={arXiv},
      primaryClass={cs.RO}
}
```