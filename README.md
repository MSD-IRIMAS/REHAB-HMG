

<div align="center">

# HMG: Human Motion Generation
## Generating diverse Rehabilitation human motions from textual descriptions with a score.

</div>

## Installation :construction_worker:

<details><summary>Click to expand</summary>

### 1. Create conda environment

<details><summary>Instructions</summary>

```
conda create python=3.10 --name py310
conda activate py310
```
Install the following packages:
```bash

pip install torchvisison = 0.17
pip install numpy
pip install imageio
pip install pandas
pip install seaborn
pip install matplotlib
```
The code was tested on Python 3.10.13 and PyTorch 2.2

</details>


## How to generate motions:
### How to train :rocket:

<details><summary>Click to expand</summary>

The command to launch a training experiment is the folowing:
```bash
python train_vae.py [OPTIONS]
```
#### Training
- ``trainer=cuda``: training with CUDA, on an automatically selected GPU (default)
- ``trainer=mps``: training with MPS, training on GPU for MacOS devices with Metal programming framework.
- ``trainer=cpu``: training on the CPU (not recommended)

</details>




</details>