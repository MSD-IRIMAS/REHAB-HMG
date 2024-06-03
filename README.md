

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


<details><summary>Click to expand</summary>

The command to launch a training experiment and generating samples is the folowing:
```bash
python3 train_vae.py --generative-model CVAE --dataset Kimore --output-directory results/ --runs 5 --weight-rec 0.9 --weight-kl 1e-3 --epochs 2000 --device cuda

```
#### Model
- `` generative-model=CVAE``: select which generative model to use to generate samples.
#### runs
- `` runs = int``: number of times to train the model.

#### Device
- ``Device=cuda``: training with CUDA, on an automatically selected GPU (default)
- ``Device=mps``: training with MPS, training on GPU for MacOS devices with Metal programming framework.
- ``Device=cpu``: training on the CPU (not recommended)

</details>




</details>