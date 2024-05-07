# Lie_nODEs
Repository for intrinsic neural ODEs on Lie groups. Currently based on the article [Optimal Potential Energy Shaping on SE(3) via Neural ODEs on Lie groups](https://arxiv.org/abs/2401.15107), showing examples of optimal potential energy shaping and damping injection control for a rigid body on the Lie group SE(3).

## Dependencies:
- [pytorch](https://github.com/pytorch/pytorch)  
  - Pytorch 2.0 or higher to include [functorch](https://github.com/pytorch/functorch/releases)
- [pytorch-lightning](https://github.com/Lightning-AI/pytorch-lightning)
- [torchdyn](https://github.com/DiffEqML/torchdyn)
- [torchcde](https://github.com/patrick-kidger/torchcde)

## Code:

### Application to learning potential energy and damping injection:
- quadratic_shaping.ipynb contains example from Section 6.1.2 
- general_shaping.ipynb contains examples from Section 6.1.3, 6.2.2 and 6.2.3

### lietorch.py 
Library for exponential maps on SO(3) and SE(3), derivative of exponential map, minimal exponential Atlas, partition of unity w.r.t. minimal expotential Atlas

### models.py 
Dynamics of rigid body on SE(3) with potential energy and damping injection

### learners.py
Wrapper for optimization of dynamics in models.py

### utils.py
Probability density on SE(3), distribution for sampling points on SE(3)

### Others (adapted from [torchdyn](https://github.com/DiffEqML/torchdyn))
- sensitivity.py
- HybridODE.py
- odeint.py

### Plots
- Generate plots by executing all_figures.py or all_figures2.py

### Included Training examples 
For various hyperparameters not necessarily agreeing with [[1]](https://arxiv.org/abs/2401.15107)
- Corresponding to Section 6.1.2: IBV_fShaping_quadratic_13_10_11:04.pt, IBV_fShaping_quadratic_26_07_10:33.pt
- Corresponding to Section 6.1.3: IBV_fShaping_20_07_16:50.pt, IBV_fShaping_21_07_10:07.pt, IBV_fShaping_25_07_13:07.pt
- Corresponding to Section 6.2.2: IBV_fShaping_04_08_10:57.pt, IBV_fShaping_07_08_17:16.pt
- Corresponding to Section 6.2.3: IBV_fShaping_A_01_09_16:46.pt, IBV_fShaping_A_18_08_12:09.pt

