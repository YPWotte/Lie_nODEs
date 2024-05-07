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
- general_shaping.ipynb
- general_shaping_2.ipynb
- general_shaping_3.ipynb
- general_shaping_4.ipynb

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
