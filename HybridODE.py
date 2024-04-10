# Required Imports for ODEProblem
import torch
from torch.autograd import Function
from torch import Tensor
import torch.nn as nn
from typing import Callable, Union, List
from torchdyn.core.defunc import DEFuncBase
from torchdyn.numerics.odeint import odeint, odeint_mshooting, str_to_solver
from torchdyn.core.utils import standardize_vf_call_signature

# Remaining imports for NeuralODE
from torchdyn.numerics import odeint, odeint_hybrid
from torchdyn.core.defunc import DEFunc, SDEFunc
import pytorch_lightning as pl
import warnings

# Remaining for specific hybrid implementations with adapted/removed bisection:
#import sys; sys.path.append('../../')
#from torchdyn_alt.torchdyn.numerics.odeint import odeint_hybrid_alt, odeint_hybrid_raw
import sys; sys.path.append('../'); from sensitivity import _gather_odefunc_interp_adjoint, _gather_odefunc_hybrid_adjoint_light, _gather_odefunc_interp_adjoint_old, _gather_odefunc_hybrid_adjoint_light_old, generic_odeint # last one for debugging
from odeint import odeint_hybrid_alt, odeint_hybrid_raw


#######
### Versions of ODEProblem & NeuralODE classes
#######

class ODEProblem_Hybrid(nn.Module):
    def __init__(self, vector_field:Union[Callable, nn.Module], solver:Union[str, nn.Module], callbacks, callbacks_adjoint, sensitivity:str='hybrid_adjoint_light', 
                jspan:int = 1000, jspan_adjoint:int = 1000, interpolator:Union[str, Callable, None]=None, order:int=1, atol:float=1e-6, rtol:float=1e-6, dt_min=0,
                solver_adjoint:Union[str, nn.Module, None] = None, atol_adjoint:float=1e-8, rtol_adjoint:float=1e-8, dt_min_adjoint=0, 
                seminorm:bool=False, integral_loss:Union[Callable, None]=None):
        """An ODE Problem coupling a given vector field with solver and sensitivity algorithm to compute gradients w.r.t different quantities.
        Args:
            vector_field ([Callable]): the vector field, called with `vector_field(t, x)` for `vector_field(x)`. 
                                       In the second case, the Callable is automatically wrapped for consistency
            solver (Union[str, nn.Module]): 
            callbacks ([Callable]):    EventCallback(s), each with .check_event(t,x) and .jump_map(t,x) methods
            callbacks_adjoint ([Callable]): EventCallbackAugmented with .check_event(t,z) and.jump_map(t,z) methods
            sensitivity (str, optional): Sensitivity method ['autograd', 'adjoint', 'interpolated_adjoint']. Defaults to 'hybrid_adjoint_light'.
            jspan (int, optional): Maximum number of chart-switches per trajectory. Defaults to 1000
            interpolator(optional)
            order (int, optional): Order of the ODE. Defaults to 1.
            atol (float, optional): Absolute tolerance of the solver. Defaults to 1e-6.
            rtol (float, optional): Relative tolerance of the solver. Defaults to 1e-6.
            solver_adjoint (Union[str, nn.Module, None], optional): ODE solver for the adjoint. Defaults to None.
            atol_adjoint (float, optional): Defaults to 1e-8.
            rtol_adjoint (float, optional): Defaults to 1e-8.
            seminorm (bool, optional): Indicates whether the a seminorm should be used for error estimation during adjoint backsolves. Defaults to False.
            integral_loss (Union[Callable, None]): Integral loss to optimize for. Defaults to None.
        Notes:
            Integral losses can be passed as generic function or `nn.Modules`. 
        """
        super().__init__()
        # instantiate solver at initialization
        if type(solver) == str: solver = str_to_solver(solver)
        if solver_adjoint is None:
            solver_adjoint = solver
        else: solver_adjoint = str_to_solver(solver_adjoint)

        self.solver, self.interpolator, self.atol, self.rtol = solver, interpolator, atol, rtol
        self.solver_adjoint, self.atol_adjoint, self.rtol_adjoint = solver_adjoint, atol_adjoint, rtol_adjoint
        self.sensitivity, self.integral_loss = sensitivity, integral_loss
        
        # wrap vector field if `t, x` is not the call signature
        vector_field = standardize_vf_call_signature(vector_field)

        self.vf, self.order, self.sensalg = vector_field, order, sensitivity
        if len(tuple(self.vf.parameters())) > 0:
            self.vf_params = torch.cat([p.contiguous().flatten() for p in self.vf.parameters()])
        else:
            print("Your vector field does not have `nn.Parameters` to optimize.")
            dummy_parameter = self.vf_params = nn.Parameter(torch.zeros(1))
            self.vf.register_parameter('dummy_parameter', dummy_parameter)

        # instantiates an underlying autograd.Function that overrides the backward pass with the intended version
        # sensitivity algorithm
        if self.sensalg == 'hybrid_adjoint_full':  # Full state-trajectory stored for adjoint dynamics
            self.autograd_function = _gather_odefunc_interp_adjoint(self.vf, self.vf_params, 
                                                                         solver, jspan, callbacks, atol, rtol,dt_min,
                                                                         solver_adjoint, jspan_adjoint, callbacks_adjoint, atol_adjoint, rtol_adjoint, dt_min_adjoint,
                                                                         integral_loss, 'raw').apply # 'raw' is the problem_type
        elif self.sensalg == 'hybrid_adjoint_light': # only final conditions stored for adjoint dynamics
            self.autograd_function = _gather_odefunc_hybrid_adjoint_light(self.vf, self.vf_params, 
                                                                         solver, jspan, callbacks, atol, rtol, dt_min,
                                                                         solver_adjoint, jspan_adjoint, callbacks_adjoint, atol_adjoint, rtol_adjoint, dt_min_adjoint,
                                                                         integral_loss, 'raw').apply  

    def odeint(self, x:Tensor, t_span:Tensor):
        "Returns Tuple(`t_eval`, `solution`)"
        if self.sensalg == 'autograd':
            return odeint_hybrid_raw(self.vf, x, t_span, self.jspan, self.solver, self.callbacks, self.atol, self.rtol, self.event_tol, interpolator=self.interpolator)
        else:
            return self.autograd_function(self.vf_params, x, t_span) 

    def forward(self, x:Tensor, t_span:Tensor):
        "For safety redirects to intended method `odeint`"
        return self.odeint(x, t_span) 


class NeuralODE_Hybrid(ODEProblem_Hybrid, pl.LightningModule):
    def __init__(self, vector_field:Union[Callable, nn.Module], jspan, callbacks, jspan_adjoint, callbacks_adjoint, solver:Union[str, nn.Module]='dopri5', 
                atol:float=1e-6, rtol:float=1e-6, dt_min =0, atol_adjoint:float=1e-8, rtol_adjoint:float=1e-8, dt_min_adjoint=0, integral_loss:Union[Callable, None]=None, order:int=1, 
                sensitivity='hybrid_adjoint_light', solver_adjoint:Union[str, nn.Module, None] = None, 
                interpolator:Union[str, Callable, None]=None, \
                seminorm:bool=False, return_t_eval:bool=True):
        """Generic Neural Ordinary Differential Equation.
        Args:
            vector_field ([Callable]): the vector field, called with `vector_field(t, x)` for `vector_field(x)`. 
                                       In the second case, the Callable is automatically wrapped for consistency
            solver (Union[str, nn.Module]): 
            order (int, optional): Order of the ODE. Defaults to 1.
            atol (float, optional): Absolute tolerance of the solver. Defaults to 1e-4.
            rtol (float, optional): Relative tolerance of the solver. Defaults to 1e-4.
            sensitivity (str, optional): Sensitivity method ['autograd', 'adjoint', 'interpolated_adjoint']. Defaults to 'autograd'.
            solver_adjoint (Union[str, nn.Module, None], optional): ODE solver for the adjoint. Defaults to None.
            atol_adjoint (float, optional): Defaults to 1e-6.
            rtol_adjoint (float, optional): Defaults to 1e-6.
            integral_loss (Union[Callable, None], optional): Defaults to None.
            seminorm (bool, optional): Whether to use seminorms for adaptive stepping in backsolve adjoints. Defaults to False.
            return_t_eval (bool): Whether to return (t_eval, sol) or only sol. Useful for chaining NeuralODEs in `nn.Sequential`.
        Notes:
            In `torchdyn`-style, forward calls to a Neural ODE return both a tensor `t_eval` of time points at which the solution is evaluated
            as well as the solution itself. This behavior can be controlled by setting `return_t_eval` to False. Calling `trajectory` also returns
            the solution only. 
            The Neural ODE class automates certain delicate steps that must be done depending on the solver and model used. 
            The `prep_odeint` method carries out such steps. Neural ODEs wrap `ODEProblem`.
        """
        super().__init__(vector_field=standardize_vf_call_signature(vector_field, order, defunc_wrap=True), jspan = jspan, callbacks = callbacks, jspan_adjoint = jspan_adjoint,
                         callbacks_adjoint= callbacks_adjoint, order=order, sensitivity=sensitivity,
                         solver=solver, atol=atol, rtol=rtol, dt_min=dt_min, solver_adjoint=solver_adjoint, atol_adjoint=atol_adjoint, rtol_adjoint=rtol_adjoint, 
                         dt_min_adjoint=dt_min_adjoint, seminorm=seminorm, interpolator=interpolator, integral_loss=integral_loss)     
        
        self.u, self.controlled, self.t_span = None, False, None # data-control conditioning
        self.return_t_eval = return_t_eval
        if integral_loss is not None: self.vf.integral_loss = integral_loss
        self.vf.sensitivity = sensitivity
        # for debugging:
        self.callbacks = callbacks
        self.f = vector_field

    def _prep_integration(self, x:Tensor, t_span:Tensor) -> Tensor:
        "Performs generic checks before integration. Assigns data control inputs and augments state for CNFs"

        # assign a basic value to `t_span` for `forward` calls that do no explicitly pass an integration interval
        if t_span is None and self.t_span is None: t_span = torch.linspace(0, 1, 2)
        elif t_span is None: t_span = self.t_span

        # loss dimension detection routine; for CNF div propagation and integral losses w/ autograd
        excess_dims = 0
        if (not self.integral_loss is None) and self.sensitivity == 'autograd':
            excess_dims += 1

        # handle aux. operations required for some jacobian trace CNF estimators e.g Hutchinson's
        # as well as datasets-control set to DataControl module
        for _, module in self.vf.named_modules():
            if hasattr(module, 'trace_estimator'):
                if module.noise_dist is not None: module.noise = module.noise_dist.sample((x.shape[0],))
                excess_dims += 1

            # data-control set routine. Is performed once at the beginning of odeint since the control is fixed to IC
            if hasattr(module, 'u'):
                self.controlled = True
                module.u = x[:, excess_dims:].detach()
        return x, t_span

    def forward(self, x:Tensor, t_span:Tensor=None):
        # generic_odeint('raw',self.f,x,t_span,10,self.solver,self.callbacks,1e-3,1e-3) # executes without flaw, error happens in line 162
        x, t_span = self._prep_integration(x, t_span)
        t_eval, sol =  super().forward(x, t_span) # changed from super().forward(x,t_span) to self.big_forward(x, t_span), also renaming forward in super() class to big_forward 
        if self.return_t_eval: return t_eval, sol
        else: return sol

    def trajectory(self, x:torch.Tensor, t_span:Tensor):
        x, t_span = self._prep_integration(x, t_span)
        _, sol = odeint(self.vf, x, t_span, solver=self.solver, atol=self.atol, rtol=self.rtol)
        return sol

    def __repr__(self):
        npar = sum([p.numel() for p in self.vf.parameters()])
        return f"Neural ODE:\n\t- order: {self.order}\
        \n\t- solver: {self.solver}\n\t- adjoint solver: {self.solver_adjoint}\
        \n\t- tolerances: relative {self.rtol} absolute {self.atol}\
        \n\t- adjoint tolerances: relative {self.rtol_adjoint} absolute {self.atol_adjoint}\
        \n\t- num_parameters: {npar}\
        \n\t- NFE: {self.vf.nfe}"