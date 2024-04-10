# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

#######
### ADAPTED FROM https://github.com/DiffEqML/torchdyn/blob/master/torchdyn/numerics/sensitivity.py
###
### Changes by Y.P. Wotte:
### Hybrid-sensitivity with callback allowing jumps in state & co-state during time integration of dynamics
#######

from inspect import getfullargspec
import torch
from torch.autograd import Function, grad
from torchcde import CubicSpline, natural_cubic_coeffs
#import sys; sys.path.append('../../')
#from torchdyn_alt.torchdyn.numerics.odeint import odeint_hybrid_alt, odeint_hybrid_raw
import sys; sys.path.append('../')
from odeint import odeint_hybrid_alt, odeint_hybrid_raw, odeint_hybrid


def generic_odeint(problem_type, vf, x, t_span, j_span, solver, callbacks, atol, rtol, dt_min=0):
    "Dispatches to appropriate `odeint` function depending on `Problem` class (Now two flavors of HybridODEProblem, previously not for hybrid systems)"
    if problem_type == 'alt':
        return odeint_hybrid_alt(vf, x, t_span, j_span, solver, callbacks, atol=atol, rtol=rtol)   
    
    elif problem_type == 'raw':
        return odeint_hybrid_raw(vf, x, t_span, j_span, solver, callbacks, atol=atol, rtol=rtol, dt_min=dt_min)


def _gather_odefunc_hybrid_adjoint_light(vf, vf_params, solver, jspan, callbacks, atol, rtol, dt_min, 
                                             solver_adjoint, jspan_adjoint, callbacks_adjoint, atol_adjoint, rtol_adjoint, dt_min_adjoint,
                                             integral_loss, problem_type ):
    # Used to be passed to generic_odeint_problem: interpolator, maxiter=4, fine_steps=4):
    "Prepares definition of autograd.Function for adjoint sensitivity analysis of the above `ODEProblem`"
    class _ODEProblemFunc(Function):
        @staticmethod
        def forward(ctx, vf_params, x, t_span, B=None):
            with torch.set_grad_enabled(True):
                t_sol, sol = generic_odeint(problem_type, vf, x, t_span, jspan, solver, callbacks, atol, rtol, dt_min)
            ctx.save_for_backward(sol, t_sol)
            return t_sol, sol

        @staticmethod
        def backward(ctx, *grad_output):
            sol, t_sol = ctx.saved_tensors
            vf_params = torch.cat([p.contiguous().flatten() for p in vf.parameters()])
            # initialize flattened adjoint state
            xT, λT, μT = sol[-1], grad_output[-1][-1], torch.zeros_like(vf_params)
            xT_nel, λT_nel, μT_nel = xT.numel(), λT.numel(), μT.numel()
            xT_shape, λT_shape, μT_shape = xT.shape, λT.shape, μT.shape


            λT_flat = λT.flatten()
            λtT = λT_flat @ vf(t_sol[-1], xT).flatten()
            # concatenate all states of adjoint system
            A = torch.cat([xT.flatten(), λT_flat, μT.flatten(), λtT[None]])

            def adjoint_dynamics(t, A):
                if len(t.shape) > 0: t = t[0]
                x, λ, μ = A[:xT_nel], A[xT_nel:xT_nel+λT_nel], A[-μT_nel-1:-1]
                x, λ, μ = x.reshape(xT.shape), λ.reshape(λT.shape), μ.reshape(μT.shape)
                with torch.set_grad_enabled(True):
                    x, t = x.requires_grad_(True), t.requires_grad_(True)
                    dx = vf(t, x)
                    #dλ, dt, *dμ = tuple(grad(dx, (x, t) + tuple(vf.parameters()), -λ,
                    #                allow_unused=True, retain_graph=False))
                    dt, *dμ = tuple(grad(dx, (t,) + tuple(vf.parameters()), -λ,
                                    allow_unused=True, retain_graph=False))
                    dλ = vf.vf.grad(x,λ)
                    if integral_loss:
                        dg = vf.integral_loss.grad(x)
                        dλ = dλ - dg
                    dμ = torch.cat([el.flatten() if el is not None else torch.zeros(1) 
                                    for el in dμ], dim=-1)

                    if dt == None: dt = torch.zeros(1).to(t)
                    if len(t.shape) == 0: dt = dt.unsqueeze(0)
                return torch.cat([dx.flatten(), dλ.flatten(), dμ.flatten(), dt.flatten()])

            # solve the adjoint equation
            
            numels = (xT_nel, λT_nel)
            shapes = (xT_shape, λT_shape)
            callbacks_adjoint[0].des_props = (numels,shapes) 
            
            n_elements = (xT_nel, λT_nel, μT_nel)
            dLdt = torch.zeros(len(t_sol)).to(xT)
            dLdt[-1] = λtT
            for i in range(len(t_sol) - 1, 0, -1):
                t_adj_sol, A = odeint_hybrid_raw(adjoint_dynamics, A, t_sol[i - 1:i + 1].flip(0), jspan_adjoint, solver_adjoint, callbacks_adjoint,
                                                 atol=atol_adjoint, rtol=rtol_adjoint,dt_min = dt_min_adjoint, event_tol = 1e-4, priority ='jump',
                                                 seminorm=(True, xT_nel))#+λT_nel))
                
                # prepare adjoint state for next interval
                #TODO: reuse vf_eval for dLdt calculations
                xt = A[-1, :xT_nel].reshape(xT_shape)
                dLdt_ = A[-1, xT_nel:xT_nel + λT_nel]@vf(t_sol[i], xt).flatten()
                A[-1, -1:] -= grad_output[0][i - 1]
                dLdt[i-1] = dLdt_

                A = torch.cat([A[-1, :xT_nel], A[-1, xT_nel:xT_nel + λT_nel], A[-1, -μT_nel-1:-1], A[-1, -1:]])
                A[xT_nel:xT_nel + λT_nel] += grad_output[-1][i - 1].flatten()

            λ, μ = A[xT_nel:xT_nel + λT_nel], A[-μT_nel-1:-1]
            λ, μ = λ.reshape(λT.shape), μ.reshape(μT.shape)
            λ_tspan = torch.stack([dLdt[0], dLdt[-1]])
            return (μ, λ, λ_tspan, None, None, None)

    return _ODEProblemFunc


# x_spline is calculated within dynamics, to avoid issues with chart-switches
def _gather_odefunc_interp_adjoint(vf, vf_params, solver, jspan, callbacks, atol, rtol, dt_min, 
                                    solver_adjoint, jspan_adjoint, callbacks_adjoint, atol_adjoint, rtol_adjoint, dt_min_adjoint,
                                    integral_loss, problem_type ):
    # Used to be passed to generic_odeint_problem: interpolator, maxiter=4, fine_steps=4):
    "Prepares definition of autograd.Function for interpolated adjoint sensitivity analysis of the above `ODEProblem`"
    class _ODEProblemFunc(Function):
        @staticmethod
        def forward(ctx, vf_params, x, t_span, B=None):
            t_sol, sol = generic_odeint(problem_type, vf, x, t_span, jspan, solver, callbacks, atol, rtol, dt_min)
            ctx.save_for_backward(sol, t_span, t_sol)
            return t_sol, sol

        @staticmethod
        def backward(ctx, *grad_output):
            sol, t_span, t_sol = ctx.saved_tensors
            vf_params = torch.cat([p.contiguous().flatten() for p in vf.parameters()])

            # initialize adjoint state
            xT, μT = sol[-1], torch.zeros_like(vf_params)
            
            #λiT = grad_output[-1][-1]
            #λT = callbacks_adjoint[0].batch_jump_to_Id(λiT,xT)
            λT = vf.vf.loss.grad_cost_vmapped(xT) #TODO: Neatly implement ad-hoc computation of the cost-gradient, possibly see where grad_output is computed
            
            
            λT_nel, μT_nel = λT.numel(), μT.numel()
            xT_shape, λT_shape, μT_shape = xT.shape, λT.shape, μT.shape
            A = torch.cat([λT.flatten(), μT.flatten()])

            # define adjoint dynamics
            def adjoint_dynamics(t, A):
                if len(t.shape) > 0: t = t[0]
                x = x_spline.evaluate(t).requires_grad_(True)
                t = t.requires_grad_(True)
                λ, μ = A[:λT_nel], A[-μT_nel:]
                λ, μ = λ.reshape(λT.shape), μ.reshape(μT.shape)
                with torch.set_grad_enabled(True):
                    dx = vf.vf.forward_light(t, x)
                    dt, *dμ = tuple(grad(dx, (t,) + tuple(vf.parameters()), -λ,
                                    allow_unused=True, retain_graph=False))
                    dλ = -vf.vf.grad_light(t,x,λ) 
                    
                    z1 = torch.zeros(1).to(x)
                    dμ = torch.cat([el.flatten() if el is not None else z1 
                                    for el in dμ], dim=-1)
                    
                    if integral_loss:
                        dg = integral_loss.grad(t,x);
                        dλ = dλ - dg
                        dt_2, *dμ_2 = tuple(grad(integral_loss.forward(t,x).sum(), (t,) + tuple(vf.parameters()),
                                        allow_unused=True, retain_graph=False))
                        dμ_2 = torch.cat([el.flatten() if el is not None else z1 
                                        for el in dμ_2], dim=-1)
                        dμ = dμ - dμ_2
                        #H = torch.inner(λ,dx).diag()+integral_loss.forward(t,x).transpose(-1,-2)
                    #if torch.isnan(dx).sum()+ torch.isnan(dλ).sum()+ torch.isnan(dμ).sum()>0:
                    #    dx = vf(t, x)
                    #    dλ = -vf.vf.grad_light(x,λ)
                    #    dμ = torch.cat([el.flatten() if el is not None else torch.zeros(1) 
                    #                for el in dμ], dim=-1)
                return torch.cat([dλ.flatten(), dμ.flatten()])

            # solve the adjoint equation
            n_elements = (λT_nel, μT_nel)
            for i in range(len(t_sol) - 1, 0, -1):
                if t_sol[i-1] == t_sol[i]:
                    continue
                
                #TODO: Make sure spline is well-defined independent of chart
                x=sol[i - 1:i + 2].permute(1, 0, 2).detach()
                t = torch.tensordot(torch.ones(x.size(0)).to(x),t_sol[i-1:i+2],dims=0)
                j = x[:,:,12]
                j[:,0] = j[:,-1]
                j[:,1] = j[:,-1] 
                #if torch.norm(x[:,0,12]-x[:,-1,12]) > 0:
                #    1+1 #tell me
                
                #q0,q1,q2 = x[0,:6],x[1,:6],x[2,:6]
                #i0,i1,i3 = x[0,12],x[1,12],x[2,12]
                #H0,H1,H2 = lie.unchart(q0, i0),lie.unchart(q1, i1),lie.unchart(q2, i2)
                
                spline_coeffs = natural_cubic_coeffs(x = callbacks[0].batch_jump_forced(t,x,j), t=t_sol[i - 1:i + 2])
                x_spline = CubicSpline(coeffs=spline_coeffs, t=t_sol[i - 1:i + 2])
            
                t_adj_sol, A = odeint_hybrid_raw(adjoint_dynamics, A, t_sol[i - 1:i + 1].flip(0), jspan_adjoint, solver_adjoint, callbacks_adjoint,
                                                 atol=atol_adjoint, rtol=rtol_adjoint,dt_min= dt_min_adjoint, event_tol = 1e-4, priority ='jump',
                                                 seminorm=(True, λT_nel)) # odeint(adjoint_dynamics, A, t_span[i - 1:i + 1].flip(0), solver, atol=atol, rtol=rtol)
                # prepare adjoint state for next interval
                A = torch.cat([A[-1, :λT_nel], A[-1, -μT_nel:]])
                A[:λT_nel] += grad_output[-1][i - 1].flatten()

            λ, μ = A[:λT_nel], A[-μT_nel:]
            λ, μ = λ.reshape(λT.shape), μ.reshape(μT.shape)
            return (μ, λ, None, None, None)

    return _ODEProblemFunc


#######
## OLD VERSIONS:
#######

# LIGHT (state) METHOD: Only final condition of forward pass for state is given to solver, dynamics solved backwards together with adjoint state
# OLD: does not use analytic gradients
def _gather_odefunc_hybrid_adjoint_light_old(vf, vf_params, solver, jspan, callbacks, atol, rtol, dt_min, 
                                             solver_adjoint, jspan_adjoint, callbacks_adjoint, atol_adjoint, rtol_adjoint, dt_min_adjoint,
                                             integral_loss, problem_type ):
    # Used to be passed to generic_odeint_problem: interpolator, maxiter=4, fine_steps=4):
    "Prepares definition of autograd.Function for adjoint sensitivity analysis of the above `ODEProblem`"
    class _ODEProblemFunc(Function):
        @staticmethod
        def forward(ctx, vf_params, x, t_span, B=None):
            with torch.set_grad_enabled(True):
                t_sol, sol = generic_odeint(problem_type, vf, x, t_span, jspan, solver, callbacks, atol, rtol, dt_min)
            ctx.save_for_backward(sol, t_sol)
            return t_sol, sol

        @staticmethod
        def backward(ctx, *grad_output):
            sol, t_sol = ctx.saved_tensors
            vf_params = torch.cat([p.contiguous().flatten() for p in vf.parameters()])
            # initialize flattened adjoint state
            xT, λT, μT = sol[-1], grad_output[-1][-1], torch.zeros_like(vf_params)
            xT_nel, λT_nel, μT_nel = xT.numel(), λT.numel(), μT.numel()
            xT_shape, λT_shape, μT_shape = xT.shape, λT.shape, μT.shape

            λT_flat = λT.flatten()
            λtT = λT_flat @ vf(t_sol[-1], xT).flatten()
            # concatenate all states of adjoint system
            A = torch.cat([xT.flatten(), λT_flat, μT.flatten(), λtT[None]])
            # WANT: A
            def adjoint_dynamics(t, A):
                if len(t.shape) > 0: t = t[0]
                x, λ, μ = A[:xT_nel], A[xT_nel:xT_nel+λT_nel], A[-μT_nel-1:-1]
                x, λ, μ = x.reshape(xT.shape), λ.reshape(λT.shape), μ.reshape(μT.shape)
                with torch.set_grad_enabled(True):
                    x, t = x.requires_grad_(True), t.requires_grad_(True)
                    dx = vf(t, x)
                    dλ, dt, *dμ = tuple(grad(dx, (x, t) + tuple(vf.parameters()), -λ,
                                    allow_unused=True, retain_graph=False))

                    if integral_loss:
                        dg = torch.autograd.grad(integral_loss(t, x).sum(), x, allow_unused=True, retain_graph=True)[0]
                        dλ = dλ - dg

                    dμ = torch.cat([el.flatten() if el is not None else torch.zeros(1) 
                                    for el in dμ], dim=-1)
                    if dt == None: dt = torch.zeros(1).to(t)
                    if len(t.shape) == 0: dt = dt.unsqueeze(0)
                return torch.cat([dx.flatten(), dλ.flatten(), dμ.flatten(), dt.flatten()])

            # solve the adjoint equation
            n_elements = (xT_nel, λT_nel)
            shape_elements = (xT_shape, λT_shape)
            callbacks_adjoint[0].des_props = (n_elements,shape_elements) 

            dLdt = torch.zeros(len(t_sol)).to(xT)
            dLdt[-1] = λtT
            for i in range(len(t_sol) - 1, 0, -1):
                t_adj_sol, A = odeint_hybrid_raw(adjoint_dynamics, A, t_sol[i - 1:i + 1].flip(0), jspan_adjoint, solver_adjoint, callbacks_adjoint,
                                                 atol=atol_adjoint, rtol=rtol_adjoint, dt_min=dt_min_adjoint, event_tol = 1e-4, priority ='jump',
                                                 seminorm=(True, xT_nel+λT_nel))
                
                # prepare adjoint state for next interval
                #TODO: reuse vf_eval for dLdt calculations
                xt = A[-1, :xT_nel].reshape(xT_shape)
                dLdt_ = A[-1, xT_nel:xT_nel + λT_nel]@vf(t_sol[i], xt).flatten()
                A[-1, -1:] -= grad_output[0][i - 1]
                dLdt[i-1] = dLdt_

                A = torch.cat([A[-1, :xT_nel], A[-1, xT_nel:xT_nel + λT_nel], A[-1, -μT_nel-1:-1], A[-1, -1:]])
                A[xT_nel:xT_nel + λT_nel] += grad_output[-1][i - 1].flatten()

            λ, μ = A[xT_nel:xT_nel + λT_nel], A[-μT_nel-1:-1]
            λ, μ = λ.reshape(λT.shape), μ.reshape(μT.shape)
            λ_tspan = torch.stack([dLdt[0], dLdt[-1]])
            return (μ, λ, λ_tspan, None, None, None)


# FULL (state) METHOD: full state-trajectory is saved after forward pass and used in backward pass and interpolation is used: taken from torchdyn, Apr 12, 2022
# OLD: this one doesn't use analytic gradients

def _gather_odefunc_interp_adjoint_old(vf, vf_params, solver, jspan, callbacks, atol, rtol, dt_min, 
                                       solver_adjoint, jspan_adjoint, callbacks_adjoint, atol_adjoint, rtol_adjoint, dt_min_adjoint,
                                       integral_loss, problem_type ):
    "Prepares definition of autograd.Function for interpolated adjoint sensitivity analysis of the above `ODEProblem`"
    class _ODEProblemFunc(Function):
        @staticmethod
        def forward(ctx, vf_params, x, t_span, B=None):
            t_sol, sol = generic_odeint(problem_type, vf, x, t_span, jspan, solver, callbacks, atol, rtol,dt_min)
            ctx.save_for_backward(sol, t_span, t_sol)
            return t_sol, sol

        @staticmethod
        def backward(ctx, *grad_output):
            sol, t_span, t_sol = ctx.saved_tensors
            vf_params = torch.cat([p.contiguous().flatten() for p in vf.parameters()])

            # initialize adjoint state
            xT, λT, μT = sol[-1], grad_output[-1][-1], torch.zeros_like(vf_params)
            λT_nel, μT_nel = λT.numel(), μT.numel()
            xT_shape, λT_shape, μT_shape = xT.shape, λT.shape, μT.shape
            A = torch.cat([λT.flatten(), μT.flatten()])

            #spline_coeffs = natural_cubic_coeffs(x=sol.permute(1, 0, 2).detach(), t=t_sol)
            #x_spline = CubicSpline(coeffs=spline_coeffs, t=t_sol)

            # define adjoint dynamics
            def adjoint_dynamics(t, A):
                if len(t.shape) > 0: t = t[0]
                x = x_spline.evaluate(t).requires_grad_(True)
                t = t.requires_grad_(True)
                λ, μ = A[:λT_nel], A[-μT_nel:]
                λ, μ = λ.reshape(λT.shape), μ.reshape(μT.shape)
                with torch.set_grad_enabled(True):
                    dx = vf(t, x)
                    dλ, dt, *dμ = tuple(grad(dx, (x, t) + tuple(vf.parameters()), -λ,
                                        allow_unused=True, retain_graph=False))

                    if integral_loss:
                        dg = torch.autograd.grad(integral_loss(t, x).sum(), x, allow_unused=True, retain_graph=True)[0]
                        dλ = dλ - dg

                    dμ = torch.cat([el.flatten() if el is not None else torch.zeros(1) 
                                    for el in dμ], dim=-1)
                return torch.cat([dλ.flatten(), dμ.flatten()])

            # solve the adjoint equation
            n_elements = (λT_nel, μT_nel)
            for i in range(len(t_span) - 1, 0, -1):
                if t_sol[i-1] != t_sol[i]:
                    spline_coeffs = natural_cubic_coeffs(x=sol[i - 1:i + 1].permute(1, 0, 2).detach(), t=t_sol[i - 1:i + 1])
                    x_spline = CubicSpline(coeffs=spline_coeffs, t=t_sol[i - 1:i + 1])
                t_adj_sol, A = odeint_hybrid_raw(adjoint_dynamics, A, t_sol[i - 1:i + 1].flip(0), jspan_adjoint, solver_adjoint, callbacks_adjoint,
                                                 atol=atol_adjoint, rtol=rtol_adjoint,dt_min=dt_min_adjoint, event_tol = 1e-4, priority ='jump',
                                                 seminorm=(True, λT_nel)) # odeint(adjoint_dynamics, A, t_span[i - 1:i + 1].flip(0), solver, atol=atol, rtol=rtol)
                # prepare adjoint state for next interval
                A = torch.cat([A[-1, :λT_nel], A[-1, -μT_nel:]])
                A[:λT_nel] += grad_output[-1][i - 1].flatten()

            λ, μ = A[:λT_nel], A[-μT_nel:]
            λ, μ = λ.reshape(λT.shape), μ.reshape(μT.shape)
            return (μ, λ, None, None, None)

    return _ODEProblemFunc
