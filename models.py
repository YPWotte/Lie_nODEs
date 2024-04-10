import torch
import torch.nn as nn
from torch.autograd import Function

import lie_torch as lie
import time
from functorch import vmap, grad, jacrev

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
eye3 = torch.eye(3).to(device)
eye4 = torch.eye(4).to(device)
eye6 = torch.eye(6).to(device)
z1 = torch.zeros(1).to(device)
z3 = torch.zeros(3).to(device)
z6 = torch.zeros(6).to(device)
z66 = torch.zeros((6,6)).to(device)
z66eye6 = torch.cat((z66,eye6),1).to(device)

#######
### SE3_Dynamics Module, Augmented_SE3 including running cost, varios implementations for 6-by-6 damping (pos.def.sym.) matrix implementations for Damping Injection
#######

class se3_Dynamics(nn.Module):
    # Dynamics and gradients at algebra, as much as is reasonable
    def __init__(self,I,B,V,target):   
        super().__init__()
        self.I = I                                      # Diagonal inertia tensor I
        self.I_inv = torch.diag(torch.diag(I).pow_(-1)) # inverse of inertia tensor
        self.Vnet = V
        self.Bnet = B
        self.loss = target

    def forward(self, t, x):
        # Batched forward dynamics 
        t_batch = t*torch.ones(x.shape[0]).to(device)
        return vmap(self.forward_single)(t_batch,x)
          
    def forward_single(self, t, x): 
        # Unbatched forward dynamics: expects x one at a time
        qi, P, i = x[:6], x[6:12], x[12] 
        T = self.I_inv @ P
        H = lie.unchart(qi,i)  
        
        # Dynamics:
        dqi = self.K_Inv(H,i)@T
        dP = lie.ad(T).transpose(-1,-2)@P + self.W(H,P) 
        di = z1
        
        return torch.cat((dqi,dP,di),-1) 
    
    def forward_body_chart_single(self, t, x): #TODO: Express in body chart
        # Unbatched forward dynamics: expects x one at a time
        qi, P, i = x[:6], x[6:12], x[12] 
        T = self.I_inv @ P
        H = lie.unchart(qi,i)  
        q_free = qi.detach().clone()
        # Dynamics:
        dqi = torch.inverse(eye6-1/2*lie.ad(qi-q_free))@T;
        dP = lie.ad(T).transpose(-1,-2)@P - self.DampingInjection(H,P) - torch.inverse(eye6-1/2*lie.ad(qi-q_free)).transpose(-1,-2)@self.PotentialGradient(H) 
        di = z1
        
        return torch.cat((dqi,dP,di),-1) 
    
    def forward_light(self, t, x):
        # Batched "Light" forward dynamics: see forward_light_single for details 
        t_batch = t*torch.ones(x.shape[0]).to(device)
        return vmap(self.forward_light_single)(t_batch,x)
          
    def forward_light_single(self, t, x): 
        # Unbatched "Light" forward dynamics: change-rate dH is only returned as twist T at algebra, not as dqi in chart
        qi, P, i = x[:6], x[6:12], x[12] 
        T = self.I_inv @ P
        H = lie.unchart(qi,i)  

        # Dynamics:
        dP = lie.ad(T).transpose(-1,-2)@P + self.W(H,P) #
        di = z1
        
        return torch.cat((T,dP,di),-1) 
    
    def grad_light(self, t, x, λ):
        # Batched gradient of λ(f), with f the dynamics at algebra level
        t_batch = t*torch.ones(x.shape[0]).to(device)  # formality
        return vmap(self.grad_light_single)(t_batch, x, λ)
    
    def grad_light_single(self, t, x, λ):
        qi, P, i = x[:6], x[6:12], x[12] 
        λT, λP, λi = λ[:6], λ[6:12], λ[12]        
        T = self.I_inv @ P
        
        ## 1. Gradients w.r.t. H (and pulled back to dual of algebra) and P
        # 1.1 Gradient of λT(dqi)
        dqi_H = T*0; #  torch.einsum('j,ij->i', λT, torch.einsum('ijk,k->ij',-1/2*vmap(lie.ad)(eye6),T))
        dqi_P = torch.einsum('j,ij-> i', λT, self.I_inv)
        
        # 1.2 Gradient of input wrench λP(W)
        dW_HP = self.dW(x)
        dW_H = dW_HP[:6]
        dW_P = dW_HP[6:]
        
        # 1.3 Gradient of λP(dP)
        dP_H = torch.einsum('j,ij->i', λP, dW_H) 
        adP_P = torch.einsum('k, jk', λP, vmap(lie.ad)(self.I_inv).transpose(-1,-2)@P+ lie.ad(T)) 
        dP_P = adP_P + torch.einsum('k,jk', λP , dW_P) #Fixed - to +
        
        # 1.4 Full gradient of λ(f)
        dfλ = torch.cat((dqi_H + dP_H - lie.ad(T).transpose(-1,-2)@λT, dqi_P + dP_P, z1),-1) # TODO: check the following code: lambda_f_ = lambda x: self.forward_single(t,x); lambda_dot = -grad(lambda_f_)(x); lambda_dot and -dfλ should differ by 1/2*lie.ad(T).transpose(-1,-2)@λT, if all is correct
        # lambda_f_ = lambda x: torch.einsum('i,i',λ,self.forward_body_chart_single(i,x));
        # dfλ_num = grad(lambda_f_)(x)
        # dfλ_num[:6] = dfλ_num[:6] - 1/2*lie.ad(T).transpose(-1,-2)@λT
        # diff = dfλ - dfλ_num
        # # diff[:6] = diff[:6] - 1/2*lie.ad(T).transpose(-1,-2)@λT
        return dfλ
    
    def W(self,H,P):
        # Full input wrench
        return - self.DampingInjection(H,P) - self.PotentialGradient(H)
  
    def dW(self,x):
        # Gradient of input wrench at algebra level
        qi, P, i = x[:6], x[6:12], x[12] 
        H = lie.unchart(qi,i)  
        q =  qi*0 
        
        # 1. dB
        qP = torch.cat((q,P),-1)
        qP.requires_grad_(True)
        B_ = lambda qP: self.DampingInjection(H@(eye4+lie.skew_se3(qP[:6])),qP[6:])
        dB = jacrev(B_)(qP) # size [6,12], differentiated in second index

        # 2. ddV: check this
        # dV_num = lambda q: torch.matmul((eye6+1/2*lie.ad(q)).transpose(-1,-2),grad(V_)(q));
        # ddV_num = jacrev(dV_num)(q)
        # ddV[:6,:6] - ddV_num 
        q.requires_grad_(True) 
        V_ = lambda q: self.Potential(H @ ( eye4 + lie.skew_se3(q) + 1/2 *lie.skew_se3(q) @ lie.skew_se3(q)  ) ) 
        dV = grad(V_)(q)
        ddV = jacrev(grad(V_))(q) - 1/2 *torch.einsum('ijk,j-> ki',vmap(lie.ad)(eye6),dV)
        ddV = torch.cat((ddV,z66),-1) # differentiation likewise in second index

        # 3. Output
        dW = -dB-ddV
        return dW.transpose(-1,-2) # differentiation in first index, contract with costate along second index 

    def K_Inv(self, H,i):
        # Wrapper for inverse of derivative of exponential map
        return lie.K2_Inv(lie.chart(H,i))
    
    def DampingInjection(self,H,P):
        # Method with PosDefSym() only a layer in self.B:
        return torch.squeeze(self.Bnet(torch.cat((H[:3,:].reshape(12),P),-1)))@P
    
    def Potential(self,H):
        # Wrapper for potential function
        return torch.squeeze(self.Vnet(H[:3,:].reshape(12)))
    
    def PotentialGradient(self, H):
        # Gradient of potential function, at algebra level
        q =  lie.chart(H,0)*0 
        with torch.set_grad_enabled(True):
            q.requires_grad_(True) 
            V_ = lambda q: self.Potential(H@(eye4+lie.skew_se3(q)))
            dV = grad(V_)(q)
        return dV
    
    def Energy(self, x):
        qi, P, i = x[:6], x[6:12], x[12] 
        H = lie.unchart(qi,i)  
        E_Pot = self.Potential(H);
        E_Kin = 1/2*P@self.I_inv@P;
        E_Kin_rot = 1/2*P[:3]@self.I_inv[:3,:3]@P[:3]
        E_Kin_trans = E_Kin - E_Kin_rot
        E_Total = E_Kin + E_Pot
        return E_Total, E_Pot, E_Kin, E_Kin_rot, E_Kin_trans

class se3_Dynamics_2(nn.Module):
    # Dynamics and gradients at algebra, as much as is reasonable. Version 2 has an extra potential only for distance.
    def __init__(self,I,B,V1,V2,target):   
        super().__init__()
        self.I = I                                      # Diagonal inertia tensor I
        self.I_inv = torch.diag(torch.diag(I).pow_(-1)) # inverse of inertia tensor
        self.Vnet_1 = V1
        self.Vnet_2 = V2
        self.Bnet = B
        self.loss = target

    def forward(self, t, x):
        # Batched forward dynamics 
        t_batch = t*torch.ones(x.shape[0]).to(device)
        return vmap(self.forward_single)(t_batch,x)
          
    def forward_single(self, t, x): 
        # Unbatched forward dynamics: expects x one at a time
        qi, P, i = x[:6], x[6:12], x[12] 
        T = self.I_inv @ P
        H = lie.unchart(qi,i)  
        
        # Dynamics:
        dqi = self.K_Inv(H,i)@T
        dP = lie.ad(T).transpose(-1,-2)@P + self.W(H,P) 
        di = z1
        
        return torch.cat((dqi,dP,di),-1) 
    
    def forward_light(self, t, x):
        # Batched "Light" forward dynamics: see forward_light_single for details 
        t_batch = t*torch.ones(x.shape[0]).to(device)
        return vmap(self.forward_light_single)(t_batch,x)
          
    def forward_light_single(self, t, x): 
        # Unbatched "Light" forward dynamics: change-rate dH is only returned as twist T at algebra, not as dqi in chart
        qi, P, i = x[:6], x[6:12], x[12] 
        T = self.I_inv @ P
        H = lie.unchart(qi,i)  

        # Dynamics:
        dP = lie.ad(T).transpose(-1,-2)@P + self.W(H,P) #
        di = z1
        
        return torch.cat((T,dP,di),-1) 
    
    def grad_light(self, t, x, λ):
        # Batched gradient of λ(f), with f the dynamics at algebra level
        t_batch = t*torch.ones(x.shape[0]).to(device)  # formality
        return vmap(self.grad_light_single)(t_batch, x, λ)
    
    def grad_light_single(self, t, x, λ):
        qi, P, i = x[:6], x[6:12], x[12] 
        λT, λP, λi = λ[:6], λ[6:12], λ[12]        
        T = self.I_inv @ P
        
        ## 1. Gradients w.r.t. H (and pulled back to dual of algebra) and P
        # 1.1 Gradient of λT(dqi)
        dqi_H = T*0; #  torch.einsum('j,ij->i', λT, torch.einsum('ijk,k->ij',-1/2*vmap(lie.ad)(eye6),T))
        dqi_P = torch.einsum('j,ij-> i', λT, self.I_inv)
        
        # 1.2 Gradient of input wrench λP(W)
        dW_HP = self.dW(x)
        dW_H = dW_HP[:6]
        dW_P = dW_HP[6:]
        
        # 1.3 Gradient of λP(dP)
        dP_H = torch.einsum('j,ij->i', λP, dW_H) 
        adP_P = torch.einsum('k, jk', λP, vmap(lie.ad)(self.I_inv).transpose(-1,-2)@P+ lie.ad(T)) 
        dP_P = adP_P + torch.einsum('k,jk', λP , dW_P) #Fixed - to +
        
        # 1.4 Full gradient of λ(f)
        dfλ = torch.cat((dqi_H + dP_H - lie.ad(T).transpose(-1,-2)@λT, dqi_P + dP_P, z1),-1)
        return dfλ
    
    def W(self,H,P):
        # Full input wrench
        return - self.DampingInjection(H,P) - self.PotentialGradient(H)
  
    def dW(self,x):
        # Gradient of input wrench at algebra level
        qi, P, i = x[:6], x[6:12], x[12] 
        H = lie.unchart(qi,i)  
        q =  qi*0 
        
        # 1. dB
        qP = torch.cat((q,P),-1)
        qP.requires_grad_(True)
        B_ = lambda qP: self.DampingInjection(H@(eye4+lie.skew_se3(qP[:6])),qP[6:])
        dB = jacrev(B_)(qP) # size [6,12], differentiated in second index

        # 2. ddV:
        q.requires_grad_(True) 
        V_ = lambda q: self.Potential(H @ ( eye4 + lie.skew_se3(q) + 1/2 *lie.skew_se3(q) @ lie.skew_se3(q)  ) ) 
        dV = grad(V_)(q)
        ddV = jacrev(grad(V_))(q) - 1/2 *torch.einsum('ijk,j-> ki',vmap(lie.ad)(eye6),dV)
        ddV = torch.cat((ddV,z66),-1) # differentiation likewise in second index

        # 3. Output
        dW = -dB-ddV
        return dW.transpose(-1,-2) # differentiation in first index, contract with costate along second index 

    def K_Inv(self, H,i):
        # Wrapper for inverse of derivative of exponential map
        return lie.K2_Inv(lie.chart(H,i))
    
    def DampingInjection(self,H,P):
        # Method with PosDefSym() only a layer in self.B:
        return torch.squeeze(self.Bnet(torch.cat((H[:3,:].reshape(12),P),-1)))@P
    
    def Potential(self,H):
        # Wrapper for potential function
        return torch.squeeze(self.Vnet_1(H[:3,:].reshape(12)))+torch.squeeze(self.Vnet_2(H[:3,3].reshape(3)))
    
    def PotentialGradient(self, H):
        # Gradient of potential function, at algebra level
        q =  lie.chart(H,0)*0 
        with torch.set_grad_enabled(True):
            q.requires_grad_(True) 
            V_ = lambda q: self.Potential(H@(eye4+lie.skew_se3(q)))
            dV = grad(V_)(q)
        return dV

    
    
class SE3_Dynamics(nn.Module):
    # Expresses potential and damping injection as sum of chart-components, performing all computations in charts
    # Computationally inferior to se3_Dynamics, which moves computations to the Lie algebra
    def __init__(self,I,B,V,C=2):   
        super().__init__()
        self.I = I    # Diagonal inertia tensor I
        self.I_inv = torch.diag(torch.diag(I).pow_(-1)) # inverse of inertia tensor
        
        self.set_PoU(C) # Continuity of PoU (2 or inf)
        self.set_V(V)
        self.set_B(B)
        
        
    def forward(self, t, x):
        t_batch = t*torch.ones(x.shape[0]).to(device)
        return vmap(self.forward_single)(t_batch,x)
          
    def forward_single(self, t, x):
        # expects x one at a time
        #x = torch.squeeze(x) # indexing without squeeze: x[...,:6], x[...,6:12].transpose(-1,-2), x[12]
        qi, P, i = x[:6], x[6:12], x[12] 
        T = self.I_inv @ P
        H = lie.unchart(qi,i)  
        dqi = self.K_Inv(H,i)@T

        dP = lie.ad(T).transpose(-1,-2)@P + self.W(H,P) # quicker gradient by using known K matrices, check if it is indeed quicker   
        
        di = z1
        return torch.cat((dqi,dP,di),-1) #torch.unsqueeze(,0)
    
    def forward_light(self, t, x):
        t_batch = t*torch.ones(x.shape[0]).to(device)
        return vmap(self.forward_light_single)(t_batch,x)
          
    def forward_light_single(self, t, x):
        # expects x one at a time
        #x = torch.squeeze(x) # indexing without squeeze: x[...,:6], x[...,6:12].transpose(-1,-2), x[12]
        qi, P, i = x[:6], x[6:12], x[12] 
        T = self.I_inv @ P
        H = lie.unchart(qi,i)  

        dP = lie.ad(T).transpose(-1,-2)@P + self.W(H,P) #
        
        di = z1
        return torch.cat((T,dP,di),-1) #torch.unsqueeze(,0)
    
    def grad_light(self, x, λ):
        return vmap(self.grad_light_single)(x, λ)
    
    def grad_light_single(self, x, λ):
        qi, P, i = x[:6], x[6:12], x[12] 
        λT, λP, λi = λ[:6], λ[6:12], λ[12]        
        T = self.I_inv @ P
        #H_I_B = lie.unchart(qi,i)  
        
        dqi_H = z1*T 
        dqi_P = torch.einsum('i,ij-> j', λT, self.I_inv)
        
        dW_HP = self.dW(x)
        dW_H = dW_HP[:6]
        dW_P = dW_HP[6:]
        
        dP_H = torch.einsum('j,ij->i', λP, dW_H) 
        adP_P = torch.einsum('k, jk', λP, vmap(lie.ad)(self.I_inv).transpose(-1,-2)@P+ lie.ad(T)) 
        dP_P = adP_P - torch.einsum('k,jk', λP , dW_P) 
        dfλ = torch.cat((dqi_H + dP_H + lie.ad(T).transpose(-1,-2)@λT, dqi_P + dP_P, z1),-1);
        
        #dfλ = x;  
        return dfλ
    
    def W(self,H,P):
        # Total input wrench
        return - self.DampingInjection(H,P) - self.PotentialGradient(H)
        
  
    def dW(self,x):
        qi, P, i = x[:6], x[6:12], x[12] 
        H = lie.unchart(qi,i)  
        
        PoU = self.PoU(H)
        dPoU = self.dPoU(H)
        ddPoU= self.ddPoU(H)

        dW = 0*torch.tensordot(x[:12],P,dims=0)
        for i in range(4):
            qi = lie.chart(H,i)
            KiTinv = lie.K2_Inv(qi).transpose(-1,-2)
            
            #dB
            Ai = torch.cat((torch.cat((KiTinv,z66),1),z66eye6))
            qiP = torch.cat((qi,P),-1)
            qiP.requires_grad_(True)
            Bi = self.Bi(i)(qiP) 
            dBi = jacrev(self.Bi(i))(qiP) # size [6,12], differentiated in second index
            dBsi = PoU[i]*dBi + torch.tensordot(Bi,torch.cat((dPoU[i],z6),-1),dims=0)  # differentiated in second index
            
            #ddV
            dK_q_TInv_i = lie.dK2_q_Inv(qi).transpose(-1,-2) # size [6,6,6], differentiated in first index
            qi.requires_grad_(True)
            
            Vi_unsqueezed = self.Vi(i)
            def Vi_(qi): return torch.squeeze(Vi_unsqueezed(qi))
            Vi = Vi_(qi)
            dVi = grad(Vi_)(qi)
            Vsi = PoU[i]*Vi
            dVsi = (PoU[i]*dVi+dPoU[i]*Vi)
            ddVsi = (PoU[i]*jacrev(grad(Vi_))(qi) + Vi*ddPoU[i] + 2*lie.sym(torch.tensordot(dVi,dPoU[i],dims=0))) # [6,6], symmetric, so differentiated in either index
            ddVsi = torch.einsum('ij,jd->id', KiTinv, ddVsi) + torch.einsum('djk,k->jd', dK_q_TInv_i, dVsi) # [6,6], differentiated in second index
            ddVsi = torch.cat((ddVsi,z66),-1)
            dW += torch.einsum('ij,kj->ik', Ai, -dBsi-ddVsi) # "differentiated" in first index, contract with co-state along second

        return dW 
    
    def K(self, H,i):
        # derivative of exponential map
        return lie.K2(lie.chart(H,i))
    
    def K_Inv(self, H,i):
        # inverse of derivative of exponential map
        return lie.K2_Inv(lie.chart(H,i))
    
    def DampingInjection(self,H,P):
        # Method with PosDefSym() only a layer in self.B:
        return self.B(H,P)
    
    def PotentialGradient(self, H):
        dPoU = self.dPoU(H)
        PoU = self.PoU(H)
        q =  torch.stack((lie.chart(H,0), lie.chart(H,1), lie.chart(H,2), lie.chart(H,3))) # intended: qi is the i-th row. Alternative: np.concatenate((np.expand_dims(lie.chart(H,0),0), np.expand_dims(lie.chart(H,1),0)),0) )
        with torch.set_grad_enabled(True):
            dV = 0*dPoU[0,:]
            for i in range(4):
                Ki_Inv = self.K_Inv(H,i) # Slowest Step
                Vi_unsqueezed = self.Vi(i); 
                def Vi_(qi): return torch.squeeze(Vi_unsqueezed(qi))
                qi = q[i,:];
                qi.requires_grad_(True)
                dVi = grad(Vi_)(qi) 
                dV +=  Ki_Inv.transpose(-1, -2) @ (PoU[i]*dVi + dPoU[i,:]*Vi_(q[i,:]))
        return dV
    
    def set_PoU(self,C):
        self.C = C
        if self.C == 2:
            self.PoU_q = lambda q: lie.PoU_C2q(q)   # returns PoU in current chart
            self.PoU = lambda H: lie.PoU_C2(H)      # returns PoU in all charts
            self.dPoU = lambda H: lie.dPoU_C2(H)    # returns dPoU for all charts (chart i differentiated w.r.t. q_i)
            self.ddPoU = lambda H: lie.ddPoU_C2(H)  # returns ddPoU for all charts (hessian of PoU in chart i)
        else:
            self.PoU_q = lambda q: lie.PoU_CInfq(q)
            self.PoU = lambda H: lie.PoU_CInf(H)
            self.dPoU = lambda H: lie.dPoU_CInf(H)
            self.ddPoU = lambda H: lie.ddPoUC_Inf(H) # not implemented yet
        return 0
    
    def set_V(self,V):
        self.V0 = V[0]; self.V1 = V[1]; self.V2 = V[2]; self.V3 = V[3]
        self.Vs0 = lambda x: torch.squeeze(self.PoU_q(x[:6])*self.V0(x[:6]))
        self.Vs1 = lambda x: torch.squeeze(self.PoU_q(x[:6])*self.V1(x[:6]))
        self.Vs2 = lambda x: torch.squeeze(self.PoU_q(x[:6])*self.V2(x[:6]))
        self.Vs3 = lambda x: torch.squeeze(self.PoU_q(x[:6])*self.V3(x[:6]))
        self.V = lambda H: lie.F_Lie_free(H,self.Vs0,self.Vs1,self.Vs2,self.Vs3)
        return 0
    
    def set_B(self,B):
        self.B0 = lambda x: (B[0])(x[:12])@x[6:12]; #self.B1 = B[1]; self.B2 = B[2]; self.B3 = B[3]; 
        self.B1 = lambda x: (B[1])(x[:12])@x[6:12];
        self.B2 = lambda x: (B[2])(x[:12])@x[6:12];
        self.B3 = lambda x: (B[3])(x[:12])@x[6:12];
        
        self.Bs0 = lambda x: self.PoU_q(x[:6])*self.B0(x[:12])
        self.Bs1 = lambda x: self.PoU_q(x[:6])*self.B1(x[:12])
        self.Bs2 = lambda x: self.PoU_q(x[:6])*self.B2(x[:12])
        self.Bs3 = lambda x: self.PoU_q(x[:6])*self.B3(x[:12])
        self.B = lambda H,P: lie.F_Lie_3_free(H,P,self.Bs0,self.Bs1,self.Bs2,self.Bs3)
        return 0
    
    def Vi(self, i):
        if i == 0:
            return self.V0
        elif i == 1:
            return self.V1
        elif i == 2: 
            return self.V2
        elif i == 3:
            return self.V3
        else:
            return self.V
        
    def Bi(self,i):
        if i == 0:
            return self.B0
        elif i == 1:
            return self.B1
        elif i == 2: 
            return self.B2
        elif i == 3:
            return self.B3
        else:
            return self.V

class SE3_Quadratic(nn.Module):
    # Class for quadratic control of R. Rashad et al.
    def __init__(self,I,Kd,Go,Gt,H_I_D):   
        super().__init__()
        self.I = I    # Diagonal inertia tensor I
        self.I_inv = torch.diag(torch.diag(I).pow_(-1)) # inverse of inertia tensor
        self.Kd = Kd    # damping injection
        self.Go = Go    # rotational co-stiffness matrix
        self.Gt = Gt    # translational co-stiffness matrix
        self.H_I_D = H_I_D # Destination frame D as seen from global reference I
        self.H_CS_B = eye4  # start in motion control-mode, call Change_Mode(Tool_Frame) for interaction control, Change_Mode(Identity) tor revert back to body-frame
        self.Ad_CS_B = eye6 
        
    def forward(self, t, x):
        self.Kd.to(device); self.Go.to(device); self.Gt.to(device)
        t_batch = t*torch.ones(x.shape[0]).to(device)
        return vmap(self.forward_single)(t_batch,x)
          
    def forward_single(self, t, x):
        # expects x one at a time
        #x = torch.squeeze(x) # indexing without squeeze: x[...,:6], x[...,6:12].transpose(-1,-2), x[12]
        qi, P, i = x[:6], x[6:12], x[12] 
        T = self.I_inv @ P
        H_I_B = lie.unchart(qi,i)  
        H_D_CS = lie.inv_SE3(self.H_I_D)@H_I_B@lie.inv_SE3(self.H_CS_B)
        dqi = self.K_Inv(H_I_B,i)@T

        dP = lie.ad(T).transpose(-1,-2)@P + self.W_di(T) + self.Ad_CS_B.transpose(-1,-2)@(self.W_p(H_D_CS)-self.W_grv()+self.W_grv_real()) 
        
        di = z1
        return torch.cat((dqi,dP,di),-1) #torch.unsqueeze(,0)
    
    def grad(self, x, λ):
        return vmap(self.grad_single)(x, λ)
    
    def grad_single(self, x, λ):
        qi, P, i = x[:6], x[6:12], x[12] 
        λqi, λP, λi = λ[:6], λ[6:12], λ[12]        
        T = self.I_inv @ P
        H_I_B = lie.unchart(qi,i)  
        H_D_CS = lie.inv_SE3(self.H_I_D)@H_I_B@lie.inv_SE3(self.H_CS_B)
        dH_D_CS = lie.inv_SE3(self.H_I_D)@lie.dH_q(H_I_B, qi)@lie.inv_SE3(self.H_CS_B)
        
        dqi_qi = torch.einsum('j,ijk -> ik', λqi, lie.dK_q_Inv(qi))@T #VERY SLOW: but agrees with numerical differentation (0.15s)
        dqi_P = torch.einsum('j,ij-> i', λqi, (self.K_Inv(H_I_B,i)@self.I_inv).transpose(-1,-2)) # SLOW: but agrees with numerical differentiation (0.01s)
        dP_qi = torch.einsum('j,ij->i', λP, self.dW_p(H_D_CS,dH_D_CS)) # SLOW (0.05s): but agrees with numerical differetiation
        adP_P = torch.einsum('k, jk', λP, vmap(lie.ad)(self.I_inv).transpose(-1,-2)@P+ lie.ad(T)) # agrees with numerical differentiation
        dP_P = adP_P - torch.einsum('k,jk', λP , (self.Kd(z1)@self.I_inv).transpose(-1,-2)) # agrees with numerical differentiation

        dfλ = torch.cat((dqi_qi + dP_qi, dqi_P + dP_P, z1),-1);
        # to check: 
        # r = lambda i:(self.forward_single( 0, x+(1e-3)*torch.eye(13)[i,:])@λ - self.forward_single( 0, x-(1e-3)*torch.eye(13)[i,:])@λ)/2e-3
        # torch.stack((r(0),r(1),r(2),r(3),r(4),r(5),r(6),r(7),r(8),r(9),r(10),r(11))
        
        return -dfλ
    
    def K_Inv(self, H,i):
        # inverse of derivative of exponential map
        return lie.K2_Inv(lie.chart(H,i))
    
    def W_di(self,T):
        #print(self.Kd.get_device())
        Kd = self.Kd(z1)  
        return - Kd@T
    
    def W_p(self,H_D_CS):
        zero = z1
        R_D_CS = H_D_CS[:3,:3]
        R_CS_D = R_D_CS.transpose(-1,-2)
        x_D_CS = H_D_CS[:3,3]
        t_skew = -2*lie.sk(self.Go(zero)@R_D_CS) - lie.sk(self.Gt(zero)@R_CS_D@lie.skew2_so3(x_D_CS)@R_D_CS)
        f_skew = -R_CS_D@lie.sk(self.Gt(zero)@lie.skew_so3(x_D_CS))@R_D_CS - lie.sk(self.Gt(zero)@R_CS_D@lie.skew_so3(x_D_CS)@R_D_CS)
        t = lie.unskew_so3(t_skew)
        f = lie.unskew_so3(f_skew)
        return torch.cat((t,f),-1)
    
    def dW_p(self,H_D_CS,dH_D_CS):
        zero = z1
        R_D_CS = H_D_CS[:3,:3]
        R_CS_D = R_D_CS.transpose(-1,-2)
        x_D_CS = H_D_CS[:3,3]
        dR_D_CS = dH_D_CS[...,:3,:3]
        dx_D_CS = dH_D_CS[...,:3,3]
        dR_CS_D = dR_D_CS.transpose(-1,-2)
        
        dt_skew = -2*lie.sk(self.Go(zero)@dR_D_CS) - lie.sk(self.Gt(zero)@dR_CS_D@lie.skew2_so3(x_D_CS)@R_D_CS) \
            - lie.sk(self.Gt(zero)@R_CS_D@lie.skew_so3_many(dx_D_CS)@lie.skew_so3(x_D_CS)@R_D_CS) - lie.sk(self.Gt(zero)@R_CS_D@lie.skew_so3(x_D_CS)@lie.skew_so3_many(dx_D_CS)@R_D_CS) - lie.sk(self.Gt(zero)@R_CS_D@lie.skew2_so3(x_D_CS)@dR_D_CS)
        df_skew = -dR_CS_D@lie.sk(self.Gt(zero)@lie.skew_so3(x_D_CS))@R_D_CS -R_CS_D@lie.sk(self.Gt(zero)@lie.skew_so3_many(dx_D_CS))@R_D_CS -R_CS_D@lie.sk(self.Gt(zero)@lie.skew_so3(x_D_CS))@dR_D_CS \
            - lie.sk(self.Gt(zero)@dR_CS_D@lie.skew_so3(x_D_CS)@R_D_CS) - lie.sk(self.Gt(zero)@R_CS_D@lie.skew_so3_many(dx_D_CS)@R_D_CS) - lie.sk(self.Gt(zero)@R_CS_D@lie.skew_so3(x_D_CS)@dR_D_CS)
        
        dt = vmap(lie.unskew_so3)(dt_skew)
        df = vmap(lie.unskew_so3)(df_skew)
        return torch.cat((dt,df),-1)

    def W_grv(self):
        # compensation term for gravity: perfect cancellation assumed
        return z6
    
    def W_grv_real(self):
        # actual gravity: perfect cancellation assumed
        return z6
    
    def Change_Mode(self,H_CS_B):
        self.H_CS_B = H_CS_B
        self.Ad_CS_B = lie.Ad(H_CS_B)
        return self.Ad_CS_B
            
        
class IntegralLoss(nn.Module):
    # control effort integral cost
    # f is of type SE3_Dynamics or se3_dynamics
    def __init__(self, f):
        super().__init__()
        self.scale = 1
        self.f = f
        
    def forward(self,t,x):
        t_batch = t*torch.ones(x.shape[0]).to(device)  # formality
        return vmap(self.forward_single)(t_batch,x)
    
    def forward_single(self, t, x):
        with torch.set_grad_enabled(True):
            x = torch.squeeze(x)
            qi = x[...,:6].requires_grad_(True)
            P = x[...,6:12].requires_grad_(True)
            i = x[...,12]
            H = lie.unchart(qi,i)
            u = self.f.PotentialGradient(H) + self.f.DampingInjection(H,P)
        return self.scale*torch.unsqueeze(lie.dot(u,u),0)

    def grad(self, t, x):
        t_batch = t*torch.ones(x.shape[0]).to(device)  # formality
        return vmap(self.grad_single)(t_batch,x)
    
    def grad_single(self, t, x):
        qi, P, i = x[:6], x[6:12], x[12] 
        T = self.f.I_inv @ P
        
        H = lie.unchart(qi,i)  
        u = self.f.PotentialGradient(H) + self.f.DampingInjection(H,P)
        
        dqi_H = z1*T 
        dqi_P = self.f.I_inv
        
        dW_HP = self.f.dW(x)
        dW_H = dW_HP[:6]
        dW_P = dW_HP[6:]
        
        loss_dq = torch.einsum('j,ij->i', 2*u, dW_H)
        loss_dP = torch.einsum('j,ij->i', 2*u , dW_P)
        return self.scale*torch.cat((loss_dq,loss_dP,z1),-1)

class IntegralLoss_Full(nn.Module):
    # control effort and state-norm integral cost
    # f is of type SE3_Dynamics or se3_dynamics
    def __init__(self, f,su,sw,sd,sPw,sPv,sg,H_target):
        super().__init__()
        self.scale_u = su
        self.scale_w = sw
        self.scale_d = sd
        self.scale_Pw = sPw
        self.scale_Pv = sPv
        self.scale_g = sg
        self.H_target = H_target
        self.f = f
        
    def forward(self,t,x):
        t_batch = t*torch.ones(x.shape[0]).to(device)  # formality
        return vmap(self.forward_single)(t_batch,x)
    
    def forward_single(self, t, x):
        with torch.set_grad_enabled(True):
            x = torch.squeeze(x)
            qi = x[...,:6].requires_grad_(True)
            P = x[...,6:12].requires_grad_(True)
            i = x[...,12]
            H = lie.unchart(qi,i)
            u = self.f.PotentialGradient(H) + self.f.DampingInjection(H,P)
        return self.cost_HPU(t,H,P,u)
    
    def cost_HPU(self, t, H, P, u):
        del_H = lie.inv_SE3(self.H_target)@H 
        pw = torch.linalg.vector_norm(P[...,:3],dim=-1)
        pv = torch.linalg.vector_norm(P[...,3:6],dim=-1)
        d = lie.angle(del_H[0:3,3])
        U = u-self.gravity(H)
        return self.scale_u*torch.unsqueeze(lie.dot(U,U),0) - self.scale_w*(lie.trace(del_H)-4) + self.scale_d*lie.dot(d,d) + self.scale_Pw*lie.dot(pw,pw) + self.scale_Pv*lie.dot(pv,pv)
    
    def gravity(self,H):
        return -self.scale_g*torch.cat((z3,H[3,0:3]))

    def grad(self, t, x):
        t_batch = t*torch.ones(x.shape[0]).to(device)
        return vmap(self.grad_single)(t_batch,x)
    
    def grad_single(self, t, x):
        qi, P, i = x[:6], x[6:12], x[12] 
        T = self.f.I_inv @ P
        q = qi*0
        
        H = lie.unchart(qi,i)  
        u = self.f.PotentialGradient(H) + self.f.DampingInjection(H,P)
        
        dqi_H = z1*T 
        dqi_P = self.f.I_inv
        
        dW_HP = self.f.dW(x)
        dW_H = dW_HP[:6]
        dW_P = dW_HP[6:]
        
        qPiu = torch.cat((q,P,i.unsqueeze(0),u),-1)
        qPiu.requires_grad_(True)
        C_ = lambda qPiu: self.cost_HPU(t,H@(eye4+lie.skew_se3(qPiu[:6])),qPiu[6:12],qPiu[13:])
        dC = jacrev(C_)(qPiu) # size [1,19], differentiated in second index
        
        loss_dq = torch.einsum('j,ij->i', dC[0,13:], dW_H) + dC[0,:6]
        loss_dP = torch.einsum('j,ij->i', dC[0,13:] , dW_P) + dC[0,6:12]
        return torch.cat((loss_dq,loss_dP,z1),-1) 
    
class IntegralLoss_Full2(nn.Module):
    # control effort and state-norm integral cost, with cost-functional linear instead of quadratic
    # f is of type SE3_Dynamics or se3_dynamics
    def __init__(self, f,su,sw,sd,sPw,sPv,sg,H_target):
        super().__init__()
        self.scale_u = su
        self.scale_w = sw
        self.scale_d = sd
        self.scale_Pw = sPw
        self.scale_Pv = sPv
        self.scale_g = sg
        self.H_target = H_target
        self.f = f
        
    def forward(self,t,x):
        t_batch = t*torch.ones(x.shape[0]).to(device)  # formality
        return vmap(self.forward_single)(t_batch,x)
    
    def forward_single(self, t, x):
        with torch.set_grad_enabled(True):
            x = torch.squeeze(x)
            qi = x[...,:6].requires_grad_(True)
            P = x[...,6:12].requires_grad_(True)
            i = x[...,12]
            H = lie.unchart(qi,i)
            u = self.f.PotentialGradient(H) + self.f.DampingInjection(H,P)
        return self.cost_HPU(t,H,P,u)
    
    def cost_HPU(self, t, H, P, u):
        del_H = lie.inv_SE3(self.H_target)@H 
        pw = torch.linalg.vector_norm(P[...,:3],dim=-1)
        pv = torch.linalg.vector_norm(P[...,3:6],dim=-1)
        d = lie.angle(del_H[0:3,3])
        U = u-self.gravity(H)
        return self.scale_u*torch.sqrt(torch.unsqueeze(lie.dot(U,U),0)+1e-32) + self.scale_w*torch.sqrt(4-(lie.trace(del_H))+1e-32) + self.scale_d*d + self.scale_Pw*pw + self.scale_Pv*pv
    
    def gravity(self,H):
        return -self.scale_g*torch.cat((z3,H[3,0:3]))

    def grad(self, t, x):
        t_batch = t*torch.ones(x.shape[0]).to(device)
        return vmap(self.grad_single)(t_batch,x)
    
    def grad_single(self, t, x):
        qi, P, i = x[:6], x[6:12], x[12] 
        T = self.f.I_inv @ P
        q = qi*0
        
        H = lie.unchart(qi,i)  
        u = self.f.PotentialGradient(H) + self.f.DampingInjection(H,P)
        
        dqi_H = z1*T 
        dqi_P = self.f.I_inv
        
        dW_HP = self.f.dW(x)
        dW_H = dW_HP[:6]
        dW_P = dW_HP[6:]
        
        qPiu = torch.cat((q,P,i.unsqueeze(0),u),-1)
        qPiu.requires_grad_(True)
        C_ = lambda qPiu: self.cost_HPU(t,H@(eye4+lie.skew_se3(qPiu[:6])),qPiu[6:12],qPiu[13:])
        dC = jacrev(C_)(qPiu) # size [1,19], differentiated in second index
        
        loss_dq = torch.einsum('j,ij->i', dC[0,13:], dW_H) + dC[0,:6]
        loss_dP = torch.einsum('j,ij->i', dC[0,13:] , dW_P) + dC[0,6:12]
        return torch.cat((loss_dq,loss_dP,z1),-1) 
    
class IntegralLoss_Quadratic(nn.Module):
    # control effort integral cost
    # f is of type SE3_Quadratic
    def __init__(self, f):
        super().__init__()
        self.f = f
    def forward(self,t,x):
        t_batch = t*torch.ones(x.shape[0]).to(device)  # formality
        return vmap(self.forward_single)(t_batch,x)
    
    def forward_single(self, t, x):
        with torch.set_grad_enabled(True):
            x = torch.squeeze(x)
            qi, P, i = x[...,:6].requires_grad_(True), x[...,6:12].requires_grad_(True), x[...,12] 
            T = self.f.I_inv @ P
            H_B_I = lie.unchart(qi,i)  
            H_D_CS = lie.inv_SE3(self.f.H_CS_B@H_B_I@self.f.H_I_D)
            u = self.f.W_di(T) + self.f.Ad_CS_B.transpose(-1,-2)@(self.f.W_p(H_D_CS)-self.f.W_grv()+self.f.W_grv_real()) 
        return torch.unsqueeze(torch.sqrt(lie.dot(u,u)),0)
    
    def grad(self, x):
        return vmap(self.grad_single)(x)
    
    def grad_single(self, x):
        qi, P, i = x[:6], x[6:12], x[12] 
        T = self.f.I_inv @ P
        H_I_B = lie.unchart(qi,i)  
        H_D_CS = lie.inv_SE3(self.f.H_I_D)@H_I_B@lie.inv_SE3(self.f.H_CS_B)
        dH_D_CS = lie.inv_SE3(self.f.H_I_D)@lie.dH_q(H_I_B, qi)@lie.inv_SE3(self.f.H_CS_B)
        u = self.f.W_di(T) + self.f.Ad_CS_B.transpose(-1,-2)@(self.f.W_p(H_D_CS)-self.f.W_grv()+self.f.W_grv_real()) 

        loss_dq = torch.einsum('j,ij->i', 2*u, self.f.dW_p(H_D_CS,dH_D_CS))
        loss_dP = torch.einsum('k,jk->j', 2*u , (self.f.Kd(z1)@self.f.I_inv).transpose(-1,-2))
        return torch.cat((loss_dq,loss_dP,z1),-1)
    

class AugmentedSE3(nn.Module):
    def __init__(self,f,integral_cost,terminal_cost):
        super().__init__()
        self.f = f
        self.l = integral_cost
        self.term = terminal_cost
        self.nfe = 0 #number of function evaluations, interesting for logger 
        
    def forward(self,t,x):
        self.nfe += 1
        dxdt = self.f(t, x)
        dldt = self.l(t, x) #torch.unsqueeze(,0)
        return torch.cat((dxdt, dldt), -1)

class PosDefSym(nn.Module):
    # 21 dim input to 6by6 positive definite, symmetric matrix
    # via cholesky-decomposition 
    def __init__(self):
        super().__init__()
        
    def forward(self,x):
        return lie.makeB2(x)
    
class PosDefTriv(nn.Module):
    # 6 dim input to 6by6 diagonal matrix with positive elements
    def __init__(self):
        super().__init__()
    def forward(self,x):
        return torch.diag(torch.exp(x))
    
class PosDefSym_Small(nn.Module):
    # 6 dim input to 3by3 positive definite, symmetric matrix
    def __init__(self):
        super().__init__()
    
    def forward(self,x):
        R = lie.exp_SO3(x[:3])
        invR = lie.inv_SO3(R)
        D = torch.diag(x[3:])
        return(R@D@invR)

class Embed_SE3_in_R12(nn.Module):
    # H in SE(3) is mapped to vector in R^12 
    def __init__(self):
        super().__init__()
    def forward(self,H):
        return torch.flatten(H[:3,:])
    
class Quadratic_Potential_SE3(nn.Module):
    def __init__(self):
        super().__init__()
        self.Kt_params = torch.nn.Linear(1,6)
        self.Go_params = torch.nn.Linear(1,6)
        self.Kt = PosDefSym_Small()
        self.Go = PosDefSym_Small()
    def forward(self,x):
        # assumes that x = Embed_SE3_in_R12(H), i.e., that x is in R^12
        params_Kt = self.Kt_params(z1.unsqueeze(-1));
        params_Go = self.Go_params(z1.unsqueeze(-1));
        Kt = self.Kt(params_Kt.squeeze());
        Go = self.Go(params_Go.squeeze());
        Rv = x.reshape(3,4)
        R = Rv[:3,:3]
        v = Rv[:3,3]
        
        return 1/4*v@Kt@v + 1/4* v@R@Kt@R@v - lie.trace(Go@(R-eye3))
    
class state_to_zero(nn.Module):
    # maps input to zero, needed for quadratic potential and linear damping injection to only dependent on biases
    def __init__(self):
        super().__init__()
    def forward(self,x):
        return x*0
    