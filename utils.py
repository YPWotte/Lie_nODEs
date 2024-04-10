import torch
from torch.utils import data as data
from torch.distributions import MultivariateNormal, Normal
import lie_torch as lie

from functorch import vmap, grad, jacrev

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
eye4 = torch.eye(4).to(device)

#######
### 1. Custom Distribution Functions:
#######

class prior_dist_SE3():
    # uniform "prior" distribution of initial conditions q(0),p(0) and initial chart i0
    def __init__(self,th_max, d_max, pw_max, p_max, ch_min, ch_max, device ='cpu'):
        #super().__init__()
        devz = torch.tensor(0).to(device)
        self.device = device
        self.th = Uniform(devz,th_max,device) # angle
        self.d = Uniform(devz,d_max,device)   # distance
        self.pw = Uniform(devz,pw_max,device) # angular momentum
        self.pv = Uniform(devz,p_max,device)  # linear momentum
        self.chart = Uniform(ch_min,ch_max+1,device)
        self.q = Normal(torch.zeros(6).to(device),torch.ones(6).to(device))
        self.p = Normal(torch.zeros(6).to(device),torch.ones(6).to(device))
        
    def sample(self,n=1):
        # Sample coordinates
        q0 = self.q.sample([n]); q0[:,:3] = q0[:,:3]/torch.linalg.vector_norm(q0[:,:3],2,1)[:,None]; q0[:,:3] = self.th.sample([n])[:,None]*q0[:,:3]; 
        q0[:,3:] = q0[:,3:]/torch.linalg.vector_norm(q0[:,3:],2,1)[:,None]; q0[:,3:] = self.d.sample([n])[:,None]*q0[:,3:]; 

        # Sample momentum
        p0 = self.p.sample([n]); p0[:,:3] = p0[:,:3]/torch.linalg.vector_norm(p0[:,:3],2,1)[:,None]; p0[:,:3] = self.pw.sample([n])[:,None]*p0[:,:3];
        p0[:,3:] = p0[:,3:]/torch.linalg.vector_norm(p0[:,3:],2,1)[:,None]; p0[:,3:] = self.pv.sample([n])[:,None]*p0[:,3:];

        # Sample chart
        i0 = torch.floor(self.chart.sample([n]))[:,None]
        return torch.cat((q0,p0,i0),1).to(self.device)

class target_dist_SE3():
    # Gaussian target, chart-agnostic 
    # BUT INCOMPLETE: numerical gradient needs to propagate through all steps
    # See instead target_dist_SE3_HP below
    def __init__(self,H_target, sigma_th, sigma_d, sigma_pw, sigma_p, device ='cpu'):
        #super().__init__()
        devz = torch.tensor(0).to(device)
        self.device = device
        self.H_target = H_target
        self.th = Normal(devz,sigma_th) # angle: Distribution should have finite support
        self.d = Normal(devz,sigma_d)   # distance
        self.pw = Normal(devz,sigma_pw) # angular momentum
        self.pv = Normal(devz,sigma_p)  # linear momentum
        self.chart = Uniform(devz,torch.tensor(4).to(device),device)
        self.q = Normal(torch.zeros(6).to(device),torch.ones(6).to(device))
        self.p = Normal(torch.zeros(6).to(device),torch.ones(6).to(device))
    
    def sample(self,n=1):
        # Sample coordinates
        q0 = self.q.sample([n]); q0[:,:3] = q0[:,:3]/torch.linalg.vector_norm(q0[:,:3],2,1)[:,None]; q0[:,:3] = self.th.sample([n])[:,None]*q0[:,:3]; 
        q0[:,3:] = q0[:,3:]/torch.linalg.vector_norm(q0[:,3:],2,1)[:,None]; q0[:,3:] = self.d.sample([n])[:,None]*q0[:,3:]; 

        # Sample momentum
        p0 = self.p.sample([n]); p0[:,:3] = p0[:,:3]/torch.linalg.vector_norm(p0[:,:3],2,1)[:,None]; p0[:,:3] = self.pw.sample([n])[:,None]*p0[:,:3];
        p0[:,:3] = p0[:,3:]/torch.linalg.vector_norm(p0[:,3:],2,1)[:,None]; p0[:,3:] = self.pv.sample([n])[:,None]*p0[:,3:];

        # Sample chart
        i0 = torch.floor(self.chart.sample([n]))[:,None]
        
        k = 0
        for q,i in zip(q0,i0):
            # Map into coordinates, treating q0 as collection of coordinates in chart centered on H_target
            q0[k,:] = lie.chart(self.H_target@lie.exp(q),i) # Collect result in q0 again
            k+=1
            
        return torch.cat((q0,p0,i0),1).to(self.device)
    
    def log_prob(self,x):
        # minorly flawed log_prob, instead of normal distributions this should use distributions with finite support for SO(3) part, e.g. a scaled beta distribution
        th_d = self.distance(x)
        th = th_d[0] #torch.linalg.vector_norm(x[...,:3],dim=-1) #Alternative: th = lie.angle(x[:,:3])
        d = th_d[1]  #torch.linalg.vector_norm(x[...,3:6],dim=-1)
        pw = torch.linalg.vector_norm(x[...,6:9],dim=-1)
        pv = torch.linalg.vector_norm(x[...,9:12],dim=-1)
        # Log of probability: Probability (density) of output is product of probability (densities) of independent parts: 
        return torch.log(torch.tensor(2))*(self.th.log_prob(th)+self.d.log_prob(d)+self.pw.log_prob(pw)+self.pv.log_prob(pv)+torch.log(torch.tensor(1/4)))
    
    def distance(self,x):
        # angle and distance to H_target frame
        def dist(x):
            q = x[:6]
            i = x[12]
            d_raw = lie.log_SE3(lie.inv_SE3(self.H_target)@lie.unchart(q,i))
            return lie.angle(d_raw[:3]), lie.angle(d_raw[3:])
        return vmap(dist)(x)
    
    def grad_log_prob(self,x):
        raise NotImplementedError
        
        
class target_dist_SE3_HP():
    # Gaussian target, chart-agnostic, log_prob implemented differently and grad_log_prob defined 
    def __init__(self,H_target, sigma_th, sigma_d, sigma_pw, sigma_p, device ='cpu'):
        #super().__init__()
        devz = torch.tensor(0).to(device)
        self.device = device
        self.H_target = H_target
        self.th = Normal(devz,sigma_th) # angle: Distribution should have finite support
        self.d = Normal(devz,sigma_d)   # distance
        self.pw = Normal(devz,sigma_pw) # angular momentum
        self.pv = Normal(devz,sigma_p)  # linear momentum
        self.chart = Uniform(devz,torch.tensor(4).to(device),device)
        self.q = Normal(torch.zeros(6).to(device),torch.ones(6).to(device))
        self.p = Normal(torch.zeros(6).to(device),torch.ones(6).to(device))
    
    def sample(self,n=1):
        # Sample coordinates
        q0 = self.q.sample([n]); q0[:,:3] = q0[:,:3]/torch.linalg.vector_norm(q0[:,:3],2,1)[:,None]; q0[:,:3] = self.th.sample([n])[:,None]*q0[:,:3]; 
        q0[:,3:] = q0[:,3:]/torch.linalg.vector_norm(q0[:,3:],2,1)[:,None]; q0[:,3:] = self.d.sample([n])[:,None]*q0[:,3:]; 

        # Sample momentum
        p0 = self.p.sample([n]); p0[:,:3] = p0[:,:3]/torch.linalg.vector_norm(p0[:,:3],2,1)[:,None]; p0[:,:3] = self.pw.sample([n])[:,None]*p0[:,:3];
        p0[:,:3] = p0[:,3:]/torch.linalg.vector_norm(p0[:,3:],2,1)[:,None]; p0[:,3:] = self.pv.sample([n])[:,None]*p0[:,3:];

        # Sample chart
        i0 = torch.floor(self.chart.sample([n]))[:,None]
        
        k = 0
        for q,i in zip(q0,i0):
            # Map into coordinates, treating q0 as collection of coordinates in chart centered on H_target
            q0[k,:] = lie.chart(self.H_target@lie.exp(q),i) # Collect result in q0 again
            k+=1
            
        return torch.cat((q0,p0,i0),1).to(self.device)
    
    def log_prob(self,x):
        return vmap(self.log_prob_inner)(x)
    
    def log_prob_inner(self,x):
        qi, P, i = x[:6], x[6:12], x[12] 
        H = lie.unchart(qi,i)  
        return self.log_prob_HP(H,P)

    def log_prob_HP(self,H,P):
        dH = lie.inv_SE3(self.H_target)@H
        
        th = lie.acos((lie.trace(dH)-2)/2)
        d = lie.angle(H[3,0:3])
        pw = torch.linalg.vector_norm(P[...,:3],dim=-1)
        pv = torch.linalg.vector_norm(P[...,3:6],dim=-1)
        

        return torch.log(torch.tensor(2))*(self.th.log_prob(th)+self.d.log_prob(d)+self.pw.log_prob(pw)+self.pv.log_prob(pv)+torch.log(torch.tensor(1/4)))
    
    def distance(self,x):
        # angle and distance to H_target frame, but vmapped
        def dist(x):
            q = x[:6]
            i = x[12]
            d_raw = lie.log_SE3(lie.inv_SE3(self.H_target)@lie.unchart(q,i))
            return lie.angle(d_raw[:3]), lie.angle(d_raw[3:])
        return vmap(dist)(x)
    
    def grad_log_prob(self,x):
        #return gradient at algebra
        qi, P, i = x[:6], x[6:12], x[12] 
        H = lie.unchart(qi,i)  
        q =  qi*0
        
        qP = torch.cat((q,P),-1)
        qP.requires_grad_(True)
        C_ = lambda qP: self.log_prob_HP(H@(eye4+lie.skew_se3(qP[:6])),qP[6:])
        dC = jacrev(C_)(qP) # size [1,12], differentiated in second index
        return dC
    

class target_cost_SE3():
    # Arbitrary target cost, not a "normalized" probability density and chart-agnostic 
    def __init__(self,H_target, weight_th, weight_d, weight_pw, weight_p, device ='cpu'):
        #super().__init__()
        devz = torch.tensor(0).to(device)
        self.device = device
        self.H_target = H_target
        self.weight_th, self.weight_d, self.weight_pw, self.weight_p = weight_th, weight_d, weight_pw, weight_p
        
    def cost(self,x):
        return vmap(self.cost_inner)(x)
    
    def cost_inner(self,x):
        qi, P, i = x[:6], x[6:12], x[12] 
        H = lie.unchart(qi,i)  
        return self.cost_HP(H,P)

    def cost_HP(self,H,P):
        # Implement arbitrary cost using H in SE(3) and P in se*(3)
        del_H = lie.inv_SE3(self.H_target)@H 
        
        pw = torch.linalg.vector_norm(P[...,:3],dim=-1)
        pv = torch.linalg.vector_norm(P[...,3:6],dim=-1)
        d = lie.angle(del_H[0:3,3])

        return 1/2*(-self.weight_th*(lie.trace(del_H)-4) + self.weight_d*torch.pow(d,2) + self.weight_pw*torch.pow(pw,2) + self.weight_p*torch.pow(pv,2))
    
    def distance_single(self,x):
        q = x[:6]
        i = x[12]
        d_raw = lie.log_SE3(lie.inv_SE3(self.H_target)@lie.unchart(q,i))
        return lie.angle(d_raw[:3]), lie.angle(d_raw[3:])
    
    def distance(self,x):
        # angle and distance to H_target frame, but vmapped
        return vmap(self.distance_single)(x)
    
    def grad_cost_vmapped(self,xcol):
        return vmap(self.grad_cost)(xcol)
    
    def grad_cost(self,x):
        #return gradient at algebra
        qi, P, i = x[:6], x[6:12], x[12] 
        H = lie.unchart(qi,i)  
        q =  qi*0
        
        qPi = torch.cat((q,P,i.unsqueeze(0)),-1)
        qPi.requires_grad_(True)
        C_ = lambda qPi: self.cost_HP(H@(eye4+lie.skew_se3(qPi[:6])),qPi[6:]) # qPi[6:] also passes on i, but cost_HP explicitly only uses qPi[6:12], so this is fine
        dC = jacrev(C_)(qPi) # size [1,12], differentiated in second index
        return dC
    
class target_cost2_SE3():
    # Arbitrary target cost, not a "normalized" probability density and chart-agnostic;
    # linear instead of quadratic terms
    def __init__(self,H_target, weight_th, weight_d, weight_pw, weight_p, device ='cpu'):
        #super().__init__()
        devz = torch.tensor(0).to(device)
        self.device = device
        self.H_target = H_target
        self.weight_th, self.weight_d, self.weight_pw, self.weight_p = weight_th, weight_d, weight_pw, weight_p
        
    def cost(self,x):
        return vmap(self.cost_inner)(x)
    
    def cost_inner(self,x):
        qi, P, i = x[:6], x[6:12], x[12] 
        H = lie.unchart(qi,i)  
        return self.cost_HP(H,P)

    def cost_HP(self,H,P):
        # Implement arbitrary cost using H in SE(3) and P in se*(3)
        del_H = lie.inv_SE3(self.H_target)@H 
        
        pw = torch.linalg.vector_norm(P[...,:3],dim=-1)
        pv = torch.linalg.vector_norm(P[...,3:6],dim=-1)
        d = lie.angle(del_H[0:3,3])

        return self.weight_th*torch.sqrt(4-(lie.trace(del_H))+1e-32) + self.weight_d*d + self.weight_pw*pw + self.weight_p*pv
    
    def distance_single(self,x):
        q = x[:6]
        i = x[12]
        d_raw = lie.log_SE3(lie.inv_SE3(self.H_target)@lie.unchart(q,i))
        return lie.angle(d_raw[:3]), lie.angle(d_raw[3:])
    
    def distance(self,x):
        # angle and distance to H_target frame, but vmapped
        return vmap(self.distance_single)(x)
    
    def grad_cost_vmapped(self,xcol):
        return vmap(self.grad_cost)(xcol)
    
    def grad_cost(self,x):
        #return gradient at algebra
        qi, P, i = x[:6], x[6:12], x[12] 
        H = lie.unchart(qi,i)  
        q =  qi*0
        
        qPi = torch.cat((q,P,i.unsqueeze(0)),-1)
        qPi.requires_grad_(True)
        C_ = lambda qPi: self.cost_HP(H@(eye4+lie.skew_se3(qPi[:6])),qPi[6:]) # qPi[6:] also passes on i, but cost_HP explicitly only uses qPi[6:12], so this is fine
        dC = jacrev(C_)(qPi) # size [1,12], differentiated in second index
        return dC

class Uniform():
    def __init__(self,lower,upper,device):
        self.lower = lower
        self.upper = upper
        self.device = device
        
    def sample(self,N):
        # N = [n0,n1,n2, ..., ni]
        return torch.rand(N).to(self.device)*(self.upper-self.lower)+self.lower
    
    def log_prob(self,x):
        return torch.ones(x.shape).to(self.device)*torch.log(1/(self.upper-self.lower))

#######
### 2. Taken from utils.py for optimal_energy_shaping.ipynb:
#######

def dummy_trainloader():
    # dummy trainloader for Lightning learner
    dummy = data.DataLoader(
        data.TensorDataset(
            torch.Tensor(1, 1),
            torch.Tensor(1, 1)
        ),
        batch_size=1,
        shuffle=False
    )
    return dummy

def log_likelihood_loss(x, target):
    # negative log likelihood loss
    return -torch.mean(target.log_prob(x))

def weighted_log_likelihood_loss(x, target, weight):
    # weighted negative log likelihood loss
    log_prob = target.log_prob(x)
    weighted_log_p = weight * log_prob
    return -torch.mean(weighted_log_p.sum(1))

def weighted_L2_loss(xh, x, weight):
    # weighted squared error loss
    e = xh - x
    return torch.einsum('ij, jk, ik -> i', e, weight, e).mean()

def target_dist(mu, sigma, device='cpu'):
    # normal target distribution of terminal states x(T)
    mu, sigma = torch.Tensor(mu).reshape(1, 2), torch.Tensor(sigma).reshape(1, 2)
    return Normal(mu, torch.sqrt(sigma))

def multinormal_target_dist(mu, sigma, device='cpu'):
    # normal target distribution of terminal states x(T)
    mu, sigma = torch.Tensor(mu), torch.Tensor(sigma)
    return MultivariateNormal(mu, sigma*torch.eye(mu.shape[0]))

