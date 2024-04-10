import torch
import torch.nn as nn
from torch.autograd import grad 

import pytorch_lightning as pl

from torchdyn.numerics.odeint import odeint_hybrid
from torchdyn.numerics.solvers import DormandPrince45
import torchdyn.numerics.sensitivity


from lie_torch import angle 
from utils import dummy_trainloader, log_likelihood_loss

import sys; sys.path.append('../')
from odeint import odeint_hybrid_alt, odeint_hybrid_raw

from functorch import vmap

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class EnergyShapingLearner(pl.LightningModule):
    def __init__(self, model: nn.Module, t_span, prior_dist, target_dist, aug_model: nn.Module,):
        super().__init__()
        self.model = model
        self.t_span = t_span
        self.prior, self.target = prior_dist, target_dist
        self.aug_model = aug_model
        
        self.batch_size = 2048 
        self.lr = 1e-3
        # Unadapted:
        #self.weight = torch.Tensor([1., 1.]).reshape(1, 2)
        self.iterations = 0

    def forward(self, x):
        return self.model(x, self.t_span)

    def training_step(self, batch, batch_idx): 
        self.iterations+=1
        # 1. Sample a batch of initial conditions
        x0 = self.prior.sample((self.batch_size)).to(device)

        
        # 2. Integrate the model
        t, xT = self(x0) 

        # 3. Compute loss
        terminal_loss = torch.mean(self.target.cost(xT[-1])) # Alternatively, when target_dist_SE3_HP is used for the target: log_likelihood_loss(xT[-1], self.target)
        loss = terminal_loss # + 0.01 * integral_loss
        

        # 4. Logging:

        
        if self.iterations % 10 == 0:
            # 4.1 compute integral loss explicitly by augmentation of state with 
            # an aux. variable. See `Dissecting Neural ODEs` for more information
            xl0 = torch.cat([x0,torch.zeros(x0.shape[0], 1).to(x0)], 1)
            _, sol = self.aug_model(xl0, self.t_span)
            integral_loss = sol[-1, :, -1].mean()
            self.logger.experiment.log({'integral_loss': integral_loss,
                                        'loss': terminal_loss + integral_loss})
        
        # 4.2 Chart-Occurences:
        n_chart_0 = (xT[:,:,12] == 0).sum()
        n_chart_1 = (xT[:,:,12] == 1).sum()
        n_chart_2 = (xT[:,:,12] == 2).sum()
        n_chart_3 = (xT[:,:,12] == 3).sum()
        n_chart_switches = (torch.diff(xT[:,:,12].transpose(-1,-2))!=0).sum() # Number of Chart-Switches
        
        # 4.3 Maximum & Minimum Linear & Angular Momentum
        Pw = angle((xT[:, :,6:9])); Pw_max = Pw.max(); Pw_min = Pw.min()
        Pv = angle((xT[:, :,9:12])); Pv_max = Pv.max(); Pv_min = Pv.min()
        Pw_Start = angle((xT[0, :,6:9])).mean()
        Pv_Start = angle((xT[0, :,9:12])).mean()
        Pw_Final = angle((xT[-1, :,6:9])).mean()
        Pv_Final = angle((xT[-1, :,9:12])).mean()
        
        # 4.4 Minimum possible terminal loss:
        #term_loss_fun = lambda xT_n: self.target.log_prob(xT_n) 
        #min_Term_Losses, Indices = vmap(term_loss_fun)(xT).min(0)
        #avg_min_Term_Loss = (min_Term_Losses).mean()
        #best_loss_scaled_time = Indices.float().mean()/xT.shape[0]  # Point of occurence of minimal loss, 0 being start and 1 being end of trajectory
        
        # 4.5 Distance from target: TODO: Check if these are the intended quantities..
        End_th_to_target, End_d_to_target = self.target.distance(xT[-1,:,:])
        Start_th_to_target, Start_d_to_target = self.target.distance(xT[0,:,:])
        
        self.logger.experiment.log(
            {
                'terminal loss': terminal_loss,
                'pw_max': Pw_max,
                'pw_min': Pw_min,
                'pv_max': Pv_max,
                'pv_min': Pv_min,
                'Pw_Start': Pw_Start,
                'Pv_Start':Pv_Start,
                'Pw_Final': Pw_Final,
                'Pv_Final':Pv_Final,
                'th_start': Start_th_to_target.mean(),
                'd_start': Start_d_to_target.mean(),
                'th_final':End_th_to_target.mean(),
                'd_final':End_d_to_target.mean(),
                'avg_n_chart_switches': n_chart_switches/self.batch_size
                #'avg_min_terminal': avg_min_Term_Loss,
                #'avg_min_terminal_scaled_time': best_loss_scaled_time
            }
        )

        self.model.nfe = 0
        return {'loss': loss}

    def configure_optimizers(self): # '#'
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=.999)
        return [optimizer], [scheduler]

    def train_dataloader(self): # '#'
        return dummy_trainloader()
