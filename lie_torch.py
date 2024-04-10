#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Last modified on Apr 28 2022

@author: Y. P. Wotte

Custom library for lie groups SO(3) and SE(3) using pytorch for compatibility 
with pytorch, torchdyn  and functorch/vmap. 

Features: 
    - minimal covering Atlas using exponential charts
    - numerical stability for edge cases
    - construction of aribtrary smooth functions on SE(3) via partition of unity
    - efficient 1st derivative of exponential map on SE(3), as well as their inverses
    - small and big Adjoint on SE(3), twists and skew forms 
    - create random positive definite symmetric matrices, with parametrizations covering the full space of such matrices, via cholesky 
    - higher powers of skew-forms
    - 2nd derivative of exponential map on SE(3)
    - wrapper for automatic differentiation of functions on SE(3), with gradient on se*(3) 
    
Below features removed since prior version:
    - exponential map on SO(6) (adapted from D.G.A.A. Galvez (2018)) 
    - create random positive definite symmetric matrices via svd parametrization
"""

import torch  
import math      
from functorch import vmap
import time

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

#######
### 0. Auxilliary Tensors / numbers 
#######

# Identity and zero-matrices
eye1 = torch.tensor(1.).to(device)
eye3 = torch.eye(3).to(device)
eye4 = torch.eye(4).to(device)
eye6 = torch.eye(6).to(device)
zero1 = torch.tensor(0).to(device);
zero3 = torch.zeros(3).to(device);
zero33 = torch.zeros((3,3)).to(device)
zero66 = torch.zeros((6,6)).to(device)

# 3 by 3 diagonal selection matrices
I0 = torch.zeros((3,3)).to(device); I0[0,0]=1
I1 = torch.zeros((3,3)).to(device); I1[1,1]=1
I2 = torch.zeros((3,3)).to(device); I2[2,2]=1

# Basis R matrices
R0 = eye3
R1 = torch.tensor(( (1, 0, 0), (0, -1, 0), (0, 0, -1))).to(device)
R2 = torch.tensor(( (-1, 0, 0), (0, 1, 0), (0, 0, -1))).to(device)
R3 = torch.tensor(( (-1, 0, 0), (0, -1, 0), (0, 0, 1))).to(device)

# Basis H matrices:
H0 = eye4
H1 = torch.tensor(( (1, 0, 0, 0), (0, -1, 0, 0), (0, 0, -1, 0), (0, 0, 0, 1) )).to(device)
H2 = torch.tensor(( (-1, 0, 0, 0), (0, 1, 0, 0), (0, 0, -1, 0), (0, 0, 0, 1) )).to(device)
H3 = torch.tensor(( (-1, 0, 0, 0), (0, -1, 0, 0), (0, 0, 1, 0), (0, 0, 0, 1) )).to(device)

# other:
arange10 = torch.arange(11).to(device)
r0001 = torch.tensor((0,0,0,1)).to(device)

# For skew & unskew:
A_sk_so3 = torch.tensor((( (0, 0, 0), (0, 0, -1), (0, 1, 0) ),
                      ( (0, 0, 1), (0, 0, 0 ), (-1, 0, 0)),
                      ( (0, -1, 0), (1, 0, 0), (0, 0, 0) ) ), dtype=torch.float).to(device)
A_usk_so3 = torch.tensor((( (0, 0, 0), (0, 0, 0), (0, 1, 0) ),
                      ( (0, 0, 1), (0, 0, 0 ), (0, 0, 0)),
                      ( (0, 0, 0), (1, 0, 0), (0, 0, 0) ) ), dtype=torch.float).to(device)
A_sk_se3 = torch.tensor((( (0, 0, 0, 0, 0, 0), (0, 0, -1, 0, 0, 0), (0, 1, 0, 0, 0, 0), (0, 0, 0, 1, 0, 0)),
                      ( (0, 0, 1, 0, 0, 0), (0, 0, 0, 0, 0, 0 ), (-1, 0, 0, 0, 0, 0),(0, 0, 0, 0, 1, 0)),
                      ( (0, -1, 0, 0, 0, 0), (1, 0, 0, 0, 0, 0), (0, 0, 0, 0, 0, 0), (0, 0, 0, 0, 0, 1)),
                      ( (0, 0, 0, 0, 0, 0) , (0, 0, 0, 0, 0, 0), (0, 0, 0, 0, 0, 0), (0, 0, 0, 0, 0, 0))), dtype=torch.float).to(device)
A_usk_se3 = torch.tensor((( (0, 0, 0, 0, 0, 0), (0, 0, 0, 0, 0, 0), (0, 1, 0, 0, 0, 0), (0, 0, 0, 1, 0, 0)),
                      ( (0, 0, 1, 0, 0, 0), (0, 0, 0, 0, 0, 0 ), (0, 0, 0, 0, 0, 0),(0, 0, 0, 0, 1, 0)),
                      ( (0, 0, 0, 0, 0, 0), (1, 0, 0, 0, 0, 0), (0, 0, 0, 0, 0, 0), (0, 0, 0, 0, 0, 1)),
                      ( (0, 0, 0, 0, 0, 0) , (0, 0, 0, 0, 0, 0), (0, 0, 0, 0, 0, 0), (0, 0, 0, 0, 0, 0))), dtype=torch.float).to(device) 

#######
### 1. Basic Auxilliary Functions:
#######

def dot(v,w):
    # executes the dot-product between two tensors of the last dimension
    return (v*w).sum(-1)

def angle(t):
    # return angle |w| for twists t = [w,v] in se(3) or t = w in so(3)
    # t can be a collection of twists, then last index of t are treated as twist
    w,v = t.tensor_split([3],-1)
    return torch.sqrt(dot(w,w)+1e-32)-1e-16     # add small number to avoid gradient issues, subtract it outside because th>0 is an important condition in many places

def dangle (t,dt):
    # returns changerate of angle(t), with dt the change rate of t. 
    th = angle(t)
    dth_origin = angle(dt)
    return torch.where(th>0,torch.einsum('i,i',t[:3],dt[:3])/th, dth_origin)    # torch.where() for compatibility with vmap.

def factorial(A):
    # element-wise factorial of integer tensor A
    A = A.int();
    # vmap compatible options are:
    # 1. f = lambda a: torch.prod(torch.arange(1,a)); A.apply_(f); #(but apply is slow and only works on CPU)
    # 2. Ramanujan approximation: faster, but more inaccurate than 3.
    # 3. 2-term stirling-type approximation: exact until n = 11, still pretty good after
    R = torch.round(torch.sqrt(2*math.pi*A)*torch.pow(A/math.e,A)*torch.exp(1/(12*A)-1/(360*A**3)))
    return torch.where(A>0,R,eye1)

def sign(x):
    return (x<0)*(-1) + (x>0)

def acos(x):
    # Clipped torch.acos to circumvent numerical issues if x is outside range of validity
    xg = (x>1).float()          # Clip to +1
    xl = (x<-1).float()         # Clip to -1
    xngl = (x>=-1)*(x<=1).float() # No clipping
    x = xg - xl + xngl*x
    return torch.acos(x)

def sinc(x):
    # sinc(x) = sin(x)/x, with well-behaved limit
    return torch.sinc(x/math.pi)
    
def sumf(f,th):
    # evaluate infinite series in style sum_(i=0)^n f(x,i) given f until n = 10
    i = arange10 # incompatible with vmap: torch.arange(n+1).to(device)
    return torch.sum(f(th,i))

def trace(M):
    # return trace of square matrix M
    Tr = 0
    for i in range(len(M)):
        Tr += M[i,i]
    return Tr

def sk(M):
    # returns skew-symmetric part of matrix M
    return (M - M.transpose(-1,-2))/2

def sym(M):
    # returns symmetric part of matrix M
    return (M+M.transpose(-1,-2))/2

#######
### 2. Auxilliary Lie Functions, in order of importance:
#######
"""    
def H_i(i,dim):
    # returns i-th basis R-Matrix or H-Matrix for minimal Atlas, but incompatible with vmap
    if dim == 3:       
        if i == 0:
            R = torch.eye(3)
        elif i == 1:
            R = torch.tensor(( (1, 0, 0), (0, -1, 0), (0, 0, -1)))
        elif i == 2:
            R = torch.tensor(( (-1, 0, 0), (0, 1, 0), (0, 0, -1)))
        elif i == 3:
            R = torch.tensor(( (-1, 0, 0), (0, -1, 0), (0, 0, 1)))
        return R
    else: 
        H = torch.eye(4)
        H[:3,:3] = H_i(i,3)
        return H
"""
    
def R_i(i):
    # returns i-th basis R-Matrix for minimal Atlas
    R = R0*(i==0)\
    +(i==1)*R1\
    +(i==2)*R2\
    +(i==3)*R3
    return R

def H_i(i):
    # returns i-th basis H-Matrix for minimal Atlas
    H = H0*(i==0)\
    +(i==1)*H1\
    +(i==2)*H2\
    +(i==3)*H3
    return H

def inv_SO3(R):
    #inverse of R in SO3
    return R.transpose(-1,-2)

def inv_SE3(H):
    #inverse of H in SE3
    invH = eye4
    R = H[:3,:3]; invR = inv_SO3(R); 
    P = H[:3,3]; invP = torch.unsqueeze(-invR@P,-1)
    return torch.cat((torch.cat((invR,invP),-1), torch.unsqueeze(H[3,:],0)))

def skew_so3(t):
    # return skew form of twists in so3
    A = A_sk_so3
    return torch.einsum('ijk,k->ij',A,t)

def skew_se3(t):
    # return skew form of twists in se3
    A = A_sk_se3
    return torch.einsum('ijk,k->ij',A,t)

def skew_so3_many(t):
    # skew_so3 for collection of t
    A = A_sk_so3
    return torch.einsum('ijk,lk->lij',A,t)

def skew_se3_many(t):
    # skew_se3 for collection of t
    A = A_sk_se3
    return torch.einsum('ijk,lk->lij',A,t)

def unskew_so3(st):
    # inverse operation of skew
    A = A_usk_so3
    return torch.einsum('ijk,jk -> i', A, st)

def unskew_se3(st):
    # inverse operation of skew
    A = A_usk_se3   
    return torch.einsum('ijk,ij -> k', A, st)

    
def skew2_so3(t):
    # return skew(t) @ skew(t) 
    st = skew_so3(t)
    return st @ st 

def skew2_se3(t):
    # return skew(t) @ skew(t) 
    st = skew_se3(t)
    return st @ st 

def skewn_so3(w,n):
    # return n-th power of skew(w) for w in so(3)
    th = angle(w)
    n2 = n % 2
    i = (n-2+n2)/2 
    w_hat = w/(th+(th==0));                                                     # makes sure divison by 0 doesn't occur
    out = torch.pow(-1,i)*torch.pow(th,n)*torch.where(n2==0, skew2_so3(w_hat),skew_so3(w_hat))
    return (n==0)*eye3+(n!=0)*(th!=0)*out


def Ad(H):
    # return big adjoint for H in SE(3)
    R = H[:3,:3]
    p = H[:3, 3]
    sp = skew_so3(p)
    zero = zero33
    r1 = torch.cat((R, zero),-1)
    r2 = torch.cat((sp @ R, R),-1)
    return torch.cat((r1,r2))
    

def ad(t):
    # return small adjoint of twists in se(3)
    w = t[:3]
    v = t[3:]
    sw = skew_so3(w)
    sv = skew_so3(v)
    zero = zero33
    r1 = torch.cat((sw, zero),-1)
    r2 = torch.cat((sv,sw),-1)
    return torch.cat((r1,r2))

def unad(ad_t):
    # inverse operation of ad
    w = unskew_so3(ad_t[:3,:3])
    v = unskew_so3(ad_t[3:,:3])
    return torch.cat((w,v))

def ad_n(t,n):
    # efficient implementation of ad(t)^n
    th = angle(t)
    w = t[:3]
    v = t[3:]
    w_hat = w/(th+(th==0)); v_hat = v/(th+(th==0));

    n2 = n % 2
    k = n-1
    
    sw = skew_so3(w_hat)
    sw2 = sw@sw
    swn = skewn_so3(w,n)
    sv = skew_so3(v_hat)
    
    a = (n2 == 0)*sw@sv + (n2 != 0)*sw@sv@sw
    b = (n2 == 0)*sw@sv@sw2 + (n2 != 0)*sw2@sv
    rn_1  = ((-1)**((k-1)/2))*(th**n)*(a+a.transpose(-1, -2) +(1-k)/2*(b+b.transpose(-1, -2) ))
    rn_2  = ((-1)**(k/2))*(th**n)*((1-k)*a - b + b.transpose(-1, -2) )
    rn = torch.where((n2 == 0),rn_1,rn_2)
    r1 = torch.cat((swn,zero33),-1)
    r2 = torch.cat((rn,swn),-1)
    return torch.where(n==0, eye6, torch.cat((r1,r2)))

def ad_0_5(t):
    # ad_n for all n from 0 to 5
    n = arange10[1:6]
    th = angle(t)
    w = t[:3]
    v = t[3:]
    w_hat = w/(th+(th==0)); v_hat = v/(th+(th==0));

    n2 = n % 2
    k = n-1
    
    sw = skew_so3(w_hat)
    sw2 = sw@sw
    swn = vmap(lambda n: skewn_so3(w,n))(n)
    sv = skew_so3(v_hat)
    
    a0,a1 = sw@sv, sw@sv@sw
    b0,b1 = sw@sv@sw2, sw2@sv
    c_rn0 = torch.pow(-1,((k-1)/2))*torch.pow(th,n)
    c_rn1 = torch.pow(-1,(k/2))*torch.pow(th,n)
    c_rn = torch.where((n2 == 0),c_rn0,c_rn1)
    d_rn0 = vmap(lambda k:(a0+a0.transpose(-1, -2) +(1-k)/2*(b0+b0.transpose(-1, -2) )))(k)
    d_rn1 = vmap(lambda k:((1-k)*a1 - b1 + b1.transpose(-1, -2) ))(k)
    # incomplete from here
    rn  = torch.einsum('i,ij->j', c_rn, torch.stack((d_rn1,d_rn0,d_rn1,d_rn0,d_rn1)))
    r1 = torch.cat((swn,zero33),-1)
    r2 = torch.cat((rn,swn),-1)
    return torch.where(n==0, eye6, torch.cat((r1,r2)))

def dad_n(t,dt,n):
    # implementation of d/dx ad_n(t(x),n), argument dt denotes dt/dx
    th = angle(t)
    dth = dangle(t,dt)
    w = t[:3]
    v = t[3:]
    dw = dt[:3]
    dv = dt[3:]
    
    n2 = n % 2
    k = n-1
    
    sw = skew_so3(w)
    sw2 = sw@sw
    sdw = skew_so3(dw)
    dsw2 = sdw@sw; dsw2 = dsw2+dsw2.transpose(-1, -2) 
    sv = skew_so3(v)
    sdv = skew_so3(dv)
    
    ## case n2 == 0
    a1 = sw@sv
    da1 = sdw@sv + sw@sdv 
    a = a1 + a1.transpose(-1, -2) 
    da = da1 +da1.transpose(-1, -2) 
    
    b1 = sw@sv@sw2
    db1 = sdw@sv@sw2 + sw@sdv@sw2 + sw@sv@dsw2 
    b = b1 + b1.transpose(-1, -2) 
    db = db1 + db1.transpose(-1, -2)            
    
    dw_n1 = (-1)**((n-2)/2)*((n-2)*th**(n-3)*dth*sw2+ th**(n-2)*dsw2);
    dr_n1 = (-1)**((k-1)/2) * ((k-1)*th**(k-2)*dth*a + th**(k-1)*da
                - (k-1)/2*((k-3)*th**(k-4)*dth*b + th**(k-3)*db));
    
    ## case n2 != 0
    a = sw@sv@sw
    da = sdw@sv@sw + sw@sdv@sw + sw@sv@sdw
    b1 = sw2@sv            
    db1 = dsw2@sv + sw2@sdv
    b = b1 - b1.transpose(-1, -2) 
    db = db1 - db1.transpose(-1, -2) 
    dw_n2 = (n2 != 0)*(-1)**((n-1)/2)*((n-1)*th**(n-2)*dth*sw+ th**(n-1)*sdw);
    dr_n2 = (n2 != 0)*(-1)**(k/2) * ( (k-2)*th**(k-3)*dth * ((1-k)*a -b)
                          + th**(k-2) * ((1-k)*da - db))
    
    ## output
    dw_n = torch.where((n2 == 0),dw_n1,dw_n2)
    dr_n = torch.where((n2 ==0 ),dr_n1,dr_n2)
    r1 = torch.cat((dw_n,zero33),1)
    r2 = torch.cat((dr_n,dw_n),1)
    return torch.where(n!=0,torch.cat((r1,r2)),zero66)
     
def LowerTriangular(Vec):
    # Create 6 by  6 Lower Triangular Matrix from Vec with 26 elements
    tril_indices = torch.tril_indices(row=6, col=6, offset=0).to(device)
    diag_ind = (tril_indices[0]==tril_indices[1])
    Vec[diag_ind] = torch.exp(Vec[diag_ind])
    A = torch.zeros(21,6,6); i = range(21)
    A[i,tril_indices[0,i],tril_indices[1,i]] = 1
    return torch.einsum('ijk,i->jk', A, Vec)

#######
### 2.1 Coefficient Functions for derivative of exp and exp SO(6)
#######

   
def KC(th):#,n):
    # Coefficients for Cayley-Hamilton method to calculate derivative of exponential map:
    #       return n-th coefficient of K = sum_n KC(theta,n) ad(t)^n
    p0 = eye1
    p1 = -1/2*eye1
    p2 = 1/(4*th**2)*(8 + 2*torch.cos(th)-10*sinc(th))
    p3 = 1/(4*th**3)*( (12-12*torch.cos(th))/th - 2*torch.sin(th) - 4*th)
    p4 = 1/(4*th**4)*(4+2*torch.cos(th)- 6*sinc(th))
    p5 = 1/(4*th**5)*( (8-8*torch.cos(th))/th - 2*torch.sin(th) - 2*th)
    return torch.stack((p0,p1,p2,p3,p4,p5))#p0*(n == 0)+p1*(n == 1)+p2*(n == 2)+p3*(n == 3)+p4*(n == 4)+p5*(n == 5)
   
def dKC(th,dth):#,n):
    # return n-th coefficient of dK = sum_n dKC(theta,n)*ad(t/theta)^n + KC(theta,n) dad_n
    s = torch.sin(th)
    sc = sinc(th)
    c = torch.cos(th)
    dp0 = zero1
    dp1 = zero1
    dp2 = -(8 - 8*sc + th*s + 7*c-7*sc)/(2*th**3)
    dp3 = (24*(c - 1)/th**2 - c + 9*sc + 4 )/(2*th**3)
    dp4 = -(8 - 8*sc + th*s + 7*c-7*sc)/(2*th**5)
    dp5 =  (24*(c - 1)/th**2 - c + 9*sc + 4)/(2*th**5)
    return torch.stack((dp0,dp1,dp2,dp3,dp4,dp5))*dth#(dp1*(n==1)+dp2*(n==2)+dp3*(n==3)+dp4*(n==4)+dp5*(n==5))*dth

def K2C(th):
    # Coefficients for simplified Cayley-Hamilton to calculate derivative of exponential map
    # Adapted to be more numerically stable from: Eade E. (https://ethaneade.com/exp_diff.pdf)
    
    aS = lambda th, i : torch.pow(-1, i)/factorial(2*i+1)*torch.pow(th,2*i)
    bS = lambda th, i : torch.pow(-1, i)/factorial(2*i+2)*torch.pow(th,2*i)  
    cS = lambda th, i : torch.pow(-1, i)/factorial(2*i+3)*torch.pow(th,2*i)  
    fS = lambda th, i : torch.pow(-1, i+1)*(1/factorial(2*i+3) -2/factorial(2*i+4))*torch.pow(th,2*i)
    gS = lambda th, i : torch.pow(-1, i+1)*(1/factorial(2*i+4) -3/factorial(2*i+5))*torch.pow(th,2*i) 
            
    Bool1 = th > 0.1                                                            # Conservative numerically stable range for trig functions in a,b,c,f,g,e
    Bool3 = torch.abs(torch.sin(th))<0.1                                        # Numerically stable range for e
    
    a = torch.where( Bool1, sinc(th), sumf(aS,th))       
    b = torch.where( Bool1, (1 - torch.cos(th))/th**2, sumf(bS,th) )
    c = torch.where( Bool1, (1 - a)/th**2, sumf(cS,th) )
    f = torch.where( Bool1, (a-2*b)/th**2, sumf(fS,th) )
    g = torch.where( Bool1, (b-3*c)/th**2, sumf(gS,th) )
    e = torch.where( Bool1, (torch.where(Bool3, (b - a/2)/(1-torch.cos(th)), (b-2*c)/(2*a) ) ) , -f/(2*b))

    return torch.stack((a,b,c,e,f,g))

def K2C2(th):
    # K2C with minimal numerical measures
    Bool = torch.abs(torch.sin(th))<0.1
    a = sinc(th)     
    b = (1 - torch.cos(th))/th**2
    c = (1 - a)/th**2
    f = (a-2*b)/th**2
    g = (b-3*c)/th**2
    e = torch.where(Bool, (b - a/2)/(1-torch.cos(th)), (b-2*c)/(2*a) ) 

    return torch.stack((a,b,c,e,f,g))

def dK2C(th):
    # Gradients of terms in K2C w.r.t. input th, including numerical measures
    Bool = th > 0.1
    Bool_e = torch.abs(torch.sin(th))<0.1
    
    aS = lambda th, i : torch.pow(-1, i)/factorial(2*i+1)*torch.pow(th,2*i)
    bS = lambda th, i : torch.pow(-1, i)/factorial(2*i+2)*torch.pow(th,2*i)  
    cS = lambda th, i : torch.pow(-1, i)/factorial(2*i+3)*torch.pow(th,2*i)  
    fS = lambda th, i : torch.pow(-1, i+1)*(1/factorial(2*i+3) -2/factorial(2*i+4))*torch.pow(th,2*i)
    gS = lambda th, i : torch.pow(-1, i+1)*(1/factorial(2*i+4) -3/factorial(2*i+5))*torch.pow(th,2*i) 
    
    daS = lambda th, i : (2*i+2)*torch.pow(-1, i+1)/factorial(2*i+3)*torch.pow(th,2*i+1)
    dbS = lambda th, i : (2*i+2)*torch.pow(-1, i+1)/factorial(2*i+4)*torch.pow(th,2*i+1)  
    dcS = lambda th, i : (2*i+2)*torch.pow(-1, i+1)/factorial(2*i+5)*torch.pow(th,2*i+1)  
    dfS = lambda th, i : (2*i+2)*torch.pow(-1, i+2)*(1/factorial(2*i+5) -2/factorial(2*i+6))*torch.pow(th,2*i+1)
    dgS = lambda th, i : (2*i+2)*torch.pow(-1, i+2)*(1/factorial(2*i+6) -3/factorial(2*i+7))*torch.pow(th,2*i+1)
    
    a = torch.where( Bool, sinc(th), sumf(aS,th))       
    b = torch.where( Bool, (1 - torch.cos(th))/th**2, sumf(bS,th) )
    c = torch.where( Bool, (1 - a)/th**2, sumf(cS,th) )
    f = torch.where( Bool, (a-2*b)/th**2, sumf(fS,th) )
    g = torch.where( Bool, (b-3*c)/th**2, sumf(gS,th) )
    e = torch.where( Bool, (torch.where(Bool_e, (b - a/2)/(1-torch.cos(th)), (b-2*c)/(2*a) ) ) , -f/(2*b))
    
    da = (th*torch.cos(th)-torch.sin(th))/th**2     
    db = (th*torch.sin(th) + 2*torch.cos(th) - 2)/th**3
    dc = (-2 +2*a - th*da) /th**3
    df = (-2*(a-2*b) + th*(da-2*db))/th**3
    dg = (-2*(b-3*c) + th*(db-3*dc))/th**3
    
    da = torch.where(Bool, da, sumf(daS,th))
    db = torch.where(Bool, db, sumf(dbS,th))
    dc = torch.where(Bool, dc, sumf(dcS,th))
    df = torch.where(Bool, df, sumf(dfS,th))
    dg = torch.where(Bool, dg, sumf(dgS,th))

    de1 = -torch.sin(th)*(b - a/2)/(1-torch.cos(th))**2 + (db - da/2)/(1-torch.cos(th))
    de2 = -2*da*(b-2*c)/(2*a)**2 + (db-2*dc)/(2*a)
    de = torch.where(Bool_e, de1, de2 ) 

    return torch.stack((da,db,dc,de,df,dg))

def dK2C2(th):
    # dK2C with minimal numerical measures
    Bool = torch.abs(torch.sin(th))<0.1
    a = sinc(th)     
    b = (1 - torch.cos(th))/th**2
    c = (1 - a)/th**2
    f = (a-2*b)/th**2
    g = (b-3*c)/th**2
    e = torch.where(Bool, (b - a/2)/(1-torch.cos(th)), (b-2*c)/(2*a) ) 
    da = (th*torch.cos(th)-torch.sin(th))/th**2     
    db = (th*torch.sin(th) + 2*torch.cos(th) - 2)/th**3
    dc = (-2 +2*a - th*da) /th**3
    df = (-2*(a-2*b) + th*(da-2*db))/th**3
    dg = (-2*(b-3*c) + th*(db-3*dc))/th**3
    de1 = -torch.sin(th)*(b - a/2)/(1-torch.cos(th))**2 + (db - da/2)/(1-torch.cos(th))
    de2 = -2*da*(b-2*c)/(2*a)**2 + (db-2*dc)/(2*a)
    de = torch.where(Bool, de1, de2 ) 

    return torch.stack((da,db,dc,de,df,dg))


#######
### 3. Important Lie Functions: exp, log, their derivatives, exp_so6, Atlases
#######

def exp_SO3(t):
    # Exponential map for SO(3) and SE(3)
    th = angle(t)
    I = eye3
    return torch.where(th > 1e-10, I + torch.sin(th)*skew_so3(t/th)+(1 - torch.cos(th))*skew2_so3(t/th), I+skew_so3(t))
    

def exp_SE3(t):
    # Exponential map for SO(3) and SE(3)
    th = angle(t)
    I = eye3
    w = t[:3]
    v = t[3:]
    R = exp_SO3(w)
    x = torch.where(th > 1e-10, ( 1/th**2 )*( torch.einsum('ij,j->i', (I - R) @ skew_so3(w) + torch.outer(w,w), v) ), v)
    x = torch.unsqueeze(x,1)
    b_row = torch.unsqueeze(r0001,0)
    return torch.cat((torch.cat((R,x),1),b_row)) 

def log_SO3(H):
    # Logarithmic map for SO(3): adapted vmap-compatible version without if-statements
    
    Tr = trace(H)
    A = (H-H.transpose(-1, -2) )/2
    A2 = A@A
    TrA2 = trace(A2)
    S = (H+H.transpose(-1, -2) )/2
    
    # Case 1: (Tr>-1) & (Tr<3)
    w_Case1 = unskew_so3(acos((Tr-1)/2)/torch.sqrt(-TrA2/2)*A) 
    
    # Case 2: Tr == 3: 
    w_Case2 = unskew_so3(A) # Tr == 3 means R = eye(3)
    
    # Case 3: Tr == -1: vmap-compatible version of human-readable log_SO3_simple further below

    wii = torch.sign(S)
    w_hat2 = (S - eye3)/2
    wi = torch.sqrt(w_hat2+1); wi = torch.diag(torch.diag(wi))
    C = ((S+1)!=0).float()
    C0 = torch.einsum('ij,ij',I0,C)
    C1 = torch.einsum('ij,ij',I1,C)
    notC0 = torch.abs(C0-1)
    notC1 = torch.abs(C1-1)
    z1 = zero1;
    z3 = zero3;
    z33 = zero33; 
    CMat_10 = torch.stack((z1,C0,z1)); CMat_20 = torch.stack((z1,z1,C0));
    CMat_21 = torch.stack((z1,z1,notC0*C1))
    CMat_1 = torch.stack((CMat_10, z3, z3)); CMat_2 = torch.stack((CMat_20,CMat_21,z3))
    CMat = torch.stack((z33, CMat_1, CMat_2)) 
    w_Case3 = math.pi*wi@(torch.einsum('ijk,jk -> i', CMat, wii) + torch.stack((C0, notC0*C1, notC0*notC1)))
    ###

    return torch.where((-TrA2>0), w_Case1, torch.where((Tr>1), w_Case2, w_Case3)) # (-TrA2>0) to avoid nan in case 1, Tr>1 to decide between Case 2 and 3 by which is closer


def log_SE3(H):
    # Logarithmic map for SE(3)
    Tr = trace(H)
    w = log_SO3(H[:3,:3])
    sw = skew_so3(w)
    sw2 = skew2_so3(w)
    th = angle(w)
    sin = torch.sin(th)
    ScaleS_Num = lambda th,i: torch.pow(-1, i+1)*(2/factorial(2*i+3) - 1/factorial(2*i+2))*torch.pow(th,2*i)
    ScaleS_Den = lambda th, i : torch.pow(-1, i)/factorial(2*i+1)*torch.pow(th,2*i)
    Scale = torch.where(sin>0,(2*sin-th*(1+torch.cos(th)))/(2*th**2*sin), 1/th**2)
    Scale = torch.where(th>0.1,Scale,sumf(ScaleS_Num,th)/sumf(ScaleS_Den,th))
    A = eye3 - sw/2 +Scale*sw2
    v = torch.einsum('ij,j->i',A,H[:3,3])
    return torch.cat((w,v))

'''
def log_SO3_simple(R):
    # Logarithmic map for SO(3), not vmap compatible
    Tr = trace(R)
    A = (R-R.transpose(-1, -2) )/2                                              #antisymmetric part of R
    A2 = A@A
    TrA2 = trace(A2)
    S = (R+R.transpose(-1, -2) )/2                                              # symmetric part of R
    if (Tr > -1)& (Tr<3):                                                       # most common case: random R
        sw = math.acos((Tr-1)/2)/math.sqrt(-TrA2/2)*A
        return unskew(sw)
    elif Tr == 3:                                                               # Case R == identity
        return unskew(A)
    else:                                                                       # Case where R rotates through an angle of pi
        w01 = sign(S[0,1])                                                      # w01 = sign(w0*w1) is contained in S
        w02 = sign(S[0,2])                                                      # wjk are used to recover w (not unique, only up to sign of vector)
        w12 = sign(S[1,2])
        w_hat2 = (S - torch.eye(3))/2                                           # all wi**2 are known
        if S[0,0]+1 != 0: 
            w0 = math.sqrt(w_hat2[0,0]+1)
            w1 = w01*math.sqrt(w_hat2[1,1]+1)
            w2 = w02*math.sqrt(w_hat2[2,2]+1)
        elif S[1,1]+1 != 0:           
            w0 = 0
            w1 = math.sqrt(w_hat2[1,1]+1)
            w2 = w12*math.sqrt(w_hat2[2,2]+1)
        else:
            w0 = 0
            w1 = 0
            w2 = math.sqrt(w_hat2[2,2]+1)
        return math.pi*torch.tensor( (w0, w1, w2) )
'''

def K2(t):
    # returns K, the derivative of the exponential map on SE3

    w = t[:3]
    v = t[3:]
    sw = skew_so3(w)
    sw2= sw@sw
    sv = skew_so3(v)
    
    th = torch.unsqueeze(angle(w),0)
    P = K2C(th)
    a, b, c, e, f, g = P[0], P[1], P[2], P[3], P[4], P[5]     
    
    Q = f*sw + g*sw2
    W = Q - 2*c*eye3

    Dw = a*eye3 + b*sw + c*torch.outer(w,w)
    Dv = b*sv + c*(torch.outer(w,v)+torch.outer(v,w))+torch.einsum('i,i',w,v)*W
    r1 = torch.cat((Dw.transpose(-1, -2) , zero33), 1)
    r2 = torch.cat((Dv.transpose(-1, -2) , Dw.transpose(-1, -2) ), 1)
    return torch.cat((r1,r2))

def K2_Inv(t):
    # returns inverse of K using closed-form solution

    w = t[:3]
    v = t[3:]
    sw = skew_so3(w)
    sw2= sw@sw
    sv = skew_so3(v)
    
    th = torch.unsqueeze(angle(w),0)
    P = K2C(th)
    a, b, c, e, f, g = P[0], P[1], P[2], P[3], P[4], P[5]     
    
    Q = f*sw + g*sw2
    W = Q - 2*c*eye3

    Dw = eye3 - sw/2 + e*sw2
    Dv = b*sv + c*(torch.outer(w,v)+torch.outer(v,w))+torch.einsum('i,i',w,v)*W
    Dv = -Dw@Dv@Dw        

    r1 = torch.cat((Dw.transpose(-1, -2) , zero33), 1)
    r2 = torch.cat((Dv.transpose(-1, -2) , Dw.transpose(-1, -2) ), 1)
    return torch.cat((r1,r2))


def dK2(t,dt):
    # returns dK, the second derivative of the exponential map on SE3

    w = t[:3]
    v = t[3:]
    dw = dt[:3]
    dv = dt[3:]
    sw = skew_so3(w)
    sw2= sw@sw
    sv = skew_so3(v)
    dsw = skew_so3(dw)
    dsw2 = sw@dsw + dsw@sw
    dsv = skew_so3(dv)
    
    th = torch.unsqueeze(angle(w),0)
    dth = dangle(t,dt)
    P = K2C(th)
    dP = dK2C(th)*dth
    
    a, b, c, e, f, g = P[0], P[1], P[2], P[3], P[4], P[5]     
    da, db, dc, de, df, dg = dP[0], dP[1], dP[2], dP[3], dP[4], dP[5] 
    
    Q = f*sw + g*sw2
    W = Q - 2*c*eye3
    dQ = df*sw + dg*sw2 + f*dsw + g*dsw2
    dW = dQ - 2*dc*eye3 
    
    dDw = da*eye3 + db*sw + b*dsw + dc*torch.outer(w,w) + c*(torch.outer(dw,w)+ torch.outer(w,dw))
    dDv = db*sv + b*dsv + dc*(torch.outer(w,v)+torch.outer(v,w)) \
            + c*(torch.outer(dw,v)+torch.outer(dv,w)+torch.outer(w,dv) \
            + torch.outer(v,dw))+(torch.einsum('i,i',dw,v) \
            + torch.einsum('i,i',w,dv))*W + torch.einsum('i,i',w,v)*dW
    r1 = torch.cat((dDw.transpose(-1, -2) , zero33), 1)
    r2 = torch.cat((dDv.transpose(-1, -2) , dDw.transpose(-1, -2) ), 1)
    return torch.cat((r1,r2))

def dK2_q(t):
    # returns d/dt K2(t), the second derivative of the exponential map on SE3

    w = t[:3]
    v = t[3:]
    dw = eye6[...,:3]
    dv = eye6[...,3:]
    sw = skew_so3(w)
    sw2= sw@sw
    sv = skew_so3(v)
    dsw = vmap(skew_so3)(dw)
    dsw2 = torch.einsum('ij,kjl->kil', sw,dsw) + torch.einsum('jlk,ki->jli', dsw,sw)
    dsv = vmap(skew_so3)(dv)
    
    th = torch.unsqueeze(angle(w),0)
    dangle_ = lambda dt: dangle(t,dt)
    dth = vmap(dangle_)(eye6)
    
    P = K2C(th)
    dP = dK2C(th)*dth
    a, b, c, e, f, g = P[0], P[1], P[2], P[3], P[4], P[5]     
    da, db, dc, de, df, dg = dP[0], dP[1], dP[2], dP[3], dP[4], dP[5] 
    
    Q = f*sw + g*sw2
    W = Q - 2*c*eye3
    dQ = torch.tensordot(df,sw,dims=0) + torch.tensordot(dg,sw2,dims=0) + f*dsw + g*dsw2
    dW = dQ - 2*torch.tensordot(dc,eye3,dims=0) 
    
    dDw = torch.tensordot(da,eye3,dims=0) + torch.tensordot(db,sw,dims=0) + b*dsw + torch.tensordot(dc,torch.outer(w,w),dims=0) + c*(torch.tensordot(dw,w,dims=0)+torch.tensordot(dw,w,dims=0).transpose(-1,-2))
    dDv = torch.tensordot(db,sv,dims=0) + b*dsv + torch.tensordot(dc,(torch.outer(w,v)+torch.outer(v,w)),dims=0) \
            + c*(torch.tensordot(dw,v,dims=0)+torch.tensordot(dv,w,dims=0)+ torch.tensordot(dw,v,dims=0).transpose(-1,-2)+torch.tensordot(dv,w,dims=0).transpose(-1,-2)) \
            +torch.tensordot((torch.einsum('ji,i',dw,v) + torch.einsum('i,ji',w,dv)),W,dims=0) + torch.tensordot(dW,torch.einsum('i,i',w,v),dims=0)
    r1 = torch.cat((dDw.transpose(-1, -2) , dDw*0), -1)
    r2 = torch.cat((dDv.transpose(-1, -2) , dDw.transpose(-1, -2) ), -1)
    return torch.cat((r1,r2),1)

def dK2_Inv(t,dt):
    # returns dK, the second derivative of the exponential map on SE3

    w = t[:3]
    v = t[3:]
    dw = dt[:3]
    dv = dt[3:]
    sw = skew_so3(w)
    sw2= sw@sw
    sv = skew_so3(v)
    dsw = skew_so3(dw)
    dsw2 = sw@dsw + dsw@sw
    dsv = skew_so3(dv)
    
    th = torch.unsqueeze(angle(w),0)
    dth = dangle(t,dt)
    P = K2C(th)
    dP = dK2C(th)*dth
    
    a, b, c, e, f, g = P[0], P[1], P[2], P[3], P[4], P[5]     
    da, db, dc, de, df, dg = dP[0], dP[1], dP[2], dP[3], dP[4], dP[5] 
    
    Q = f*sw + g*sw2
    W = Q - 2*c*eye3
    dQ = df*sw + dg*sw2 + f*dsw + g*dsw2
    dW = dQ - 2*dc*eye3 
    
    dDw = da*eye3 + db*sw + b*dsw + dc*torch.outer(w,w) + c*(torch.outer(dw,w)+ torch.outer(w,dw))
    dDv = db*sv + b*dsv + dc*(torch.outer(w,v)+torch.outer(v,w)) \
            + c*(torch.outer(dw,v)+torch.outer(dv,w)+torch.outer(w,dv) \
            + torch.outer(v,dw))+(torch.einsum('i,i',dw,v) \
            + torch.einsum('i,i',w,dv))*W + torch.einsum('i,i',w,v)*dW
                                  
    Dw = eye3 - sw/2 + e*sw2
    Dv = b*sv + c*(torch.outer(w,v)+torch.outer(v,w))+torch.einsum('i,i',w,v)*W
    Dv = -dDw@Dv@Dw - Dw@dDv@Dw -Dw@Dv@dDw
    
    r1 = torch.cat((dDw.transpose(-1, -2) , zero33), 1)
    r2 = torch.cat((dDv.transpose(-1, -2) , dDw.transpose(-1, -2) ), 1)
    return torch.cat((r1,r2))

def dK2_q_Inv(t):
    # returns d/dt K2(t), the second derivative of the exponential map on SE3

    w = t[:3]
    v = t[3:]
    dw = eye6[...,:3]
    dv = eye6[...,3:]
    sw = skew_so3(w)
    sw2= sw@sw
    sv = skew_so3(v)
    dsw = vmap(skew_so3)(dw)
    dsw2 = torch.einsum('ij,kjl->kil', sw,dsw) + torch.einsum('jlk,ki->jli', dsw,sw)
    dsv = vmap(skew_so3)(dv)
    
    th = torch.unsqueeze(angle(w),0)
    dangle_ = lambda dt: dangle(t,dt)
    dth = vmap(dangle_)(eye6)
    
    P = K2C(th)
    dP = dK2C(th)*dth
    a, b, c, e, f, g = P[0], P[1], P[2], P[3], P[4], P[5]     
    da, db, dc, de, df, dg = dP[0], dP[1], dP[2], dP[3], dP[4], dP[5] 
    
    Q = f*sw + g*sw2
    W = Q - 2*c*eye3
    dQ = torch.tensordot(df,sw,dims=0) + torch.tensordot(dg,sw2,dims=0) + f*dsw + g*dsw2
    dW = dQ - 2*torch.tensordot(dc,eye3,dims=0) 

    dDw = -dsw/2 + torch.tensordot(de,sw2,dims=0) + e*dsw2
    dDv = torch.tensordot(db,sv,dims=0) + b*dsv + torch.tensordot(dc,(torch.outer(w,v)+torch.outer(v,w)),dims=0) \
            + c*(torch.tensordot(dw,v,dims=0)+torch.tensordot(dv,w,dims=0)+ torch.tensordot(dw,v,dims=0).transpose(-1,-2)+torch.tensordot(dv,w,dims=0).transpose(-1,-2)) \
            +torch.tensordot((torch.einsum('ji,i',dw,v) + torch.einsum('i,ji',w,dv)),W,dims=0) + torch.tensordot(dW,torch.einsum('i,i',w,v),dims=0)
    Dw = eye3 - sw/2 + e*sw2
    Dv = b*sv + c*(torch.outer(w,v)+torch.outer(v,w))+torch.einsum('i,i',w,v)*W
    dDv = -dDw@(Dv@Dw) - Dw@dDv@Dw -(Dw@Dv)@dDw #torch.einsum('ijk,kl->ijl',dDw,(Dv@Dw)) - torch.einsum('ijk,kl->ijl',torch.einsum('ij,kjl->kil',Dw,dDv),Dw) - torch.einsum('ij,kjl->kil',(Dw@Dv),dDw)
    
    r1 = torch.cat((dDw.transpose(-1, -2) , dDw*0), -1)
    r2 = torch.cat((dDv.transpose(-1, -2) , dDw.transpose(-1, -2) ), -1)
    return torch.cat((r1,r2),1)



def dH(H,t,dt):
    # return d/dx H(t(x)), works regardless of chart coordinates t
    return H@skew_se3(K2(t)@dt)

def dH_Inv(H,t,dt):
    # return d/dx H^-1(t(x)), works regardless of chart coordinates t
    H_Inv = inv_SE3(H)
    return H_Inv@dH(H,t,dt)@H_Inv

def dH_q(H,t):
    # return d/dt H(t), works regardless of chart coordinates t
    K = K2(t)
    return H@torch.stack((skew_se3(K[:,0]),skew_se3(K[:,1]),skew_se3(K[:,2]),skew_se3(K[:,3]),skew_se3(K[:,4]),skew_se3(K[:,5])))

def dH_q_Inv(H,t):
    # return d/dt H^-1(t), works regardless of chart coordinates t
    H_Inv = inv_SE3(H)
    return H_Inv@dH_q(H,t)@H_Inv

def dK(t,dt):
    # return d/dx K(t(x)) 
    
    i = arange10[0:6]
    ad_n_all = vmap(lambda n: ad_n(t,n))(i)
    dad_n_all = vmap(lambda n: dad_n(t,dt,n))(i)
    th = angle(t)
    dth = dangle(t,dt)
    KC_th = KC(th) 
    dKC_th = dKC(th,dth)
    Sum = torch.einsum('i,ijk->jk', KC_th, dad_n_all) \
        + torch.einsum('i,ijk->jk', dKC_th, ad_n_all)
    return Sum

def dK_Inv(t,dt):
    # return d/dx K^-1(t(x))
    KInv = K2_Inv(t)
    return -KInv@dK(t,dt)@KInv

def dK_q(t):
    # return d/dt K(t)
    # old method: # torch.stack((dK(t,eye6[0,:]),dK(t,eye6[1,:]),dK(t,eye6[2,:]),dK(t,eye6[3,:]),dK(t,eye6[4,:]),dK(t,eye6[5,:])))
    i = arange10[0:6]
    i4 = arange10[2:6]
    ad_n_all = vmap(lambda n: ad_n(t,n))(i)
    dad_n_all = torch.stack((vmap(lambda n: dad_n(t,eye6[0,:],n))(i),
                             vmap(lambda n: dad_n(t,eye6[1,:],n))(i),
                             vmap(lambda n: dad_n(t,eye6[2,:],n))(i),
                             vmap(lambda n: dad_n(t,eye6[3,:],n))(i),
                             vmap(lambda n: dad_n(t,eye6[4,:],n))(i),
                             vmap(lambda n: dad_n(t,eye6[5,:],n))(i)))
    th = angle(t)
    dth = vmap(lambda dt: dangle(t,dt))(eye6)
    KC_th = KC(th) 
    dKC_th = vmap(lambda dth: dKC(th,dth))(dth)
    return torch.einsum('j,ijkl->ikl', KC_th, dad_n_all) + torch.einsum('ij,jkl->ikl', dKC_th, ad_n_all)

def dK_q_Inv(t):
    # return d/dt K^-1(t)
    KInv = K2_Inv(t)
    return -KInv@dK_q(t)@KInv    

#######
### 3.1 ATLAS, PARTITION OF UNITY AND FUNCTION DEFINITION ON SO(3) & SE(3)
#######

def chart(H,i):
    # return chart coordinates of H in SE(3), in chart i = 0, 1, 2 or 3
    return log_SE3(H_i(i).transpose(-1, -2) @H)
    
def unchart(ti,i):
    # return H in SO(3) or SE(3) correspondeing to chart coordinates ti
    return H_i(i)@exp_SE3(ti) 

def chart_trans(ti,i,j):
    # transition chart coordinates form chart i to chart j
    return chart(unchart(ti,i),j)
 
def dchart_trans_Vec(dti,ti,i,j):
    # transition change-rate of chart coordinates from chart i to chart j, only for SE(3)
    tj = chart_trans(ti,i,j)
    return torch.einsum('i,i', K2_Inv(tj)@K2(ti),dti)
    
def dchart_trans_Co(λi,ti,i,j):
    # transition dual of change-rate of chart coordinates from chart i to chart j, only for SE(3)
    tj = chart_trans(ti,i,j)
    return torch.einsum('ij,j->i', K2(tj).transpose(-1, -2) @K2_Inv(ti).transpose(-1, -2) ,λi)

def dchart_trans_Co_to_Id(λi,ti,i):
    # transition dual of change-rate of chart coordinates from T*T*SE(3) to T*T*_I SE(3)
    return torch.einsum('ij,j->i', K2_Inv(ti).transpose(-1, -2) ,λi)

def chart_trans_mix(xi,i,j):
    # transform xi = (qi,P) 
    q = xi[:6]
    q = chart_trans(q,i,j)
    return torch.cat((q,xi[6:]))

def chart_trans_mix_Co(xi,λi,i,j):
    # transform λi, co-state of of dxi/dt, with xi = (qi,P) 
    q = xi[:6]
    lp = λi[:6]
    lp = dchart_trans_Co(lp,q,i,j)
    return torch.cat((lp,λi[6:])) 

def dchart_trans_mix_Co_to_Id(λi,xi,i):
    # transition co-state of of dxi/dt, with xi = (qi,P) , from T*T*SE(3) to T*T*_I SE(3)
    q = xi[:6]
    lp = λi[:6]
    lp = dchart_trans_Co_to_Id(lp,q,i)
    return torch.cat((lp,λi[6:])) 

def PoU_C2(H):
    # return C2 partition of unity at H in SE(3) 
    return torch.stack((trace(H_i(0)@H), 
                        trace(H_i(1)@H),
                        trace(H_i(2)@H),
                        trace(H_i(3)@H)),0)/4

def PoU_CInf(H):
    # return CInfinity partition of unity at H in SE(3) 
    pC2 = PoU_C2(H)**2
    P = torch.exp(-1/pC2)        
    return P/sum(P)

def PoU_C2q(q):
    # C2 partition of chart i, expressed in coordinates of chart i
    th = angle(q)
    return (1+torch.cos(th))/2

def PoU_CInfq(qi):
    # C Infinity partition of chart i, expressed in coordinates of chart i
    th = angle(qi)
    pi_C2 = PoU_C2q(qi) # value of i-th C2 partition function
    pj_C2 = (1 - pi_C2)*(torch.abs(qi[:3])/th)**2 
    Pj = torch.exp(-1/pj_C2**2)
    Pi = torch.exp(-1/pi_C2**2)
    return Pi/(Pi+torch.sum(Pj))

def dPoU_C2q(q,dq):
    # differential of C 2 partition applied to changerate dq
    th = angle(q)
    dth = dangle(q,dq)
    return -torch.sin(th)*dth

def dPoU_C2(H):
    # return derivative of C2 partition functions at H in SE(3) 
    q0, q1, q2, q3 = chart(H,0), chart(H,1), chart(H,2), chart(H,3)
    dPoU_dq = lambda q : -sinc(angle(q))*torch.cat((q[:3],zero3))
    return torch.stack((dPoU_dq(q0),dPoU_dq(q1),dPoU_dq(q2),dPoU_dq(q3)))

def dPoU_CInf(H):
    # return derivative of CInfinity partition functions at H in SE(3) : Currently Flawed
    #PoU2 = PoU_C2(H)
    #PoU_Inf = PoU_CInf(H)
    #dPoU2 = dPoU_C2(H)
    #dPoU_InfT = PoU_Inf/(PoU2+(PoU2==0))**2*dPoU2.transpose(-1, -2) # element wise, dPoU[i,:]
    raise NotImplementedError
    #return dPoU_InfT.transpose(-1, -2) # i-th row is derivative of i-th partition w.r.t qi 

def ddPoU_C2(H):
    # return hessians of C2 partition functions at H in SE(3) 
    q0, q1, q2, q3 = chart(H,0), chart(H,1), chart(H,2), chart(H,3)
    dsinc_over_th = lambda th: -(th*torch.cos(th)-torch.sin(th))/th**3 # actual dsinc would be divided by th**2, not th**3 
    ddPoU_dq = lambda q : -sinc(angle(q))*torch.cat((torch.cat((eye3,zero33),-1),torch.cat((zero33,zero33),-1))) - dsinc_over_th(angle(q))*(torch.tensordot(torch.cat((q[:3],zero3)),torch.cat((q[:3],zero3)),dims=0))  
    return torch.stack((ddPoU_dq(q0),ddPoU_dq(q1),ddPoU_dq(q2),ddPoU_dq(q3)))

def ddPoU_CInf(H): # NOT IMPLEMENTED
    # return hessians of CInf partition functions at H in SE(3) 
    raise NotImplementedError

def bestChart(H): 
    # return index of chart with largest partition function, so i = 0,1,2 or 3
    # if two have largest partition, return lower chart index (unique answer desired)
    # independent of style of (implemented!) partition of unity
    return torch.argmax(PoU_C2(H)) # == torch.argmax(PoU(H,3))
    
def F_Lie(H,f0,f1,f2,f3):
    # evaluate function as sum of 4 chart components summing with partition of unity
    p = PoU_CInf(H)
    return p[0]*f0(chart(H,0))+p[1]*f1(chart(H,1))+p[2]*f2(chart(H,2))+p[3]*f3(chart(H,3))

def F_Lie2(H,P,U,f0,f1,f2,f3):
    # like F_Lie, but fi(qi,P,U) instead of just qi
    p = PoU_CInf(H)
    return p[0]*f0(chart(H,0),P,U)+p[1]*f1(chart(H,1),P,U)+p[2]*f2(chart(H,2),P,U)+p[3]*f3(chart(H,3),P,U)

def F_Lie3(H,P,f0,f1,f2,f3):
    # like F_Lie, but fi([qi,P]) instead of just qi
    q0, q1, q2, q3 = chart(H,0), chart(H,1), chart(H,2), chart(H,3)
    x0, x1, x2, x3 = torch.cat((q0,P),-1), torch.cat((q1,P),-1), torch.cat((q2,P),-1), torch.cat((q3,P),-1)
    p = PoU_CInf(H)
    return p[0]*f0(x0)+p[1]*f1(x1)+p[2]*f2(x2)+p[3]*f3(x3)

def F_Lie_3_free(H,P,f0,f1,f2,f3):
    # like F_Lie3, but does not use PoU
    q0, q1, q2, q3 = chart(H,0), chart(H,1), chart(H,2), chart(H,3)
    x0, x1, x2, x3 = torch.cat((q0,P),-1), torch.cat((q1,P),-1), torch.cat((q2,P),-1), torch.cat((q3,P),-1)
    return f0(x0) + f1(x1) + f2(x2) + f3(x3) 

def F_Lie_free(H,f0,f1,f2,f3):
    # like F_Lie, but does not use PoU
    q0, q1, q2, q3 = chart(H,0), chart(H,1), chart(H,2), chart(H,3)
    return f0(q0) + f1(q1) + f2(q2) + f3(q3) 

def makeB2(BVec):
    # 6 by 6 pos. def symmetric matrix from 21 dim vector BVec, faster than makeB
    # Uses cholesky decomposition
    L = LowerTriangular(BVec)
    return L@L.transpose(-1,-2)

