#%%
# BEGINNING(GPU calculation on cluster)
# Data are stored as 4-D array. The first dimension stores data of different parameter
# parameter L: angular momentum of the beam

import cupy as cp
import numpy as np
import h5py
import utils.dirutils as dd
import os
from os.path import join, expanduser
from scipy.special import genlaguerre


def computation(parameter,nj,stepJ,fileformat):
    param = cp.array(parameter)
    fileformat = "{}_" + fileformat
    base = dd.base()
    path = join(base, 'tmp')
    try: 
        os.mkdir(path)
    except OSError as error: 
        print(error)
    # Meta-parameters and parameters
    Nx = 121
    Ny = 121
    Nz = 121
    Lx = 20
    Ly = 20
    Lz = 20
    x = cp.linspace(-Lx,Lx,Nx)
    y = cp.linspace(-Ly,Ly,Ny)
    z = cp.linspace(-Lz,Lz,Nz)
    dx = cp.diff(x)[0]
    dy = cp.diff(y)[0]
    dz = cp.diff(z)[0]
    dw = 1e-5   # condition for converge : <1e-3*dx**2

    [X,Y,Z] = cp.meshgrid(x,y,z)
    
    # Some constants
    pi = 3.14159265359
    hbar = 1.054571800139113e-34 
    m = 1.411000000000000e-25  # Rb atoms
    # BEC parameters
    As = 5.82e-09
    Nbec = 400000
    Rabi = 57
    d = 0  #Raman detuning
    Wx = 452
    Wy = 452
    Wz = 509
    # unit = 1.222614572474304e-06;
    unit = cp.sqrt(hbar/m/Wz)

    Ggg = cp.array((4*pi*hbar**2*As*Nbec/m)*unit**  -3*(hbar*Wz)**-1)
    Gee = cp.array(Ggg)
    Gge = cp.array(0)
    Geg = cp.array(0)
    
    
    Epot = ( (Wx**2*X**2 + Wy**2*Y**2 + Wz**2*Z**2 )
                / (2*Wz**2) )


    psiGmu = (15*Ggg / ( 16*pi*cp.sqrt(2) )  )**(2/5)  
    psiEmu = (15*Gee / ( 16*pi*cp.sqrt(2) )  )**(2/5) 

    # psiGmu = (15*Ggg/(64*cp.sqrt(2)*cp.pi))**(2/5) # for oval potential
    TF_amp = cp.array((psiGmu-Epot)/Ggg)
    cp.clip(TF_amp, 0, cp.inf,out=TF_amp)
    TF_pbb = cp.sqrt(TF_amp)
    total = cp.sum(cp.abs(TF_pbb)**2*dx*dy*dz)
    n_TF_pbb = TF_pbb/cp.sqrt(total,dtype=cp.complex128)

    psiG = cp.array(cp.abs(n_TF_pbb),dtype=cp.complex128)     
    psiE = cp.zeros_like(n_TF_pbb)
    # psiG = cp.array(cp.ones(TF_pbb.shape)+5,dtype=cp.complex128)
    # psiE = cp.array(cp.ones(TF_pbb.shape)+5,dtype=cp.complex128)

    # boradcast
    psiGmu = cp.repeat(psiGmu, cp.size(param))
    psiEmu = cp.repeat(psiEmu,cp.size(param))
    X = cp.repeat(X[cp.newaxis,...],cp.size(param),axis=0)
    Y = cp.repeat(Y[cp.newaxis,...],cp.size(param),axis=0)
    Z = cp.repeat(Z[cp.newaxis,...],cp.size(param),axis=0)
    n_TF_pbb = cp.repeat(n_TF_pbb[cp.newaxis,...],cp.size(param),axis=0)
    psiG = cp.repeat(psiG[cp.newaxis,...],cp.size(param),axis=0)
    psiE = cp.repeat(psiE[cp.newaxis,...],cp.size(param),axis=0)
    Lap = cp.zeros_like(psiG) 
    
    # Laguerre-Gaussian laser
    L = cp.array(param)
    W0 = 5e-6/unit
    Lambda = 790e-9/unit
    P = 0
    Zrl = cp.pi*W0**2/Lambda                         #Rayleigh length
    W= W0*cp.sqrt(1+(Z/Zrl)**2)  
    Rz = Z + Zrl**2/Z 
    Guoy = cp.einsum("i,ijkl->ijkl",(cp.abs(L)+2*P+1),cp.arctan2(Z,Zrl) )
    R = cp.sqrt(X**2 + Y**2)
    Phi = cp.arctan2(Y,X)
    AL = cp.array([data**(cp.abs(L[i])) for (i,data) in enumerate(cp.sqrt(2)*R/W)])
    _L = L.get()
    _R = R.get()
    _W = W.get()
    ALpoly = cp.array( [ genlaguerre( P, np.abs( _L[i] ) ) ( 2 * ( _R[i] / _W[i] ) ** 2 ) for i in range( np.size( _L ) ) ] )
    AGauss = cp.exp(-(R/W)**2)
    Ptrans1 = cp.exp(-1j*(2*cp.pi/Lambda)*R**2/(2*Rz)) # Here
    Ptrans2 = cp.exp(-1j*  cp.einsum("i,ijkl->ijkl",L, Phi))
    PGuoy = cp.exp(1j*Guoy)
    LG = (W0/W)*AL*ALpoly*AGauss*Ptrans1*Ptrans2*PGuoy
    


    #Cover light in half along x-axis
    # LG[:, :, 60:, :] = 0

    
    #TODO: correct the Gaussian beam
    # if (L == 0 and P == 0):
    #     Plong = cp.exp(-1j*((2*cp.pi/Lambda)*Z - Guoy))
    #     LG = (W0/W)*AGauss*Ptrans1*Ptrans2*Plong
    
    
    LG = 1*LG/cp.max(cp.abs(LG)) 
    LG = 0.5*Rabi*LG


    psiEmuArray = cp.zeros((len(param), int( np.ceil(nj/stepJ) )), dtype=cp.float32)
    psiGmuArray = cp.zeros((len(param), int( np.ceil(nj/stepJ) )), dtype=cp.float32)
    J = 0
    psiGmuArray[:,J] = psiGmu
    psiEmuArray[:,J] = psiEmu
    
    

    for j in range(nj):
        Lap[:, 1:Ny-1,1:Nx-1,1:Nz-1] = (
                (0.5/dy**2)*(
                        psiG[:, 2:Ny,   1:Nx-1, 1:Nz-1] 
                    - 2*psiG[:, 1:Ny-1, 1:Nx-1, 1:Nz-1] 
                    + psiG[:, 0:Ny-2, 1:Nx-1, 1:Nz-1])
                +(0.5/dx**2)*(
                        psiG[:, 1:Ny-1, 2:Nx,   1:Nz-1] 
                    - 2*psiG[:, 1:Ny-1, 1:Nx-1, 1:Nz-1] 
                    + psiG[:, 1:Ny-1, 0:Nx-2, 1:Nz-1])
                +(0.5/dz**2)*(
                        psiG[:, 1:Ny-1, 1:Nx-1, 2:Nz]
                    - 2*psiG[:, 1:Ny-1, 1:Nx-1, 1:Nz-1] 
                    + psiG[:, 1:Ny-1, 1:Nx-1, 0:Nz-2]))
        
        psiG_n = dw * (Lap - (Epot + Ggg*cp.abs(psiG)**2 + Gge*cp.abs(psiE)**2) * psiG \
                        - cp.conjugate(LG)*psiE + cp.einsum("i,ijkl->ijkl",psiGmu,psiG)) + psiG 
        

        Lap[:, 1:Ny-1,1:Nx-1,1:Nz-1] = (
                (0.5/dy**2)*(
                        psiE[:, 2:Ny,   1:Nx-1, 1:Nz-1] 
                    - 2*psiE[:, 1:Ny-1, 1:Nx-1, 1:Nz-1] 
                    + psiE[:, 0:Ny-2, 1:Nx-1, 1:Nz-1])
                +(0.5/dx**2)*(
                        psiE[:, 1:Ny-1, 2:Nx,   1:Nz-1] 
                    - 2*psiE[:, 1:Ny-1, 1:Nx-1, 1:Nz-1] 
                    + psiE[:, 1:Ny-1, 0:Nx-2, 1:Nz-1])
                +(0.5/dz**2)*(
                        psiE[:, 1:Ny-1, 1:Nx-1, 2:Nz]
                    - 2*psiE[:, 1:Ny-1, 1:Nx-1, 1:Nz-1] 
                    + psiE[:, 1:Ny-1, 1:Nx-1, 0:Nz-2]))
        psiE_n = dw * ( Lap - (Epot + Gee*cp.abs(psiE)**2 + Geg*cp.abs(psiG)**2 - 1j*d/2) * psiE \
                        - LG*psiG  +cp.einsum("i,ijkl->ijkl",psiEmu,psiE)) + psiE
        
        if ((j+1) % stepJ) == 0 or j == 0:
            # convergence test
            lmaxE = cp.abs(cp.max(psiE, axis=(1,2,3)))
            cmaxE = cp.abs(cp.max(psiE_n, axis=(1,2,3)))
            lmaxG = cp.abs(cp.max(psiG, axis=(1,2,3)))
            cmaxG = cp.abs(cp.max(psiG_n, axis=(1,2,3)))
            diffG = cp.abs(cmaxG - lmaxG)/cmaxG
            diffE = cp.abs(cmaxE - lmaxE)/cmaxE
            diffG = diffG[:,cp.newaxis]
            diffE = diffE[:,cp.newaxis]
            if (j == 0):
                convergeG = cp.zeros((len(param),1))
                convergeE = cp.zeros((len(param),1))
                convergeG[:,0] = diffG[:,0]
                convergeE[:,0] = diffE[:,0]
            else:
                convergeG = cp.append(convergeG, diffG,axis=1)
                convergeE = cp.append(convergeE, diffE,axis=1)
            
        psiE = psiE_n
        psiG = psiG_n
        
            
        
        if (j % stepJ) == 0 and j != 0:
            #  update energy constraint 
            SumPsiG = cp.sum( cp.abs(psiG)**2*dx*dy*dz, axis=(1,2,3))
            SumPsiE = cp.sum( cp.abs(psiE)**2*dx*dy*dz, axis=(1,2,3))
            Nfactor = SumPsiG +  SumPsiE  
            # Update energy
            psiGmu = psiGmu/(Nfactor)
            psiEmu = psiEmu/(Nfactor)  
            J = J + 1
            psiGmuArray[:,J] = psiGmu
            psiEmuArray[:,J] = psiEmu
            
        if ((j+1) % stepJ) == 0 and j != 0: #last step must store
            # storing data
            fs =  [ h5py.File( join( expanduser(path), fileformat.format(j+1,i) ) , "w" ) for i in param ]
            _psiG  = psiG.get()
            _psiE  = psiE.get()
            _LG  = LG.get()
            _psiGmuArray = psiGmuArray.get()
            _psiEmuArray = psiEmuArray.get()
            convergeG = convergeG.get()
            convergeE = convergeE.get()
            _Ggg = Ggg.get()
            _Gee = Gee.get()
            _Gge = Gge.get()
            _Geg = Geg.get()
            _W0 = W0.get()
            _Lambda = Lambda.get()
            
            # one file store one case
            for idx, f in enumerate(fs):
                f['psiG'] = _psiG[idx,...]
                f['psiE'] = _psiE[idx,...]
                f['LG'] = _LG[idx,...]
                f['psiGmuArray'] = _psiGmuArray[idx,...]
                f['psiEmuArray'] = _psiEmuArray[idx,...]
                f['convergeG'] = convergeG[idx,...]
                f['convergeE'] = convergeE[idx,...]
                f['Metaparameters/j'] = j
                f['Metaparameters/Nx'] = Nx
                f['Metaparameters/Ny'] = Ny
                f['Metaparameters/Nz'] = Nz
                f['Metaparameters/Lx'] = Lx
                f['Metaparameters/Ly'] = Ly
                f['Metaparameters/Lz'] = Lz
                f['Metaparameters/dw'] = dw
                f['Metaparameters/nj'] = nj
                f['Metaparameters/stepJ'] = stepJ
                f['Parameters/As'] = As
                f['Parameters/Nbec'] = Nbec
                f['Parameters/Rabi'] = Rabi
                f['Parameters/m'] = m
                f['Parameters/Wx'] = Wx
                f['Parameters/Wy'] = Wy
                f['Parameters/Wz'] = Wz
                f['Parameters/dw'] = dw
                f['Parameters/Ggg'] = _Ggg
                f['Parameters/Gee'] = _Gee
                f['Parameters/Gge'] = _Gge
                f['Parameters/Geg'] = _Geg 
                f['Parameters/L'] = _L[idx]
                f['Parameters/W0'] = _W0
                f['Parameters/Lambda'] = _Lambda
                # print("storing succeeded!")
    
            _fs = [ f.close() for f in fs ]
                
        
    return 
    
#%%
if __name__ == "__main__":
    L1 = np.arange(2,7)
    fileformat = "L{}_test.h5"
    n = 1000
    computation(L1,n,500,fileformat)
# %%
