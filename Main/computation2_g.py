#%%
# BEGINNING(GPU calculation on cluster)
# Data are stored as 4-D array. The first dimension stores data of different parameter
# parameter g: interaction constant
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

    unit = cp.sqrt(hbar/m/Wz)

    Ggg_tmp = cp.array((4*pi*hbar**2*As*Nbec/m)*unit** -3*(hbar*Wz)**-1)
    Ggg = Ggg_tmp[cp.newaxis, ...]
    Ggg = cp.repeat(Ggg, len(param), axis=0)
    Ggg = Ggg*param
    Gee = cp.zeros_like(Ggg)
    Gee[...] = Ggg
    Geg = cp.zeros_like(Ggg)
    Gge = cp.zeros_like(Ggg)



    Epot = ( (Wx**2*X**2 + Wy**2*Y**2 + Wz**2*Z**2 )
                / (2*Wz**2) )
    Epot = Epot[cp.newaxis, ...]
    Epot = cp.repeat(Epot, len(param), axis=0)

    psiGmu = (15*Ggg / ( 16*pi*cp.sqrt(2) )  )**(2/5)
    psiEmu = (15*Gee / ( 16*pi*cp.sqrt(2) )  )**(2/5)

    # psiGmu = (15*Ggg/(64*cp.sqrt(2)*cp.pi))**(2/5) # for oval potential
    TF_amp = cp.array([(psiGmu[i]-Epot[i])/Ggg[i] for i in range(len(Ggg))])
    cp.clip(TF_amp, 0, cp.inf,out=TF_amp)
    TF_pbb = cp.sqrt(TF_amp)
    total = cp.sum(cp.abs(TF_pbb)**2*dx*dy*dz, axis=(1,2,3))
    n_TF_pbb = cp.array([TF_pbb[i]/cp.sqrt(total[i],dtype=cp.complex128) for i in range(len(Ggg))])

    psiG = cp.array(cp.abs(n_TF_pbb),dtype=cp.complex128)
    psiE = cp.zeros_like(n_TF_pbb)
    # psiG = cp.array(cp.ones(TF_pbb.shape)+5,dtype=cp.complex128)
    # psiE = cp.array(cp.ones(TF_pbb.shape)+5,dtype=cp.complex128)



    # Laguerre-Gaussian laser
    L = 1
    P = 0
    W0 = 5e-6/unit
    Lambda = 790e-9/unit
    Zrl = cp.pi*W0**2/Lambda                         #Rayleigh length
    W= W0*cp.sqrt(1+(Z/Zrl)**2)  
    # Rz = Z + cp.divide(Zrl**2, z, out=cp.zeros_like(z), where=z!=0.0) #use numpy ufunc
    Rz = Z + Zrl**2/Z 
    Guoy = (abs(L)+2*P+1)*cp.arctan2(Z,Zrl) 

    R = cp.sqrt(X**2 + Y**2)
    Phi = cp.arctan2(Y,X)
    AL =((cp.sqrt(2)*R/W))**abs(L)
    _R = R.get()
    _W = W.get()
    ALpoly = cp.array(genlaguerre(P,abs(L))(2*(_R/_W)**2))
    AGauss = cp.exp(-(R/W)**2)
    Ptrans1 = cp.exp(-1j*(2*cp.pi/Lambda)*R**2/(2*Rz)) # Here
    Ptrans2 = cp.exp(-1j*L*Phi)
    PGuoy = cp.exp(1j*Guoy)
    LG = (W0/W)*AL*ALpoly*AGauss*Ptrans1*Ptrans2*PGuoy
    if (L == 0 and P == 0):
        Plong = cp.exp(-1j*((2*cp.pi/Lambda)*Z - Guoy))
        LG = (W0/W)*AGauss*Ptrans1*Ptrans2*Plong
    
    
    LG = 1*LG/cp.max(cp.abs(LG)) 
    LG = 0.5*Rabi*LG
    
    # boradcast
    X = cp.repeat(X[cp.newaxis,...],cp.size(param),axis=0)
    Y = cp.repeat(Y[cp.newaxis,...],cp.size(param),axis=0)
    Z = cp.repeat(Z[cp.newaxis,...],cp.size(param),axis=0)
    Lap = cp.zeros_like(psiG) 


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
        
        psiG_n = dw * (Lap - (Epot + cp.einsum("i,ijkl->ijkl",Ggg,cp.abs(psiG)**2)  \
            + cp.einsum("i,ijkl->ijkl",Gge,cp.abs(psiE)**2)) * psiG \
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
        
        psiE_n = dw * ( Lap - (Epot + cp.einsum("i,ijkl->ijkl",Gee,cp.abs(psiE)**2) + \
            cp.einsum("i,ijkl->ijkl",Geg,cp.abs(psiG)**2) - 1j*d/2) * psiE \
                        - LG*psiG  + cp.einsum("i,ijkl->ijkl",psiEmu,psiE)) + psiE
        
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
            # psiGmu = psiGmu/(Nfactor)
            # psiEmu = psiEmu/(Nfactor)  
            J = J + 1
            psiGmuArray[:,J] = psiGmu
            psiEmuArray[:,J] = psiEmu
            
        if ((j+1) % stepJ) == 0 and j != 0: #last step must store
            # storing data
            fs =  [ h5py.File( join( expanduser(path), fileformat.format(j+1,i) ) , "w" ) for i in param ]
            _psiG  = psiG.get()
            _psiE  = psiE.get()
            _psiGmuArray = psiGmuArray.get()
            _psiEmuArray = psiEmuArray.get()
            _convergeG = convergeG.get()
            _convergeE = convergeE.get()
            _LG = LG.get()
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
                f['LG'] = _LG
                f['psiGmuArray'] = _psiGmuArray[idx,...]
                f['psiEmuArray'] = _psiEmuArray[idx,...]
                f['convergeG'] = _convergeG[idx,...]
                f['convergeE'] = _convergeE[idx,...]
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
                f['Parameters/Ggg'] = _Ggg[idx,...]
                f['Parameters/Gee'] = _Gee[idx,...]
                f['Parameters/Gge'] = _Gge[idx,...]
                f['Parameters/Geg'] = _Geg[idx,...] 
                f['Parameters/L'] = L
                f['Parameters/W0'] = _W0
                f['Parameters/Lambda'] = _Lambda
                # print("storing succeeded!")
    
            _fs = [ f.close() for f in fs ]
                
        
    return 
    
#%%
if __name__ == "__main__":
    Gs = [1,0.8,0.5,0.001]
    fileformat = "g{}_test.h5"
    n = 10000
    computation(Gs,n, n/2, fileformat )
# %%
