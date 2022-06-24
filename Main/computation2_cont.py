#%%
#BEGINNING(GPU calculation on cluster)
#Data are stored as 4-D array. The first dimension stores data of different parameter
#Reading data in ~/Data and continue to run


import cupy as cp
import numpy as np
import h5py
import utils.dirutils as dd
import os
from os.path import join, expanduser
from scipy.special import genlaguerre
import sys

def computation(read, added_n ,fileformat):

    
    # reading previous case from ~/Data (stored in h5py format)
    data = dd.retrieve(read)
    module = sys.modules[__name__]
    for name, value in data.items():
        setattr(module, name, value)
        
    psiG = cp.array(globals()['psiG'])
    psiE = cp.array(globals()['psiE'])
    LG = cp.array(globals()['LG'])
    nj = globals()['nj'] + added_n
    start = globals()['j']

        
    x = cp.linspace(-Lx,Lx,Nx)
    y = cp.linspace(-Ly,Ly,Ny)
    z = cp.linspace(-Lz,Lz,Nz)
    dx = cp.diff(x)[0]
    dy = cp.diff(y)[0]
    dz = cp.diff(z)[0]


    [X,Y,Z] = cp.meshgrid(x,y,z)
    
    # Some constants
    pi = 3.14159265359
    hbar = 1.054571800139113e-34 

    
    unit = cp.sqrt(hbar/m/Wz)

    Ggg = cp.array((4*pi*hbar**2*As*Nbec/m)*unit**  -3*(hbar*Wz)**-1)
    Gee = cp.array(Ggg)
    Gge = cp.array(0)
    Geg = cp.array(0)
    
    Epot = ( (Wx**2*X**2 + Wy**2*Y**2 + Wz**2*Z**2 )
                / (2*Wz**2) )
    psiGmu = (15*Ggg / ( 16*pi*cp.sqrt(2) )  )**(2/5)  
    psiEmu = (15*Gee / ( 16*pi*cp.sqrt(2) )  )**(2/5) 

    Lap = cp.zeros_like(psiG) 
    J = 0


    fileformat = "{}_" + fileformat
    base = dd.base()
    path = join(base, 'tmp')
    try: 
        os.mkdir(path)
    except OSError as error: 
        print(error)
        
        
        
    for j in range(start+1, nj):
        Lap[1:Ny-1,1:Nx-1,1:Nz-1] = (
                (0.5/dy**2)*(
                        psiG[2:Ny,   1:Nx-1, 1:Nz-1] 
                    - 2*psiG[1:Ny-1, 1:Nx-1, 1:Nz-1] 
                    + psiG[0:Ny-2, 1:Nx-1, 1:Nz-1])
                +(0.5/dx**2)*(
                        psiG[1:Ny-1, 2:Nx,   1:Nz-1] 
                    - 2*psiG[1:Ny-1, 1:Nx-1, 1:Nz-1] 
                    + psiG[1:Ny-1, 0:Nx-2, 1:Nz-1])
                +(0.5/dz**2)*(
                        psiG[1:Ny-1, 1:Nx-1, 2:Nz]
                    - 2*psiG[1:Ny-1, 1:Nx-1, 1:Nz-1] 
                    + psiG[1:Ny-1, 1:Nx-1, 0:Nz-2]))
        
        psiG_n = dw * (Lap - (Epot + Ggg*cp.abs(psiG)**2 + Gge*cp.abs(psiE)**2) * psiG \
                        - cp.conjugate(LG)*psiE + psiGmu*psiG) + psiG 
        

        Lap[1:Ny-1,1:Nx-1,1:Nz-1] = (
                (0.5/dy**2)*(
                        psiE[2:Ny,   1:Nx-1, 1:Nz-1] 
                    - 2*psiE[1:Ny-1, 1:Nx-1, 1:Nz-1] 
                    + psiE[0:Ny-2, 1:Nx-1, 1:Nz-1])
                +(0.5/dx**2)*(
                        psiE[1:Ny-1, 2:Nx,   1:Nz-1] 
                    - 2*psiE[1:Ny-1, 1:Nx-1, 1:Nz-1] 
                    + psiE[1:Ny-1, 0:Nx-2, 1:Nz-1])
                +(0.5/dz**2)*(
                        psiE[1:Ny-1, 1:Nx-1, 2:Nz]
                    - 2*psiE[1:Ny-1, 1:Nx-1, 1:Nz-1] 
                    + psiE[1:Ny-1, 1:Nx-1, 0:Nz-2]))
        psiE_n = dw * ( Lap - (Epot + Gee*cp.abs(psiE)**2 + Geg*cp.abs(psiG)**2 - 1j*d/2) * psiE \
                        - LG*psiG  + psiEmu*psiE) + psiE
        
        
        psiE[...] = psiE_n
        psiG[...] = psiG_n
        
            
        
        if (j % stepJ) == 0 and j != 0:
            #  update energy constraint 
            SumPsiG = cp.sum( cp.abs(psiG)**2*dx*dy*dz, axis=(0,1,2))
            SumPsiE = cp.sum( cp.abs(psiE)**2*dx*dy*dz, axis=(0,1,2))
            Nfactor = SumPsiG +  SumPsiE  
            # Update energy
            # psiGmu = psiGmu/(Nfactor)
            # psiEmu = psiEmu/(Nfactor)  
            J = J + 1

            
        if ((j+1) % stepJ) == 0 and j != 0: #last step must store
            # storing data
            fs =   h5py.File( join( expanduser(path), fileformat.format(j+1,L) ) , "w" )
            _psiG  = psiG.get()
            _psiE  = psiE.get()
            _LG  = LG.get()

            _Ggg = Ggg.get()
            _Gee = Gee.get()
            _Gge = Gge.get()
            _Geg = Geg.get()
        
            
            
            fs['psiG'] = _psiG[...]
            fs['psiE'] = _psiE[...]
            fs['LG'] = _LG[...]
            fs['Metaparameters/j'] = j
            fs['Metaparameters/Nx'] = Nx
            fs['Metaparameters/Ny'] = Ny
            fs['Metaparameters/Nz'] = Nz
            fs['Metaparameters/Lx'] = Lx
            fs['Metaparameters/Ly'] = Ly
            fs['Metaparameters/Lz'] = Lz
            fs['Metaparameters/dw'] = dw
            fs['Metaparameters/nj'] = nj
            fs['Metaparameters/stepJ'] = stepJ
            fs['Parameters/As'] = As
            fs['Parameters/Nbec'] = Nbec
            fs['Parameters/Rabi'] = Rabi
            fs['Parameters/m'] = m
            fs['Parameters/Wx'] = Wx
            fs['Parameters/Wy'] = Wy
            fs['Parameters/Wz'] = Wz
            fs['Parameters/dw'] = dw
            fs['Parameters/Ggg'] = _Ggg
            fs['Parameters/Gee'] = _Gee
            fs['Parameters/Gge'] = _Gge
            fs['Parameters/Geg'] = _Geg 
            fs['Parameters/L'] = L
            fs['Parameters/W0'] = W0
            fs['Parameters/Lambda'] = Lambda
    
            _fs = fs.close()
                
        
    return 
    
#%%
if __name__ == "__main__":
    file = os.path.join(dd.base(), 'tmp', '100000_L7_10um_1e-6.h5')
    fileformat = "L{}_10um_1e-6.h5"
    addn = 900000
    computation(file, addn, fileformat)
# %%
