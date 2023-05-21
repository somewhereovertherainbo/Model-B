# Model-B 

Model B is a phase separation model of two mixed liquids into separated ones.

The equations of motion for Model B are:

$$
\dot{\phi} =-\boldsymbol{\nabla}\cdot \left(J^\phi +J^{\Lambda}\right).
$$

$J^\Lambda$ is the current arising from 'Thermal Noise'

$J^\phi$ is the current that minimizes the Free energy of the system. 

the currents are written as follows:

$$
\boldsymbol{J}^\phi=-M^{\phi}\boldsymbol{\nabla}\mu^{\phi},\qquad\mu^{\phi}=\frac{\delta\mathcal{F}}{\delta\phi}
$$

$\boldsymbol{\Lambda}$ is a zero-mean, unit-variance Gaussian white noise. 


The equilibrium $\mathcal{F}$ is the Landau-Ginzburg
free energy functional given by:

$$
\mathcal{F}[\phi]=\int\left(\frac{a}{2}\phi^{2}+\frac{b}{4}\phi^{4}+\frac{\kappa}{2}(\boldsymbol{\nabla}\phi)^{2}
%+\frac{D}{2}c^{2}-\beta\phi c
\right)d\boldsymbol{r} = \int f \ d\boldsymbol{r} 
$$

So, the final dynamical equation in the variable $\phi$ is 

$$
\dot{\phi} =-\nabla^2 \left[a\phi + b \phi^3 - \kappa \nabla^2 \phi \right] - \sqrt{2D^{\phi}M^{\phi}}\nabla\cdot\Lambda
$$

# Numerical Solution using Pseudo-Spectral Method:

In Fourier space, the dynamical equations are of the form:

$$
\dot{\phi}_q=\alpha(q) \phi_q+\hat{N}_q
$$

Multiplying both sides by $\exp (-\alpha(k) t)$ gives

$$
\frac{d\left(\phi_q e^{-\alpha t}\right)}{dt}=\frac{d \psi}{dt}=e^{-\alpha t} \hat{N}_q
$$

Thus, we can solve for $\psi_q=\phi_q e^{-\alpha t}$ and then compute
  
$$
\phi_q(t+dt)=\psi_q(t+dt) e^{\alpha(t+dt)}=\left[\psi_q(t)+dt\left(e^{-\alpha t} \hat{N}_q\right)\right] e^{\alpha(t+dt)}
$$

This can be simplified to 
  
$$
\phi_q (t+dt) = \left[\phi_q (t) + dt\hat{N}_q   \right] e^{\alpha dt}
$$

The code for simulation and the results:

```
import numpy as np
import sys, time
import matplotlib.pyplot as plt
from scipy.io import savemat, loadmat
fft2  = np.fft.fft2
ifft2 = np.fft.ifft2
randn = np.random.randn

from matplotlib import rc
rc('text', usetex=True)
fSA=20
font = {'family' : 'normal',
         'weight' : 'bold',
         'size'   : 32}  
rc('font', **font)
```

```
class FPS():
    '''Class to simulate field theories using a PSEUDO-SPECTRAL TECHNIQUE WITH FILTERING'''
    def __init__(self, param):
        self.Nt = param['Nt']
        self.dt = param['dt']
        self.Nf = param['Nf']
        self.h  = param['h']
        self.a  = param['a']
        self.b  = param['b']
        self.kp = param['kp']

        self.Ng = param['Ng']
        self.Df = param['Df']
        Ng = self.Ng
        
        self.XX  = np.zeros((int(self.Nf+1), Ng*Ng)) 
        
        self.kx  = np.fft.fftfreq(Ng)*(2*np.pi/self.h)
        self.ky  = np.fft.fftfreq(Ng)*(2*np.pi/self.h)
        self.kx, self.kx = np.meshgrid(self.kx, self.kx) 
        self.ksq = self.kx*self.kx + self.ky*self.ky

        alpha = -self.ksq*(self.a + self.kp*self.ksq)
        self.eA = np.exp(alpha*self.dt) 

        
    def integrate(self, u ):
        '''  simulates the equation and plots it at different instants '''
        ii=0;  t=0;  dt=self.dt;    Df=self.Df; simC=(100/self.Nt)
        if Df==0:
            rhs=self.rhs
        else:
            rhs=self.rhsW
        for i in range(self.Nt):          
            u = rhs(u, dt)

            if i%(int(self.Nt/self.Nf))==0:  
                self.XX[ii,:] = (np.real(np.fft.ifft2(u))).flatten()
                ii += 1   
                if ii%50==0:
                    print (int(simC*i), '% done', end=' ')
  
                
    def rhs(self, u, dt):
        '''
        returns the right hand side of \dot{phi} in active model H
        \dot{phi} = Δ(a*u + b*u*u*u + kΔu + λ(∇u)^2) 
        '''
        uc = ifft2(u);      N_u=-self.ksq*self.b*fft2(uc*uc*uc)        
        u = (u + N_u*dt)*self.eA
        return u  
    
    
    def rhsW(self, u, dt):
        '''
        returns the right hand side of \dot{phi} in active model H
        \dot{phi} = Δ(a*u + b*u*u*u + kΔu + λ(∇u)^2) 
        '''
        uc = ifft2(u);      N_u=-self.ksq*self.b*fft2(uc*uc*uc)        
        u = (u + N_u*dt)*self.eA
        duN = self.Df*( 1j*self.kx*randn(Ng,Ng) + 1j*self.ky*(randn(Ng,Ng)))
        return u + duN 

                    
    def dW(self):
        '''
        returns the right hand side of \dot{phi} in active model H
        \dot{phi} = Δ(a*u + b*u*u*u + kΔu + λ(∇u)^2) 
        '''
        duN = self.Df*(1j*self.kx*(randn(Ng,Ng)) + 1j*self.ky*(randn(Ng,Ng)))

        return duN 
    
    
    
    def configPlot(U, fig, n_, i):
        import matplotlib.pyplot as plt
        sp =  fig.add_subplot(1, 5, n_ )   

        im=plt.pcolor(U, cmap=plt.cm.RdBu_r);  plt.clim(-1.1, 1.1);
        cbar = plt.colorbar(im,fraction=0.04, pad=0.05, orientation="horizontal", 
                            ticks=[-1, 0, 1])

        plt.axis('off'); plt.axis('equal'); plt.title('T = %1.2E'%(i))
```

```
fig = plt.figure(num=None, figsize=(11, 6), dpi=100)
plt.rcParams.update({'font.size': 12})

xx=1.56;  x = np.linspace(-xx,xx,128); y = -0.5*x*x + .25*x**4
y1 = np.linspace(-.245,0,128, );   y2 = np.linspace(-.245,.28,128, )

plt.plot(x, y, color='tab:red',lw=4)
plt.plot(x, y*0, color='slategray',lw=4, alpha=0.6); 
plt.plot(x*0, y2, color='slategray',lw=4, alpha=0.6)
plt.plot(0*x+1, y1, '--', color='tab:blue',lw=4, label=r'binodals $\phi_b$')
plt.plot(0*x-1, y1, '--', color='tab:blue',lw=4)

plt.plot(0*x+np.sqrt(1/3), y1*0.55, '--', color='tab:olive',lw=4, label=r'spinodals $\phi_s$')
plt.plot(0*x-np.sqrt(1/3), y1*0.55, '--', color='tab:olive',lw=4 )
plt.axis('off'); #plt.legend(loc='upper left');
plt.legend(fontsize="20", bbox_to_anchor=(.12, .85))

print ('spinodal:', np.sqrt(1/3)) 
```

```
Deff, Ng = .0, 64; 
Nt, dt, Nf = int(1.25e6), .02, 200

phi0 = 0; Df=0.0;
phi0 = phi0+ (1-phi0)*(1-2*np.random.random((Ng,Ng)));  


param = {'h':1, 'Ng':Ng, 'a':-0.25, 'b':0.25, 'kp':1,
         'Nt':Nt, 'dt':dt, 'Nf':Nf, 'Df':Df}


am = FPS(param);   
t1 = time.perf_counter();     am.integrate(fft2(phi0) )

print ('total time taken: ', time.perf_counter()-t1)
```



```
phi0 = 0.6 ; Df=2;
phi0 = phi0+ (1-phi0)*(1-2*np.random.random((Ng,Ng)));  


param = {'h':1, 'Ng':Ng, 'a':-0.25, 'b':0.25, 'kp':1,
         'Nt':Nt, 'dt':dt, 'Nf':Nf, 'Df':Df}


am = FPS(param);   
t1 = time.perf_counter();     am.integrate(fft2(phi0) )

print ('total time taken: ', time.perf_counter()-t1)
```

```
fig = plt.figure(num=None, figsize=(22, 4), dpi=200);
ti=0;     configPlot(am.XX[ti,::].reshape(Ng, Ng), fig, 1, ti)
ti=5;     configPlot(am.XX[ti,::].reshape(Ng, Ng), fig, 2, ti)
ti=50;    configPlot(am.XX[ti,::].reshape(Ng, Ng), fig, 3, ti)
ti=100;   configPlot(am.XX[ti,::].reshape(Ng, Ng), fig, 4, ti)
ti=199;   configPlot(am.XX[ti,::].reshape(Ng, Ng), fig, 5, ti);
```
