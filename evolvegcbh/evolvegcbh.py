from __future__ import division
import numpy
from pylab import log, sqrt, exp
from scipy.integrate import ode
from numpy import array
from pylab import pi, sqrt, log, log10, sin
from readcol import readcol
from scipy.special import erf

class evolvegcbh:
    def __init__(self, N, rhoh, **kwargs):
        self.G  = 0.004499 # pc^3 /Msun /Myr^2

        # Cluster ICs
        self.m0 = 0.638    # For Kroupa (2001) IMF 0.1-100 Msun
        self.N = N
        self.M0 = self.m0*N
        self.rh0 = (3*self.M0/(8*pi*rhoh))**(1./3)
        self.vesc0 = 50*(self.M0/1e5)**(1./3)*(rhoh/1e5)**(1./6)

        # Model parameters
        self.v0 = 31.5
        self.zeta = 0.1
        self.a0 = 1
        self.f0 = 0.06
        self.kick = False
        self.fret = 1        
        
        self.tsev = 2
        self.alpha =  6.22
        self.beta = 0.00259
        self.nu = 0.0765
        self.a1 = 1.91
        self.a2 = 0 # ingore 2nd order for now

        # Check input parameters
        if kwargs is not None:
            for key, value in kwargs.iteritems():
                setattr(self, key, value)

        if (self.kick):
            self.fret = erf(self.vesc0/self.v0)**3

        self.trh0 = self._trh(self.M0, self.rh0, self.f0*self.fret)
                    
        self.evolve(N, rhoh)

    def rk4(self, t, y, ydot, dt):
        # Take RK4 step
        k1 = ydot(t, y)*dt
        k2 = ydot(t + 0.5*dt, y + 0.5*k1)*dt
        k3 = ydot(t + 0.5*dt, y + 0.5*k2)*dt
        k4 = ydot(t + dt, y + k3)*dt
        y += (k1 + 2*(k2 + k3) + k4)/6
        return y

    def _psi(self, fbh):
        return self.a0  + self.a1*abs(fbh)/0.01 + self.a2*(abs(fbh)/0.01)**2 
    
    def _trh(self, M, rh, fbh):
        m = M/self.N
        if M>0 and rh>0:
            return 0.138*sqrt(M*rh**3/self.G)/(m*self._psi(fbh)*10)
        else:
            return 1e-99

    def odes(self, t, y):
        Mst = y[0]
        Mbh = y[1]
        rh = y[2]
        
        M = Mst + Mbh
        fbh = Mbh/M
        
        trh = self._trh(M, rh, fbh)
        tcc = self.alpha*self.trh0
        tsev = self.tsev

        Mst_dot, rh_dot, Mbh_dot = 0, 0, 0
        
        # Stellar mass loss
        if t>tsev:
            Mst_dot -= self.nu*Mst/t 
            rh_dot -= Mst_dot/M*rh 
        
        # BH escape
        if t>tcc:
            rh_dot = 0 # Note reset of rh_dot
            rh_dot += self.zeta*rh/trh
            rh_dot += 2*Mst_dot/M * rh
            
            if Mbh > 0:
                Mbh_dot = -self.beta*M/trh 
                rh_dot += 2*Mbh_dot/M * rh

        derivs = [Mst_dot]
        derivs.append(Mbh_dot)
        derivs.append(rh_dot)

        return numpy.array(derivs)

    def evolve(self, N, rhoh):
        Mst = [self.M0]
        Mbh = [self.fret*self.f0*self.M0]
        rh = [self.rh0]
        
        y = [Mst[0], Mbh[0], rh[0]]

        sol = ode(self.odes)
        sol.set_integrator('dopri5') 
        sol.set_initial_value(y,0)
    
        t = [0]
        tnext = self.tsev

        while t[-1] < 15e3:
            dt =  0.05*max([t[-1],self.tsev])
            y = self.rk4(t[-1], y, self.odes, dt)
            Mst.append(y[0])
            Mbh.append(y[1])
            rh.append(y[2])
            t.append(t[-1]+dt)

        self.t = array(t)
        self.Mst = array(Mst)
        self.Mbh = array(Mbh)
        self.rh = array(rh)
        
        # Some derived quantities
        self.M = self.Mst + self.Mbh
        self.fbh = self.Mbh/self.M

