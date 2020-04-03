from __future__ import division
import numpy
from pylab import log, sqrt, exp
from scipy.integrate import ode
from scipy.special import hyp2f1
from numpy import array
from pylab import pi, sqrt, log, log10, sin
from scipy.special import erf

class clusterBH:
    def __init__(self, N, rhoh, **kwargs):
        self.G  = 0.004499 # pc^3 /Msun /Myr^2

        # Cluster ICs
        self.m0 = 0.638    # For Kroupa (2001) IMF 0.1-100 Msun
        self.N = N
        self.M0 = self.m0*N
        self.rh0 = (3*self.M0/(8*pi*rhoh))**(1./3)
        self.fc = 1 # equation (50)
        
        # BH MF
        self.mlo = 3
        self.mup = 30
        self.alpha = 0.5
        
        # Model parameters
        self.zeta = 0.1
        self.a0 = 1 # fix zeroth order
        self.a2 = 0 # ignore 2nd order for now
        self.f0 = 0.06  # for "metal-poor" GCs
        self.kick = False
        self.fretm = 1        
        self.tsev = 2.

        
        # Parameters that were fit to N-body
        self.ntrh =  3.21
        self.beta = 0.00280
        self.nu = 0.0823
        self.a1 = 1.47

        self.sigmans = 265 # km/s
        self.mns = 1.4 # Msun

        # Some integration params
        self.tend = 12e3
        self.dtout = 2 # Myr

        self.output = False
        self.outfile = "cluster.txt"

        # Mass loss mechanism
        self.tidal = False
        
        # Check input parameters
        if kwargs is not None:
            for key, value in kwargs.items():
                setattr(self, key, value)

        self.vesc0 = 50*(self.M0/1e5)**(1./3)*(rhoh/1e5)**(1./6)
                        
        self.vesc0 *= self.fc
        
        if (self.kick):
            mb = (9*pi/2)**(1./6)*self.sigmans*self.mns/self.vesc0
            self.mb = mb
            mu, ml, a, a2 = self.mup, self.mlo, self.alpha, self.alpha + 2
            
            qul, qub, qlb = mu/ml, mu/mb, ml/mb

            b = a2/3

            h1 = hyp2f1(1,b,b+1, -qub**3)
            h2 = hyp2f1(1,b,b+1, -qlb**3)
        
            if a != -2:
                self.fretm = 1 - (qul**a2*h1 - h2)/(qul**a2-1)
            else:
                self.fretm = log( (qub**3+1)/(qlb**3+1) )/log(qul**3)

        self.Mbh0 = self.fretm*self.f0*self.M0

        self.trh0 = self._trh(self.M0, self.rh0, self.f0*self.fretm)
        self.tcc = self.ntrh*self.trh0

        self.evolve(N, rhoh)

    def _psi(self, fbh):
        psi = self.a0  + self.a1*abs(fbh)/0.01 + self.a2*(abs(fbh)/0.01)**2
        return psi
    
    def _trh(self, M, rh, fbh):
        m = M/self.N
        if M>0 and rh>0:
            return 0.138*sqrt(M*rh**3/self.G)/(m*self._psi(fbh)*10)
        else:
            return 1e-99


    def find_mmax(self, Mbh):
        a2 = self.alpha+2
        
        # Note that a warning for alpha = -2 is needed
        if (self.kick):
            def integr(mm, qmb, qlb):
                a2 = self.alpha+2
                b = a2/3
                h1 = hyp2f1(1,b,b+1, -qmb**3)
                h2 = hyp2f1(1,b,b+1, -qlb**3)
                
                return mm**a2*(1-h1) - self.mlo**a2*(1-h2)

            # invert eq. 52 from AG20
            Np = 1000
            mmax_ = numpy.linspace(self.mlo, self.mup, Np)
            qml, qmb, qlb  = mmax_/self.mlo, mmax_/self.mb, self.mlo/self.mb

            A = Mbh[0]/integr(self.mup, self.mup/self.mb, qlb)

            Mbh_ = A * integr(mmax_, qmb, qlb)
            mmax = numpy.interp(Mbh, Mbh_, mmax_)
        else:
            # eq 51 in AG20
            mmax = (Mbh/self.Mbh0 * (self.mup**a2 - self.mlo**a2) + self.mlo**a2)**(1./a2)
        return mmax
            
    def odes(self, t, y):
        Mst = y[0]
        Mbh = y[1]
        rh = y[2]
        
        M = Mst + Mbh
        fbh = Mbh/M
        
        trh = self._trh(M, rh, fbh)
        tcc = self.tcc 
        tsev = self.tsev

        Mst_dot, rh_dot, Mbh_dot = 0, 0, 0
        
        # Stellar mass loss
        if t>=tsev:
            Mst_dot -= self.nu*Mst/t
            rh_dot -= Mst_dot/M*rh 
            
            # Add tidal mass loss
        if (self.tidal):
            # Remove mass needed to turn over GCMF
            # Delta M ~ 2e5 Msun
            # Dekta t ~ 10 Gyr
            # Mdot ~ 20 Msun/Myr
            Mst_dot -= 20

        # BH escape
        if t>tcc:
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
        Mst = [self.M0]   #  MG 29/1/2020 should be self.M0-self.Mbh0 ???
        Mbh = [self.Mbh0]
        rh = [self.rh0]

        y = [Mst[0], Mbh[0], rh[0]]

        sol = ode(self.odes)
        sol.set_integrator('dopri5') 
        sol.set_initial_value(y,0)
    
        t = [0]

        while t[-1] <= self.tend:
            tnext = t[-1] + self.dtout

            sol.integrate(tnext)

            Mst.append(sol.y[0])
            Mbh.append(sol.y[1])
            rh.append(sol.y[2])
            t.append(tnext)

        self.t = array(t)
        self.Mst = array(Mst)
        self.Mbh = array(Mbh)
        self.rh = array(rh)
        self.mmax = self.find_mmax(self.Mbh)


        
        # Some derived quantities
        self.M = self.Mst + self.Mbh
        self.fbh = self.Mbh/self.M

        self.E = -self.G*self.M**2/(2*self.rh)
        self.trh = numpy.zeros_like(self.M)
        for i in range(len(self.trh)):
            self.trh[i] = self._trh(self.M[i], self.rh[i], self.fbh[i])

        if (self.output):
            f = open(self.outfile,"w")
            for i in range(len(self.t)):
                f.write("%12.5e %12.5e %12.5e %12.5e %12.5e\n"%(self.t[i]/1e3, self.Mbh[i],
                                                                self.M[i], self.rh[i],
                                                                self.mmax[i]))
            f.close()





