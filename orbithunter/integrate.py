from __future__ import print_function, division
from math import pi
import numpy as np
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import h5py
warnings.resetwarnings()
from scipy.fftpack import fft,ifft
from orbithunter import *
import os
from scipy.linalg import eig,inv


def spatial_integration(orbit,*args,**kwargs):
    return None


def ETDRK4(orbit,**kwargs):
    symmetry=kwargs.get('symmetry','ppo')
    UU,N,M,T,L,S = orbit
    nstp = kwargs.get('nstp',N*M)

    if N==1:
        u = np.reshape(UU,M)
    else:
        u = UU[-1,:]
    v = fft(u)/np.sqrt(M)
    u = np.reshape(u,[M,1])
    v = np.reshape(v,[M,1])
    h = T/nstp

    k = (2*pi*M/L)*np.fft.fftfreq(M)
    k = np.reshape(k,[M,1])
    k[int(M//2)] = 0


    L_op = (k**2.0)-(k**4.0)
    E = np.exp(h*L_op)
    E2 = np.exp(h*L_op/2.0)

    RoU = 16
    RoUarray = np.arange(1,RoU+1,1)

    r = np.exp(1.0j*pi*(RoUarray-0.5)/RoU)
    r = np.reshape(r,(1,RoU))

    LR = h*np.tile(L_op, (1,RoU)) + np.tile(r,(M,1))
    Q =  h*np.real(np.mean( (np.exp(LR/2)-1.0)/LR,axis=1))
    f1 = h*np.real(np.mean( (-4.0-LR+np.exp(LR)*(4.0-3.0*LR+LR**2))/LR**3 , axis=1))
    f2 = h*np.real(np.mean( (2.0+LR+np.exp(LR)*(-2.0+LR))/LR**3 , axis=1))
    f3 = h*np.real(np.mean( (-4.0-3.0*LR-LR**2+np.exp(LR)*(4.0-LR))/LR**3 , axis=1))

    Q = np.reshape(Q,[M,1])
    f1 = np.reshape(f1,[M,1])
    f2 = np.reshape(f2,[M,1])
    f3 = np.reshape(f3,[M,1])

    tmax = T
    nmax = int(tmax/h)
    g = -0.5j*k
    uu = np.zeros([M,nmax],float)
    tt = np.zeros([nmax,1],float)
    vv = np.zeros([M,nmax],complex)
    uu[0:M,0] = u[0:M,0]
    vv[0:M,0]= v[0:M,0]

    for step in range(0,nmax):
        t = step*h
        Nv = g*fft(np.real(ifft(v,axis=0)*np.sqrt(M))**2,axis=0)/np.sqrt(M)
        a = E2*v + Q*Nv
        Na = g*fft(np.real(ifft(a,axis=0)*np.sqrt(M))**2,axis=0)/np.sqrt(M)
        b = E2*v + Q*Na
        Nb = g*fft(np.real(ifft(b,axis=0)*np.sqrt(M))**2,axis=0)/np.sqrt(M)
        c = E2*a + Q*(2.0*Nb-Nv)
        Nc = g*fft(np.real(ifft(c,axis=0)*np.sqrt(M))**2,axis=0)/np.sqrt(M)
        v = E*v + Nv*f1 + 2.0*(Na+Nb)*f2 + Nc*f3
        v[0] = 0
        v[int(M // 2)] = 0
        u = np.sqrt(M)*np.real(ifft(v,axis=0))
        uu[0:M,step] = u[0:M,0]
        tt[step] = t
        if np.mod(step,3000)==0:
            print(t)
    uu = np.flipud(uu.transpose())
    # vv = np.flipud(vv.transpose())
    # plt.figure()
    # plt.imshow(uu,cmap='jet')
    # plt.show()
    orbit_tmp = (uu,nstp,M,T,L,S)
    orbit = orbit_init.relevant_symmetry_operation(orbit_tmp,symmetry=symmetry)

    return orbit


def ETDRK4_reproduce_L22():
    for n in range(1,241):
        symmetry='ppo'
        nname = str(n)
        dataname = ''.join(['/',symmetry,'/',nname])
        filename = ''.join([symmetry,nname])
        # file1 = h5py.File('ks22h02t100E.h5', 'r')
        file1 = h5py.File(''.join([symmetry,"_L22.h5"]))
        datasetr = file1[''.join([dataname,'/r'])]
        dataseti = file1[''.join([dataname,'/i'])]
        S0 = float(file1[''.join([dataname,'/shift'])][0])
        T = float(file1[''.join([dataname,'/time'])][0])
        r = np.array(datasetr[:])
        c = np.array(dataseti[:])
        file1.close()
        L0=22
        N=32
        v = np.reshape(1*r + 1j*c,[15,1])
        v = np.concatenate(([[0]],v,np.zeros([N-31,1]),np.flipud(np.conj(v))),axis=0)
        N = np.size(v)
        v = np.reshape(v,[N,1])
        u = np.real(ifft(v))*N

        h = T/4096

        k = (2*pi*N/L0)*np.fft.fftfreq(N)
        k = np.reshape(k,[N,1])
        k[int(N//2)] = 0

        k = (2*pi*N/L0)*np.fft.fftfreq(N)
        k = np.reshape(k,[N,1])
        k[int(N//2)] = 0

        L = (k**2.0)-(k**4.0)
        E = np.exp(h*L)
        E2 = np.exp(h*L/2.0)

        M = 16
        Marray = np.arange(1,M+1,1)

        r = np.exp(1.0j*pi*(Marray-0.5)/M)
        r = np.reshape(r,(1,M))

        LR = h*np.tile(L, (1,M)) + np.tile(r,(N,1))
        Q =  h*np.real(np.mean( (np.exp(LR/2)-1.0)/LR,axis=1))
        f1 = h*np.real(np.mean( (-4.0-LR+np.exp(LR)*(4.0-3.0*LR+LR**2))/LR**3 , axis=1))
        f2 = h*np.real(np.mean( (2.0+LR+np.exp(LR)*(-2.0+LR))/LR**3 , axis=1))
        f3 = h*np.real(np.mean( (-4.0-3.0*LR-LR**2+np.exp(LR)*(4.0-LR))/LR**3 , axis=1))

        Q = np.reshape(Q,[N,1])
        f1 = np.reshape(f1,[N,1])
        f2 = np.reshape(f2,[N,1])
        f3 = np.reshape(f3,[N,1])

        tmax = T
        nmax = int(tmax/h)
        g = -0.5j*k*N
        uu = np.zeros([N,nmax],float)
        tt = np.zeros([nmax,1],float)
        vv = np.zeros([N,nmax],complex)
        uu[0:N,0] = u[0:N,0]
        vv[0:N,0]= v[0:N,0]

        for step in range(0,nmax):
            t = step*h
            Nv = g*fft(np.real(ifft(v,axis=0))**2,axis=0)
            a = E2*v + Q*Nv
            Na = g*fft(np.real(ifft(a,axis=0))**2,axis=0)
            b = E2*v + Q*Na
            Nb = g*fft(np.real(ifft(b,axis=0))**2,axis=0)
            c = E2*a + Q*(2.0*Nb-Nv)
            Nc = g*fft(np.real(ifft(c,axis=0))**2,axis=0)
            v = E*v + Nv*f1 + 2.0*(Na+Nb)*f2 + Nc*f3
            v[0] = 0
            v[int(N // 2)] = 0
            u = N*np.real(ifft(v,axis=0))
            uu[0:N,step] = u[0:N,0]
            vv[0:N,step] = v[0:N,0]
            tt[n] = t

        uu = np.flipud(uu.transpose())
        N,M = np.shape(uu)

        if symmetry=='rpo':
            k_vec = ((2*pi*M)/L0)*np.fft.fftfreq(M)
            k_vec[int(M//2)]=0

            orbit_tuple = (uu,N,M,T,L0,0)
            S0 = rpo.calculate_shift2(orbit_tuple)
            vvcorrect = fft(uu,axis=1)/M
            tt = np.flipud(tt)
            complex_rotation = np.exp(1j*tt*k_vec*(S0/T))
            vv_movingframe = np.multiply(complex_rotation,vvcorrect)
            uu_movingframe = np.real(ifft(vv_movingframe,axis=1))*M
            uu = uu_movingframe
            S = -S0
        elif symmetry=='ppo':
            uu = np.concatenate((-1.0*np.roll(np.fliplr(uu),1,axis=1),uu),axis=0)
            T = 2*T
            S0 = 0


        time_trunc_number=128
        folder = ''.join("C:\\Users\\matth\\Desktop\\GuBuCv17\\")
        save_filename = ''.join([folder,filename,'_',str(time_trunc_number),'b',str(32),'.h5'])
        N,M = np.shape(uu)
        if symmetry=='rpo':
            orbit_tuple = (uu,N,M,T,L0,S0)
            orbit_tuple = rpo.mvf_rotate_orbit(orbit_tuple)
            uu,N,M,T,L0,S0 = orbit_tuple
            orbit_tuple = (uu,N,M,T,L0,-S0)
            orbit_tuple_full = rpo.mvf_rotate_orbit(orbit_tuple)
        # uu,N,M,T,L0,S0 = orbit_tuple
        # plt.figure()
        # plt.imshow(uu,interpolation='nearest',extent=[0,L0,0,T],cmap='jet')
        # plt.colorbar()
        # plt.xlabel('x')
        # plt.ylabel('t')
        # plt.show()
        orbit_tuple=(uu,N,M,T,L0,S0)
        orbit_tuple = disc.rediscretize(orbit_tuple, newN=time_trunc_number)
        uu_truncated,N,M,T,L0,S0 = orbit_tuple
        # plt.figure()
        # plt.imshow(uu_truncated,interpolation='nearest',extent=[0,L0,0,T],cmap='jet')
        # plt.colorbar()
        # plt.xlabel('x')
        # plt.ylabel('t')
        # plt.show()
        orbit_io.export_orbit(save_filename, orbit_tuple,symmetry=symmetry)
    return None


def stability_matrix(Lop,QX,FFT,IFFT,v):
    u = np.dot(IFFT,v)
    Du = np.diag(np.reshape(u,np.size(u)))
    Dux = np.diag(np.reshape(np.dot(IFFT,np.dot(QX,v)),np.size(u)))
    NL0 = np.dot(FFT,np.dot(Du,np.dot(IFFT,QX)))
    NL1 = np.dot(FFT,np.dot(Dux,IFFT))
    A_n = Lop-(NL0+NL1)
    return A_n


def nonlinear_stability_matrix(DX,FFT,IFFT,v):
    return np.dot(DX,np.dot(FFT,np.dot(np.diag(np.reshape(np.dot(IFFT,v),np.size(v))),IFFT)))
    # u = np.dot(IFFT,v)
    # Du = np.diag(np.reshape(u,np.size(u)))
    # Dux = np.diag(np.reshape(np.dot(IFFT,np.dot(QX,v)),np.size(u)))
    # NL0 = np.dot(FFT,np.dot(Du,np.dot(IFFT,QX)))
    # NL1 = np.dot(FFT,np.dot(Dux,IFFT))
    # A_NL = -1.0*(NL0+NL1)
    # return NONLINEARMATRIX


def ustability_matrix(LINEAR,DX,FFT,IFFT,u):
    Du = np.diag(np.reshape(u,np.size(u)))
    NL = np.dot(FFT,np.dot(Du,IFFT))
    A_n = LINEAR-0.5*np.dot(DX,NL)
    return A_n


def ETDRK4_jacobian(orbit0,**kwargs):
    symmetry=kwargs.get('symmetry','ppo')
    nstp=kwargs.get('nstp',4096)
    stepsbetweensave=kwargs.get('stepsbetweensaves',1)

    U,N,M,T,L,S = orbit0
    orbit0 = disc.rediscretize(orbit0,newN=8*N)

    # if symmetry=='ppo':
    #     T=T/2
    #     orbit0 = (U,N,M,T,L,S)
    if symmetry=='rpo':
        orbit0 = symm.frame_rotation(orbit0)
        U,N,M,T,L,S = orbit0

    FFT = ksdm.FFT_x_matrix(1,M,symmetry=symmetry)
    IFFT = ksdm.IFFT_x_matrix(1,M,symmetry=symmetry)
    DX = ksdm.DX(1,M,L,symmetry=symmetry)

    u = U[-1,:]
    u = np.reshape(u,[M,1])
    v = np.dot(FFT,u)
    v = np.reshape(v,[M-2,1])

    h = T/nstp

    k = (2*pi*M/L)*np.fft.fftfreq(M)
    k = np.reshape(k,[M,1])
    k=k[1:int(M//2)]
    k = np.concatenate((k,k),axis=0)
    k = np.reshape(k,[M-2,1])

    l_op = (k**2.0)-(k**4.0)
    e = np.exp(h*l_op)
    e2 = np.exp(h*l_op/2.0)

    RoU = 16
    RoUarray = np.arange(1,RoU+1,1)
    r = np.exp(1.0j*pi*(RoUarray-0.5)/RoU)
    r = np.reshape(r,(1,RoU))

    lr = h*np.tile(l_op, (1,RoU)) + np.tile(r,(M-2,1))
    q =  h*np.real(np.mean( (np.exp(lr/2)-1.0)/lr,axis=1))
    f1 = h*np.real(np.mean( (-4.0-lr+np.exp(lr)*(4.0-3.0*lr+lr**2))/lr**3 , axis=1))
    f2 = h*np.real(np.mean( (2.0+lr+np.exp(lr)*(-2.0+lr))/lr**3 , axis=1))
    f3 = h*np.real(np.mean( (-4.0-3.0*lr-lr**2+np.exp(lr)*(4.0-lr))/lr**3 , axis=1))

    q = np.reshape(q,[M-2,1])
    f1 = np.reshape(f1,[M-2,1])
    f2 = np.reshape(f2,[M-2,1])
    f3 = np.reshape(f3,[M-2,1])

    '''jacobian quantities'''
    # Jdot quantities given by unraveled A(x).

    L_OP = np.reshape(np.tile((k**2.0)-(k**4.0),(1,M-2)),(M-2)**2)
    E = np.exp(h*L_OP)
    E2 = np.exp(h*L_OP/2.0)

    ROU = 16
    ROUarray = np.arange(1,RoU+1,1)
    R = np.exp(1.0j*pi*(RoUarray-0.5)/RoU)
    R = np.reshape(R,(1,ROU))

    LR = h*np.tile(L_OP, (1,ROU)) + np.tile(R,((M-2)**2,1))
    Q =  h*np.real(np.mean( (np.exp(LR/2)-1.0)/LR,axis=1))
    F1 = h*np.real(np.mean( (-4.0-LR+np.exp(LR)*(4.0-3.0*LR+LR**2))/LR**3 , axis=1))
    F2 = h*np.real(np.mean( (2.0+LR+np.exp(LR)*(-2.0+LR))/LR**3 , axis=1))
    F3 = h*np.real(np.mean( (-4.0-3.0*LR-LR**2+np.exp(LR)*(4.0-LR))/LR**3 , axis=1))

    Q = np.reshape(Q,[(M-2)**2,1])
    F1 = np.reshape(F1,[(M-2)**2,1])
    F2 = np.reshape(F2,[(M-2)**2,1])
    F3 = np.reshape(F3,[(M-2)**2,1])


    # diagQ = np.diag(np.reshape(Q,np.size(Q)))
    # diagF1 = np.diag(np.reshape(f1,np.size(f1)))
    # diagF2 = np.diag(np.reshape(f2,np.size(f2)))
    # diagF3 = np.diag(np.reshape(f3,np.size(f3)))
    # diagE = np.diag(np.reshape(E,np.size(E)))
    # diagE2 = np.diag(np.reshape(E2,np.size(E2)))

    # Qtotal = np.concatenate((Q,q),axis=0)
    # F1total = np.concatenate((F1,f1),axis=0)
    # F2total = np.concatenate((F2,f2),axis=0)
    # F3total = np.concatenate((F3,f3),axis=0)


    '''
    Am I doing this completely wrong? Jdot = AJ is linear in J. Shouldn't I just make unraveled
    A(x) equal to the linear operator L_OP?? Use built in routine???
    
    I think the key point is that L_OP is independent of all variables, NL_OP is not.
    
    '''
    # Do I have J as a matrix? if so that's wrong; want a NxN dimensional vector;
    # Really want N**2 + N dimensional vector which stores J and u.

    uu = np.zeros([M,nstp],float)
    vv = np.zeros([M-2,nstp],float)
    uu[:,0] = u[:,0]
    vv[:,0]= v[:,0]
    J = np.reshape(np.eye(M-2),[M**2,1])
    v0 = np.copy(v)
    savecounter = 0
    savenumber = int(nstp/stepsbetweensave)
    JJ = np.zeros([M-2,M-2,savenumber])



    for step in range(0,nstp):
        t = step*h
        Nv = np.dot(-0.5*DX,np.dot(FFT,np.real(np.dot(IFFT,v))**2))
        a = e2*v + q*Nv
        Na = np.dot(-0.5*DX,np.dot(FFT,np.real(np.dot(IFFT,a))**2))
        b = e2*v + q*Na
        Nb = np.dot(-0.5*DX,np.dot(FFT,np.real(np.dot(IFFT,b))**2))
        c = e2*a + q*(2.0*Nb-Nv)
        Nc = np.dot(-0.5*DX,np.dot(FFT,(np.real(np.dot(IFFT,c))**2)))

        Av = np.reshape(nonlinear_stability_matrix(DX,FFT,IFFT,v),(M**2))
        Aa = np.reshape(nonlinear_stability_matrix(DX,FFT,IFFT,a),(M**2))
        Ab = np.reshape(nonlinear_stability_matrix(DX,FFT,IFFT,b),(M**2))
        Ac = np.reshape(nonlinear_stability_matrix(DX,FFT,IFFT,c),(M**2))

        JNv = np.multiply(Av,J)
        Ja = np.multiply(E,J) + np.multiply(Q,JNv)
        JNa = np.multiply(Aa,Ja)
        Jb = np.multiply(E2,J) + np.dot(Q,JNa)
        JNb = np.multiply(Ab,Jb)
        Jc = np.multiply(E2,Ja) + np.multiply(Q,(2*JNb-JNv))
        JNc = np.multiply(Ac,Jc)

        J = np.multiply(E, J) + np.multiply(F1, JNv) + np.multiply(2 * F2, (JNa + JNb)) + np.multiply(F3, JNc)

        v = e*v + Nv*f1 + 2.0*(Na+Nb)*f2 + Nc*f3
        u = np.dot(IFFT,v)
        uu[:,step] = u[:,0]
        step+=1

        if np.mod(step,stepsbetweensave)==0 and step!=0:
            JJ[:,:,savecounter] = J
            J = np.eye(M-2)
            savecounter+=1


    uu = np.flipud(uu.transpose())

    Jp = np.eye(M-2)
    for n in range(0,savenumber):
        Jp = np.dot(JJ[:,:,n],Jp)

    eigenvalues,eigenvectors = eig(J)
    real_exponents = (1/T)*np.log(np.abs(eigenvalues))

    return J,eigenvalues,real_exponents


def jacobianeigenvalues(orbit_filename,**kwargs):
    symmetry=kwargs.get('symmetry','ppo')
    nstp=kwargs.get('nstp',4096)
    orbit0 = orbit_io.import_orbit(orbit_filename)
    U,N,M,T,L,S = orbit0
    # if symmetry=='ppo':
    #     T=T/2
    #     orbit0 = (U,N,M,T,L,S)


    FFT = ksdm.ppo.FFT_x_matrix(1,M)
    IFFT = ksdm.ppo.IFFT_x_matrix(1,M)
    DX = -1.0*ksdm.ppo.DX(1,M,L)
    LINEAR = -1.0*ksdm.ppo.LINEAR(1,M,L)
    k = (2*pi*M/L)*np.fft.fftfreq(M)
    k = np.reshape(k,[M,1])
    k=k[1:int(M//2)]
    k = np.concatenate((k,k),axis=0)
    k = np.reshape(k,[M-2,1])
    L_op = np.reshape((k**2.0)-(k**4.0),M-2)
    LINEAR = np.diag(L_op)

    uu = ETDRK4_timeseries(orbit0,nstp=nstp)
    orbit_integrated = (uu,np.shape(uu)[0],np.shape(uu)[1],T,L,S)
    uuSR = -1.0*np.roll(np.roll(np.fliplr(uu),1,axis=1),(np.shape(uu)[0])//2,axis=0)
    orbit_integratedSR = (uuSR,np.shape(uu)[0],np.shape(uu)[1],T,L,S)
    orbit_diff = (uu-uuSR,np.shape(uu)[0],np.shape(uu)[1],T,L,S)
    ksplot.plot_spatiotemporal_field(orbit_integrated,symmetry='none',display_flag=True,padding=False)
    ksplot.plot_spatiotemporal_field(orbit_integratedSR,symmetry='none',display_flag=True,padding=False)
    ksplot.plot_spatiotemporal_field(orbit_integratedSR,symmetry='none',display_flag=True,padding=False)
    Jn = np.eye(M-2)
    nmax = int(nstp)-1
    nmax = int(nstp//2)-1
    h = (T/nstp)
    for step in range(0,nmax):
        # k1 = np.dot(stability_matrix(Q1,Q2,FFT,IFFT,uu[-(2*step)-1,:],M),Jn)
        # k2 = np.dot(stability_matrix(Q1,Q2,FFT,IFFT,uu[-2*step-2,:],M),(Jn+0.5*k1))
        # k3 = np.dot(stability_matrix(Q1,Q2,FFT,IFFT,uu[-2*step-2,:],M),(Jn+0.5*k2))
        # k4 = np.dot(stability_matrix(Q1,Q2,FFT,IFFT,uu[-2*step-3,:],M),(Jn+k3))
        # Jn = Jn + (h/6)*(k1+2*k2+2*k3+k4)
        An1 = ustability_matrix(LINEAR,DX,FFT,IFFT,uu[-step-2,:])
        invFactor = inv(np.eye(M-2)-h*An1)
        Jn = np.dot(invFactor,Jn)
        # Jn = Jn + h*np.dot(stability_matrix(Q1,Q2,FFT,IFFT,uu[-step-1,:],M),Jn)
        # test = np.linalg.norm(Jn0-Jn)
        print(step*h)

    if symmetry == 'rpo':
        Gp = real_valued_rotation(N,M,T,L,S)
        Jn = np.dot(Gp,Jn)
    elif symmetry =='ppo':
        Rp = real_valued_reflection(N,M,T,L,S)
        Jn = np.dot(Rp,Jn)
    else:
        pass

    eigenvalues,eigenvectors = eig(Jn)
    real_exponents = (1/T)*np.log(np.abs(eigenvalues))

    return real_exponents


def main():
    PWD = os.path.dirname(__file__)
    init_dir = os.path.abspath(os.path.join(PWD, "../../../../../../.././dump/"))
    init_dir= os.path.join(init_dir,'')

    filename = ''.join([init_dir,"ppo1_128b128.h5"])
    symmetry='rpo'
    nstp = 8192*2
    orbit = orbit_io.import_orbit(filename)
    uu=ETDRK4_timeseries(orbit,symmetry=symmetry,nstp=nstp)

    ETDRK4_jacobian(orbit,symmetry=symmetry,stepsbetweensaves=50)


    return None

if __name__=='__main__':
    main()

