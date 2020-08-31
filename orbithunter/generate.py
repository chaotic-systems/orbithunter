from __future__ import print_function, division
import numpy as np
import warnings
import os
from math import pi
import random
warnings.simplefilter(action='ignore', category=ImportWarning)
warnings.resetwarnings()



def random_initial_condition(param_tuple, **kwargs):
    if T == 0.:
        self.T = 20 + 100*np.random.rand(1)
    else:
        self.T = T
    if L == 0.:
        self.L = 22 + 44*np.random.rand(1)
    else:
        self.L = L
    self.N = kwargs.get('N', np.max([32, 2**(int(np.log2(self.T)-1))]))
    self.M = kwargs.get('M', np.max([2**(int(np.log2(self.L))), 32]))
    self.n, self.m = int(self.N // 2) - 1, int(self.M // 2) - 1
    self.random_modes(**kwargs)
    self.convert(to='field', inplace=True)
    tmp = self // (1.0/4.0)
    self.state = tmp.state
    self.convert(to='modes', inplace=True)
    symmetry = kwargs.get('symmetry','ppo')
    amplitude = kwargs.get('amplitude',5.)
    scale_type = kwargs.get('scale_type','random')
    tms = kwargs.get('tms',False)
    N,M,T,L = param_tuple
    if symmetry == 'ppo':
        uu = ppo.initial_condition_generator(N,M,T,L,amplitude=amplitude,scale_type=scale_type,tms=tms)
        Orbit = (uu,N,M,T,L,0)
    elif symmetry == 'rpo':
        uu = rpo.initial_condition_generator(N,M,T,L,amplitude=amplitude,scale_type=scale_type,tms=tms)
        if L < 88:
            Orbit = (uu, N, M, T, L, random.randint(1,int(M//2))*(L/M))
        elif L > 500:
            Orbit = (uu, N, M, T, L, 0)
        else:
            Orbit = (uu, N, M, T, L, (L/30))
    elif symmetry == 'anti':
        uu = anti.initial_condition_generator(N,M,T,L,amplitude=amplitude,scale_type=scale_type,tms=tms)
        Orbit = (uu, N, M, T, L, 0)
    elif symmetry == 'none':
        uu = none.initial_condition_generator(N,M,T,L,amplitude=amplitude,scale_type=scale_type,tms=tms)
        Orbit = (uu, N, M, T, L, 0)
    else:
        Orbit = (np.zeros([N,M]),N,M,T,L,1)
    return Orbit


def average_spectrum_initial_condition(param_tuple,*args,**kwargs):
    symmetry=kwargs.get('symmetry','rpo')
    amplitude=kwargs.get('amplitude',5.)
    PWD = os.path.dirname(__file__)
    data_dir = os.path.abspath(os.path.join(PWD, "../../../data_and_figures/"))
    data_dir= os.path.join(data_dir,'')
    parent_folder = os.path.abspath(os.path.join(PWD, ''.join(["../../../data_and_figures/trawl/",str(symmetry),'/'])))
    Np,Mp,Tp,Lp = param_tuple
    Tstar,Lstar = 10.,2*pi*np.sqrt(2)
    if symmetry=='rpo' or symmetry=='full':
        n=Np-1
        m=Mp-2
        average_spectrum=np.zeros([n,m])
        nstar,mstar=int(Tp/10),int(Lp/(2*pi*np.sqrt(2)))
        Nstar,Mstar=2*int(nstar*n/2.)-1,int(mstar*m/2.)-1
    else:
        n=Np-1
        m=int(Mp/2)-1
        average_spectrum=np.zeros([n,m])
        nstar,mstar=int(Tp/10),int(Lp/(2*pi*np.sqrt(2)))
        Nstar,Mstar=2*int(nstar*n/2.)-1,int(mstar*Mp/2.)-1
    for files in os.listdir(parent_folder):
        folder = os.path.join(os.path.abspath(os.path.join(parent_folder, files)),'')
        check_spectrum_filepath = os.path.join(os.path.abspath(os.path.join(parent_folder,'other_data/spectrum')))
        if os.path.isfile(''.join([check_spectrum_filepath,'.h5'])):
            Orbit=Orbit_io.import_Orbit(check_spectrum_filepath)
            U,N,M,T,L,S = Orbit
            Orbit_spectrum = ks.fft_(Orbit,N,M,symmetry=symmetry)
        elif os.path.isdir(folder):
            print('accessing',folder)
            for data_file in os.listdir(folder):
                if data_file.endswith(".h5"):
                    basename = data_file.split('.h5')[0]
                    print(basename)
                    Orbit = Orbit_io.import_Orbit(''.join([folder,data_file]))
                    Ua,Na,Ma,Ta,La,Sa = Orbit

                    maxw,maxq = 2*pi*Nstar/Tstar,2*pi*Mstar/Lstar
                    wa,qa = 2*pi/Ta,2*pi/La
                    na,ma=int(maxw/wa)*Nstar,int(maxq/qa)*Mstar
                    Natilde,Matilde=na*Nstar,ma*Mstar
                    Orbit = disc.rediscretize(Orbit,newN=Natilde,newM=Matilde)
                    U,_,_,_,_,_ = Orbit
                    uvec=np.reshape(U,[Natilde*Matilde,1])
                    if symmetry=='rpo' or symmetry=='full':
                        nav,mav=Natilde-1,Matilde-2
                    else:
                        nav,mav=Natilde-1,int(Matilde/2)-1
                    Orbit_spectral_tmp = np.reshape(ks.fft_(uvec,Natilde,Matilde,symmetry=symmetry),[nav,mav])
                    Orbit_spectral_tmp = Orbit_spectral_tmp[::na,ma::ma]
                    average_spectrum +=Orbit_spectral_tmp
                    average_spectrum = average_spectrum/np.linalg.norm(average_spectrum)
            average_spectrum_Orbit = (ks.ifft_(average_spectrum,navpad,mavpad,symmetry=symmetry),navpad,navpad,10,2*pi*np.sqrt(2),0)
            ksplot.plot_spatiotemporal_field(average_spectrum_Orbit,symmetry=symmetry,display_flag=True,filename=''.join([check_spectrum_filepath,'.png']))
            Orbit_io.export_Orbit(average_spectrum_Orbit,check_spectrum_filepath,symmetry=symmetry)
            Orbit_spectrum=average_spectrum[:nstar*(Np-1):nstar,mstar:mstar*(int(Mp/2.)):mstar]

    random_spectral = np.random.randn([n,m])*Orbit_spectrum
    random_Orbit_uu = ks.ifft_(random_spectral,N,M,symmetry=symmetry)
    initial_condition_uu = amplitude*random_Orbit_uu/np.linalg.norm(random_Orbit_uu)
    initial_condition = (initial_condition_uu,N,M,T,L,S)
    return initial_condition


def seeded_initial_condition(param_tuple,**kwargs):
    symmetry = kwargs.get('symmetry','ppo')
    amplitude = kwargs.get('amplitude',5.)
    scale_type = kwargs.get('scale_type','random')
    tms = kwargs.get('tms',False)
    N,M,T,L = param_tuple
    if symmetry == 'ppo':
        uu = ppo.initial_condition_generator(N,M,T,L,amplitude=amplitude,scale_type=scale_type,tms=tms)
        Orbit = (uu,N,M,T,L,0)
    elif symmetry == 'rpo':
        uu = rpo.tile_seeded_noise(N,M,T,L,amplitude=amplitude,scale_type=scale_type,tms=tms)
        if L < 88:
            Orbit = (uu, N, M, T, L, (L/6))
        elif L > 500:
            Orbit = (uu, N, M, T, L, 0)
        else:
            Orbit = (uu, N, M, T, L, (L/30))
    elif symmetry == 'anti':
        uu = anti.initial_condition_generator(N,M,T,L,amplitude=amplitude,scale_type=scale_type,tms=tms)
        Orbit = (uu, N, M, T, L, 0)
    else:
        Orbit = (np.zeros([N,M]),N,M,T,L,1)
    return Orbit


def symbolic_initial_condition(symbol_block_list,period,speriod,*args,**kwargs):
    combine_method = kwargs.get('combine_method','tile')
    tileN = kwargs.get('tileN',128)
    tileM = kwargs.get('tileM',128)
    # if combine_method == 'tile':
    block_Orbit = tile.tile(symbol_block_list,period,speriod,block_symmetry='full')

    return block_Orbit


def glued_initial_condition(OrbitA,OrbitB,*args,**kwargs):
    # gluetype = kwargs.get('gluetype',1)
    # symmetry = kwargs.get('symmetry','ppo')
    # gluecomplexity = kwargs.get('gluecomplexity','complex')
    # buffertype=kwargs.get('buffertype','dynamic')
    # resolution=kwargs.get('resolution','low')
    # OrbitA0,OrbitB0 = validate_discretization(OrbitA,OrbitB)
    # ua,Na,Ma,Ta,La,Sa= OrbitA
    # ub,Nb,Mb,Tb,Lb,Sb= OrbitB
    #
    # if gluetype:
    #     Orbit_tmp = (ua,Nb,Ma+Mb,(Ta+Tb)/2.,Lb+La,Sa+Sb)
    #     Na,Nb = np.max([Na,Nb]),np.max([Na,Nb])
    #     nfinal,mfinal = disc.parameter_based_discretization(Orbit_tmp,resolution=resolution)
    # else:
    #     Orbit_tmp = (ua,Nb+Na,Ma,Ta+Tb,(Lb+La)/2,Sa+Sb)
    #     Ma,Mb = np.max([Ma,Mb]),np.max([Ma,Mb])
    #     nfinal,mfinal = disc.parameter_based_discretization(Orbit_tmp,resolution=resolution)
    #
    # if gluecomplexity=='simple':
    #     ARtori,Rtori,Ctori,CBtori = ((),),((),),((),),((),)
    #     GOrbit = merge_fields((OrbitA,OrbitB),gluetype=gluetype,tori_pair=True)
    #     return ARtori,Rtori,Ctori,CBtori,GOrbit
    #
    # if symmetry=='rpo':
    #     if np.sign(Sa)!=np.sign(Sb):
    #         Sa = -Sa
    #         ua = -1.0*np.fliplr(np.roll(ua,1,axis=1))
    #         OrbitA0 = (ua,Na,Ma,Ta,La,Sa)
    #     if np.abs(Sa)<np.abs(Sb):
    #         OrbitA,OrbitB = OrbitB,OrbitA
    #         ua,Na,Ma,Ta,La,Sa = OrbitA
    #         ub,Nb,Mb,Tb,Lb,Sb = OrbitB
    #     OrbitA_AR,OrbitB_AR = discretization_ratios(OrbitA,OrbitB,gluetype=gluetype)
    #     _,NAAR,MAAR,_,_,_=OrbitA_AR
    #     _,NBAR,MBAR,_,_,_=OrbitB_AR
    #     # OrbitA_AR_large = disc.rediscretize(OrbitA_AR,newN=8*Na0,newM=8*Ma0)
    #     # OrbitB_AR_large = disc.rediscretize(OrbitB_AR,newN=8*Nb0,newM=8*Mb0)
    #     # OrbitAfull=rpodm.mvf_rotate_Orbit(OrbitA)
    #     # OrbitBfull=rpodm.mvf_rotate_Orbit(OrbitB)
    #     # ARtori = (OrbitAfull,OrbitBfull)
    #     # OrbitA_R,OrbitB_R = find_best_rotation(OrbitAfull,OrbitBfull,gluetype=gluetype)
    #     # ARtori = (OrbitA_R,OrbitB_R)
    #
    #     OrbitA_R,OrbitB_R = find_best_rotation(OrbitA_AR,OrbitB_AR,gluetype=gluetype)
    #     ARtori = (OrbitA_AR,OrbitB_AR)
    #     Rtori = (OrbitA_R,OrbitB_R)
    #     Ctori = chop_fields(OrbitA_R,OrbitB_R,symmetry=symmetry,gluetype=gluetype,buffertype=buffertype)
    #     CBtori = convex_buffer(Ctori,symmetry=symmetry,gluetype=gluetype)
    #     merged_tori= merge_fields(CBtori,symmetry=symmetry,gluetype=gluetype)
    #     GOrbit = disc.rediscretize(merged_tori,newN=nfinal,newM=mfinal)
    #     # GOrbit = disc.residual_guided_discretization(GOrbit,symmetry=symmetry)
    # else:
    #     OrbitA = disc.rediscretize(OrbitA0,newN=8*Na,newM=8*Ma)
    #     OrbitB = disc.rediscretize(OrbitB0,newN=8*Nb,newM=8*Mb)
    #     OrbitA_AR,OrbitB_AR = discretization_ratios(OrbitA0,OrbitB0,gluetype=gluetype)
    #     ua,Na,Ma,Ta,La,Sa = OrbitA_AR
    #     ub,Nb,Mb,Tb,Lb,Sb = OrbitB_AR
    #     ua_star = -1.0*np.fliplr(np.roll(ua,1,axis=1))
    #     ub_star = -1.0*np.fliplr(np.roll(ub,1,axis=1))
    #     OrbitASTAR = (ua_star,Na,Ma,Ta,La,Sa)
    #     OrbitBSTAR = (ub_star,Nb,Mb,Tb,Lb,Sb)
    #     Orbitcombinations = ((OrbitA_AR,OrbitB_AR),(OrbitB_AR,OrbitA_AR),(OrbitASTAR,OrbitB_AR),(OrbitBSTAR,OrbitA_AR))
    #     ncombo = 0
    #     combolist = ['AB','BA','(RA)B','A(RB)']
    #     reslist = np.zeros([np.size(combolist),1])
    #     residualprev = 0
    #     for tori_pair in Orbitcombinations:
    #         Orbit0,Orbit1 = tori_pair
    #         ARtori0 = (Orbit0,Orbit1)
    #         Rtori0 = (Orbit0,Orbit1)
    #         Ctori0 = chop_fields(Orbit0,Orbit1,symmetry=symmetry,gluetype=gluetype,buffertype=buffertype)
    #         CBtori0 = convex_buffer(Ctori0,symmetry=symmetry,gluetype=gluetype)
    #         merged_tori = merge_fields(CBtori0,symmetry=symmetry,gluetype=gluetype)
    #         GOrbit0 = disc.rediscretize(merged_tori,newN=nfinal,newM=mfinal)
    #         # GOrbit0 = disc.residual_guided_discretization(GOrbit0,symmetry=symmetry)
    #         residual = ks.compute_residual_fromtuple(GOrbit0,symmetry=symmetry)
    #         reslist[ncombo]=residual
    #         print(combolist[ncombo],"residual",residual)
    #         if residualprev==0:
    #             ARtori = ARtori0
    #             Rtori = Rtori0
    #             Ctori = Ctori0
    #             CBtori = CBtori0
    #             GOrbit = GOrbit0
    #             residualprev = residual
    #         elif float(residual)<float(residualprev):
    #             ARtori = ARtori0
    #             Rtori = Rtori0
    #             Ctori = Ctori0
    #             CBtori = CBtori0
    #             GOrbit = GOrbit0
    #             residualprev=residual
    #         ncombo+=1
    return ARtori,Rtori,Ctori,CBtori,GOrbit

def initial_condition_generator(N,M,T,L,**kwargs):
    amplitude = kwargs.get('amplitude',5)
    scale_type = kwargs.get('scale_type','random')
    n = int(N-1)
    m = int(M//2)-1
    L = float(np.real(L))
    sms = int(L/(2*pi*np.sqrt(2)))
    u = np.random.randn(M*N)
    st_mat = np.reshape(fft_(u,N,M),[n,m])
    spacetime_mollifier_grid = np.zeros([n,m])
    tms = 2
    if L < 16:
        for i in range(0,n):
            for j in range(0,m):
                spacetime_mollifier_grid[i,j]-=(np.sign((j)-(sms))*((j)-(sms)))
        exp_mollifier = np.exp(spacetime_mollifier_grid)
        smoothed = np.multiply(exp_mollifier, st_mat)
        smoothed[tms:int(n//2)+1, :] = 0
        smoothed[int(n//2)+tms:, :] = 0
        st_vec_smooth = np.reshape(smoothed, [m * n, 1])
        u_vec = ifft_(st_vec_smooth, N, M)
        renormed_u = 3 * u_vec / np.max(np.abs(u_vec))
    else:
        for i in range(0,n):
            for j in range(0,int(m//2)):
                if scale_type == 'random':
                    spacetime_mollifier_grid[i,j]-=(np.sign((j)-(sms))*((j)-(sms)))/sms
                elif scale_type == 'physical':
                    spacetime_mollifier_grid[i,j]-=((2*pi*sms/L)**2-(2*pi*sms/L)**4) -(((2*pi*j/L)**2-(2*pi*j/L)**4))
        exp_mollifier = np.exp(spacetime_mollifier_grid)
        smoothed = np.multiply(exp_mollifier, st_mat)
        smoothed[tms+1:int(n//2), :] = 0
        smoothed[-int(n//2):-tms, :] = 0
        st_vec_smooth = np.reshape(smoothed, [m * n, 1])
        u_vec = ifft_(st_vec_smooth, N, M)
        renormed_u = amplitude * u_vec / np.max(np.abs(u_vec))
    renormed_uu = np.reshape(renormed_u,[N,M])
    return renormed_uu

def initial_condition_generator(N,M,T,L,**kwargs):
    amplitude = kwargs.get('amplitude',5)
    scale_type = kwargs.get('scale_type','random')
    tms = kwargs.get('tms',False)
    n = int(N-1)
    m = int(M/2)-1
    L = float(np.real(L))
    sms = int(L/(2*pi*np.sqrt(2)))
    u = np.random.randn(M*N)
    st_mat = np.reshape(fft_(u,N,M),[n,M-2])
    if tms == False:
        tms_short = int(T/20)
        tms_long = int(T/10)
    else:
        tms_short,tms_long = 1,1
        tms=1
    if scale_type == 'basic':
        spatial_range=np.arange(1,m+1)
        spatial_spectrum_tmp = -1.*np.abs(spatial_range-sms)
        spatial_spectrum = np.tile(spatial_spectrum_tmp,(n,1))
        spatial_spectrum = np.concatenate((spatial_spectrum,spatial_spectrum),axis=1)
        spatially_smoothed = np.multiply(np.exp(spatial_spectrum), st_mat)

        temporal_range=np.reshape(np.arange(0,int(N//2)),[int(N//2),1])
        temporal_range_nozero = np.reshape(temporal_range[1:],[int(N//2)-1,1])
        temporal_spectrum_tmp = -1.*(temporal_range-tms)**2
        temporal_spectrum_tmp = np.tile(temporal_spectrum_tmp,(1,m))
        temporal_spectrum_nozero_tmp = -1.*(temporal_range_nozero-tms)**2
        temporal_spectrum_nozero_tmp = np.tile(temporal_spectrum_nozero_tmp,(1,m))
        temporal_spectrum = np.concatenate((np.concatenate((temporal_spectrum_tmp,temporal_spectrum_tmp),axis=1)
                                            ,np.concatenate((temporal_spectrum_nozero_tmp,temporal_spectrum_nozero_tmp),axis=1)),axis=0)

        spatiotemporal_spectrum = np.multiply(np.exp(temporal_spectrum),spatially_smoothed)

        st_vec_smooth = np.reshape(spatiotemporal_spectrum, [(M-2) * n, 1])
        u_vec = ifft_(st_vec_smooth, N, M)
        renormed_u = amplitude * u_vec / np.max(np.abs(u_vec))
        renormed_uu = np.reshape(renormed_u,[N,M])
    elif scale_type == 'gaussian':
        sigma_time = 2
        sigma_space = 5
        tms=2
        time = np.tile(np.reshape(np.concatenate((np.arange(0,int(N/2)),np.arange(1,int(N/2))),axis=0),[n,1]),(1,M-2))
        space = np.tile(np.reshape(np.concatenate((np.arange(1,m+1),np.arange(1,m+1)),axis=0),[1,M-2]),(n,1))
        spatiotemporal_gaussian = 1./np.sqrt(2*pi**2*sigma_space**2*sigma_time**2)*np.exp(-(space-sms)**2/(2*sigma_space**2)-(time-tms)**2/(2*sigma_time**2))
        spacetime_spectrum = np.multiply(spatiotemporal_gaussian, st_mat)
        st_vec_smooth = np.reshape(spacetime_spectrum, [(M-2) * n, 1])
        u_vec = ifft_(st_vec_smooth, N, M)
        renormed_u = amplitude * u_vec / np.max(np.abs(u_vec))
        renormed_uu = np.reshape(renormed_u,[N,M])
    return renormed_uu

def initial_condition_generator(N,M,T,L,**kwargs):
    amplitude = kwargs.get('amplitude',5)
    scale_type = kwargs.get('scale_type','random')
    tms = kwargs.get('tms',False)
    n = int(N-1)
    m = int(M/2)-1
    L = float(np.real(L))
    sms = int(L/(2*pi*np.sqrt(2)))
    u = np.random.randn(M*N)
    st_mat = np.reshape(fft_(u,N,M),[n,M-2])
    sms = int(L/(2*pi*np.sqrt(2)))
    spacetime_mollifier_grid = np.zeros([n,m])
    if tms == False:
        tms_short = int(T/20)
        tms_long = int(T/10)
    else:
        tms_short,tms_long = 1,1
        tms=1

    if scale_type == 'gaussian':
        sigma_time = 2
        sigma_space = 5
        tms=2
        time = np.tile(np.reshape(np.concatenate((np.arange(0,int(N/2)),np.arange(1,int(N/2))),axis=0),[n,1]),(1,M-2))
        space = np.tile(np.reshape(np.concatenate((np.arange(1,m+1),np.arange(1,m+1)),axis=0),[1,M-2]),(n,1))
        spatiotemporal_gaussian = 1./np.sqrt(2*pi**2*sigma_space**2*sigma_time**2)*np.exp(-(space-sms)**2/(2*sigma_space**2)-(time-tms)**2/(2*sigma_time**2))
        spacetime_spectrum = np.multiply(spatiotemporal_gaussian, st_mat)
        modes_smooth = np.reshape(spacetime_spectrum, [(M-2) * n, 1])
        u_vec = ifft_(modes_smooth, N, M)
        renormed_u = amplitude * u_vec / np.max(np.abs(u_vec))
        renormed_uu = np.reshape(renormed_u,[N,M])
    elif scale_type=='modulated':
        spatial_range=np.arange(1,m+1)
        spatial_spectrum_tmp = -1.*np.abs(spatial_range-sms)
        spatial_spectrum = np.tile(spatial_spectrum_tmp,(n,1))
        spatial_spectrum = np.concatenate((spatial_spectrum,spatial_spectrum),axis=1)
        spatially_smoothed = np.multiply(np.exp(spatial_spectrum), st_mat)

        temporal_range=np.reshape(np.arange(0,int(N//2)),[int(N//2),1])
        temporal_range_nozero = np.reshape(temporal_range[1:],[int(N//2)-1,1])
        temporal_spectrum_tmp = -1.*(temporal_range-tms)**2
        temporal_spectrum_tmp = np.tile(temporal_spectrum_tmp,(1,m))
        temporal_spectrum_nozero_tmp = -1.*(temporal_range_nozero-tms)**2
        temporal_spectrum_nozero_tmp = np.tile(temporal_spectrum_nozero_tmp,(1,m))
        temporal_spectrum = np.concatenate((np.concatenate((temporal_spectrum_tmp,temporal_spectrum_tmp),axis=1)
                                            ,np.concatenate((temporal_spectrum_nozero_tmp,temporal_spectrum_nozero_tmp),axis=1)),axis=0)

        spatiotemporal_spectrum = np.multiply(np.exp(temporal_spectrum),spatially_smoothed)

        modes_smooth = np.reshape(spatiotemporal_spectrum, [(M-2) * n, 1])
        u_vec = ifft_(modes_smooth, N, M)
        renormed_u = amplitude * u_vec / np.max(np.abs(u_vec))
        renormed_uu = np.reshape(renormed_u,[N,M])
    elif scale_type =='random':
        u = np.random.randn(M*N)
        renormed_uu = np.reshape(amplitude*u/np.max(np.abs(u)),[N,M])
    else:
        tms = 2
        jrange = np.arange(1,m+1)
        spacetime_mollifier_grid=-1*np.tile(np.sqrt((np.abs(jrange-sms)/sms)),(n,1))
        spacetime_mollifier_grid = np.concatenate((spacetime_mollifier_grid,spacetime_mollifier_grid),axis=1)
        mollifier = np.exp(spacetime_mollifier_grid)
        smoothed = np.multiply(mollifier, st_mat)
        smoothed[tms:int(n//2)+1, :] = 0
        smoothed[int(n//2)+tms:, :] = 0
        modes_smooth = np.reshape(smoothed, [(M-2) * n, 1])
        u_vec = ifft_(modes_smooth, N, M)
        renormed_u = amplitude * u_vec / np.max(np.abs(u_vec))
        renormed_uu = np.reshape(renormed_u,[N,M])
    return renormed_uu

def initial_condition_generator(N,M,T,L,**kwargs):
    amplitude = kwargs.get('amplitude',5)
    scale_type = kwargs.get('scale_type','nonphysical')
    n = int(N-1)
    m = int(M//2)-1
    L = float(np.real(L))
    sms = int(L/pi)
    u = np.random.randn(M*N)
    st_mat = np.reshape(fft_(u,N,M),[n,m])
    spacetime_mollifier_grid = np.zeros([n,m])
    tms = 2
    if L < 16:
        for i in range(0,n):
            for j in range(0,m):
                spacetime_mollifier_grid[i,j]-=(np.sign((j)-(sms))*((j)-(sms)))
        exp_mollifier = np.exp(spacetime_mollifier_grid)
        smoothed = np.multiply(exp_mollifier, st_mat)
        smoothed[tms:int(n//2)+1, :] = 0
        smoothed[int(n//2)+tms:, :] = 0
        st_vec_smooth = np.reshape(smoothed, [m * n, 1])
        u_vec = ifft_(st_vec_smooth, N, M)
        renormed_u = 3 * u_vec / np.max(np.abs(u_vec))
    else:
        for i in range(0,n):
            for j in range(0,int(m)):
                if scale_type == 'nonphysical':
                    #if j > sms:
                    spacetime_mollifier_grid[i,j]-=np.sqrt((np.abs(j-sms)/sms))
                elif scale_type == 'physical':
                    if j > sms:
                        spacetime_mollifier_grid[i,j]-=((2*pi*sms/L)**2-(2*pi*sms/L)**4) -(((2*pi*j/L)**2-(2*pi*j/L)**4))
        mollifier = np.exp(spacetime_mollifier_grid)
        smoothed = np.multiply(mollifier, st_mat)
        smoothed[tms:int(n//2)+1, :] = 0
        smoothed[int(n//2)+tms:, :] = 0
        st_vec_smooth = np.reshape(smoothed, [m * n, 1])
        u_vec = ifft_(st_vec_smooth, N, M)
        renormed_u = amplitude * u_vec / np.max(np.abs(u_vec))
    renormed_uu = np.reshape(renormed_u,[N,M])
    return renormed_uu

def initial_condition_generator(N,M,T,L,**kwargs):
    amplitude = kwargs.get('amplitude',5)
    scale_type = kwargs.get('scale_type','random')
    tms = kwargs.get('tms',False)
    n = int(N-1)
    m = int(M/2)-1
    L = float(np.real(L))
    sms = int(L/(2*pi*np.sqrt(2)))
    u = np.random.randn(M*N)
    st_mat = np.reshape(fft_(u,N,M),[n,M-2])
    sms = int(L/(2*pi*np.sqrt(2)))
    spacetime_mollifier_grid = np.zeros([n,m])
    if tms == False:
        tms_short = int(T/20)
        tms_long = int(T/10)
    else:
        tms_short,tms_long = 1,1
        tms=1

    if scale_type == 'gaussian':
        sigma_time = 2
        sigma_space = 5
        tms=2
        time = np.tile(np.reshape(np.concatenate((np.arange(0,int(N/2)),np.arange(1,int(N/2))),axis=0),[n,1]),(1,M-2))
        space = np.tile(np.reshape(np.concatenate((np.arange(1,m+1),np.arange(1,m+1)),axis=0),[1,M-2]),(n,1))
        spatiotemporal_gaussian = 1./np.sqrt(2*pi**2*sigma_space**2*sigma_time**2)*np.exp(-(space-sms)**2/(2*sigma_space**2)-(time-tms)**2/(2*sigma_time**2))
        spacetime_spectrum = np.multiply(spatiotemporal_gaussian, st_mat)
        st_vec_smooth = np.reshape(spacetime_spectrum, [(M-2) * n, 1])
        u_vec = ifft_(st_vec_smooth, N, M)
        renormed_u = amplitude * u_vec / np.max(np.abs(u_vec))
        renormed_uu = np.reshape(renormed_u,[N,M])
    elif scale_type=='modulated':
        spatial_range=np.arange(1,m+1)
        spatial_spectrum_tmp = -1.*np.abs(spatial_range-sms)
        spatial_spectrum = np.tile(spatial_spectrum_tmp,(n,1))
        spatial_spectrum = np.concatenate((spatial_spectrum,spatial_spectrum),axis=1)
        spatially_smoothed = np.multiply(np.exp(spatial_spectrum), st_mat)

        temporal_range=np.reshape(np.arange(0,int(N//2)),[int(N//2),1])
        temporal_range_nozero = np.reshape(temporal_range[1:],[int(N//2)-1,1])
        temporal_spectrum_tmp = -1.*(temporal_range-tms)**2
        temporal_spectrum_tmp = np.tile(temporal_spectrum_tmp,(1,m))
        temporal_spectrum_nozero_tmp = -1.*(temporal_range_nozero-tms)**2
        temporal_spectrum_nozero_tmp = np.tile(temporal_spectrum_nozero_tmp,(1,m))
        temporal_spectrum = np.concatenate((np.concatenate((temporal_spectrum_tmp,temporal_spectrum_tmp),axis=1)
                                            ,np.concatenate((temporal_spectrum_nozero_tmp,temporal_spectrum_nozero_tmp),axis=1)),axis=0)

        spatiotemporal_spectrum = np.multiply(np.exp(temporal_spectrum),spatially_smoothed)

        st_vec_smooth = np.reshape(spatiotemporal_spectrum, [(M-2) * n, 1])
        u_vec = ifft_(st_vec_smooth, N, M)
        renormed_u = amplitude * u_vec / np.max(np.abs(u_vec))
        renormed_uu = np.reshape(renormed_u,[N,M])
    elif scale_type =='random':
        u = np.random.randn(M*N)
        renormed_uu = np.reshape(amplitude*u/np.max(np.abs(u)),[N,M])
    else:
        tms = 2
        jrange = np.arange(1,m+1)
        spacetime_mollifier_grid=-1*np.tile(np.sqrt((np.abs(jrange-sms)/sms)),(n,1))
        spacetime_mollifier_grid = np.concatenate((spacetime_mollifier_grid,spacetime_mollifier_grid),axis=1)
        mollifier = np.exp(spacetime_mollifier_grid)
        smoothed = np.multiply(mollifier, st_mat)
        smoothed[tms:int(n//2)+1, :] = 0
        smoothed[int(n//2)+tms:, :] = 0
        st_vec_smooth = np.reshape(smoothed, [(M-2) * n, 1])
        u_vec = ifft_(st_vec_smooth, N, M)
        renormed_u = amplitude * u_vec / np.max(np.abs(u_vec))
        renormed_uu = np.reshape(renormed_u,[N,M])
    return renormed_uu

def initial_condition_generator(N,M,T,L):
    n = int(N//2)-1
    m = int(M//2)-1
    L = float(np.real(L))
    sms = int(L/(2*pi*np.sqrt(2)))
    v = np.random.randn(n*m)+1j*np.random.randn(m*n)
    st_mat = np.reshape(v,[n,m])
    spacetime_mollifier_grid = np.zeros([n,m])
    tms = 2
    if L < 16:
        for i in range(0,n):
            for j in range(0,m):
                spacetime_mollifier_grid[i,j]-=(np.sign((j)-(sms))*((j)-(sms)))
        exp_mollifier = np.exp(spacetime_mollifier_grid)
        smoothed = np.multiply(exp_mollifier, st_mat)
        smoothed[tms:int(n//2)+1, :] = 0
        smoothed[int(n//2)+tms:, :] = 0
        st_vec_smooth = np.reshape(smoothed, [m * n, 1])
        u_vec = ifft_(st_vec_smooth, N, M)
        renormed_u = 3 * u_vec / np.max(np.abs(u_vec))
    else:
        for i in range(0,n):
            for j in range(0,int(m//2)):
                spacetime_mollifier_grid[i,j]-=(np.sign((j)-(sms))*((j)-(sms)))/sms
        exp_mollifier = np.exp(spacetime_mollifier_grid)
        smoothed = np.multiply(exp_mollifier, st_mat)
        smoothed[tms:int(n//2)+1, :] = 0
        smoothed[int(n//2)+tms:, :] = 0
        st_vec_smooth = np.reshape(smoothed, [m * n, 1])
        u_vec = ifft_(st_vec_smooth, N, M)
        renormed_u = 5 * u_vec / np.max(np.abs(u_vec))
    renormed_uu = np.reshape(renormed_u,[N,M])
    return renormed_uu

def initial_condition_generator(N,M,T,L):
    n = int(N//2)-1
    m = int(M//2)-1
    L = float(np.real(L))
    sms = int(L/(2*pi*np.sqrt(2)))
    v = np.random.randn(n*m)+1j*np.random.randn(m*n)
    st_mat = np.reshape(v,[n,m])
    spacetime_mollifier_grid = np.zeros([n,m])
    tms = 2
    if L < 16:
        for i in range(0,n):
            for j in range(0,m):
                spacetime_mollifier_grid[i,j]-=(np.sign((j)-(sms))*((j)-(sms)))
        exp_mollifier = np.exp(spacetime_mollifier_grid)
        smoothed = np.multiply(exp_mollifier, st_mat)
        smoothed[tms:int(n//2)+1, :] = 0
        smoothed[int(n//2)+tms:, :] = 0
        st_vec_smooth = np.reshape(smoothed, [m * n, 1])
        u_vec = ifft_(st_vec_smooth, N, M)
        renormed_u = 3 * u_vec / np.max(np.abs(u_vec))
    else:
        for i in range(0,n):
            for j in range(0,int(m//2)):
                spacetime_mollifier_grid[i,j]-=(np.sign((j)-(sms))*((j)-(sms)))/sms
        exp_mollifier = np.exp(spacetime_mollifier_grid)
        smoothed = np.multiply(exp_mollifier, st_mat)
        smoothed[tms:int(n//2)+1, :] = 0
        smoothed[int(n//2)+tms:, :] = 0
        st_vec_smooth = np.reshape(smoothed, [m * n, 1])
        u_vec = ifft_(st_vec_smooth, N, M)
        renormed_u = 5 * u_vec / np.max(np.abs(u_vec))
    renormed_uu = np.reshape(renormed_u,[N,M])
    return renormed_uu

def initial_condition_generator(N,M,T,L):
    n = int(N-1)
    m = int(M//2)-1
    L = float(np.real(L))
    sms = int(L/(2*pi))
    u = np.random.randn(M*N)
    st_mat = np.reshape(fft_(u,N,M),[n,m])
    spacetime_mollifier_grid = np.zeros([n,m])
    tms = 2
    for i in range(0,n):
        for j in range(0,int(m//2)):
            spacetime_mollifier_grid[i,j]-=(np.sign((j)-(sms))*((j)-(sms)))
    exp_mollifier = np.exp(spacetime_mollifier_grid)
    smoothed = np.multiply(exp_mollifier, st_mat)
    smoothed[tms:int(n//2)+1, :] = 0
    smoothed[int(n//2)+tms:, :] = 0
    st_vec_smooth = np.reshape(smoothed, [m * n, 1])
    u_vec = ifft_(st_vec_smooth, N, M)
    renormed_u = 6 * u_vec / np.max(np.abs(u_vec))
    renormed_uu = np.reshape(renormed_u,[N,M])
    return renormed_uu

def main(*args,**kwargs):
    # symmetry=kwargs.get('symmetry','ppo')
    # gluetype=kwargs.get('gluetype',0)
    # buffersize=kwargs.get('buffersize',1)
    # rotateOrbit = kwargs.get('rotateOrbit',True)
    # mindiscdirection = kwargs.get('mindisc','none')
    # file0 = "C:\\Users\\matt\\Desktop\\gudorf\\KS\\python\\data_and_figures\\L22_initial_conditions\\rpo\\rpo1_32b32.h5"
    # file1 = "C:\\Users\\matt\\Desktop\\gudorf\\KS\\python\\data_and_figures\\L22_initial_conditions\\rpo\\rpo2_32b32.h5"
    # # file0 = "C:\\Users\\matt\\Desktop\\gudorf\\KS\\python\\data_and_figures\\L22_initial_conditions\\ppo\\ppo1_32b32.h5"
    # # file1 = "C:\\Users\\matt\\Desktop\\gudorf\\KS\\python\\data_and_figures\\L22_initial_conditions\\ppo\\ppo2_32b32.h5"
    # # file1 = "C:\\Users\\matt\\Desktop\\gudorf\\KS\\python\\data_and_figures\\converged\\rpo\\data\\rpo_L21p74_T78.h5"
    # if gluetype:
    #     gluename='space'
    # else:
    #     gluename='time'
    #
    # conv_figs_dir = ''.join(["C:\\Users\\matt\\Desktop\\tests\\conv\\"])
    # fail_figs_dir = ''.join(["C:\\Users\\matt\\Desktop\\tests\\fail\\"])
    # OrbitLT = Orbit_io.import_Orbit(file0)
    # OrbitRB = Orbit_io.import_Orbit(file1)
    # base0_tmp=file0.split("\\")[-1]
    # base1_tmp=file1.split("\\")[-1]
    # base0 = base0_tmp.split('.h5')[0]
    # base1 = base1_tmp.split('.h5')[0]
    #
    # glued_Orbit = glued_initial_condition(OrbitLT,OrbitRB,symmetry=symmetry,gluetype=gluetype,buffersize=buffersize,slicenotchop=slicenotchop,rotateOrbit=rotateOrbit)
    # ksplot.plot_spatiotemporal_field(glued_Orbit,symmetry='none',display_flag=True)
    # glued_Orbit_final = disc.minimize_discretization(glued_Orbit,symmetry=symmetry,direction=mindiscdirection)
    # ksplot.plot_spatiotemporal_field(glued_Orbit_final,symmetry='none',display_flag=True)
    # u,N,M,T,L,S = glued_Orbit_final

    return None

if __name__=='__main__':
    main()