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
        torus = (uu,N,M,T,L,0)
    elif symmetry == 'rpo':
        uu = rpo.initial_condition_generator(N,M,T,L,amplitude=amplitude,scale_type=scale_type,tms=tms)
        if L < 88:
            torus = (uu, N, M, T, L, random.randint(1,int(M//2))*(L/M))
        elif L > 500:
            torus = (uu, N, M, T, L, 0)
        else:
            torus = (uu, N, M, T, L, (L/30))
    elif symmetry == 'anti':
        uu = anti.initial_condition_generator(N,M,T,L,amplitude=amplitude,scale_type=scale_type,tms=tms)
        torus = (uu, N, M, T, L, 0)
    elif symmetry == 'none':
        uu = none.initial_condition_generator(N,M,T,L,amplitude=amplitude,scale_type=scale_type,tms=tms)
        torus = (uu, N, M, T, L, 0)
    else:
        torus = (np.zeros([N,M]),N,M,T,L,1)
    return torus


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
            torus=torus_io.import_torus(check_spectrum_filepath)
            U,N,M,T,L,S = torus
            torus_spectrum = ks.fft_(torus,N,M,symmetry=symmetry)
        elif os.path.isdir(folder):
            print('accessing',folder)
            for data_file in os.listdir(folder):
                if data_file.endswith(".h5"):
                    basename = data_file.split('.h5')[0]
                    print(basename)
                    torus = torus_io.import_torus(''.join([folder,data_file]))
                    Ua,Na,Ma,Ta,La,Sa = torus

                    maxw,maxq = 2*pi*Nstar/Tstar,2*pi*Mstar/Lstar
                    wa,qa = 2*pi/Ta,2*pi/La
                    na,ma=int(maxw/wa)*Nstar,int(maxq/qa)*Mstar
                    Natilde,Matilde=na*Nstar,ma*Mstar
                    torus = disc.rediscretize(torus,newN=Natilde,newM=Matilde)
                    U,_,_,_,_,_ = torus
                    uvec=np.reshape(U,[Natilde*Matilde,1])
                    if symmetry=='rpo' or symmetry=='full':
                        nav,mav=Natilde-1,Matilde-2
                    else:
                        nav,mav=Natilde-1,int(Matilde/2)-1
                    torus_spectral_tmp = np.reshape(ks.fft_(uvec,Natilde,Matilde,symmetry=symmetry),[nav,mav])
                    torus_spectral_tmp = torus_spectral_tmp[::na,ma::ma]
                    average_spectrum +=torus_spectral_tmp
                    average_spectrum = average_spectrum/np.linalg.norm(average_spectrum)
            average_spectrum_torus = (ks.ifft_(average_spectrum,navpad,mavpad,symmetry=symmetry),navpad,navpad,10,2*pi*np.sqrt(2),0)
            ksplot.plot_spatiotemporal_field(average_spectrum_torus,symmetry=symmetry,display_flag=True,filename=''.join([check_spectrum_filepath,'.png']))
            torus_io.export_torus(average_spectrum_torus,check_spectrum_filepath,symmetry=symmetry)
            torus_spectrum=average_spectrum[:nstar*(Np-1):nstar,mstar:mstar*(int(Mp/2.)):mstar]

    random_spectral = np.random.randn([n,m])*torus_spectrum
    random_torus_uu = ks.ifft_(random_spectral,N,M,symmetry=symmetry)
    initial_condition_uu = amplitude*random_torus_uu/np.linalg.norm(random_torus_uu)
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
        torus = (uu,N,M,T,L,0)
    elif symmetry == 'rpo':
        uu = rpo.tile_seeded_noise(N,M,T,L,amplitude=amplitude,scale_type=scale_type,tms=tms)
        if L < 88:
            torus = (uu, N, M, T, L, (L/6))
        elif L > 500:
            torus = (uu, N, M, T, L, 0)
        else:
            torus = (uu, N, M, T, L, (L/30))
    elif symmetry == 'anti':
        uu = anti.initial_condition_generator(N,M,T,L,amplitude=amplitude,scale_type=scale_type,tms=tms)
        torus = (uu, N, M, T, L, 0)
    else:
        torus = (np.zeros([N,M]),N,M,T,L,1)
    return torus


def symbolic_initial_condition(symbol_block_list,period,speriod,*args,**kwargs):
    combine_method = kwargs.get('combine_method','tile')
    tileN = kwargs.get('tileN',128)
    tileM = kwargs.get('tileM',128)
    # if combine_method == 'tile':
    block_torus = tile.tile(symbol_block_list,period,speriod,block_symmetry='full')

    return block_torus


def glued_initial_condition(torusA,torusB,*args,**kwargs):
    # gluetype = kwargs.get('gluetype',1)
    # symmetry = kwargs.get('symmetry','ppo')
    # gluecomplexity = kwargs.get('gluecomplexity','complex')
    # buffertype=kwargs.get('buffertype','dynamic')
    # resolution=kwargs.get('resolution','low')
    # torusA0,torusB0 = validate_discretization(torusA,torusB)
    # ua,Na,Ma,Ta,La,Sa= torusA
    # ub,Nb,Mb,Tb,Lb,Sb= torusB
    #
    # if gluetype:
    #     torus_tmp = (ua,Nb,Ma+Mb,(Ta+Tb)/2.,Lb+La,Sa+Sb)
    #     Na,Nb = np.max([Na,Nb]),np.max([Na,Nb])
    #     nfinal,mfinal = disc.parameter_based_discretization(torus_tmp,resolution=resolution)
    # else:
    #     torus_tmp = (ua,Nb+Na,Ma,Ta+Tb,(Lb+La)/2,Sa+Sb)
    #     Ma,Mb = np.max([Ma,Mb]),np.max([Ma,Mb])
    #     nfinal,mfinal = disc.parameter_based_discretization(torus_tmp,resolution=resolution)
    #
    # if gluecomplexity=='simple':
    #     ARtori,Rtori,Ctori,CBtori = ((),),((),),((),),((),)
    #     Gtorus = merge_fields((torusA,torusB),gluetype=gluetype,tori_pair=True)
    #     return ARtori,Rtori,Ctori,CBtori,Gtorus
    #
    # if symmetry=='rpo':
    #     if np.sign(Sa)!=np.sign(Sb):
    #         Sa = -Sa
    #         ua = -1.0*np.fliplr(np.roll(ua,1,axis=1))
    #         torusA0 = (ua,Na,Ma,Ta,La,Sa)
    #     if np.abs(Sa)<np.abs(Sb):
    #         torusA,torusB = torusB,torusA
    #         ua,Na,Ma,Ta,La,Sa = torusA
    #         ub,Nb,Mb,Tb,Lb,Sb = torusB
    #     torusA_AR,torusB_AR = discretization_ratios(torusA,torusB,gluetype=gluetype)
    #     _,NAAR,MAAR,_,_,_=torusA_AR
    #     _,NBAR,MBAR,_,_,_=torusB_AR
    #     # torusA_AR_large = disc.rediscretize(torusA_AR,newN=8*Na0,newM=8*Ma0)
    #     # torusB_AR_large = disc.rediscretize(torusB_AR,newN=8*Nb0,newM=8*Mb0)
    #     # torusAfull=rpodm.mvf_rotate_torus(torusA)
    #     # torusBfull=rpodm.mvf_rotate_torus(torusB)
    #     # ARtori = (torusAfull,torusBfull)
    #     # torusA_R,torusB_R = find_best_rotation(torusAfull,torusBfull,gluetype=gluetype)
    #     # ARtori = (torusA_R,torusB_R)
    #
    #     torusA_R,torusB_R = find_best_rotation(torusA_AR,torusB_AR,gluetype=gluetype)
    #     ARtori = (torusA_AR,torusB_AR)
    #     Rtori = (torusA_R,torusB_R)
    #     Ctori = chop_fields(torusA_R,torusB_R,symmetry=symmetry,gluetype=gluetype,buffertype=buffertype)
    #     CBtori = convex_buffer(Ctori,symmetry=symmetry,gluetype=gluetype)
    #     merged_tori= merge_fields(CBtori,symmetry=symmetry,gluetype=gluetype)
    #     Gtorus = disc.rediscretize(merged_tori,newN=nfinal,newM=mfinal)
    #     # Gtorus = disc.residual_guided_discretization(Gtorus,symmetry=symmetry)
    # else:
    #     torusA = disc.rediscretize(torusA0,newN=8*Na,newM=8*Ma)
    #     torusB = disc.rediscretize(torusB0,newN=8*Nb,newM=8*Mb)
    #     torusA_AR,torusB_AR = discretization_ratios(torusA0,torusB0,gluetype=gluetype)
    #     ua,Na,Ma,Ta,La,Sa = torusA_AR
    #     ub,Nb,Mb,Tb,Lb,Sb = torusB_AR
    #     ua_star = -1.0*np.fliplr(np.roll(ua,1,axis=1))
    #     ub_star = -1.0*np.fliplr(np.roll(ub,1,axis=1))
    #     torusASTAR = (ua_star,Na,Ma,Ta,La,Sa)
    #     torusBSTAR = (ub_star,Nb,Mb,Tb,Lb,Sb)
    #     toruscombinations = ((torusA_AR,torusB_AR),(torusB_AR,torusA_AR),(torusASTAR,torusB_AR),(torusBSTAR,torusA_AR))
    #     ncombo = 0
    #     combolist = ['AB','BA','(RA)B','A(RB)']
    #     reslist = np.zeros([np.size(combolist),1])
    #     residualprev = 0
    #     for tori_pair in toruscombinations:
    #         torus0,torus1 = tori_pair
    #         ARtori0 = (torus0,torus1)
    #         Rtori0 = (torus0,torus1)
    #         Ctori0 = chop_fields(torus0,torus1,symmetry=symmetry,gluetype=gluetype,buffertype=buffertype)
    #         CBtori0 = convex_buffer(Ctori0,symmetry=symmetry,gluetype=gluetype)
    #         merged_tori = merge_fields(CBtori0,symmetry=symmetry,gluetype=gluetype)
    #         Gtorus0 = disc.rediscretize(merged_tori,newN=nfinal,newM=mfinal)
    #         # Gtorus0 = disc.residual_guided_discretization(Gtorus0,symmetry=symmetry)
    #         residual = ks.compute_residual_fromtuple(Gtorus0,symmetry=symmetry)
    #         reslist[ncombo]=residual
    #         print(combolist[ncombo],"residual",residual)
    #         if residualprev==0:
    #             ARtori = ARtori0
    #             Rtori = Rtori0
    #             Ctori = Ctori0
    #             CBtori = CBtori0
    #             Gtorus = Gtorus0
    #             residualprev = residual
    #         elif float(residual)<float(residualprev):
    #             ARtori = ARtori0
    #             Rtori = Rtori0
    #             Ctori = Ctori0
    #             CBtori = CBtori0
    #             Gtorus = Gtorus0
    #             residualprev=residual
    #         ncombo+=1
    return ARtori,Rtori,Ctori,CBtori,Gtorus


def main(*args,**kwargs):
    # symmetry=kwargs.get('symmetry','ppo')
    # gluetype=kwargs.get('gluetype',0)
    # buffersize=kwargs.get('buffersize',1)
    # rotatetorus = kwargs.get('rotatetorus',True)
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
    # torusLT = torus_io.import_torus(file0)
    # torusRB = torus_io.import_torus(file1)
    # base0_tmp=file0.split("\\")[-1]
    # base1_tmp=file1.split("\\")[-1]
    # base0 = base0_tmp.split('.h5')[0]
    # base1 = base1_tmp.split('.h5')[0]
    #
    # glued_torus = glued_initial_condition(torusLT,torusRB,symmetry=symmetry,gluetype=gluetype,buffersize=buffersize,slicenotchop=slicenotchop,rotatetorus=rotatetorus)
    # ksplot.plot_spatiotemporal_field(glued_torus,symmetry='none',display_flag=True)
    # glued_torus_final = disc.minimize_discretization(glued_torus,symmetry=symmetry,direction=mindiscdirection)
    # ksplot.plot_spatiotemporal_field(glued_torus_final,symmetry='none',display_flag=True)
    # u,N,M,T,L,S = glued_torus_final

    return None


if __name__=='__main__':
    main()