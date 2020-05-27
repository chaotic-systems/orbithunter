from __future__ import print_function, division, absolute_import
import os,sys
sys.path.insert(0, os.path.abspath(os.path.join(sys.argv[0], '../..')))
import numpy as np


def windowed_subdomain(self, Tmin=0, Tmax=0, Xmin=0, Xmax=0,**kwargs):
    symmetry = kwargs.get('symmetry','rpo')
    rotatex = kwargs.get('rotatex',0)
    rotatet = kwargs.get('rotatet',0)

    if symmetry =='rpo':
        tori_tuple = symm.frame_rotation(tori_tuple)

    if rotatex != 0:
        tori_tuple=symm.rotation(tori_tuple,rotatex,direction=1)
    if rotatet !=0:
        tori_tuple=symm.rotation(tori_tuple,rotatet,direction=0)
    uu,N,M,T,L,S0 = tori_tuple

    Lp = L/(2*pi)
    nmin,nmax = -int(N*(Tmin/T)),-int(N*(Tmax/T))
    mmin,mmax = int(M*(Xmin/Lp)),int(M*(Xmax/Lp))

    if nmax == 0 and nmin == 0:
        uu_windowed = uu[:,:]
    elif nmax!=0 and nmin!=0:
        if np.mod(nmin-nmax,2)==1:
            nmax += 1
        uu_windowed = uu[nmax:nmin,:]
    elif nmax == 0:
        if np.mod(nmin-nmax,2)==1:
            nmin += 1
        uu_windowed = uu[:nmin,:]
    elif nmin == 0:
        if np.mod(nmin-nmax,2)==1:
            nmax += 1
        uu_windowed = uu[nmax:,:]


    if mmax == 0 and mmin == 0:
        uu_windowed = uu_windowed[:,:]
    elif mmax!=0 and mmin!=0:
        if np.mod(mmin-mmax,2)==1:
            mmin -= 1
        uu_windowed = uu_windowed[:,mmin:mmax]
    elif mmax == 0:
        if np.mod(mmin-mmax,2)==1:
            mmin -= 1
        uu_windowed = uu_windowed[:,mmin:]
    elif mmin == 0:
        if np.mod(mmin-mmax,2)==1:
            mmax -= 1
        uu_windowed = uu_windowed[:,:mmax]

    nwin,mwin = np.shape(uu_windowed)
    nwin,mwin = int(nwin),int(mwin)
    twin,lwin = nwin*(T/N),mwin*(L/M)
    new_tori_tuple = (uu_windowed,nwin,mwin,twin,lwin,S0)

    if symmetry=='rpo':
        new_S0 = symm.calculate_shift(new_tori_tuple)
        new_tori_tuple = (uu_windowed,nwin,mwin,twin,lwin,new_S0)
        new_tori_tuple = symm.frame_rotation(new_tori_tuple,new_S0)


    return new_tori_tuple


def masked_window_fields(self, Tmin=0, Tmax=0, Xmin=0, Xmax=0,*args,**kwargs):
    symmetry = kwargs.get('symmetry','rpo')
    rotation = kwargs.get('rotation',0)
    if rotation!=0:
        orbit = rpo.rotate_torus(torus,rotation)
    if symmetry =='rpo':
        torus = rpo.mvf_rotate_torus(torus)
    uu,N,M,T,L,S0 = torus
    # cutout_counter = 0
    # nmin,nmax,mmin,mmax = [],[],[],[]
    # field_without_cutout= ()
    # cutouts = ()
    # if isinstance(Tmin,list):
    #     while cutout_counter < np.length(Tmin):
    #         nmin_tmp,nmax_tmp,mmin_tmp,mmax_tmp = discretization_window(torus,Xmin[cutout_counter],Xmax[cutout_counter]
    #                                                     ,Tmin[cutout_counter],Tmax[cutout_counter])
    #         nmin,nmax = np.append(nmin,nmin_tmp),np.append(nmax,nmax_tmp)
    #         mmin,mmax = np.append(mmin,mmin_tmp),np.append(mmax,mmax_tmp)
    #         cutout_counter+=1
    #         mask_cutout= np.ones(np.shape(uu),dtype=int)
    #         mask_not_cutout= np.zeros(np.shape(uu),dtype=int)
    #         mask_cutout[nmax:nmin,mmin:mmax] = 0
    #         mask_not_cutout[nmax:nmin,mmin:mmax] = 1
    #         uu_mask_cutout = np.ma.masked_array(uu,mask_cutout)
    #         uu_mask_not_cutout = np.ma.masked_array(uu,mask_not_cutout)
    #
    #         cutouts += ((uu_mask_not_cutout,N,M,T,L,S0),)
    #         fields_without_cutouts += ((uu_mask_cutout,N,M,T,L,S0),)
    # else:
    nmin,nmax,mmin,mmax = discretization_window(torus,Xmin,Xmax
                                                ,Tmin,Tmax)
    cutout= np.ones(np.shape(uu),dtype=int)
    not_cutout= np.zeros(np.shape(uu),dtype=int)
    cutout[nmax:nmin,mmin:mmax] = 0
    not_cutout[nmax:nmin,mmin:mmax] = 1
    uu_not_cutout = np.ma.masked_array(uu,not_cutout)
    uu_cutout = np.ma.masked_array(uu,cutout)
    cutout_torus = (uu_cutout,N,M,T,L,S0)
    not_cutout_torus =(uu_not_cutout,N,M,T,L,S0)
    return not_cutout_torus,cutout_torus


def discretization_window(self, Tmin=0, Tmax=0, Xmin=0, Xmax=0):
    u,N,M,T,L,S = torus
    Lp = L/(2*pi)
    nmin,nmax = -int(N*(Tmin/T)),-int(N*(Tmax/T))
    mmin,mmax = int(M*(Xmin/Lp)),int(M*(Xmax/Lp))
    if nmax == 0:
        nmax = -N
    if mmax == 0:
        mmax = M
    if nmin==0:
        nmin = -1
    if np.mod(nmin-nmax,2)==1:
        nmax += 1
    if np.mod(mmin-mmax,2)==1:
        mmax -= 1
    dimensions = [nmin,nmax,mmin,mmax]
    return dimensions