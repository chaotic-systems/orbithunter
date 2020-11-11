from __future__ import print_function, division, absolute_import
import os,sys
sys.path.insert(0, os.path.abspath(os.path.join(sys.argv[0], '../../..')))
from orbithunter import *
import time
import numpy as np
from math import pi
import matplotlib.pyplot as plt
import glob
from scipy.linalg import pinv

def plot_cartoon_Orbit():
    PWD = os.path.dirname(__file__)
    data_dir = os.path.abspath(os.path.join(PWD, "../../../data_and_figures/"))
    data_dir= os.path.join(data_dir,'')
    savedir = os.path.abspath(os.path.join(PWD, ''.join(["../../../../../.././GuBuCv17_figs/"])))
    #Draw approximate Orbit with non-matching tangent space and then
    #with underlying Orbit with correct tangent space.
    N = 400
    nu = np.reshape(np.linspace(0,2*pi,num=N),[N,1])
    u = np.reshape(np.linspace(0,2*pi,num=N),[1,N])
    a=1
    c=4*a
    x = (c+np.cos(nu))*np.cos(u)
    y =  (c+np.cos(nu))*np.sin(u)
    xnoise = (c+1.5*np.cos(nu))*np.cos(u)
    ynoise =  (2*c+2*np.cos(nu))*np.sin(u)
    z=np.tile(np.reshape(np.sin(nu),[N,1]),(1,N))
    znoise=np.transpose(np.tile(np.reshape(-np.cos(nu)-np.sin(3*nu),[N,1]),(1,N)))
    znoise = z + znoise
    fig =plt.figure()
    ax = plt.axes(projection='3d')
    ax.plot_wireframe(x,y,z,color='k',alpha=0.5,linewidth=0.5)
    ax.plot_wireframe(xnoise,ynoise,znoise,color='red',alpha=0.25,linewidth=0.5)
    ax.set_xlim3d(-c,c)
    ax.set_ylim3d(-c,c)
    ax.set_zlim3d(-4,4)
    ax.set_axis_off()
    filename0 = os.path.abspath(os.path.join(savedir,"./tori.png"))
    print(filename0)
    plt.savefig(filename0,dpi=250,bbox_inches='tight')
    #plt.show()
    return None


def plot_trawling_initial_to_final():
    return None


def plot_tile_guesses_and_cutouts():
    PWD = os.path.dirname(__file__)
    data_dir = os.path.abspath(os.path.join(PWD, "../../../data_and_figures/"))
    data_dir= os.path.join(data_dir,'')
    savedir = os.path.abspath(os.path.join(PWD, ''.join(["../../../../../.././GuBuCv17/"])))
    '''Tiling'''
    #Need qualitative guesses for tiles here (And the masked scalar fields?)
    dpi=1000

    '''Tile Guesses'''
    halfdefect_and_defect_import_filepath = os.path.abspath(os.path.join(data_dir,"./trawl/rpo/data/rpo_L21p99_T94p59.h5"))
    halfdefect_and_defect_initial_filename = os.path.abspath(os.path.join(savedir,"./MNG_halfdefect_defect_initial.png"))
    halfdefect_and_defect_guess = Orbit_io.import_Orbit(halfdefect_and_defect_import_filepath)
    u,n,m,t,L,s = halfdefect_and_defect_guess
    halfdefect_filename = os.path.abspath(os.path.join(savedir,"./MNG_halfdefect_guess.png"))
    field_minus_halfdefect_filename = os.path.abspath(os.path.join(savedir,"./MNG_halfdefect_cutout.png"))

    rotation = (0.2/(L/(2*pi)))*L
    Tmin,Tmax,Xmin,Xmax = 0,20,0,2.2
    ksplot.import_and_plot(halfdefect_and_defect_import_filepath,Tmin,Tmax,Xmin,Xmax,cutout=True
                           ,filename_init=halfdefect_and_defect_initial_filename,filename=field_minus_halfdefect_filename,cutoutfilename=halfdefect_filename
                           ,dpi=dpi,symmetry='rpo',rotation=rotation)
    ''''''
    Tmin,Tmax,Xmin,Xmax = 55,72,0.7,2.7
    defect_filename = os.path.abspath(os.path.join(savedir,"./MNG_defect_guess.png"))
    field_minus_defect_filename = os.path.abspath(os.path.join(savedir,"./MNG_defect_cutout.png"))
    ksplot.import_and_plot(halfdefect_and_defect_import_filepath,Tmin,Tmax,Xmin,Xmax,cutout=True
                           ,filename=field_minus_defect_filename,cutoutfilename=defect_filename
                           ,padding=False,dpi=dpi,symmetry='rpo')

    ''''''
    halfdefect2_import_filepath = os.path.abspath(os.path.join(data_dir,"./trawl/ppo/data/ppo_L21p94_t85p73.h5"))
    halfdefect2_initial_filename = os.path.abspath(os.path.join(savedir,"./halfdefect2_initial.png"))
    field_minus_halfdefect2_filename = os.path.abspath(os.path.join(savedir,"./halfdefect2guess.png"))
    halfdefect2_filename = os.path.abspath(os.path.join(savedir,"./halfdefect2guess_cutout.png"))
    halfdefect2_Orbit_guess = Orbit_io.import_Orbit(halfdefect2_import_filepath)
    Tmin,Tmax,Xmin,Xmax = 18,35,0.6,3
    ksplot.import_and_plot(halfdefect2_import_filepath,Tmin,Tmax,Xmin,Xmax,cutout=True
                           ,filename_init=halfdefect2_initial_filename,filename=field_minus_halfdefect2_filename,cutoutfilename=halfdefect2_filename
                           ,dpi=dpi,symmetry='ppo')
    ''''''


    hook_import_filepath = os.path.abspath(os.path.join(data_dir,"./trawl/ppo/data/ppo_L21p99_T10p25.h5"))
    hook_initial_filename = os.path.abspath(os.path.join(savedir,"./MNG_hook_initial.png"))
    hook_filename = os.path.abspath(os.path.join(savedir,"./MNG_hook_guess.png"))
    field_minus_hook_filename = os.path.abspath(os.path.join(savedir,"./MNG_hook_cutout.png"))
    hook_Orbit_guess = Orbit_io.import_Orbit(hook_import_filepath)
    u,n,m,t,L,s = hook_Orbit_guess
    rotation = L/2.
    #Half cell shift then x[1,2.5] t[0,10.25]
    Tmin,Tmax,Xmin,Xmax = 0,10.5,0.6,2.7
    ksplot.import_and_plot(hook_import_filepath,Tmin,Tmax,Xmin,Xmax,cutout=True
                           ,filename_init=hook_initial_filename,filename=field_minus_hook_filename,cutoutfilename=hook_filename
                           ,dpi=dpi,symmetry='ppo',rotation=rotation)
    ''''''
    gap_import_filepath = os.path.abspath(os.path.join(data_dir,"./trawl/po/data/full_L26.7_T54.h5"))
    gap_initial_filename = os.path.abspath(os.path.join(savedir,"./MNG_gap_initial.png"))
    gap_filename = os.path.abspath(os.path.join(savedir,"./MNG_gap_guess.png"))
    field_minus_gap_filename = os.path.abspath(os.path.join(savedir,"./MNG_gap_cutout.png"))
    Tmin,Tmax,Xmin,Xmax = 0,15,0,2.7
    ksplot.import_and_plot(gap_import_filepath,Tmin,Tmax,Xmin,Xmax,cutout=True
                           ,filename_init=gap_initial_filename,filename=field_minus_gap_filename,cutoutfilename=gap_filename
                           ,dpi=dpi,symmetry='none')
    ''''''
    gap2_import_filepath =os.path.abspath(os.path.join(data_dir,"./trawl/ppo/data/ppo_L24p33_T19p53.h5"))
    gap2_initial_filename = os.path.abspath(os.path.join(savedir,"./MNG_gap2_initial.png"))
    gap2_filename = os.path.abspath(os.path.join(savedir,"./MNG_gap2_guess.png"))
    field_minus_gap2_filename = os.path.abspath(os.path.join(savedir,"./MNG_gap2_cutout.png"))
    Tmin,Tmax,Xmin,Xmax = 8,20,0,2.7
    ksplot.import_and_plot(gap2_import_filepath,Tmin,Tmax,Xmin,Xmax,cutout=True
                           ,filename_init=gap2_initial_filename,filename=field_minus_gap2_filename,cutoutfilename=gap2_filename
                           ,dpi=dpi,symmetry='ppo')
    ''''''
    hookondefect_import_filepath =os.path.abspath(os.path.join(data_dir,"./trawl/ppo/data/ppo_L21p97_T73p52.h5"))
    hookondefect_initial_filename = os.path.abspath(os.path.join(savedir,"./MNG_hookondefect_initial.png"))
    hookondefect_filename = os.path.abspath(os.path.join(savedir,"./MNG_hookondefect_guess.png"))
    field_minus_hookondefect_filename = os.path.abspath(os.path.join(savedir,"./MNG_hookondefect_cutout.png"))
    hookondefect_Orbit_guess = Orbit_io.import_Orbit(hookondefect_import_filepath)
    #rotate by 1pi, xnew[.8,3] t=[30,50]
    _,_,_,_,L,_ = hookondefect_Orbit_guess
    rotation = 1./(L/(2*pi))*L
    Tmin,Tmax,Xmin,Xmax = 30,50,0.6,3.0
    ksplot.import_and_plot(hookondefect_import_filepath,Tmin,Tmax,Xmin,Xmax,cutout=True
                           ,filename_init=hookondefect_initial_filename,filename=field_minus_hookondefect_filename,cutoutfilename=hookondefect_filename
                           ,dpi=dpi,symmetry='ppo',rotation=rotation)


    ''''''
    # Import Tile data/full orbits with embedded tiles for tile collection data
    hookondefect2_import_filepath = os.path.abspath(os.path.join(data_dir,"./trawl/ppo/data/ppo_L21p93_t92p77.h5"))
    hookondefect2_initial_filename = os.path.abspath(os.path.join(savedir,"./MNG_hookondefect2_initial.png"))
    hookondefect2_filename = os.path.abspath(os.path.join(savedir,"./MNG_hookondefect2_guess.png"))
    field_minus_hookondefect2_filename = os.path.abspath(os.path.join(savedir,"./MNG_hookondefect2_cutout.png"))
    hookondefect2_Orbit_guess = Orbit_io.import_Orbit(hookondefect2_import_filepath)
    #half-cell, xnew[.8,3] t=[30,50]
    _,_,_,_,L,_ = hookondefect2_Orbit_guess
    rotation = -L/2.
    #rotate by 1pi, xnew[.25,3] t=[30,50]
    Tmin,Tmax,Xmin,Xmax = 35,55,0.25,2.55
    ksplot.import_and_plot(hookondefect2_import_filepath,Tmin,Tmax,Xmin,Xmax,cutout=True
                           ,filename_init=hookondefect2_initial_filename,filename=field_minus_hookondefect2_filename,cutoutfilename=hookondefect2_filename
                           ,dpi=dpi,symmetry='ppo',rotation=rotation)
    ''''''
    hookondefect3_import_filepath =os.path.abspath(os.path.join(data_dir,"./trawl/ppo/data/ppo_L21p99_T47p77.h5"))
    hookondefect3_initial_filename = os.path.abspath(os.path.join(savedir,"./MNG_hookondefect3_initial.png"))
    hookondefect3_filename = os.path.abspath(os.path.join(savedir,"./MNG_hookondefect3_guess.png"))
    field_minus_hookondefect3_filename = os.path.abspath(os.path.join(savedir,"./MNG_hookondefect3_cutout.png"))
    hookondefect3_Orbit_guess = Orbit_io.import_Orbit(hookondefect3_import_filepath)
    Tmin,Tmax,Xmin,Xmax = 0,25,0,2.5
    ksplot.import_and_plot(hookondefect3_import_filepath,Tmin,Tmax,Xmin,Xmax,cutout=True
                           ,filename_init=hookondefect3_initial_filename,filename=field_minus_hookondefect3_filename,cutoutfilename=hookondefect3_filename
                           ,dpi=dpi,symmetry='ppo')



    return None


def converge_tiles():
    save_folder_pathnames = fh.create_data_infrastructure('GuBuCv17','data')
    dpi=1000



    '''Tile Guesses'''

    # halfdefect_and_defect_import_filepath = os.path.abspath(os.path.join(data_dir,"./trawl/rpo/data/rpo_L21p99_T94p59.h5"))
    # halfdefect_and_defect_initial_filename = os.path.abspath(os.path.join(savedir,"./MNG_halfdefect_defect_initial.png"))
    # halfdefect_and_defect_guess = Orbit_io.import_Orbit(halfdefect_and_defect_import_filepath)
    # u,n,m,t,L,s = halfdefect_and_defect_guess
    # halfdefect_filename = os.path.abspath(os.path.join(savedir,"./MNG_halfdefect_guess.png"))
    # field_minus_halfdefect_filename = os.path.abspath(os.path.join(savedir,"./MNG_halfdefect_cutout.png"))
    # final_dataname= os.path.abspath(os.path.join(savedir,"./data/MNG_halfdefectf.h5"))
    # final_figname = os.path.abspath(os.path.join(savedir,"./figs/MNG_halfdefectf.png"))
    #
    # rotatex = (0.2/(L/(2*pi)))*L
    # Tmin,Tmax,Xmin,Xmax = 0,15,0,2.2
    # Orbit = Orbit_io.import_Orbit(halfdefect_and_defect_import_filepath)
    # u,n,m,t,l,s = Orbit
    # new_N,new_M=16*n,16*m
    # Orbit=disc.rediscretize(Orbit,new_N=new_N,new_M=new_M)
    # Orbit=sub.windowed_subdomain(Orbit,Tmin,Tmax,Xmin,Xmax,rotatex=rotatex)
    # ksplot.plot_spatiotemporal_field(Orbit,symmetry=symmetry,display_flag=True)
    # ksplot.plot_spatiotemporal_field(Orbit,symmetry='none',display_flag=True)
    #
    # uw,nw,mw,tw,lw,sw = Orbit
    # smalln,smallm = 2*int((nw/(2*new_N))*n),2*int((mw/(2*new_M))*m)
    # # smalln,smallm = 2,4
    #
    # Orbit=disc.rediscretize(Orbit,new_N=smalln,new_M=smallm)
    # Orbit_adj,retcode,res = ks.find_Orbit(Orbit,symmetry=symmetry)
    # Orbit_f,retcode,stats = ksdm.find_Orbit(Orbit_adj,symmetry=symmetry)
    # newsymm=test.retest_symmetry_type(Orbit_f,symmetry=symmetry)
    # if retcode and newsymm==symmetry:
    #     # Orbit_io.export_data_and_fig(Orbit_f,savedir,symmetry=symmetry)
    #     ksplot.plot_spatiotemporal_field(Orbit_f,symmetry=symmetry,filename=final_figname)
    #     Orbit_io.export_Orbit(Orbit_f,final_dataname,symmetry=symmetry)
    #
    # ''''''
    #
    # Tmin,Tmax,Xmin,Xmax = 55,72,0.7,2.7
    # defect_filename = os.path.abspath(os.path.join(savedir,"./MNG_defect_guess.png"))
    # field_minus_defect_filename = os.path.abspath(os.path.join(savedir,"./MNG_defect_cutout.png"))
    # final_figname = os.path.abspath(os.path.join(savedir,"./figs/MNG_defectf.png"))
    # final_dataname = os.path.abspath(os.path.join(savedir,"./data/MNG_defectf.h5"))
    # Orbit = Orbit_io.import_Orbit(halfdefect_and_defect_import_filepath)
    # u,n,m,t,l,s = Orbit
    # new_N,new_M=16*n,16*m
    # Orbit=disc.rediscretize(Orbit,new_N=new_N,new_M=new_M)
    # Orbit=sub.windowed_subdomain(Orbit,Tmin,Tmax,Xmin,Xmax)
    # uw,nw,mw,tw,lw,sw = Orbit
    # smalln,smallm = 2*int((nw/new_N)*n),2*int((mw/new_M)*m)
    # Orbit=disc.rediscretize(Orbit,new_N=smalln,new_M=smallm)
    # Orbit_adj,retcode,res = ks.find_Orbit(Orbit,symmetry=symmetry)
    # Orbit_f,retcode,stats = ksdm.find_Orbit(Orbit_adj,symmetry=symmetry)
    # newsymm=test.retest_symmetry_type(Orbit_f,symmetry=symmetry)
    # if retcode and newsymm==symmetry:
    #     # Orbit_io.export_data_and_fig(Orbit_f,savedir,symmetry=symmetry)
    #     ksplot.plot_spatiotemporal_field(Orbit_f,symmetry=symmetry,filename=final_figname)
    #     Orbit_io.export_Orbit(Orbit_f,final_dataname,symmetry=symmetry)
    #
    #
    # ''''''
    #
    #
    # halfdefect2_import_filepath = os.path.abspath(os.path.join(data_dir,"./trawl/ppo/data/ppo_L21p94_t85p73.h5"))
    # halfdefect2_initial_filename = os.path.abspath(os.path.join(savedir,"./halfdefect2_initial.png"))
    # field_minus_halfdefect2_filename = os.path.abspath(os.path.join(savedir,"./halfdefect2guess.png"))
    # halfdefect2_filename = os.path.abspath(os.path.join(savedir,"./halfdefect2guess_cutout.png"))
    # final_figname = os.path.abspath(os.path.join(savedir,"./figs/MNG_halfdefect2f.png"))
    # final_dataname = os.path.abspath(os.path.join(savedir,"./data/MNG_halfdefect2f.h5"))
    # halfdefect2_Orbit_guess = Orbit_io.import_Orbit(halfdefect2_import_filepath)
    # Tmin,Tmax,Xmin,Xmax = 18,35,0.6,3
    # Orbit = Orbit_io.import_Orbit(halfdefect2_import_filepath)
    # u,n,m,t,l,s = Orbit
    # new_N,new_M=16*n,16*m
    # Orbit=disc.rediscretize(Orbit,new_N=new_N,new_M=new_M)
    # Orbit=sub.windowed_subdomain(Orbit,Tmin,Tmax,Xmin,Xmax)
    # uw,nw,mw,tw,lw,sw = Orbit
    # smalln,smallm = 2*int((nw/new_N)*n),2*int((mw/new_M)*m)
    # Orbit=disc.rediscretize(Orbit,new_N=smalln,new_M=smallm)
    # Orbit_adj,retcode,res = ks.find_Orbit(Orbit,symmetry=symmetry)
    # Orbit_f,retcode,stats = ksdm.find_Orbit(Orbit_adj,symmetry=symmetry)
    # newsymm = retest_symmetry_type(Orbit_f,symmetry=symmetry)
    # if retcode and newsymm==symmetry:
    #     # Orbit_io.export_data_and_fig(Orbit_f,savedir,symmetry=symmetry)
    #     ksplot.plot_spatiotemporal_field(Orbit_f,symmetry=symmetry,filename=final_figname)
    #     Orbit_io.export_Orbit(Orbit_f,final_dataname,symmetry=symmetry)
    #
    #
    # hook_import_filepath = os.path.abspath(os.path.join(data_dir,"./trawl/ppo/data/ppo_L21p99_T10p25.h5"))
    # hook_initial_filename = os.path.abspath(os.path.join(savedir,"./MNG_hook_initial.png"))
    # hook_filename = os.path.abspath(os.path.join(savedir,"./MNG_hook_guess.png"))
    # field_minus_hook_filename = os.path.abspath(os.path.join(savedir,"./MNG_hook_cutout.png"))
    # final_figname = os.path.abspath(os.path.join(savedir,"./figs/MNG_hookf.png"))
    # final_dataname = os.path.abspath(os.path.join(savedir,"./data/MNG_hookf.h5"))
    # Orbit = Orbit_io.import_Orbit(hook_import_filepath)
    #
    # u,n,m,t,L,s = Orbit
    # rotatex = L/2.
    # #Half cell shift then x[1,2.5] t[0,10.25]
    # Tmin,Tmax,Xmin,Xmax = 0,10.5,0.5,2.7
    # u,n,m,t,l,s = Orbit
    # new_N,new_M=16*n,16*m
    # Orbit=disc.rediscretize(Orbit,new_N=new_N,new_M=new_M)
    # Orbit=sub.windowed_subdomain(Orbit,Tmin,Tmax,Xmin,Xmax,rotatex=rotatex)
    # uw,nw,mw,tw,lw,sw = Orbit
    # smalln,smallm = 2*int((nw/new_N)*n),2*int((mw/new_M)*m)
    # Orbit=disc.rediscretize(Orbit,new_N=smalln,new_M=smallm)
    #
    # # smalln,smallm = int((nw/new_N)*n),int((mw/new_M)*m)
    # smalln,smallm = 16,20
    # Orbit=disc.rediscretize(Orbit,new_N=smalln,new_M=smallm)
    # Orbit_adj,retcode,res = ks.find_Orbit(Orbit,symmetry=symmetry)
    # Orbit_f,retcode,stats = ksdm.find_Orbit(Orbit_adj,symmetry=symmetry)
    # newsymm=test.retest_symmetry_type(Orbit_f,symmetry=symmetry)
    #
    # if retcode and newsymm==symmetry:
    #     # Orbit_io.export_data_and_fig(Orbit_f,savedir,symmetry=symmetry)
    #     ksplot.plot_spatiotemporal_field(Orbit_f,symmetry=symmetry,filename=final_figname)
    #     Orbit_io.export_Orbit(Orbit_f,final_dataname,symmetry=symmetry)
    #
    # ''''''
    #
    #
    # # # Import Tile data/full orbits with embedded tiles for tile collection data
    # hookondefect2_import_filepath = os.path.abspath(os.path.join(data_dir,"./trawl/ppo/data/ppo_L21p93_t92p77.h5"))
    # hookondefect2_initial_filename = os.path.abspath(os.path.join(savedir,"./MNG_hookondefect2_initial.png"))
    # hookondefect2_filename = os.path.abspath(os.path.join(savedir,"./MNG_hookondefect2_guess.png"))
    # final_figname = os.path.abspath(os.path.join(savedir,"./figs/MNG_hookondefect2f.png"))
    # final_dataname = os.path.abspath(os.path.join(savedir,"./data/MNG_hookondefect2f.h5"))
    # #half-cell, xnew[.8,3] t=[30,50]
    # #rotate by 1pi, xnew[.25,3] t=[30,50]
    # Tmin,Tmax,Xmin,Xmax = 35,55 ,0.25,2.55
    # Orbit = Orbit_io.import_Orbit(hookondefect2_import_filepath)
    # _,_,_,_,L,_ = Orbit
    # rotatex = -L/2.
    # u,n,m,t,l,s = Orbit
    # new_N,new_M=16*n,16*m
    # Orbit=disc.rediscretize(Orbit,new_N=new_N,new_M=new_M)
    # Orbit=sub.windowed_subdomain(Orbit,Tmin,Tmax,Xmin,Xmax,rotatex=rotatex)
    # ksplot.plot_spatiotemporal_field(Orbit,symmetry=symmetry,display_flag=True)
    #
    # uw,nw,mw,tw,lw,sw = Orbit
    # smalln,smallm = 4*int((nw/new_N)*n),2*int((mw/new_M)*m)
    # Orbit=disc.rediscretize(Orbit,new_N=smalln,new_M=smallm)
    # Orbit_adj,retcode,res = ks.find_Orbit(Orbit,symmetry=symmetry)
    # Orbit_f,retcode,stats = ksdm.find_Orbit(Orbit_adj,symmetry=symmetry)
    # newsymm=test.retest_symmetry_type(Orbit_f,symmetry=symmetry)
    # if retcode and newsymm==symmetry:
    #     # Orbit_io.export_data_and_fig(Orbit_f,savedir,symmetry=symmetry)
    #     ksplot.plot_spatiotemporal_field(Orbit_f,symmetry=symmetry,filename=final_figname)
    #     Orbit_io.export_Orbit(Orbit_f,final_dataname,symmetry=symmetry)
    #
    #
    # ''''''
    #
    #
    # hookondefect3_import_filepath =os.path.abspath(os.path.join(data_dir,"./trawl/ppo/data/ppo_L21p99_T47p77.h5"))
    # hookondefect3_initial_filename = os.path.abspath(os.path.join(savedir,"./MNG_hookondefect3_initial.png"))
    # hookondefect3_filename = os.path.abspath(os.path.join(savedir,"./MNG_hookondefect3_guess.png"))
    # final_figname = os.path.abspath(os.path.join(savedir,"./figs/MNG_hookondefect3f.png"))
    # final_dataname = os.path.abspath(os.path.join(savedir,"./data/MNG_hookondefect3f.h5"))
    # Orbit = Orbit_io.import_Orbit(hookondefect3_import_filepath)
    # Tmin,Tmax,Xmin,Xmax = 0,25,0,2.5
    # u,n,m,t,l,s = Orbit
    # new_N,new_M=16*n,16*m
    # Orbit=disc.rediscretize(Orbit,new_N=new_N,new_M=new_M)
    # Orbit=sub.windowed_subdomain(Orbit,Tmin,Tmax,Xmin,Xmax)
    # uw,nw,mw,tw,lw,sw = Orbit
    # smalln,smallm = 2*int((nw/new_N)*n),2*int((mw/new_M)*m)
    # Orbit=disc.rediscretize(Orbit,new_N=smalln,new_M=smallm)
    # Orbit_adj,retcode,res = ks.find_Orbit(Orbit,symmetry=symmetry)
    # Orbit_f,retcode,stats = ksdm.find_Orbit(Orbit_adj,symmetry=symmetry)
    # newsymm=test.retest_symmetry_type(Orbit_f,symmetry=symmetry)
    # if retcode and newsymm==symmetry:
    #     # Orbit_io.export_data_and_fig(Orbit_f,savedir,symmetry=symmetry)
    #     ksplot.plot_spatiotemporal_field(Orbit_f,symmetry=symmetry,filename=final_figname)
    #     Orbit_io.export_Orbit(Orbit_f,final_dataname,symmetry=symmetry)
    #
    #
    # ''''''
    #
    #
    # ''''''
    #
    #
    # hookondefect_import_filepath =os.path.abspath(os.path.join(data_dir,"./trawl/ppo/data/ppo_L21p97_T73p52.h5"))
    # hookondefect_initial_filename = os.path.abspath(os.path.join(savedir,"./MNG_hookondefect_initial.png"))
    # hookondefect_filename = os.path.abspath(os.path.join(savedir,"./MNG_hookondefect_guess.png"))
    # final_figname = os.path.abspath(os.path.join(savedir,"./figs/MNG_hookondefectf.png"))
    # final_dataname = os.path.abspath(os.path.join(savedir,"./data/MNG_hookondefectf.h5"))
    # Orbit = Orbit_io.import_Orbit(hookondefect_import_filepath)
    # #rotate by 1pi, xnew[.8,3] t=[30,50]
    # _,n,m,_,L,_ = Orbit
    # rotatex = 1./(L/(2*pi))*L
    # Tmin,Tmax,Xmin,Xmax = 30,50,0.6,3.0
    # u,n,m,t,l,s = Orbit
    # new_N,new_M=16*n,16*m
    # Orbit=disc.rediscretize(Orbit,new_N=new_N,new_M=new_M)
    # Orbit=sub.windowed_subdomain(Orbit,Tmin,Tmax,Xmin,Xmax,rotatex=rotatex)
    # ksplot.plot_spatiotemporal_field(Orbit,symmetry=symmetry,display_flag=True)
    # uw,nw,mw,tw,lw,sw = Orbit
    # smalln,smallm = 4*int((nw/new_N)*n),2*int((mw/new_M)*m)
    # Orbit=disc.rediscretize(Orbit,new_N=smalln,new_M=smallm)
    # Orbit_adj,retcode,res = ks.find_Orbit(Orbit,symmetry=symmetry)
    # Orbit_f,retcode,stats = ksdm.find_Orbit(Orbit_adj,symmetry=symmetry)
    # newsymm=test.retest_symmetry_type(Orbit_f,symmetry=symmetry)
    # if retcode and newsymm==symmetry:
    #     # Orbit_io.export_data_and_fig(Orbit_f,savedir,symmetry=symmetry)
    #     ksplot.plot_spatiotemporal_field(Orbit_f,symmetry=symmetry,filename=final_figname)
    #     Orbit_io.export_Orbit(Orbit_f,final_dataname,symmetry=symmetry)
    #
    #
    #
    # minN,maxN = 8,48
    # minM,maxM = 16,48
    # rangearray = np.outer(np.zeros([len(range(minN,maxN,4)),len(range(minM,maxM,4))]),np.zeros([1,2]))
    # i,j =0,0
    # rangearray=np.reshape(rangearray,[len(range(minN,maxN,4)),len(range(minM,maxM,4)),2])
    # for j in np.arange(0,len(range(minM,maxM,4))):
    #     for i in np.arange(0,len(range(minN,maxN,4))):
    #         rangearray[i,j,0]=4*i+8
    #         rangearray[i,j,1]=4*j+16
    # rangearray=np.flipud(rangearray)
    # convarray = np.zeros([len(range(minN,maxN,4)),len(range(minM,maxM,4))])
    # ncount = 1
    # mcount = 0
    # for new_N in range(minN,maxN,4):
    #     for new_M in range(minM,maxM,4):
    #         step_max = 32*new_N*new_M
    #         Orbit=disc.rediscretize(Orbit,new_N=new_N,new_M=new_M)
    #         Orbit_adj,retcode,res = ks.find_Orbit(Orbit,symmetry=symmetry,step_max=step_max)
    #         Orbit_f,retcode,stats = ksdm.find_Orbit(Orbit_adj,symmetry=symmetry)
    #         newsymm=test.retest_symmetry_type(Orbit_f,symmetry=symmetry)
    #         if retcode and newsymm==symmetry:
    #             convarray[-ncount,mcount]=1
    #         else:
    #             print('################################## FAILURE ##############################################################')
    #         ksplot.plot_spatiotemporal_field(Orbit_f,symmetry=symmetry,filename=''.join([str(new_N),'p',str(new_M),'.png']))
    #
    #         mcount+=1
    #     ncount+=1
    #     mcount=0
    #     print(convarray)
    # convx,convy=[],[]
    # unconvx,unconvy = [],[]
    # irange,jrange = np.shape(convarray)
    # # convarray=np.array([[1,1,1,0,0,0],[1,1,0,0,0,0],[1,1,0,0,0,0],[1,1,1,1,0,0],[1,1,1,1,1,1],[1,1,1,1,1,1]])
    # for i in np.arange(-1,-irange-1,-1):
    #     for j in np.arange(0,jrange):
    #         if convarray[i,j]==1:
    #             convx=np.append(convx,rangearray[i,j,1])
    #             convy=np.append(convy,rangearray[i,j,0])
    #         else:
    #             unconvx = np.append(unconvx,rangearray[i,j,1])
    #             unconvy = np.append(unconvy,rangearray[i,j,0])
    #
    # plt.figure()
    # plt.scatter(convx,convy,marker='o',c='black')
    # plt.scatter(unconvx,unconvy,marker='<',c='red')
    # plt.savefig('NMplot.png', bbox_inches='tight',pad_inches=0)
    # plt.show()
    # ''''''
    #
    #
    # symmetry='anti'
    # gap_import_filepath = os.path.abspath(os.path.join(data_dir,"./trawl/po/data/full_L26.7_T54.h5"))
    # gap_initial_filename = os.path.abspath(os.path.join(savedir,"./MNG_gap_initial.png"))
    # gap_filename = os.path.abspath(os.path.join(savedir,"./MNG_gap_guess.png"))
    # final_figname = os.path.abspath(os.path.join(savedir,"./figs/MNG_gap.png"))
    # final_dataname = os.path.abspath(os.path.join(savedir,"./data/MNG_gapf.h5"))
    # Tmin,Tmax,Xmin,Xmax = 0,15,0,2.7
    # Orbit = Orbit_io.import_Orbit(gap_import_filepath)
    # u,n,m,t,l,s = Orbit
    # new_N,new_M=16*n,16*m
    # Orbit=disc.rediscretize(Orbit,new_N=new_N,new_M=new_M)
    # Orbit=sub.windowed_subdomain(Orbit,Tmin,Tmax,Xmin,Xmax)
    # # ksplot.plot_spatiotemporal_field(Orbit,symmetry=symmetry,display_flag=True)
    # uw,nw,mw,tw,lw,sw = Orbit
    # smalln,smallm = 2*int((nw/new_N)*n),2*int((mw/new_M)*m)
    # Orbit=disc.rediscretize(Orbit,new_N=smalln,new_M=smallm)
    # Orbit_adj,retcode,res = ks.find_Orbit(Orbit,symmetry=symmetry)
    # Orbit_f,retcode,stats = ksdm.find_Orbit(Orbit_adj,symmetry=symmetry)
    # # Orbit_f,retcode,stats = ksdm.find_Orbit(Orbit,symmetry=symmetry)
    #
    # newsymm=test.retest_symmetry_type(Orbit_f,symmetry=symmetry)
    # if retcode and newsymm==symmetry:
    #     ksplot.plot_spatiotemporal_field(Orbit_f,symmetry=symmetry,filename=final_figname)
    #     Orbit_io.export_Orbit(Orbit_f,final_dataname,symmetry=symmetry)
    #     # Orbit_io.export_data_and_fig(Orbit_f,savedir,symmetry=symmetry)
    # ''''''
    #
    # symmetry='anti'
    # gap2_import_filepath =os.path.abspath(os.path.join(data_dir,"./trawl/ppo/data/ppo_L24p33_T19p53.h5"))
    # gap2_initial_filename = os.path.abspath(os.path.join(savedir,"./MNG_gap2_initial.png"))
    # gap2_filename = os.path.abspath(os.path.join(savedir,"./MNG_gap2_guess.png"))
    # final_figname = os.path.abspath(os.path.join(savedir,"./figs/MNG_gap.png"))
    # final_dataname = os.path.abspath(os.path.join(savedir,"./data/MNG_gap2f.h5"))
    # Tmin,Tmax,Xmin,Xmax = 5,25,0,2.8
    # Orbit = Orbit_io.import_Orbit(gap2_import_filepath)
    # u,n,m,t,l,s = Orbit
    # new_N,new_M=16*n,16*m
    # rotatex=0.1*(2*pi)
    # Orbit=disc.rediscretize(Orbit,new_N=new_N,new_M=new_M)
    # Orbit=sub.windowed_subdomain(Orbit,Tmin,Tmax,Xmin,Xmax,rotatex=rotatex)
    # ksplot.plot_spatiotemporal_field(Orbit,symmetry='none',display_flag=True)
    # uw,nw,mw,tw,lw,sw = Orbit
    # smalln,smallm = 4*int((nw/new_N)*n),2*int((mw/new_M)*m)
    # Orbit=disc.rediscretize(Orbit,new_N=smalln,new_M=smallm)
    # Orbit_adj,retcode,res = ks.find_Orbit(Orbit,symmetry=symmetry)
    # Orbit_f,retcode,stats = ksdm.find_Orbit(Orbit_adj,symmetry=symmetry)
    # # Orbit_f,retcode,stats = ksdm.find_Orbit(Orbit,symmetry=symmetry)
    # newsymm=test.retest_symmetry_type(Orbit_f,symmetry=symmetry)
    # ksplot.plot_spatiotemporal_field(Orbit_f,symmetry='none',display_flag=True)
    # if retcode and newsymm==symmetry:
    #     ksplot.plot_spatiotemporal_field(Orbit_f,symmetry=symmetry,filename=final_figname)
    #     Orbit_io.export_Orbit(Orbit_f,final_dataname,symmetry=symmetry)
    #     # Orbit_io.export_data_and_fig(Orbit_f,savedir,symmetry=symmetry)
    ''''''


    #3-to-1
    symmetry='none'
    threetoone_import_filepath = fh.make_proper_pathname((save_folder_pathnames.root,"./trawl/ppo/data/ppo_L20p053_T31p323.h5"),folder=False)
    threetoone_initial = Orbit_io.import_Orbit(threetoone_import_filepath)
    ksplot.plot_spatiotemporal_field(threetoone_initial,symmetry=symmetry,display_flag=True)

    symmetry='rpo'
    u,n,m,t,L,s = threetoone_initial
    rotatex = -1.4*2*pi
    Tmin,Tmax,Xmin,Xmax = 10,t/2,0,0
    new_N,new_M=16*n,16*m
    Orbit=disc.rediscretize(threetoone_initial,new_N=new_N,new_M=new_M)
    Orbit=sub.windowed_subdomain(Orbit,Tmin,Tmax,Xmin,Xmax,rotatex=rotatex)
    ksplot.plot_spatiotemporal_field(Orbit,symmetry=symmetry,display_flag=True)

    uw,nw,mw,tw,lw,sw = Orbit
    # smalln,smallm = 4*int((nw/(2*new_N))*n),2*int((mw/(2*new_M))*m)
    # smalln,smallm = 2**(int(np.log2(lw))),2**(int(np.log2(tw))+1)
    smalln,smallm = 24,32
    Orbit=disc.rediscretize(Orbit,new_N=smalln,new_M=smallm)
    Orbit_adj,retcode,res = ks.find_Orbit(Orbit,symmetry=symmetry,step_max=64*smalln*smallm)
    Orbit_f,retcode,stats = ksdm.find_Orbit(Orbit_adj,symmetry=symmetry)
    if retcode:
        TL_pathnames,NM_pathnames,custom_pathnames = fh.create_save_data_pathnames(Orbit_f,save_folder_pathnames,custom_name='MNG_321')
        ksplot.plot_spatiotemporal_field(Orbit_f,symmetry=symmetry,filename=custom_pathnames.finalpng)
        Orbit_io.export_Orbit(Orbit_f,custom_pathnames.h5,symmetry=symmetry)
        _ = ksplot.import_and_plot(threetoone_import_filepath,Tmin,Tmax,Xmin,Xmax,cutout=True
                               ,filename_init=TL_pathnames.png,filename=custom_pathnames.png,cutoutfilename=custom_pathnames.initpng
                               ,dpi=dpi,symmetry='ppo')

    return None


def plot_converged_tiles():
    PWD = os.path.dirname(__file__)
    data_dir = os.path.join(os.path.abspath(os.path.join(PWD, "../../../data_and_figures/")),'')
    savedir = os.path.abspath(os.path.join(PWD, ''.join(["../../../../../.././GuBuCv17/"])))
    savedir = os.path.abspath(os.path.join(PWD, ''.join(["../../../data_and_figures/GuBuCv17/"])))
    dpi=1000

    '''Converged Tiles'''
    '''Defect 1'''
    defect1_filepath = os.path.abspath(os.path.join(data_dir,"./tiles/defects/defect1/final_tile/rpo_L13p02_T15.h5"))
    defect1_Orbit = Orbit_io.import_Orbit(defect1_filepath)
    defect1_Orbit = disc.rediscretize(defect1_Orbit,new_N=32,new_M=32)
    uu,n,m,t,l,s = defect1_Orbit
    ld,td = l,t

    vvdefect = np.reshape(ks.fft_(uu,n,m,symmetry='rpo'),[31,30])

    '''Defect 2'''
    defect2_filepath = os.path.abspath(os.path.join(data_dir,"./tiles/defects/defect2/final_tile/rpo_L17p5_T17.h5"))
    defect2_Orbit = Orbit_io.import_Orbit(defect2_filepath)
    '''Hook'''
    hook_filepath = os.path.abspath(os.path.join(data_dir,"./tiles/hook/final_tile/rpo_L13p07_T10.h5"))
    hook_Orbit = Orbit_io.import_Orbit(hook_filepath)
    hook_Orbit = disc.rediscretize(hook_Orbit,new_N=32,new_M=32)
    uu,n,m,t,l,s = hook_Orbit
    lh,th = l,t
    # uu = -1*np.fliplr(np.roll(uu,-3,axis=1))
    # hook_Orbit = (uu,n,m,t,l,-s)
    vvhook = np.reshape(ks.fft_(uu,n,m,symmetry='rpo'),[31,30])
    ksplot.plot_spatiotemporal_field(defect1_Orbit,symmetry='rpo',display_flag=True)
    ksplot.plot_spatiotemporal_field(hook_Orbit,symmetry='rpo',display_flag=True)

    fig,(ax1,ax2,ax3)=plt.subplots(1,3)
    ax1.imshow(np.abs(vvdefect),cmap='jet',interpolation='none',extent=[0,ld,0,td])
    ax2.imshow(np.abs(vvhook),cmap='jet',interpolation='none',extent=[0,lh,0,th])
    ax3.imshow(np.abs(vvhook)-np.abs(vvdefect),cmap='jet',interpolation='none',extent=[0,lh,0,th])

    plt.show()
    '''Gap'''
    gap_filepath = os.path.abspath(os.path.join(data_dir,"./tiles/gap/final_tile/anti_L17p5_T17.h5"))
    gap_Orbit = Orbit_io.import_Orbit(gap_filepath)
    '''Streak'''
    streak_filepath = os.path.abspath(os.path.join(data_dir,"./tiles/streak/final_tile/eqva_L3p195.h5"))
    streak_Orbit = Orbit_io.import_Orbit(streak_filepath)

    '''Figure save filenames'''
    defect1_fig_filename = os.path.abspath(os.path.join(savedir,"./MNG_defect.png"))
    defect2_fig_filename = os.path.abspath(os.path.join(savedir,"./MNG_defect2.png"))
    hook_fig_filename = os.path.abspath(os.path.join(savedir,"./MNG_hook.png"))
    gap_fig_filename = os.path.abspath(os.path.join(savedir,"./MNG_gap.png"))
    streak_fig_filename = os.path.abspath(os.path.join(savedir,"./MNG_streak.png"))

    ksplot.plot_spatiotemporal_field(defect1_Orbit,symmetry='rpo',filename=defect1_fig_filename,dpi=dpi)
    ksplot.plot_spatiotemporal_field(defect2_Orbit,symmetry='rpo',filename=defect2_fig_filename,dpi=dpi)
    ksplot.plot_spatiotemporal_field(hook_Orbit,symmetry='rpo',filename=hook_fig_filename,dpi=dpi)
    ksplot.plot_spatiotemporal_field(gap_Orbit,symmetry='anti',filename=gap_fig_filename,dpi=dpi)
    ksplot.plot_spatiotemporal_field(streak_Orbit,symmetry='eqvatime',filename=streak_fig_filename,dpi=dpi)
    return None


def plot_largeL(*args,**kwargs):
    '''
    :param args:
    :param kwargs: init=['constant','random'],
                   N= number of space points [INT]
                   T = integration time [FLOAT]
                   L0 = spatial domain size [FLOAT]
                   transient_T = guess at transient lengths (default value works well for init='constant' [FLOAT]
                   h = time step size [FLOAT]
                   save_filename_uu = filename or directory where you want to save u(x,t) field
                   save_filename_minmax = filename or directory where you want to save minmax field



    :return None
    '''
    init = kwargs.get('init','constant')
    N = int(kwargs.get('spatial_discretization',512))
    T = float(kwargs.get('integration_time',700.))
    L0 = float(kwargs.get('domain_size',512.))
    transient_T = float(kwargs.get('transient_time',200.))
    h = float(kwargs.get('step_size',0.1))
    ''''''

    if init == 'constant':
        v = np.array([0.17381466942642404+0.20216528393098937j
                         ,0.3250035087036375+0.6475450177597617j
                         ,0.07976473183256287-0.16619035167909507j
                         ,0.14548968325432+0.3078808401567302j
                         ,0.016225579870999363-0.0455458284976563j
                         ,0.047509729850359804+0.04955372131034326j
                         ,0.0014865227735564573-0.014804485332668177j
                         ,0.00842185187391528+0.006136799675022924j
                         ,-0.000490175543760971-0.0026015150583640276j
                         ,0.0012961361231364355+0.0006271857898673046j
                         ,0.00017014552669643144-0.0003932687355025244j
                         ,0.00017259399157549456+5.074928303922347e-05j
                         ,-3.746010380153342e-05-4.98473434375198e-05j
                         ,2.0615063496483646e-05+3.3317154671965435e-06j
                         ,-5.0666892964102674e-06-6.093819100348774e-06j])
        v = np.reshape(v,[15,1])
        v = np.concatenate(([[0]], v, np.zeros([N-31,1]), np.flipud(np.conj(v))), axis=0)
        # print(np.linalg.norm(v))
    else:
        v = np.random.randn(int(N//2)-1,1) + 1j*np.random.randn(int(N//2)-1,1)
        v = np.concatenate(([[0]], v, [[0]], np.flipud(np.conj(v))), axis=0)
        '''galilean invariance'''
        v[0,0],v[int(N//2),0]=0,0
        '''get rid of higher modes'''
        v[int(N//6):-int(N//6)]=0
        '''make the norm small to avoid annoying overflow issues'''
        v = v/np.linalg.norm(v)

    u = np.real(ifft(v))*N
    k = (2*pi*N/L0)*np.fft.fftfreq(N)
    k = np.reshape(k,[N,1])
    k[int(N//2)] = 0

    L = (k**2.0)-(k**4.0)
    E = np.exp(h*L)
    E2 = np.exp(h*L/2.0)

    complexroots = 16
    Marray = np.arange(1,complexroots+1,1)

    r = np.exp(1.0j*pi*(Marray-0.5)/complexroots)
    r = np.reshape(r,(1,complexroots))

    LR = h*np.tile(L, (1,complexroots)) + np.tile(r,(N,1))
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

    for n in range(0,nmax):
        t = n*h
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
        uu[0:N,n] = u[0:N,0]
        vv[0:N,n] = v[0:N,0]
        tt[n] = t
        if np.mod(n,1000)==0:
            print('.',end='')

    '''Truncate off the guess for transient period'''
    uu = uu[:,int(transient_T/h):]
    uu = np.flipud(np.transpose(uu))
    vv = vv[:,int(transient_T/h):]
    vv = np.flipud(np.transpose(vv))
    N0,M0 = np.shape(uu)


    T = T - transient_T
    print(np.max(np.max(uu)),np.min(np.min(uu)))



    N,M = np.shape(uu)

    qk_vec = 1j*(2*pi*M0/L0)*np.fft.fftfreq(M)
    qk_vec[int(M//2)]=0
    qk_vec = np.reshape(qk_vec,[1,M])
    vvx = np.multiply(np.tile(qk_vec, (N,1)),vv)
    vvxx = np.multiply(np.tile((qk_vec**2), (N,1)),vv)
    vvxxxx = np.multiply(np.tile((qk_vec**4), (N,1)),vv)
    uux = M*np.real(ifft(vvx,axis=1))
    uuxx= M*np.real(ifft(vvxx,axis=1))
    uuxxxx = M*np.real(ifft(vvxxxx,axis=1))
    uut = -np.multiply(uu,uux)-uuxx-uuxxxx

    wj = np.reshape(1j*(2*pi*N/T)*np.fft.fftfreq(N),[N,1])
    vv_t = fft(uu,axis=0)/np.sqrt(N)
    vv_ut = np.multiply(np.tile(wj, (1,M)),vv_t)
    uut2 = np.sqrt(N)*np.real(ifft(vv_ut,axis=0))

    ux_tuple = (uux,N,M,T,L0,0)
    ut_tuple = (uut,N,M,T,L0,0)
    uxxxx_tuple= disc.rediscretize((uuxxxx,N,M,T,L0,0),new_N=8192,new_M=8192)
    uux_tuple = disc.rediscretize((np.multiply(uu,uux),N,M,T,L0,0),new_N=8192,new_M=8192)
    D_tuple = disc.rediscretize((np.multiply(uux,uux),N,M,T,L0,0),new_N=8192,new_M=8192)
    P_tuple = disc.rediscretize((np.multiply(uuxx,uuxx),N,M,T,L0,0),new_N=8192,new_M=8192)
    uxx_tuple = disc.rediscretize((uuxx,N,M,T,L0,0),new_N=8192,new_M=8192)



    # ksplot.plot_spatiotemporal_field(ux_tuple,symmetry='none',dpi=250,padding=False,filename="ks_ux_largeL.png")
    # ksplot.plot_spatiotemporal_field(ut_tuple,symmetry='none',dpi=250,padding=False,filename="ks_ut_largeL.png")
    # ksplot.plot_spatiotemporal_field(uxxxx_tuple,symmetry='none',dpi=250,display_flag=True,padding=False,filename="ks_uxxxx_largeL.png")
    # ksplot.plot_spatiotemporal_field(uux_tuple,symmetry='none',dpi=250,display_flag=True,padding=0,filename="ks_uux_largeL.png")
    # ksplot.plot_spatiotemporal_field(D_tuple,symmetry='none',dpi=250,display_flag=True,padding=False,filename="ks_D_largeL.png")
    # ksplot.plot_spatiotemporal_field(P_tuple,symmetry='none',dpi=250,display_flag=True,padding=0,filename="ks_P_largeL.png")
    # ksplot.plot_spatiotemporal_field(uxx_tuple,symmetry='none',dpi=250,display_flag=True,padding=0,filename="ks_uxx_largeL.png")

    return None


def plot_frankenstein_tiling(**kwargs):
    symmetry=kwargs.get('symmetry','ppo')
    symmetry='none'
    PWD = os.path.dirname(__file__)
    data_dir = os.path.abspath(os.path.join(PWD, "../../data_and_figures/",''))
    hookondefect = os.path.abspath(os.path.join(data_dir,"./trawl/ppo/data/ppo_L21p93_t92p77.h5"))
    halfdefect = os.path.abspath(os.path.join(data_dir,"./trawl/ppo/data/ppo_L21p94_t85p73.h5"))
    streak = os.path.abspath(os.path.join(data_dir,"./tiles/streak/final_tile/eqva_L3p195.h5"))
    dpi = 1000

    hookondefect = Orbit_io.import_Orbit(hookondefect)
    halfdefect = Orbit_io.import_Orbit(halfdefect)
    streak = Orbit_io.import_Orbit(streak)
    U,N,M,T,L,S = hookondefect

    hookondefect = disc.rediscretize(hookondefect,new_N=16*N,new_M=16*M)
    hookondefect = symm.rotation(hookondefect,-2*2*pi,direction=1)
    hookondefect = sub.windowed_subdomain(hookondefect,35,55,0,2.4)
    hookondefect = symm.frame_rotation(hookondefect)
    halfdefect = disc.rediscretize(halfdefect,new_N=16*N,new_M=16*M)
    halfdefect = sub.windowed_subdomain(halfdefect,18,38,0.8,3)


    tilingdir = os.path.abspath(os.path.join(PWD,"../../data_and_figures/GuBuCv17"))
    tilingdir = os.path.join(tilingdir,'')

    Orbit_io.export_Orbit(hookondefect,os.path.abspath(os.path.join(tilingdir,'./HOD.h5')),symmetry='rpo')
    Orbit_io.export_Orbit(halfdefect,os.path.abspath(os.path.join(tilingdir,'./HD.h5')),symmetry='rpo')

    doublestreak= disc.rediscretize_eqva(streak,new_N=hookondefect[1],new_M=hookondefect[2]//2)
    u,n,m,t,l,s = doublestreak
    u = 2*u/np.max(u)
    dst = (np.concatenate((u,u),axis=1),n,2*m,2*l,2*l,0)
    HOD_ds_t = (np.concatenate((dst[0],hookondefect[0]),axis=1),dst[1],hookondefect[2]+dst[2],15,hookondefect[4]+dst[4],0)
    u,n,m,t,l,s = HOD_ds_t
    HOD_ds_t  = symm.frame_rotation(HOD_ds_t ,0.24*l)

    u,n,m,t,l,s =HOD_ds_t
    HODdoublerefl = (-1.0*np.fliplr(np.roll(u,1,axis=1)),n,m,t,l,0)
    HODR_ds_t = symm.rotation(HODdoublerefl,.5*l)
    HODds_HODRds_t = (np.concatenate((HODR_ds_t[0],HOD_ds_t[0]),axis=0),2*HODR_ds_t[1],HODR_ds_t[2],2*HODR_ds_t[3],HODR_ds_t[4],0)


    u,n,m,t,l,s = halfdefect
    HDRt = (-1.0*np.fliplr(np.roll(halfdefect[0],1,axis=1)),n,m,t,l,0)
    HDRt = symm.frame_rotation(HDRt,-5)
    hodpng=os.path.abspath(os.path.join(tilingdir,'./MNG_hookondefect.png'))
    hdpng = os.path.abspath(os.path.join(tilingdir,'./MNG_halfdefect.png'))
    ksplot.plot_spatiotemporal_field(hookondefect, symmetry='none',filename=hodpng,padding=False,dpi=dpi)
    ksplot.plot_spatiotemporal_field(HDRt, symmetry='none',filename=hdpng,padding=False,dpi=dpi)

    dsttwo = disc.rediscretize(dst,new_N=HDRt[1])
    HDR_ds_t =(np.concatenate((dsttwo[0],HDRt[0]),axis=1),dsttwo[1],HDRt[2]+dsttwo[2],15,HDRt[4]+dsttwo[4],0)
    HDR_ds_t = symm.rotation(HDR_ds_t,-0.25*HDR_ds_t[4])
    HDR_ds_t = disc.rediscretize(HDR_ds_t,new_M=HODds_HODRds_t[2])
    fundamental_domain = (np.concatenate((HDR_ds_t[0],HODds_HODRds_t[0]),axis=0),HDR_ds_t[1]+HODds_HODRds_t[1],HODds_HODRds_t[2],88.62262688790827/2.,30.141720076724777,0)

    ppo_initial = symm.relevant_symmetry_operation(fundamental_domain,symmetry='ppo')

    # tilingdir = os.path.abspath(os.path.join(PWD,'../../../../../.././GuBuCv17/',''))
    subdomain_zero_filename = os.path.abspath(os.path.join(tilingdir,'./MNG_tiling_subdomain0.png'))
    subdomain_one_filename = os.path.abspath(os.path.join(tilingdir,'./MNG_tiling_subdomain1.png'))
    subdomain_two_filename = os.path.abspath(os.path.join(tilingdir,'./MNG_tiling_subdomain2.png'))
    subdomain_double_filename = os.path.abspath(os.path.join(tilingdir,'./MNG_tiling_twosubdomains.png'))
    subdomain_fundamental_filename = os.path.abspath(os.path.join(tilingdir,'./MNG_tiling_fundamental.png'))
    initial_figure_filename = os.path.abspath(os.path.join(tilingdir,'./MNG_ppo_tiling_initial.png'))
    final_figure_filename = os.path.abspath(os.path.join(tilingdir,'./ppo_tiling_final.png'))
    finaldata = os.path.abspath(os.path.join(tilingdir,'./ppo_tiling_final.h5'))

    ksplot.plot_spatiotemporal_field(HOD_ds_t, symmetry='none',filename=subdomain_zero_filename,padding=False,dpi=dpi)
    ksplot.plot_spatiotemporal_field(HODR_ds_t, symmetry='none',filename=subdomain_one_filename,padding=False,dpi=dpi)
    ksplot.plot_spatiotemporal_field(HDR_ds_t, symmetry='none',filename=subdomain_two_filename,padding=False,dpi=dpi)
    ksplot.plot_spatiotemporal_field(HODds_HODRds_t, symmetry='none',filename=subdomain_double_filename,padding=False,dpi=dpi)
    ksplot.plot_spatiotemporal_field(fundamental_domain, symmetry='none',filename=subdomain_fundamental_filename,padding=False,dpi=dpi)
    ksplot.plot_spatiotemporal_field(ppo_initial, symmetry='ppo',filename=initial_figure_filename,dpi=dpi,padding=False)

    ppo_adjoint,ret,res = ks.find_Orbit(ppo_initial,symmetry='ppo')
    ppo_final,retcode,_ = ksdm.find_Orbit(ppo_adjoint,symmetry='ppo')
    # Orbit_io.export_Orbit(ppo_final,'ppo_tiling_final_2.h5',symmetry='ppo')
    u,n,m,T,L,S = ppo_final
    # print('final T,L',T,L)
    # if retcode:
    #     ksplot.plot_spatiotemporal_field(ppo_final, symmetry='ppo',display_flag=True,filename='ppo_tiling_final_2.png',dpi=1000)
    #     Orbit_io.export_Orbit(ppo_final,finaldata)
    return None


def plot_ppo1ppo2ppo3_gluing(*args,**kwargs):
    runtype=kwargs.get('runtype','automatic')
    gluetype=kwargs.get('gluetype',1)
    symmetry=kwargs.get('symmetry','ppo')
    searchdepth = kwargs.get('searchdepth','thorough')
    fixedL = kwargs.get('fixedL',False)
    buffertype=kwargs.get('buffertype','dynamic')
    resolution=kwargs.get('resolution','low')
    glue_number = kwargs.get('glue_number','single')
    initial_condition_directory = kwargs.get('initial_condition_directory',
                                            "C:\\Users\\matth\\Desktop\\GuBuCv17\\tiling\\initial\\")
    if gluetype:
        gluename='space'
    else:
        gluename='time'
    PWD = os.path.dirname(__file__)
    init_dir = os.path.abspath(os.path.join(PWD, initial_condition_directory,''))
    savedir = kwargs.get("save_directory",''.join(["../../../data_and_figures/GuBuCv17/"]))
    savedatadir =os.path.abspath(os.path.join(savedir,'./save/',''))
    savegluedir =os.path.abspath(os.path.join(savedir,'./save/',''))
    savefigsdir =os.path.abspath(os.path.join(savedir,'./save/',''))
    saveFAILdir =os.path.abspath(os.path.join(savedir,'./save/',''))
    symmetry='ppo'
    gluetype=1
    initial_condition_directory =savedir
    import_filename_zero = 'ppo_L21p99_T10p25.h5'
    import_filename_one ='ppo_L21p99_T14p33.h5'
    '''glue ppo1 to ppo2 to create ppo1ppo2'''
    filename_zero,filename_one=import_filename_zero,import_filename_one
    basename_zero,basename_one = filename_zero.split('.h5')[0],filename_one.split('.h5')[0]
    filename_zero = os.path.join(initial_condition_directory,filename_zero)
    filename_one =os.path.join(initial_condition_directory,filename_one)
    filegluenamebase = ''.join([basename_zero,"_",basename_one,"_",gluename])
    dataname = ''.join([savedatadir,filegluenamebase,".h5"])
    datanameFAIL = ''.join([saveFAILdir,filegluenamebase,".h5"])
    Orbit_zero = Orbit_io.import_Orbit(filename_zero)
    Orbit_one = Orbit_io.import_Orbit(filename_one)
    print('Attempting to glue',basename_zero, basename_one, 'in', gluename)
    tori_pair_initial,aspect_ratio_tori,\
    rotated_tori,chopped_tori,convex_buffer_tori,glued_Orbit \
        = Orbit_init.glued_initial_condition(Orbit_zero,Orbit_one,symmetry=symmetry,fixedL=fixedL,
                                             gluetype=gluetype,buffertype=buffertype,resolution=resolution)
    initial_Orbit = Orbit_init.merge_fields(tori_pair_initial,symmetry=symmetry,gluetype=gluetype,tori_pair=True)
    aspect_ratio_Orbit = Orbit_init.merge_fields(aspect_ratio_tori,symmetry=symmetry,gluetype=gluetype,tori_pair=True)
    rotated_Orbit = Orbit_init.merge_fields(rotated_tori,symmetry=symmetry,gluetype=gluetype,tori_pair=True)
    chopped_Orbit = Orbit_init.merge_fields(chopped_tori,symmetry=symmetry,gluetype=gluetype)
    convex_buffer_Orbit = Orbit_init.merge_fields(convex_buffer_tori,symmetry=symmetry,gluetype=gluetype)

    ksplot.plot_gluing_process(initial_Orbit,aspect_ratio_Orbit,rotated_Orbit,chopped_Orbit,convex_buffer_Orbit,glued_Orbit,glued_Orbit,symmetry=symmetry,gluetype=gluetype,display_flag=True)

    glued_Orbit_adjoint,retcode,res = ks.find_Orbit(glued_Orbit,symmetry=symmetry)
    converged_Orbit,retcode,res = ksdm.find_Orbit(glued_Orbit_adjoint,symmetry=symmetry)

    newsymm = test.retest_symmetry_type(converged_Orbit,symmetry=symmetry)
    print('Tf,Lf,Sf',converged_Orbit[-3],converged_Orbit[-2],converged_Orbit[-1])
    if retcode and newsymm==symmetry:
        gluing_filename = ''.join([savegluedir,filegluenamebase,"_gluing.png"])
        final_figname = ''.join([savefigsdir,filegluenamebase,".png"])
        final_dataname = ''.join([savedatadir,filegluenamebase,".h5"])
        print('Solution Converged: Saving to ', dataname)
        ksplot.plot_gluing_process(initial_Orbit,aspect_ratio_Orbit,rotated_Orbit,chopped_Orbit,convex_buffer_Orbit,glued_Orbit,converged_Orbit,symmetry=symmetry,gluetype=gluetype,filename=gluing_filename)
        ksplot.plot_spatiotemporal_field(converged_Orbit,symmetry=symmetry,filename=final_figname)
        Orbit_io.export_Orbit(converged_Orbit,final_dataname,symmetry=symmetry)
        dataname_ppo1ppo2 = final_dataname

    '''glue ppo1ppo2 to ppo3 to create ppo1ppo2ppo3'''
    import_filename_two ='ppo_L22p00_T32p35.h5'
    import_filename_zero_plus_one = 'ppo_L21p99_T10p25_ppo_L21p99_T14p33_space.h5'
    import_filename123 = 'MNGppo123space.h5'
    filename_zero,filename_one=import_filename_zero_plus_one,import_filename_two
    basename_zero,basename_one = filename_zero.split('.h5')[0],filename_one.split('.h5')[0]
    initial_condition_directory = savedatadir
    filename_zero = os.path.join(initial_condition_directory,filename_zero)
    filename_one =os.path.join(initial_condition_directory,filename_one)
    filegluenamebase = ''.join([basename_zero,"_",basename_one,"_",gluename])
    dataname = ''.join([savedatadir,filegluenamebase,".h5"])
    Orbit_zero = Orbit_io.import_Orbit(filename_zero)
    Orbit_one = Orbit_io.import_Orbit(filename_one)
    print('Attempting to glue',basename_zero, basename_one, 'in', gluename)

    tori_pair_initial,aspect_ratio_tori,\
    rotated_tori,chopped_tori,convex_buffer_tori,glued_Orbit \
        = Orbit_init.glued_initial_condition(Orbit_zero,Orbit_one,symmetry=symmetry,fixedL=fixedL,
                                             gluetype=gluetype,buffertype=buffertype,resolution=resolution)
    initial_Orbit = Orbit_init.merge_fields(tori_pair_initial,symmetry=symmetry,gluetype=gluetype,tori_pair=True)
    aspect_ratio_Orbit = Orbit_init.merge_fields(aspect_ratio_tori,symmetry=symmetry,gluetype=gluetype,tori_pair=True)
    rotated_Orbit = Orbit_init.merge_fields(rotated_tori,symmetry=symmetry,gluetype=gluetype,tori_pair=True)
    chopped_Orbit = Orbit_init.merge_fields(chopped_tori,symmetry=symmetry,gluetype=gluetype)
    convex_buffer_Orbit = Orbit_init.merge_fields(convex_buffer_tori,symmetry=symmetry,gluetype=gluetype)
    glued_Orbit_adjoint,retcode,res = ks.find_Orbit(glued_Orbit,symmetry=symmetry)
    converged_Orbit,retcode,res = ksdm.find_Orbit(glued_Orbit_adjoint,symmetry=symmetry)

    newsymm = test.retest_symmetry_type(converged_Orbit,symmetry=symmetry)
    print('Tf,Lf,Sf',converged_Orbit[-3],converged_Orbit[-2],converged_Orbit[-1])
    if retcode and newsymm==symmetry:
        gluing_filename = ''.join([savegluedir,filegluenamebase,"_gluing.png"])
        final_figname = ''.join([savefigsdir,filegluenamebase,".png"])
        final_dataname = ''.join([savedatadir,filegluenamebase,".h5"])
        print('Solution Converged: Saving to ', dataname)
        ksplot.plot_gluing_process(initial_Orbit,aspect_ratio_Orbit,rotated_Orbit,chopped_Orbit,convex_buffer_Orbit,glued_Orbit,converged_Orbit,symmetry=symmetry,gluetype=gluetype,filename=gluing_filename)
        ksplot.plot_spatiotemporal_field(converged_Orbit,symmetry=symmetry,filename=final_figname)
        Orbit_io.export_Orbit(converged_Orbit,final_dataname,symmetry=symmetry)
        dataname_ppo1ppo2 = final_dataname

    return None


def continue_tiles(*args,**kwargs):
    dimension = kwargs.get('dimension',1)
    overwrite = kwargs.get('overwrite',1)
    increase = kwargs.get('increase',1)
    decrease = kwargs.get('decrease',1)
    symmetry = kwargs.get('symmetry','rpo')
    save = kwargs.get('save','series')
    PWD = os.path.dirname(__file__)
    data_dir = os.path.abspath(os.path.join(PWD, '../../../data_and_figures/GuBuCv17/'))
    data_dir= os.path.join(data_dir,'')
    parent_folder = os.path.abspath(os.path.join(PWD, '../../../data_and_figures/GuBuCv17/'))

    deltamodifier=kwargs.get('deltamodifier',1)

    dataname = os.path.abspath(os.path.join(parent_folder,"./data/MNG_defect_hookondefect_time.h5"))
    Orbit = Orbit_io.import_Orbit(dataname)
    continuation.pseudoarclength_continuation(dataname,axis=axis,increase=1,decrease=0,symmetry='rpo',save=save,deltamodifier=deltamodifier)
    continuation.pseudoarclength_continuation(dataname,axis=axis,increase=0,decrease=1,symmetry='rpo',save=save,deltamodifier=deltamodifier)

    dataname = os.path.abspath(os.path.join(parent_folder,"./data/MNG_hookf.h5"))
    # Orbit = Orbit_io.import_Orbit(dataname)
    # continuation.pseudoarclength_continuation(dataname,axis=axis,increase=1,decrease=0,symmetry=symmetry,save=save,deltamodifier=2)
    continuation.pseudoarclength_continuation(dataname,axis=axis,increase=0,decrease=1,symmetry=symmetry,save=save,deltamodifier=deltamodifier)

    dataname = os.path.abspath(os.path.join(parent_folder,"./data/MNG_hookondefect2f.h5"))
    # Orbit = Orbit_io.import_Orbit(dataname)
    continuation.pseudoarclength_continuation(dataname,axis=axis,increase=0,decrease=decrease,symmetry=symmetry,save=save,deltamodifier=0.5)

    dataname = os.path.abspath(os.path.join(parent_folder,"./data/MNG_defectf.h5"))
    # Orbit = Orbit_io.import_Orbit(dataname)
    continuation.pseudoarclength_continuation(dataname,axis=axis,increase=increase,decrease=decrease,symmetry=symmetry,save=save,deltamodifier=deltamodifier)

    dataname = os.path.abspath(os.path.join(parent_folder,"./data/MNG_gapf.h5"))
    Orbit = Orbit_io.import_Orbit(dataname)
    continuation.pseudoarclength_continuation(dataname,axis=axis,increase=1,decrease=0,symmetry='anti',save=save,deltamodifier=deltamodifier)
    continuation.pseudoarclength_continuation(dataname,axis=axis,increase=0,decrease=1,symmetry='anti',save=save,deltamodifier=deltamodifier)

    dataname = os.path.abspath(os.path.join(parent_folder,"./data/MNG_reallylong.h5"))
    # Orbit = Orbit_io.import_Orbit(dataname)
    continuation.pseudoarclength_continuation(dataname,axis=axis,increase=increase,decrease=decrease,symmetry='anti',save=save,deltamodifier=deltamodifier)

    return None


def quantized_family(*args,**kwargs):
    symmetry=kwargs.get('symmetry','rpo')
    runtype=kwargs.get('runtype','automatic')
    gluetype=kwargs.get('gluetype',1)
    # symmetry=kwargs.get('symmetry','ppo')
    searchdepth = kwargs.get('searchdepth','thorough')
    fixedL = kwargs.get('fixedL',False)
    buffertype=kwargs.get('buffertype','dynamic')
    resolution=kwargs.get('resolution','low')
    glue_number = kwargs.get('glue_number','single')
    PWD = os.path.dirname(__file__)
    # data_dir = os.path.abspath(os.path.join(PWD, "../../../data_and_figures/"))
    # save_folder = os.path.join(data_dir,'')
    import_folder = os.path.join(os.path.abspath(os.path.join(PWD, ''.join(["../../../data_and_figures/GuBuCv17/data"]))),'')
    hookondefect = Orbit_io.import_Orbit(''.join([import_folder,'MNG_hookondefect2f.h5']))
    # ksplot.plot_spatiotemporal_field(hookondefect,symmetry=symmetry,display_flag=True)

    defect = Orbit_io.import_Orbit(''.join([import_folder,'MNG_defectf.h5']))
    filegluenamebase = 'MNG_defect_hookondefect_time'
    ARtori,Rtori,Ctori,CBtori,GOrbit = Orbit_init.glued_initial_condition(hookondefect,defect,symmetry=symmetry,fixedL=fixedL,gluetype=gluetype,buffertype=buffertype,resolution=resolution)
    # IOrbit = Orbit_init.merge_fields(tori_pair_init,symmetry=symmetry,gluetype=gluetype,tori_pair=True)
    AROrbit = Orbit_init.merge_fields(ARtori,symmetry=symmetry,gluetype=gluetype,tori_pair=True)
    ROrbit = Orbit_init.merge_fields(Rtori,symmetry=symmetry,gluetype=gluetype,tori_pair=True)
    COrbit = Orbit_init.merge_fields(Ctori,symmetry=symmetry,gluetype=gluetype)
    CBOrbit = Orbit_init.merge_fields(CBtori,symmetry=symmetry,gluetype=gluetype)
    # ksplot.plot_spatiotemporal_field(GOrbit,symmetry=symmetry,display_flag=True)

    GOrbit,retcode,res = ks.find_Orbit(GOrbit,symmetry=symmetry)
    FOrbit,retcode,res = ksdm.find_Orbit(GOrbit,symmetry=symmetry)
    newsymm = test.retest_symmetry_type(GOrbit,symmetry=symmetry)
    print('Tf,Lf,Sf',FOrbit[-3],FOrbit[-2],FOrbit[-1])
    if retcode and newsymm==symmetry:
        gluing_filename = ''.join([import_folder,filegluenamebase,"_gluing.png"])
        final_figname = ''.join([import_folder,filegluenamebase,".png"])
        final_dataname = ''.join([import_folder,filegluenamebase,".h5"])
        print('Solution Converged: Saving to ', final_dataname)
        ksplot.plot_gluing_process(AROrbit,ROrbit,COrbit,CBOrbit,GOrbit,FOrbit,symmetry=symmetry,gluetype=gluetype,filename=gluing_filename)
        ksplot.plot_spatiotemporal_field(FOrbit,symmetry=symmetry,filename=final_figname)
        Orbit_io.export_Orbit(FOrbit,final_dataname,symmetry=symmetry)
    else:
        print('Gluing did not converge')
    return None


def slice_and_section(*args,**kwargs):
    return None


def random_figures(*args,**kwargs):
    dimension = kwargs.get('dimension',1)
    overwrite = kwargs.get('overwrite',1)
    increase = kwargs.get('increase',1)
    decrease = kwargs.get('decrease',1)
    symmetry = kwargs.get('symmetry','ppo')
    save = kwargs.get('save','series')
    PWD = os.path.dirname(__file__)
    data_dir = os.path.abspath(os.path.join(PWD, '../../../data_and_figures/trawl/ppo/'))
    # data_dir= os.path.join(data_dir,'')
    parent_folder = os.path.abspath(os.path.join(PWD, '../../../data_and_figures/ppo/'))
    file_list = np.array(os.listdir(data_dir))
    for file0 in file_list:
        if file0.endswith(".h5"):
            filename = os.path.join(data_dir,file0)
            Orbit = Orbit_io.import_Orbit(filename)
            basename = filename.split('.h5')[0]
            save_fig_name = ''.join([basename,'.png'])
            ksplot.plot_spatiotemporal_field(Orbit,symmetry=symmetry,filename=save_fig_name)


    return None


def convergence_rate(*args,**kwargs):
    symmetry='ppo'

    PWD = os.path.dirname(__file__)
    save_directory_extension = kwargs.get("save_directory",''.join(["../../../data_and_figures/convergence_rate/",str(symmetry)]))
    save_directory = os.path.abspath(os.path.join(PWD,save_directory_extension))
    converged_figs_directory = os.path.join(os.path.abspath(os.path.join(save_directory, "./figs/")),'')
    converged_data_directory = os.path.join(os.path.abspath(os.path.join(save_directory, "./data/")),'')
    converged_otherdata_directory = os.path.join(os.path.abspath(os.path.join(save_directory, "./otherdata/")),'')


    Lrange = np.flipud(np.linspace(22,66,num=3))
    Trange = np.flipud(np.linspace(44,176,num=12))
    print(Lrange)
    print(Trange)
    Lgrid = np.tile(np.reshape(Lrange,[np.size(Lrange),1]),(1,np.size(Trange)))
    Tgrid = np.tile(np.reshape(Trange,[np.size(Trange),1]),(np.size(Lrange),1))
    Lgridvec = np.reshape(Lgrid,np.size(Lgrid))
    Tgridvec = np.reshape(Tgrid,np.size(Tgrid))
    symmetry='ppo'
    convergence_list_ppo = trawl.findsoln(Lrange,Trange,symmetry=symmetry,stats=True)
    fail_indices = np.where(convergence_list_ppo==0)
    success_indices = np.where(convergence_list_ppo==1)
    ppo_percentage = np.sum(convergence_list_ppo)/np.size(convergence_list_ppo)

    plt.figure()
    plt.scatter(Lgridvec[success_indices],Tgridvec[success_indices],marker='o',c='k')
    plt.scatter(Lgridvec[fail_indices],Tgridvec[fail_indices],marker='^',c='r')
    plt.savefig(''.join([converged_otherdata_directory,str(symmetry),'scatter.png']), bbox_inches='tight',pad_inches=0)
    plt.close()

    symmetry='rpo'
    save_directory_extension = kwargs.get("save_directory",''.join(["../../../data_and_figures/convergence_rate/",str(symmetry)]))
    save_directory = os.path.abspath(os.path.join(PWD,save_directory_extension))
    converged_figs_directory = os.path.join(os.path.abspath(os.path.join(save_directory, "./figs/")),'')
    converged_data_directory = os.path.join(os.path.abspath(os.path.join(save_directory, "./data/")),'')
    converged_otherdata_directory = os.path.join(os.path.abspath(os.path.join(save_directory, "./otherdata/")),'')

    convergence_list_rpo = trawl.findsoln(Lrange,Trange,symmetry=symmetry,stats=True)
    fail_indices = np.where(convergence_list_rpo==0)
    success_indices = np.where(convergence_list_rpo==1)
    rpo_percentage = np.sum(convergence_list_rpo)/np.size(convergence_list_rpo)
    print(rpo_percentage)

    plt.figure()
    plt.scatter(Lgridvec[success_indices],Tgridvec[success_indices],marker='o',c='k')
    plt.scatter(Lgridvec[fail_indices],Tgridvec[fail_indices],marker='^',c='r')
    plt.savefig(''.join([converged_otherdata_directory,'scatter.png']), bbox_inches='tight',pad_inches=0)
    plt.close()

    return None


def tile_properties():
    PWD = os.path.dirname(__file__)
    data_dir = os.path.join(os.path.abspath(os.path.join(PWD, "../../../data_and_figures/")),'')
    parent_folder = os.path.join(os.path.abspath(os.path.join(PWD, ''.join(["../../../data_and_figures/GuBuCv17/data/MNG_defectf/data/"]))),'')
    savedir = os.path.abspath(os.path.join(PWD, ''.join(["../../../data_and_figures/GuBuCv17/"])))
    dpi=1000

    '''Converged Tiles'''
    '''Defect'''
    defect_filepath = os.path.abspath(os.path.join(data_dir,"./tiles/defects/defect1/final_tile/rpo_L13p02_T15.h5"))
    defect_Orbit = Orbit_io.import_Orbit(defect_filepath)
    '''Gap'''
    gap_filepath = os.path.abspath(os.path.join(data_dir,"./tiles/gap/final_tile/anti_L17p5_T17.h5"))
    gap_Orbit = Orbit_io.import_Orbit(gap_filepath)
    '''Streak'''
    streak_filepath = os.path.abspath(os.path.join(data_dir,"./tiles/streak/final_tile/eqva_L3p195.h5"))
    streak_Orbit = Orbit_io.import_Orbit(streak_filepath)
    '''Hook'''
    hook_filepath = os.path.abspath(os.path.join(data_dir,"./tiles/hook/final_tile/rpo_L13p07_T10.h5"))
    hook_Orbit = Orbit_io.import_Orbit(hook_filepath)

    threetoone_import_filepath = os.path.abspath(os.path.join(data_dir,"./trawl/rpo/data/rpo_L21p97_T36p42.h5"))
    threetoone_Orbit = Orbit_io.import_Orbit(threetoone_import_filepath)

    # threetoone_import_filepath = os.path.abspath(os.path.join(data_dir,"./GuBuCv17/hookondefect.h5"))
    # threetoone_Orbit = Orbit_io.import_Orbit(threetoone_import_filepath)

    hookondefect_import_filepath =os.path.abspath(os.path.join(data_dir,"./trawl/ppo/data/ppo_L21p97_T73p52.h5"))
    hookondefect_initial_filename = os.path.abspath(os.path.join(savedir,"./MNG_hookondefect_initial.png"))
    hookondefect_filename = os.path.abspath(os.path.join(savedir,"./MNG_hookondefect_guess.png"))
    final_figname = os.path.abspath(os.path.join(savedir,"./figs/MNG_hookondefectf.png"))
    final_dataname = os.path.abspath(os.path.join(savedir,"./data/MNG_hookondefectf.h5"))
    Orbit = Orbit_io.import_Orbit(hookondefect_import_filepath)
    #rotate by 1pi, xnew[.8,3] t=[30,50]
    _,n,m,_,L,_ = Orbit
    rotatex = 1./(L/(2*pi))*L
    Tmin,Tmax,Xmin,Xmax = 30,50,0.6,3.0
    u,n,m,t,l,s = Orbit
    new_N,new_M=16*n,16*m
    Orbit=disc.rediscretize(Orbit,new_N=new_N,new_M=new_M)
    hookondefect_Orbit=sub.windowed_subdomain(Orbit,Tmin,Tmax,Xmin,Xmax,rotatex=rotatex)
    # ksplot.plot_spatiotemporal_field(Orbit,symmetry='rpo',display_flag=True)
    uw,nw,mw,tw,lw,sw = hookondefect_Orbit
    smalln,smallm = 4*int((nw/new_N)*n),2*int((mw/new_M)*m)
    Orbit=disc.rediscretize(hookondefect_Orbit,new_N=smalln,new_M=smallm)
    ksplot.plot_spatiotemporal_field(Orbit,display_flag=True,symmetry='none')

    uu,N,M,T,L,S = Orbit
    vv = fft(uu,axis=1)/np.sqrt(M)
    qk_vec = 1j*(2*pi*M/L)*np.fft.fftfreq(M)
    qk_vec[int(M//2)]=0
    qk_vec = np.reshape(qk_vec,[1,M])
    vvx = np.multiply(np.tile(qk_vec, (N,1)),vv)
    vvxx = np.multiply(np.tile((qk_vec**2), (N,1)),vv)
    vvxxxx = np.multiply(np.tile((qk_vec**4), (N,1)),vv)
    uux = np.sqrt(M)*np.real(ifft(vvx,axis=1))
    uuxx= np.sqrt(M)*np.real(ifft(vvxx,axis=1))
    uuxxxx = np.sqrt(M)*np.real(ifft(vvxxxx,axis=1))

    power = np.sum(np.sum(uux**2))
    dissipation=np.sum(np.sum(uuxx**2))
    energy = 0.5*np.sum(np.linalg.norm(np.reshape(vv,[N*M,1]))**2)
    print('HOD,      P,D,E',power,dissipation,energy)

    uu,N,M,T,L,S = defect_Orbit
    vv = fft(uu,axis=1)/np.sqrt(M)
    qk_vec = 1j*(2*pi*M/L)*np.fft.fftfreq(M)
    qk_vec[int(M//2)]=0
    qk_vec = np.reshape(qk_vec,[1,M])
    vvx = np.multiply(np.tile(qk_vec, (N,1)),vv)
    vvxx = np.multiply(np.tile((qk_vec**2), (N,1)),vv)
    vvxxxx = np.multiply(np.tile((qk_vec**4), (N,1)),vv)
    uux = np.sqrt(M)*np.real(ifft(vvx,axis=1))
    uuxx= np.sqrt(M)*np.real(ifft(vvxx,axis=1))
    uuxxxx = np.sqrt(M)*np.real(ifft(vvxxxx,axis=1))
    uut = -np.multiply(uu,uux)-uuxx-uuxxxx

    power = np.sum(np.sum(uux**2))
    dissipation=np.sum(np.sum(uuxx**2))
    energy = 0.5*np.sum(np.linalg.norm(np.reshape(vv,[N*M,1]))**2)
    print('P,D,E',power,dissipation,energy)


    uu,N,M,T,L,S = hook_Orbit
    vv = fft(uu,axis=1)/np.sqrt(M)
    qk_vec = 1j*(2*pi*M/L)*np.fft.fftfreq(M)
    qk_vec[int(M//2)]=0
    qk_vec = np.reshape(qk_vec,[1,M])
    vvx = np.multiply(np.tile(qk_vec, (N,1)),vv)
    vvxx = np.multiply(np.tile((qk_vec**2), (N,1)),vv)
    vvxxxx = np.multiply(np.tile((qk_vec**4), (N,1)),vv)
    uux = np.sqrt(M)*np.real(ifft(vvx,axis=1))
    uuxx= np.sqrt(M)*np.real(ifft(vvxx,axis=1))
    uuxxxx = np.sqrt(M)*np.real(ifft(vvxxxx,axis=1))
    uut = -np.multiply(uu,uux)-uuxx-uuxxxx

    power = np.sum(np.sum(uux**2))
    dissipation=np.sum(np.sum(uuxx**2))
    energy = 0.5*np.sum(np.linalg.norm(np.reshape(vv,[N*M,1]))**2)
    print('P,D,E',power,dissipation,energy)

    uu,N,M,T,L,S = gap_Orbit
    vv = fft(uu,axis=1)/np.sqrt(M)
    qk_vec = 1j*(2*pi*M/L)*np.fft.fftfreq(M)
    qk_vec[int(M//2)]=0
    qk_vec = np.reshape(qk_vec,[1,M])
    vvx = np.multiply(np.tile(qk_vec, (N,1)),vv)
    vvxx = np.multiply(np.tile((qk_vec**2), (N,1)),vv)
    vvxxxx = np.multiply(np.tile((qk_vec**4), (N,1)),vv)
    uux = np.sqrt(M)*np.real(ifft(vvx,axis=1))
    uuxx= np.sqrt(M)*np.real(ifft(vvxx,axis=1))
    uuxxxx = np.sqrt(M)*np.real(ifft(vvxxxx,axis=1))
    uut = -np.multiply(uu,uux)-uuxx-uuxxxx

    power = np.sum(np.sum(uux**2))
    dissipation=np.sum(np.sum(uuxx**2))
    energy = 0.5*np.sum(np.linalg.norm(np.reshape(vv,[N*M,1]))**2)
    print('P,D,E',power,dissipation,energy)

    uu,N,M,T,L,S = streak_Orbit
    vv = fft(uu,axis=1)/np.sqrt(M)
    qk_vec = 1j*(2*pi*M/L)*np.fft.fftfreq(M)
    qk_vec[int(M//2)]=0
    qk_vec = np.reshape(qk_vec,[1,M])
    vvx = np.multiply(np.tile(qk_vec, (N,1)),vv)
    vvxx = np.multiply(np.tile((qk_vec**2), (N,1)),vv)
    vvxxxx = np.multiply(np.tile((qk_vec**4), (N,1)),vv)
    uux = np.sqrt(M)*np.real(ifft(vvx,axis=1))
    uuxx= np.sqrt(M)*np.real(ifft(vvxx,axis=1))
    uuxxxx = np.sqrt(M)*np.real(ifft(vvxxxx,axis=1))
    uut = -np.multiply(uu,uux)-uuxx-uuxxxx

    power = np.sum(np.sum(uux**2))
    dissipation=np.sum(np.sum(uuxx**2))
    energy = 0.5*np.sum(np.linalg.norm(np.reshape(vv,[N*M,1]))**2)
    print('P,D,E',power,dissipation,energy)

    uu,N,M,T,L,S = threetoone_Orbit
    ksplot.plot_spatiotemporal_field(threetoone_Orbit,display_flag=True,symmetry='none')
    vv = fft(uu,axis=1)/np.sqrt(M)
    qk_vec = 1j*(2*pi*M/L)*np.fft.fftfreq(M)
    qk_vec[int(M//2)]=0
    qk_vec = np.reshape(qk_vec,[1,M])
    vvx = np.multiply(np.tile(qk_vec, (N,1)),vv)
    vvxx = np.multiply(np.tile((qk_vec**2), (N,1)),vv)
    vvxxxx = np.multiply(np.tile((qk_vec**4), (N,1)),vv)
    uux = np.sqrt(M)*np.real(ifft(vvx,axis=1))
    uuxx= np.sqrt(M)*np.real(ifft(vvxx,axis=1))
    uuxxxx = np.sqrt(M)*np.real(ifft(vvxxxx,axis=1))
    uut = -np.multiply(uu,uux)-uuxx-uuxxxx

    power = np.sum(np.sum(uux**2))
    dissipation=np.sum(np.sum(uuxx**2))
    energy = 0.5*np.sum(np.linalg.norm(np.reshape(vv,[N*M,1]))**2)
    print('P,D,E ppo',power,dissipation,energy)

    # for data_file in os.listdir(parent_folder):
    #     if data_file.endswith(".h5"):
    #         basename = data_file.split('.h5')[0]
    #         print(basename)
    #         Orbit = Orbit_io.import_Orbit(''.join([parent_folder,data_file]))
    #         # ksplot.plot_spatiotemporal_field(Orbit,display_flag=True,symmetry='rpo')
    #
    #         uu,N,M,T,L,S = Orbit
    #         # ksplot.plot_spatiotemporal_field(threetoone_Orbit,display_flag=True,symmetry='rpo')
    #         vv = fft(uu,axis=1)/np.sqrt(M)
    #         qk_vec = 1j*(2*pi*M/L)*np.fft.fftfreq(M)
    #         qk_vec[int(M//2)]=0
    #         qk_vec = np.reshape(qk_vec,[1,M])
    #         vvx = np.multiply(np.tile(qk_vec, (N,1)),vv)
    #         vvxx = np.multiply(np.tile((qk_vec**2), (N,1)),vv)
    #         vvxxxx = np.multiply(np.tile((qk_vec**4), (N,1)),vv)
    #         uux = np.sqrt(M)*np.real(ifft(vvx,axis=1))
    #         uuxx= np.sqrt(M)*np.real(ifft(vvxx,axis=1))
    #         uuxxxx = np.sqrt(M)*np.real(ifft(vvxxxx,axis=1))
    #         uut = -np.multiply(uu,uux)-uuxx-uuxxxx
    #
    #         power = np.sum(np.sum(uux**2))
    #         dissipation=np.sum(np.sum(uuxx**2))
    #         energy = 0.5*np.sum(np.linalg.norm(np.reshape(vv,[N*M,1]))**2)
    #         print('P,D,E ppo',power,dissipation,energy)

    return None


def eigenvectors():
    symmetry = 'ppo'
    PWD = os.path.dirname(__file__)
    data_dir = os.path.abspath(os.path.join(PWD, "../../../data_and_figures/"))
    data_dir= os.path.join(data_dir,'')
    # savedir = os.path.abspath(os.path.join(PWD, ''.join(["../../../../../.././GuBuCv17/"])))

    ppo_filepath = os.path.abspath(os.path.join(data_dir,"./trawl/ppo/data/ppo_L22p06_T91p40.h5"))
    # halfdefect2_initial_filename = os.path.abspath(os.path.join(savedir,"./halfdefect2_initial.png"))
    # field_minus_halfdefect2_filename = os.path.abspath(os.path.join(savedir,"./halfdefect2guess.png"))
    # halfdefect2_filename = os.path.abspath(os.path.join(savedir,"./halfdefect2guess_cutout.png"))
    Orbit = Orbit_io.import_Orbit(ppo_filepath)
    # Tmin,Tmax,Xmin,Xmax = 18,35,0.6,3
    state_vec,N,M = ks.tuple_to_statevec(Orbit,symmetry=symmetry)
    T,L,S = state_vec[-3:]
    eigvalues,eigvectors = eig(np.dot(np.transpose(ksdm.ppo.jacobian(state_vec,N,M,du_only=1)),ksdm.ppo.jacobian(state_vec,N,M,du_only=1)))
    print(np.size(eigvalues))
    counter = 0
    print(np.min(np.abs(eigvalues)))
    for vv in eigvectors:
        print(eigvalues[counter],counter)
        if np.abs(eigvalues[counter])>1:
            counter+=1
            pass
        else:
            test = np.reshape(ks.ppo.ifft_(np.imag(vv),N,M),[N,M]) + np.reshape(ks.ppo.ifft_(np.real(vv),N,M),[N,M])
            # uu = np.reshape(ks.ppo.ifft_(vv,N,M),[N,M])
            eigvOrbit = (test,N,M,T,L,S)
            ksplot.plot_spatiotemporal_field(eigvOrbit,symmetry=symmetry,display_flag=True)
            counter+=1
    return None


def reconverge_defect():
    PWD = os.path.dirname(__file__)
    data_dir = os.path.join(os.path.abspath(os.path.join(PWD, "../../data_and_figures/")),'')
    savedir = os.path.abspath(os.path.join(PWD, ''.join(["../../data_and_figures/GuBuCv17/"])))
    dpi=1000

    symmetry='none'
    defect1_filepath = os.path.abspath(os.path.join(data_dir,"./tiles/defects/defect1/final_tile/rpo_L13p02_T15.h5"))
    defect1_Orbit = Orbit_io.import_Orbit(defect1_filepath)
    Orbit = disc.rediscretize(defect1_Orbit,new_N=32,new_M=32)
    Orbit = symm.frame_rotation(Orbit)
    uu,n,m,t,l,s = Orbit
    # Orbit = (uu,n,m,t,l,0)
    # ksplot.plot_spatiotemporal_field(Orbit,symmetry='none',display_flag=True)

    Orbit_adj,retcode,res = ks.find_Orbit(Orbit,symmetry='none')
    Orbit_f,retcode,stats = ksdm.find_Orbit(Orbit_adj,symmetry='none')
    ksplot.plot_spatiotemporal_field(Orbit_f,symmetry='none',display_flag=True)

    return None


def main():
    # plot_cartoon_Orbit()
    # plot_plot_trawling_initial_to_final()
    # plot_tile_guesses_and_cutouts()
    # plot_converged_tiles()
    # plot_largeL()
    plot_frankenstein_tiling()
    # plot_ppo1ppo2ppo3_gluing()
    # converge_tiles()
    # continue_tiles()
    # quantized_family(gluetype=0)
    # slice_and_section()
    # random_figures()
    # convergence_rate()
    # tile_properties()
    # eigenvectors()
    # reconverge_defect()
    return None


if __name__=='__main__':
    main()