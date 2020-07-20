from __future__ import print_function, division, absolute_import
import os,sys
sys.path.insert(0, os.path.abspath(os.path.join(sys.argv[0], '../..')))
from torihunter import *
import numpy as np


def main(*args,**kwargs):
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
    PWD = os.path.dirname(__file__)
    save_directory_extension = kwargs.get("save_directory",''.join(["../../../data_and_figures/GuBuCv17/"]))
    save_directory = os.path.join(os.path.abspath(os.path.join(PWD,save_directory_extension)),'')
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
        torus = (uu,N0,M0,T,L0,0)
        torus_io.export_torus(torus,'C:\\Users\\matth\\Desktop\\jupyter\\MNG_uu_largeL.h5',symmetry='none')
    elif init == 'random':
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
        torus = (uu,N0,M0,T,L0,0)
        torus_io.export_torus(torus,'MNG_uu_largeLRandom.h5',symmetry='none')
    elif init =='import':
        torus_imported = torus_io.import_torus("MNG_uu_largeL.h5")
        uu,N0,M0,T,L,S0 = torus_imported
        vv = fft(uu,axis=1)/np.sqrt(M0)

    # N,M = np.shape(uu)
    # qk_vec = 1j*(2*pi*M0/L0)*np.fft.fftfreq(M)
    # qk_vec[int(M//2)]=0
    # qk_vec = np.reshape(qk_vec,[1,M])
    # vvx = np.multiply(np.tile(qk_vec, (N,1)),vv)
    # vvxx = np.multiply(np.tile((qk_vec**2), (N,1)),vv)
    # vvxxxx = np.multiply(np.tile((qk_vec**4), (N,1)),vv)
    # uux = np.sqrt(M)*np.real(ifft(vvx,axis=1))
    # uuxx= np.sqrt(M)*np.real(ifft(vvxx,axis=1))
    # uuxxxx = np.sqrt(M)*np.real(ifft(vvxxxx,axis=1))
    # uut = -np.multiply(uu,uux)-uuxx-uuxxxx
    #
    # average_power = np.sum(np.abs(vvxx),axis=0)
    # average_dissipation = np.sum(np.abs(vvxx),axis=0)
    print(uu.shape)
    u_tuple = (uu,N0,M0,T,L0,0)
    # ux_tuple = (uux,N,M,T,L0,0)
    # ut_tuple = (uut,N,M,T,L0,0)

    # power_tuple = (np.log10(uux**2),N,M,T,L0,0)
    # dissipation_tuple = (np.log10(uuxx**2),N,M,T,L0,0)
    # E_tuple = (np.log10(np.abs(uux**2-uuxx**2)),N,M,T,L0,0)
    # left = int(M//2)
    # right = 2*int(M//3)
    # top = 2*int(N//3)
    # bottom = int(N//2)

    final_path_names = fh.create_data_infrastructure('integration','large')


    # u_tuple = (uu,N,M,T,L0,0)
    fig, ax = plt.subplots()
    ax.imshow(uu,cmap='jet')
    plt.tight_layout()
    ax.set_axis_off()
    ax.set_aspect(aspect=M0/N0)
    plt.savefig('C:\\Users\\matth\\Desktop\\jupyter\\cleanlargeRandom.png', bbox_inches='tight',pad_inches=0,aspect=1)
    plt.show()

    # ux_tuple = (uux,N,M,T,L0,0)
    # ut_tuple = (uut,N,M,T,L0,0)
    # # u_tuple_subdomain =  (np.log10(np.abs(uu[-top:-bottom,left:right])),
    # #                     int(np.abs(-bottom+top)),int(np.abs(right-left)),T*np.abs(top-bottom)/N,L0*np.abs(right-left)/M,0)
    # # ux_tuple_subdomain = (np.log10(np.abs(uux[-top:-bottom,left:right])),
    # #                          int(np.abs(-bottom+top)),int(np.abs(right-left)),T*np.abs(top-bottom)/N,L0*np.abs(right-left)/M,0)
    # # ut_tuple_subdomain = (np.log10(np.abs(uut[-top:-bottom,left:right])),
    # #                          int(np.abs(-bottom+top)),int(np.abs(right-left)),T*np.abs(top-bottom)/N,L0*np.abs(right-left)/M,0)
    # u_tuple_subdomain =  (uu[-top:-bottom,left:right],
    #                     int(np.abs(-bottom+top)),int(np.abs(right-left)),T*np.abs(top-bottom)/N,L0*np.abs(right-left)/M,0)
    # ux_tuple_subdomain = (uux[-top:-bottom,left:right],
    #                          int(np.abs(-bottom+top)),int(np.abs(right-left)),T*np.abs(top-bottom)/N,L0*np.abs(right-left)/M,0)
    # ut_tuple_subdomain = (uut[-top:-bottom,left:right],
    #                          int(np.abs(-bottom+top)),int(np.abs(right-left)),T*np.abs(top-bottom)/N,L0*np.abs(right-left)/M,0)
    # originleft = L0*left/M
    # # print(originleft/(2*pi))
    # originbottom = T*bottom/N
    # power_subdomain =  (np.log10(0.5*uux[-top:-bottom,left:right]**2),
    #                     int(np.abs(-bottom+top)),int(np.abs(right-left)),T*np.abs(top-bottom)/N,L0*np.abs(right-left)/M,0)
    # dissipation_subdomain = (np.log10(0.5*uuxx[-top:-bottom,left:right]**2),
    #                          int(np.abs(-bottom+top)),int(np.abs(right-left)),T*np.abs(top-bottom)/N,L0*np.abs(right-left)/M,0)
    # E_subdomain = (np.log10(np.abs(0.5*uux[-top:-bottom,left:right]**2-0.5*uuxx[-top:-bottom,left:right]**2)),
    #                          int(np.abs(-bottom+top)),int(np.abs(right-left)),T*np.abs(top-bottom)/N,L0*np.abs(right-left)/M,0)
    # # other_subdomain =  (0.5*uuxxxx[-top:-bottom,left:right]**2,
    # #                          int(np.abs(-bottom+top)),int(np.abs(right-left)),T*np.abs(top-bottom)/N,L0*np.abs(right-left)/M,0)
    # display_flag = False

    # uname =''.join([save_directory,"MNG_u_largeL2000.png"])
    # uxname=''.join([save_directory,"MNG_ux_largeL2000.png"])
    # utname=''.join([save_directory,"MNG_ut_largeL2000.png"])
    # usubname=''.join([save_directory,"MNG_u_largeLsub2000.png"])
    # uxsubname=''.join([save_directory,"MNG_ux_largeLsub2000.png"])
    # utsubname=''.join([save_directory,"MNG_ut_largeLsub2000.png"])
    # Ename=''.join([save_directory,"MNG_E_largeL2000.png"])
    # Dname=''.join([save_directory,"MNG_D_largeL2000.png"])
    # Pname=''.join([save_directory,"MNG_P_largeL2000.png"])
    # Esubname=''.join([save_directory,"MNG_E_largeLsub2000.png"])
    # Dsubname=''.join([save_directory,"MNG_D_largeLsub2000.png"])
    # Psubname=''.join([save_directory,"MNG_P_largeLsub2000.png"])

    # ksplot.plot_spatiotemporal_field(u_tuple,symmetry='none',padding=False,display_flag=display_flag,filename=uname)
    # ksplot.plot_spatiotemporal_field(ux_tuple,symmetry='none',padding=False,display_flag=display_flag,filename=uxname)
    # ksplot.plot_spatiotemporal_field(ut_tuple,symmetry='none',padding=False,display_flag=display_flag,filename=utname)
    # ksplot.plot_spatiotemporal_field(u_tuple_subdomain,originbottom=originbottom,originleft=originleft,symmetry='none',padding=False,display_flag=display_flag,filename=usubname)
    # ksplot.plot_spatiotemporal_field(ux_tuple_subdomain,originbottom=originbottom,originleft=originleft,symmetry='none',padding=False,display_flag=display_flag,filename=uxsubname)
    # ksplot.plot_spatiotemporal_field(ut_tuple_subdomain,originbottom=originbottom,originleft=originleft,symmetry='none',padding=False,display_flag=display_flag,filename=utsubname)
    # ksplot.plot_spatiotemporal_field(E_tuple,nonnegative=True,symmetry='none',padding=False,display_flag=display_flag,filename=Ename)
    # ksplot.plot_spatiotemporal_field(power_tuple,nonnegative=True,symmetry='none',padding=False,display_flag=display_flag,filename=Pname)
    # ksplot.plot_spatiotemporal_field(dissipation_tuple,nonnegative=True,symmetry='none',padding=False,display_flag=display_flag,filename=Dname)
    # ksplot.plot_spatiotemporal_field(power_subdomain,originbottom=originbottom,originleft=originleft,nonnegative=True,symmetry='none',padding=False,display_flag=display_flag,filename=Psubname)
    # ksplot.plot_spatiotemporal_field(dissipation_subdomain,originbottom=originbottom,originleft=originleft,nonnegative=True,symmetry='none',padding=False,display_flag=display_flag,filename=Dsubname)
    # ksplot.plot_spatiotemporal_field(E_subdomain,originbottom=originbottom,originleft=originleft,nonnegative=True,symmetry='none',padding=False,display_flag=display_flag,filename=Esubname)


    # '''Assign None value to points in array that are neither minima nor maxima so that density plot (imshow) doesn't
    # color them'''
    # indicestmp = np.where(minmaxtrack_array == 0)
    # minmaxtrack_array[indicestmp] = None
    # #
    # # '''Plot and save minmax_tracking'''
    # minmaxfig = plt.figure()
    # plt.imshow(minmaxtrack_array,extent=[0,L0,0,T-transient_T],interpolation='none',cmap=plt.get_cmap('jet'))
    # # plt.savefig(save_filename_minmax,bbox_inches='tight',dpi=500)
    # plt.show()
    # plt.rc('text', usetex=True)
    # plt.rc('font', family='serif')

    N,M = np.shape(uu)
    S = 0
    field_tuple = (uu,N,M,T,L0,S)
    # field_tuple = rediscretize(field_tuple,newN=2**int(np.log2(N)-2))
    # uu_field,N,M,T,L,S = field_tuple
    # Ne,Me = np.shape(uu)




    return None





if __name__=='__main__':
    main(init='random',integration_time=2000+491,transient_time=2000,domain_size=495.02,spatial_discretization=2048,step_size=0.1)
