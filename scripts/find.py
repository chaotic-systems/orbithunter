import os,sys
sys.path.insert(0, os.path.abspath(os.path.join(sys.argv[0], '../..')))
import torihunter as th

def main(*args, method='hybrid', **kwargs):


    merger_torus_final = merger_result.torus
    # torus = th.Torus(state=torus.state, statetype='modes', L=torus.L, T=torus.T, S=torus.S)
    # torus.plot()
    # torus.convert(to='modes')
    # test = torus.convert(to='field')
    # x = th.read_h5(filename)
    # x.statetype='modes'
    # x = th.Torus(state=x.state, statetype='modes', L=x.L, T=x.T, S=x.S)
    # x = rediscretize(x, parameter_based=True)
    result = th.converge(orbit, verbose=True)

    if result.success:
        result.torus.to_h5()
        result.torus.plot(show=True)


    return None


if __name__=='__main__':
    sys.exit(main())
