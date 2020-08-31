import os,sys
sys.path.insert(0, os.path.abspath(os.path.join(sys.argv[0], '../..')))
import torihunter as th

def main(*args, method='hybrid', **kwargs):


    merger__final = merger_result.Orbit
    # Orbit = th.Orbit(state=Orbit.state, statetype='modes', L=Orbit.L, T=Orbit.T, S=Orbit.S)
    # Orbit.plot()
    # Orbit.convert(to='modes')
    # test = Orbit.convert(to='field')
    # x = th.read_h5(filename)
    # x.statetype='modes'
    # x = th.Orbit(state=x.state, statetype='modes', L=x.L, T=x.T, S=x.S)
    # x = rediscretize(x, parameter_based=True)
    result = th.converge(orbit, verbose=True)

    if result.success:
        result.Orbit.to_h5()
        result.Orbit.plot(show=True)


    return None


if __name__=='__main__':
    sys.exit(main())
