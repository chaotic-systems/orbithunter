from orbithunter.discretization import rediscretize, correct_aspect_ratios
from orbithunter.io import read_h5
import numpy as np
import os
import itertools

__all__ = ['combine', 'glue']


def best_combination(orbit, other_orbit, fundamental_domain_combinations, axis=0):

    half_combinations = list(itertools.product(half_list,repeat=2))
    residual_list = []
    glued_list = []
    for halves in half_combinations:
        orbit_domain = orbit.to_fundamental_domain(half=halves[0])
        other_orbit_domain = other_orbit.to_fundamental_domain(half=halves[1])
        glued = concat(orbit_domain, other_orbit_domain, direction=direction)
        glued_orbit = glued.from_fundamental_domain()
        glued_list.extend([glued_orbit])
        residual_list.extend([glued_orbit.residual])
    best_combi = np.array(glued_list)[np.argmin(residual_list)]
    return best_combi


def best_rotation(orbit, other_orbit, axis=0):
    field_orbit, field_other_orbit = correct_aspect_ratios(orbit, other_orbit, axis=axis)

    resmat = np.zeros([field_other_orbit.N, field_other_orbit.M])
    # The orbit only remains a converged solution if the rotations occur in
    # increments of the discretization, i.e. multiples of L / M and T / N.
    # The reason for this is because those are the only values that do not
    # actually change the field via interpolation. In other words,
    # The rotations must coincide with the collocation points.
    # for n in range(0, field_other_orbit.N):
    #     for m in range(0, field_other_orbit.M):
    #         rotated_state = np.roll(np.roll(field_other_orbit.state, m, axis=1), n, axis=0)
    #         rotated_orbit = other_orbit.__class__(state=rotated_state, state_type=field_other_orbit.state_type, T=other_orbit.T,
    #                                               L=other_orbit.L, S=other_orbit.S)
    #         resmat[n,m] = concat(field_orbit, rotated_orbit, direction=direction).residual


    bestn, bestm = np.unravel_index(np.argmin(resmat), resmat.shape)
    high_resolution_orbit = rediscretize(field_orbit, new_N=16*field_orbit.N, new_M=16*field_orbit.M)
    high_resolution_other_orbit = rediscretize(field_other_orbit, new_N=16*field_other_orbit.N, new_M=16*field_other_orbit.M)

    best_rotation_state = np.roll(np.roll(high_resolution_other_orbit.state, 16*bestm, axis=1), 16*bestn, axis=0)
    highres_rotation_orbit = other_orbit.__class__(state=best_rotation_state, state_type='field',
                                                   T=other_orbit.T, L=other_orbit.L, S=other_orbit.S)
    best_gluing = concat(high_resolution_orbit, highres_rotation_orbit, direction=direction)
    return best_gluing


def combine(orbit, other_orbit, axis=0):
    # Converts tori to best representatives for gluing by choosing from group orbit.
    continuous_classes = ['Orbit', 'RelativeOrbit']
    discrete_classes = ['ShiftReflectionOrbit', 'AntisymmetricOrbit']
    orbit_name = orbit.__class__.__name__
    other_name = other_orbit.__class__.__name__
    if (orbit_name in continuous_classes) and (other_name in continuous_classes):
        return best_rotation(orbit, other_orbit, direction=direction)
    elif (orbit_name in discrete_classes) and (other_name in discrete_classes):
        return best_combination(orbit, other_orbit, direction=direction)
    else:
        return concat(orbit, other_orbit, direction=direction)


def tile_dictionary_ks():
    directory = os.path.join(os.path.abspath(os.path.join(os.getcwd(), '../data/')), '')
    # merger Orbit
    merger = read_h5(os.path.abspath(os.path.join(directory, "./merger.h5")))

    # Wiggle Orbit
    streak = read_h5(os.path.abspath(os.path.join(directory, "./streak.h5")))

    # Streak Orbit
    wiggle = read_h5(os.path.abspath(os.path.join(directory, "./wiggle.h5")))

    tile_library = {'0': streak, '1': merger, '2': wiggle}

    return tile_library


def tile(symbol_array, tile_dict=tile_dictionary_ks):

    tile_list = [tile_dict[symbol] for symbol in symbol_array.ravel()]
    tile_array = np.reshape(tile_list, symbol_array.shape)

    '''
    Currently: spatial aspect ratio predetermined.
    need correct aspect ratios between columns AND ROWS
    This prompts us to make an aspect ratio function
    Really simplified version: No preprocessing, only sticking together
    with matching discretizations
    for aspect ratio to work i think im going to need the zero padding method'''

    blockL=0
    blockT=0
    blockS=0
    for time_index in range(0, period):
        Ttmp=0
        Ltmp=0
        Stmp=0
        symbol_array_row = list(symbol_array[time_index,:])
        row_streak_count = symbol_array_row.count('0')
        for space_index in range(0, speriod):
            if not space_index:
                block_field_row_temp = tile_array[time_index,space_index,0]
                if row_streak_count!=speriod:
                    Ttmp+=tile_array[time_index,space_index,-3]/(speriod-row_streak_count)
                else:
                    Ttmp += 0

                Ltmp+=tile_array[time_index,space_index,-2]
                Stmp+=tile_array[time_index,space_index,-1]
            else:
                block_field_row_temp = np.hstack([block_field_row_temp,tile_array[time_index,space_index,0]])
                if row_streak_count==speriod:
                    Ttmp += 0
                else:
                    Ttmp+=tile_array[time_index,space_index,-3]/(speriod-row_streak_count)
                Ltmp+=tile_array[time_index,space_index,-2]
                Stmp+=tile_array[time_index,space_index,-1]
        Ntmp,Mtmp = np.shape(block_field_row_temp)
        row_orbit = (block_field_row_temp,Ntmp,Mtmp,Ttmp,Ltmp,Stmp)

        if not time_index:
            block_orbit = row_orbit
            blockU,blockN,blockM,blockT,blockL,blockS=block_orbit

            block_orbit = (blockU,blockN,blockM,blockT,blockL,blockS)
        else:
            blockT = block_orbit[-3]+Ttmp
            blockL = (block_orbit[-2]+Ltmp)/2.
            blockS = (block_orbit[-1]+Stmp)
            blockN = block_orbit[1]+Ntmp
            blockM = 2*int((block_orbit[2]+Mtmp)/4.)
            redisc_block_orbit = rediscretize(block_orbit,new_M=blockM)
            redisc_row_orbit = rediscretize(row_orbit,new_M=blockM)
            block_orbit = (np.vstack([redisc_block_orbit[0],redisc_row_orbit[0]]),blockN,blockM,blockT,blockL,blockS)

    return block_orbit


def glue(orbit, other_orbit, axis=0, **kwargs):
    """ Function for combining spatiotemporal fields

    Parameters
    ----------
    orbit
    other_orbit
    axis
    kwargs

    Returns
    -------

    """
    newfield = np.concatenate((orbit.convert(to='field').state,
                               other_orbit.convert(to='field').state), axis=axis)

    param_zip_enum = enumerate(zip(orbit.parameters, other_orbit.parameters))
    new_parameters = [np.mean([p1, p2]) if (i != axis) else np.sum([p1, p2]) for i, (p1, p2) in param_zip_enum]

    glued_orbit = orbit.__class__(state=newfield, state_type='field', parameters=new_parameters, **kwargs)
    glued_orbit = rediscretize(glued_orbit, parameter_based=True, **kwargs)
    return glued_orbit



