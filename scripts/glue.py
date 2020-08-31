import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(sys.argv[0], '../..')))
import torihunter as th
from argparse import ArgumentParser,ArgumentTypeError, FileType, ArgumentDefaultsHelpFormatter

def str2bool(val):
    if isinstance(val, bool):
       return val
    if val.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif val.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise ArgumentTypeError('Boolean value expected.')

def main():
    parser = ArgumentParser('glue', formatter_class=ArgumentDefaultsHelpFormatter)

    parser.add_argument('--direction', default='space', help='"space" or "time", decides on gluing direction')

    example0 =  os.path.abspath(os.path.join(sys.argv[0],'../../examples/glue_example0.h5'))
    example1 =  os.path.abspath(os.path.join(sys.argv[0],'../../examples/glue_example1.h5'))
    examplenames = ' '.join([example0, example1])
    parser.add_argument('--input', default=examplenames, nargs=2, help='two *.h5 filenames separated by whitespace')

    parser.add_argument('--output', default=None, help='*.h5 filename to save result to')

    parser.add_argument('--figure', type=str2bool, default=False)


    args = parser.parse_args()
    # try:
    first_Orbit_filename, second_Orbit_filename = args.input.split()
    Orbit = th.read_h5(first_Orbit_filename, cls=th.ShiftReflectionOrbit)
    other_Orbit = th.read_h5(second_Orbit_filename, cls=th.ShiftReflectionOrbit)
    glued_Orbit = th.glue(Orbit, other_Orbit, direction=args.direction)
    result = th.converge(glued_Orbit)

    if result.success:
        result.Orbit.to_h5(filename=args.output)
        if args.figure:
            result.Orbit.plot(filename=args.output)
    #     except AttributeError:
    #         pass
    # except NameError:
    #     print('Invalid input, needs to be the names of two *.h5 files separated by whitespace')
    #     pass


if __name__=="__main__":
    sys.exit(main())




