orbithunter 1.3.3
==================

.. currentmodule:: orbithunter

Minor Changes
-------------
- Remove trim from inside map; require explicit execution elsewhere
- Remove verbose flag and replace with tqdm in places
- Change default value for remove_hull_only to False in `OrbitCovering.trim`

Patches
-------
- Was double averaging in physics calcs; now takes sum and divides by extensive parameters (space, time, etc.)

Notes
-----
- Currently trim is going to only function on pivots space array; may change in future to allow for
  mapped arrays to be trimmed.

