Future Changes
==============

- Re-do shadowing.py and how orbit scores are handled.
  Currently "orbit scores", pivot scores mapped back onto the original tile
- Write up the tutorial notebooks
- Finish writing advanced_tests.py
- Put on a Docker container

Ideas and Experiments
=====================

** Resizing OrbitKS type Orbits via scikit-image **

Currently, OrbitKS and its subclasses are resized by truncation or zero padding of its Fourier modes,
the idea being to disturb the current spectrum as little as possible but it is worth exploring
whether its field can be resized by using image analysis tools from scikit-learn. Of course
this is made easy by the KSE having 2-d scalar field solutions. Applying image resizing techniques
is good for plotting but not necessarily for numerical reasons. Not including for now because it would
require changing the requirements to include scikit-image as well which seems unreasonable this size of
an addition.
