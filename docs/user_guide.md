# RelativeOrbitKS

Similar to ```OrbitKS``` except for the inclusion of continuous spatial translation symmetry.
This adds another component to the linear term of the equation when handled via a comoving
reference frame. 

Important properties to note:
1. The spatial shift ```'S'``` is ALWAYS stored as the spatial shift FROM comoving TO physical reference frames.
2. The only method that should be used to change reference frames is ```change_reference_frame(to=[frame_type,])```
3. There are a very small number of operations allowed for state's in the physical reference frame.
4. The field state should always be (and is assumed to be) saved in the COMOVING frame.

The reason behind #3 above is because if in the physical reference frame, then the accuracy of the temporal
methods such as differentiation lose their accuracy. For nearly every operation the state NEEDS to be int
the comoving reference frame. The exceptions to this rule are ```change_reference_frame()```, and the forward and backward
spatiotemporal (Fourier) transforms via the ```convert(to=[basis], inplace=True)``` method. Note the transforms
MUST be performed inplace. 