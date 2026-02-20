# Mitigation of Coning and Motion Blur Effects in Star Trackers

This is repository contain the source code for the Master's thesis, _Mitigation of Coning and Motion Blur Effects in Star Trackers_ at DTU.

The thesis was conducted by Filip Penzar, under supervision of Associate Professor Mathias Benn, at the [_Measurement and Instrumentation Systems_](https://www.space.dtu.dk/english/research-divisions/measurement-and-instrumentation-systems), _Department of Space Research and Technology_.

The thesis is available [here](TODO link).

## Installation

To install, it is best to create a [venv](https://docs.python.org/3/library/venv.html) and then run the following:

```
pip install -r requirements.txt
```

## Source code

### Mathematical Representations of Spacecraft Attitude
* Code used in Section 2 and Appendix A, can be found in [quaternion](src/quaternion.py).

### Attitude Kinematics
* _Coning Motion Representation_ is in [gyro_simulator](src/gyro_simulator.py).
* _Attitude Propagation Methods_ are in [gyro_integrator](src/gyro_integrator.py).

### Star Tracker Image Modeling and Deblurring
* _Motion Models_ are implemented in [degradation](src/degradation.py).
* _Image Denoising_ is implemented in [denoiser](src/denoiser.py).
* _Wiener Deconvolution_ is implemented in [wiener](src/wiener.py).
* _Richardson-Lucy_ is implemented in [rl](src/rl.py).
* _Centroid Calculation_ is implementd in [centroids](src/centroids.py).

## Notes
There is no single entry point to the program, as it mostly consists of standalone scripts used throughout the project. The image deblurring was run from [deblurring](src/deblurring.py), while the coning compensation from [experiments](src/experiments.py).

The data used in the project is not publicly available.
