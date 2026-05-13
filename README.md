# XRaySimulation 

**2022-12-26 branch**:

Fork of `haoyuanli93/XRaySimulation`, based on the frozen 2022-12-26 snapshot.


**platform branch**(default):

My modified version, used as the simulation backbone in `selene-bshe/XRayOptics` via git subtree.

## Coordinate Convention
This library uses **x/y axes** swapped relative to SLAC standards: $x$ = vertical, $y$ = horizontal (SLAC is the opposite).

## Upstream

**Original repo**: https://github.com/haoyuanli93/XRaySimulation

**Original author**: Haoyuan Li, PhD (SLAC / Stanford)

## Module Overview
 
| File | Purpose |
|------|---------|
| `Crystal.py` | Crystal structure, Bragg angle, susceptibility χ₀/χₕ/χ₋ₕ |
| `Optics.py` | Optical elements: monochromator, mirror, slit, Kohzu motor |
| `Pulse.py` | X-ray pulse / wavefront (time & frequency domain) |
| `PropagationFunctions.py` | Free-space propagation via Angular Spectrum Method |
| `MultiDevice.py` | End-to-end beamline simulation — unoptimised, CPU only |
| `Geometry.py` | Rotation matrices, vector projection, coordinate transforms |
| `Util.py` | Unit conversions, physical constants, HDF5 I/O |
| `GPU/` | CuPy-accelerated kernels for large-scale computation |
 
---


