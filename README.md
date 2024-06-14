# Announce of abandon of this repo:
I am a part-time and amateur developer of hard X-ray simulation software.
As time goes by, I realize that this repo is becoming so complicated that I cannot maintain it anymore.
Therefore, I decide to create a newer version and keep everything in this repo frozen. 

# Warning:
My x-y-z coordinate convention is different from the SLAC convention!

In my simulation, the z axis is the beam propagation direction.

The y axis is the horizontal transverse direction.
This the the x axis in SLAC convension.

The x axis is the vertical transverse direction.
This is the y axis in SLAC convension.

# XRaySimulation
This is my new simulation repo.

The reason that I am abandoning the previous repo is that
the previous one is too complicated and contains too much 
information and too many functions.

I need to simplify the repo and make it more maintainable.

In this new repo, no effort has been made to include Laue 
and forward Bragg diffraction.
Later in the future, I may include those case,
however.

# Essential To-Do list
1. Add unit test module
2. Release a stable version.
3. Add functions to calculate quantities with prism
4. The one thing that I am most unsatisfied about this simulation 
    is that different functions 
    are not in most appropriate position.
    This makes it not easy to start a new simulation.
   I want to re-arrange the functions to make the location of each function 
    more reasonable. 

# Not so essential to-do list
1. Change the auto-alignment behavior
2. Change the crystal property grab behavior.

# Some thoughts
I am currently using a lot of Kohzu motors in the simulation.
However, I am not sure if I want to implement sub classes of kohzu motor models 
or not.
The good thing is that I can shorten the code.
The bad thing is that I need to create a lot of layers of code.

The most important feature of this repo is that I believe it is more readable 
than those well-developed repos.
I will choose the one that can perserve this.

# Notice
### Optimization
In this repo, not all the functions are optimized.
Functions in the GPU folder are optimized to some extend.
Functions in the MultiDevice.py file are not optimized.

### Fresnal diffraction
Previously, I have explicitly considered the Fresnel diffraction. 
However, I just realized that I do not need such things because 
in my simulation, the propagation of each monochromatic components
follow the more universal Maxwell equations. 

Therefore, in this version, I remove the Fresnel diffraction functions.

Later, I might add a function to handle a pure propagation directly. 

