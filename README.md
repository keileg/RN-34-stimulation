# RN-34 stimulation
This repository contains runscripts for simulation of a hydraulic stimulation operation at the Reykjanes geothermal field in Iceland. 
Specifically, the simulations are based on stimulation of the well RN-34, which took place at March 29, 2015.

## Prerequisites
The runscripts uses the open source simulation software [PorePy](https://github.com/pmgbergen/porepy).
The simulations should be run with PorePy v1.2.0, which can be accessed either on the PorePy GitHub page, or on [doi:10.5281/zenodo.3972034](https://zenodo.org/record/3972034#.XzpDyegzbtQ).
Instructions on how to install PorePy can be found at the PorePy GitHub page.

## How to simulate the stimulation event
This repository contains two main files:
  * Simulation_master.py is the main runscript. The different steps in the simulation are documented at the top and bottom of this file
  * models.py contains the specification of the actual simulation setup, including data, specification of numerics etc. This is a rather large file, some documantation of the content can be found at the top of the module.
