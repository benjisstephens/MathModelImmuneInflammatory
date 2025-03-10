# Mathematical Modelling of Immune and Inflammatory Responses
This repository contains the code developed for my MMath Mathematics dissertation on mathematical biology, specifically modeling immune and inflammatory responses. 

# Project Overview
The project analyses a data set supplied by Imperial College London's Bioengineering department of dendritic cell trajectories in a chemokine gradient.
Next we produce PDE models (three-variable PDE and a five-variable PDE) to model cell density and flux.
Then we produce a hybrid model (stochastic cell movement, continuous chemokine concentration) as a different way to calculate cell density and flux.
We conclude by fitting the data to the model by parameter estimation.

## Code Structure
### Code to display cell trajectory data:
- 'cell_trajectory_data.py'
- 'kde.py'
- 'visualising_density.py'
- 'visualising_flux.py'
### Code for PDE models
- 'three_pde_solve.py'
- 'three_pde_plot.py'
- 'five_pde_solve.py'
- 'five_pde_plot.py'
- 'pde_parameter_fitting.py'
### Code for hybrid model
- 'hybrid_model.py'
- 'hybrid_model_fitting.py'

## Dependencies
This project requires the following Python libraries:
- `numpy`
- `scipy`
- `matplotlib`
- `pandas`

**Author:** Benjamin Stephens  
**Dissertation Submission Date:** April 11, 2025  
**University:** University of Nottingham, School of Mathematics 
