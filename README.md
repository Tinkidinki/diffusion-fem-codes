This repository contains code to reproduce the results in Section 9 of the paper "Quantum Algorithms for Heterogeneous PDEs: The Neutron Diffusion Eigenvalue Problem" (arxiv link: https://arxiv.org/abs/2604.05098). 

See paper for full description. Briefly, we aim to test the convergence of a simple finite element element method for an elliptic problem with varying coefficients. 

To run the code:
First run the cells in `triangles_paper_code.ipynb`. This computes the data for various diffusion coefficients and stores the dictionaries to disk. 

Next run the cells in `triangles_paper_plots.ipynb`. This reads from the stored dictionaries and reproduces the plots in the paper. 

Contact Mahathi Vempati (mahathi@umd.edu) for any questions. 
