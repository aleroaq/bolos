r""" 
This scripts allows for the validation of the Boltzmann solver with electron-electron collisions turned on, 
and electron-ion collisions turned off.
"""

import numpy as np
import scipy.constants as co
from bolos import parser, solver, grid
import time

def main():
    t0 = time.time()
    # Use a linear grid from 0 to 60 eV with 500 intervals.
    gr = grid.LinearGrid(0, 100, 500)

    # Initiate the solver instance
    bsolver = solver.BoltzmannSolver(
        grid=gr,
        ee_collisions=True,
        kraphak_correction=True,
    )

    # Parse the cross-section file in BOSIG+ format and load it into the
    # solver.
    with open("O2_ISTLisbon_dataset.txt") as fp:
        bsolver.load_collisions(parser.parse(fp))

    # Set the conditions.  And initialize the solver
    T = 300.
    P = 100000
    ND = 2.14e25
    ionization_degree = 0.1
    ne = ionization_degree * ND
    bsolver.target['O2'].density = 1.0
    bsolver.kT = T * co.k / co.eV
    EN_vals = (np.linspace(0., 1., 100)**2 * 900 + 100) * solver.TOWNSEND
    Te_vals = np.zeros_like(EN_vals)
    mean_energy_vals = np.zeros_like(EN_vals)
    mobility_vals = np.zeros_like(EN_vals)
    diffusion_vals = np.zeros_like(EN_vals)

    bsolver.EN = EN_vals[0]
    bsolver.init()
    bsolver.update(T * co.k / co.eV, EN_vals[0], ne=ne, ionization_degree=ionization_degree)

    print("Initial conditions: Temperature = %.3f K, E/N = %.3e Td, N = %.3e m^-3" % (T, bsolver.EN / solver.TOWNSEND, ND))
    print()
    print()

    # Start with Maxwell EEDF as initial guess.  Here we are starting with
    # with an electron temperature of 2 eV
    for i, EN in enumerate(EN_vals):
        bsolver.EN = EN
        bsolver.update(T * co.k / co.eV, EN, ne=ne, ionization_degree=ionization_degree)
        f0 = bsolver.maxwell(2.0)

        # Solve the Boltzmann equation with a tolerance rtol and maxn iterations
        f1, iters, err  = bsolver.converge(f0, maxn=200, rtol=1e-5, full=True)
        Te_vals[i] = bsolver.electron_temperature(f1)
        mean_energy_vals[i] = bsolver.mean_energy(f1)
        mobility_vals[i] = bsolver.mobility(f1)
        diffusion_vals[i] = bsolver.diffusion(f1)

        # Calculate and print the properties
        print("E/N = %.3e Td" % (EN / solver.TOWNSEND))
        print("The final EEDF is: ", f1)
        print("Number of iterations: %d, final error: %g" % (iters, err))
        print("mobility * N  = %.3e  1/m/V/s" % (bsolver.mobility(f1)))
        print("diffusion * N = %.3e  1/m/s" % (bsolver.diffusion(f1)))
        print("average energy = %.3f  eV" % bsolver.mean_energy(f1))
        print("electron temperature = %.3f K" % bsolver.electron_temperature(f1))
        print()

    t1 = time.time()
    print(f"Time taken: {t1 - t0:.2e} seconds")

if __name__ == '__main__':
    main()
