"""This module contains the main routines to load processes, specify the
physical conditions and solve the Boltzmann equation.

The data and calculations are encapsulated into the :class:`BoltzmannSolver`
class, which you have to instantiate with a :class:`grid.Grid` instance.
Use :func:`BoltzmannSolver.load_collisions` or
:func:`BoltzmannSolver.add_process` to add processes with
their cross-sections.  Afterwards, set the density of each component
with :func:`BoltzmannSolver.set_density` or :attr:`BoltzmannSolver.target`.
The method :func:`BoltzmannSolver.maxwell` gives you a reasonable initial guess
for the electron energy distribution function (EEDF) that you can then improve
iteratively with :func:`BoltzmannSolver.converge`.  Finally, methods such as
:func:`BoltzmannSolver.rate` or :func:`BoltzmannSolver.mobility` allow you
to obtain reaction rates and transport parameters for a given EEDF.

"""

__docformat__ = "restructuredtext en"

import logging

from math import sqrt
import numpy as np
import numpy.typing as npt
from numba import njit, prange

# Units in this module will be SI units, except energies, which are expressed
# in eV.
# The scipy.constants contains the recommended CODATA for all physical
# constants in SI units.
import scipy.constants as co
from scipy import sparse
from scipy.integrate import simpson
from scipy.sparse.linalg import spsolve
from scipy.interpolate import interp1d

from bolos.process import Process
from bolos.target import Target
from bolos.grid import Grid

from typing import Iterator, Tuple

GAMMA = sqrt(2 * co.elementary_charge / co.electron_mass)
TOWNSEND = 1e-21
KB = co.k
ELECTRONVOLT = co.eV
EEDF_MIN = 1e-30


@njit(
    inline="always",
    cache=True,
)
def numba_simpson(
    y: npt.NDArray[np.float64],
    x: npt.NDArray[np.float64],
) -> float:
    """Integrates y(x) using Simpson's rule.

    Parameters
    ----------
    y : 1D-array of floats
        The y values of the function to integrate, evaluated at the x values.
    x : 1D-array of floats
        The x values of the function to integrate.

    Returns
    -------
    float
        The integral of y(x) over the range of x.
    """
    n = len(x)
    if n < 3:
        # Fallback to trapezoidal rule if there aren't enough points
        return 0.5 * (x[1] - x[0]) * (y[0] + y[1])

    integral = 0.0

    # If the number of intervals is odd (even number of points),
    # handle the last interval separately using the trapezoidal rule.
    is_even = n % 2 == 0
    end_idx = n - 1 if not is_even else n - 2

    # Simpson's 1/3 rule loop
    for i in range(1, end_idx, 2):
        h0 = x[i] - x[i - 1]
        h1 = x[i + 1] - x[i]

        h_sum = h0 + h1
        alpha = (2 * h1**3 - h0**3 + 3 * h0 * h1**2) / (6 * h1 * h_sum)
        beta = (h_sum**3) / (6 * h0 * h1)
        gamma = (2 * h0**3 - h1**3 + 3 * h1 * h0**2) / (6 * h0 * h_sum)

        integral += alpha * y[i + 1] + beta * y[i] + gamma * y[i - 1]

    # Add the trapezoidal rule for the last interval if points were even
    if is_even:
        integral += 0.5 * (x[-1] - x[-2]) * (y[-1] + y[-2])

    return integral


@njit(
    parallel=True,
    cache=True,
)
def _ee_collision_term_numba(
    benergy: npt.NDArray[np.float64],
    bf0_: npt.NDArray[np.float64],
) -> Tuple[npt.NDArray[np.float64], npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    r"""Computes the A1, A2 and A3 integrals for the electron-electron collision term in the Boltzmann equation.

    Parameters
    ----------
    benergy : 1D-array of floats
        The cell boundary energies of the grid in eV.
    bf0_ : 1D-array of floats
        The EEDF evaluated at the cell centers of the grid, and evaluated at the cell boundaries by interpolation at the midpoint between the centers.

    Returns
    -------
    1D-array of floats
        The A1 integral evaluated at each cell boundary energy.
    1D-array of floats
        The A2 integral evaluated at each cell boundary energy.
    1D-array of floats
        The A3 integral evaluated at each cell boundary energy.
    """
    n = len(benergy) - 1
    A1 = np.zeros(n + 1)
    A2 = np.zeros(n + 1)
    A3 = np.zeros(n + 1)
    for i in prange(n + 1):
        A1[i] = numba_simpson(
            np.sqrt(benergy[: i + 1]) * bf0_[: i + 1], x=benergy[: i + 1]
        )
        A2[i] = numba_simpson(
            benergy[: i + 1] ** 1.5 * bf0_[: i + 1], x=benergy[: i + 1]
        )
        A3[i] = numba_simpson(bf0_[i:], x=benergy[i:])
    return A1, A2, A3


class ConvergenceError(Exception):
    pass


class BoltzmannSolver(object):
    """Class to solve the Boltzmann equation for electrons in a gas.

    This class contains the required elements to specify the conditions
    for the solver and obtain the equilibrium electron energy distribution
    function.

    Parameters
    ----------
    grid : :class:`grid.Grid`
        The grid in energies where the distribution funcition will be
        evaluated.

    Attributes
    ----------
    benergy : array of floats
        Cell boundaries of the energy grid (set automatically at \
        initialization). Equivalent to `grid.b`.
    cenergy : array of floats
        Cell centers of the energy grid (set automatically at initialization). \
        Equivalent to `grid.c`.
    denergy : array of floats
        Cell boundaries of the energy grid (set automatically at \
        initialization). Equivalent to `grid.d`.
    denergy32 : array of floats
        Cell boundaries of the energy grid raised to the 3/2 power (set automatically at \
        initialization). Equivalent to `grid.d32`.
    sigma_eps : array of floats
        Comment
    sigma_m : array of floats
        Comment
    sigma_eps_no_coulomb : array of floats
        Comment
    sigma_m_no_coulomb : array of floats
        Comment
    n : int
        Number of cells in the energy grid (set automatically at \
        initialization). Equivalent to `grid.n`.
    kT : float
        Gas temperature in eV.  Must be set by the user.
    EN : float
        Reduced electric field in Townsend (1 Td is 1e-21 V m^2). \
        Must be set by the user.
    target : dict[str, :class:`target.Target`]
        A dictionary with targets in the set of processes.\
        The user needs to set the density (molar fraction) of the desired \
        targets using this dictionary.  E.g. synthetic air is represented by
        ``solver.target['N2'].density = 0.8`` and \
        ``solver.target['O2'].density = 0.2``.
    density : dict[str, float]
        A dictionary with the densities of each target species in m^-3.  This is set automatically when the user sets the density of each target in the ``target`` dictionary.
    ee_collisions : bool, optional
        If True, the solver will take into account electron-electron collisions [1]_. \
        By default, this is False.  Note that this is a very expensive calculation and should be used with caution.
    ei_collisions : bool, optional
        If True, the solver will take into account electron-ion collisions [1]_. \
        By default, this is False.  Note that this is a very expensive calculation and should be used with caution.
    super : bool, optional
        If True, the solver will create super-elastic processes for each inelastic process [2]_. \
        By default, this is True.
    kraphak_correction : bool, optional
        If True, the solver will apply the Kraphak correction to the Coulomb logarithm [3]_. \
        By default, this is False.

    Examples
    --------
    >>> import numpy as np
    >>> from bolos import solver, grid
    >>> grid.LinearGrid(0, 60., 400)
    >>> bsolver = solver.BoltzmannSolver(grid)
    >>> # Parse the cross-section file in BOSIG+ format and load it into the
    >>> # solver.
    >>> with open(args.input) as fp:
    >>>     processes = parser.parse(fp)
    >>> bsolver.load_collisions(processes)
    >>>
    >>> # Set the conditions.  And initialize the solver
    >>> bsolver.target['N2'].density = 0.8
    >>> bsolver.target['O2'].density = 0.2
    >>> bsolver.kT = 300 * co.k / co.eV
    >>> bsolver.EN = 300.0 * solver.TOWNSEND
    >>> bsolver.init()
    >>>
    >>> # Start with Maxwell EEDF as initial guess.  Here we are starting with
    >>> # with an electron temperature of 2 eV
    >>> f0 = bsolver.maxwell(2.0)
    >>>
    >>> # Solve the Boltzmann equation with a tolerance rtol and maxn
    >>> # iterations.
    >>> f1 = bsolver.converge(f0, maxn=50, rtol=1e-5)

    References
    ----------
    .. [1] Hagelaar, G. J. M., & Pitchford, L. C. (2005). Solving the Boltzmann equation to obtain electron transport coefficients and rate coefficients. Plasma Sources Science and Technology, 14(4), 722–733. https://doi.org/10.1088/0963-0252/14/4/011
    .. [2] Hagelaar, G. J. M. Brief Documentation of BOLSIG+ Version 07/2024. n.d.
    .. [3] Khrapak, Sergey A. “Effective Coulomb Logarithm for One Component Plasma.” Physics of Plasmas 20, no. 5 (2013): 054501. https://doi.org/10.1063/1.4804341.
    """

    ee_collisions: bool
    ei_collisions: bool
    kraphak_correction: bool
    super: bool
    density: dict[str, float]
    target: dict[str, Target]
    EN: float
    FN: float
    kT: float
    grid: Grid
    n: int
    benergy: npt.NDArray[np.float64]
    cenergy: npt.NDArray[np.float64]
    denergy: npt.NDArray[np.float64]
    denergy32: npt.NDArray[np.float64]
    sigma_eps: npt.NDArray[np.float64] = None
    sigma_m: npt.NDArray[np.float64] = None
    sigma_eps_no_coulomb: npt.NDArray[np.float64]
    sigma_m_no_coulomb: npt.NDArray[np.float64]
    electron_density: float = None
    ionization_degree: float = None

    def __init__(
        self,
        grid: Grid,
        ee_collisions: bool = False,
        ei_collisions: bool = False,
        superelastic: bool = True,
        kraphak_correction: bool = True,
    ) -> None:
        """ Initialize a solver instance.

        Use this method to initialize a solver instance with a given grid.

        Parameters
        ----------
        grid : :class:`grid.Grid`
            The grid in energies where the distribution funcition will be
            evaluated.
        ee_collisions : bool, optional
            If True, the solver will take into account electron-electron collisions [1]_. \
            By default, this is False.  Note that this is a very expensive calculation and should be used with caution.
        ei_collisions : bool, optional
            If True, the solver will take into account electron-ion collisions [1]_. \
            By default, this is False.  Note that this is a very expensive calculation and should be used with caution.
        superelastic : bool, optional
            If True, the solver will create super-elastic processes for each inelastic process [2]_. \
            By default, this is True.
        kraphak_correction : bool, optional
            If True, the solver will apply the Kraphak correction to the Coulomb logarithm [3]_. \
            By default, this is True.
        
        References
        ----------
        .. [1] Hagelaar, G. J. M., & Pitchford, L. C. (2005). Solving the Boltzmann equation to obtain electron transport coefficients and rate coefficients. Plasma Sources Science and Technology, 14(4), 722–733. https://doi.org/10.1088/0963-0252/14/4/011
        .. [2] Hagelaar, G. J. M. Brief Documentation of BOLSIG+ Version 07/2024. n.d.
        .. [3] Khrapak, Sergey A. “Effective Coulomb Logarithm for One Component Plasma.” Physics of Plasmas 20, no. 5 (2013): 054501. https://doi.org/10.1063/1.4804341.
        """

        self.density = dict()

        self.EN = None

        self.FN = 0.0

        self.grid = grid

        # Default coulomb (with e) cross sections not taken into account
        self.ee_collisions = ee_collisions

        # Default coulomb (with i) cross sections not taken into account
        self.ei_collisions = ei_collisions

        # Activation or not of the super elastic cross sections
        self.superelastic = superelastic

        # Add correction for Coulomb logarithm using Kraphak formula
        self.kraphak_correction = kraphak_correction

        # A dictionary with target_name -> target
        self.target = {}

    def _get_grid(self) -> Grid:
        return self._grid

    def _set_grid(
        self,
        grid: Grid,
    ) -> None:
        self._grid = grid

        # These are cell boundary values at i - 1/2
        self.benergy = self.grid.b

        # these are cell centers
        self.cenergy = self.grid.c

        # And these are the deltas
        self.denergy = self.grid.d

        # This is useful when integrating the growth term.
        self.denergy32 = self.benergy[1:] ** 1.5 - self.benergy[:-1] ** 1.5

        self.n = grid.n

    grid = property(_get_grid, _set_grid)

    def set_density(
        self,
        species: str,
        density: float,
    ) -> None:
        """Sets the molar fraction of a species.

        Parameters
        ----------
        species : str
           The species whose density you want to set.
        density : float
           New value of the density.

        Returns
        -------

        Examples
        --------
        These are two equivalent ways to set densities for synthetic air:

        Using :func:`set_density`::

            bsolver.set_density('N2', 0.8)
            bsolver.set_density('O2', 0.2)

        Using `bsolver.target`::

            bsolver.target['N2'].density = 0.8
            bsolver.target['O2'].density = 0.2
        """

        self.target[species].density = density

    @staticmethod
    def super_cs(
        cs: npt.NDArray[np.float64],
        thres: float,
        weight_ratio: float,
        kind: str,
    ) -> npt.NDArray[np.float64]:
        """Computes the super-elastic collision cross section as given by the Klein-Rossland formula

        Parameters
        ----------
        cs : 1D-array of floats
            The cross section of the forward process with two columns:
            - column 0 must contain energies in eV,
            - column 1 contains the cross-section in square meters for each of these energies.
        thres : float
            The energy threshold of the forward process in eV.
        weight_ratio : float
            The ratio of the statistical weights of the upper and lower levels of the transition.
        kind : str
            The kind of the forward process.  It must be 'EXCITATION'.

        Returns
        -------
        1D-array of floats
            The super elastic cross section with the same format as the input cross section.

        Raises
        ------
        ValueError
            If the kind of the forward process is not 'EXCITATION'.
        """
        nb = sum(e - thres > 0 for e in cs[:, 0])
        cs_inv = np.zeros((nb, 2))
        index = 0

        if kind == "EXCITATION":
            for i in range(len(cs[:, 0])):
                if cs[i, 0] - thres > 0:
                    cs_inv[index, 0] = cs[i, 0] - thres
                    cs_inv[index, 1] = (
                        cs[i, 1] * (1 / weight_ratio) * cs[i, 0] / (cs[i, 0] - thres)
                    )
                    index += 1
        else:
            raise ValueError(
                "Super-elastic cross sections can only be computed for EXCITATION reactions"
            )

        return cs_inv

    def create_super(
        self,
        p: dict,
        weight_ratio: float,
    ) -> dict:
        r"""Creates a super-elastic process from a given inelastic process dictionary.

        Parameters
        ----------
        p : dict
            The inelastic process from which to create the super-elastic process.
        weight_ratio : float
            The ratio of the statistical weights of the upper and lower levels of the transition.

        Returns
        -------
        dict
            The super-elastic process created from the given inelastic process.
        """
        p_super = {}
        p_super["target"] = p["product"]
        p_super["product"] = p["target"]
        p_super["kind"] = p["kind"]
        p_super["threshold"] = -p["threshold"]
        cs = np.array(p["data"])
        p_super["data"] = self.super_cs(cs, p["threshold"], weight_ratio, p["kind"])
        return p_super

    def load_collisions(
        self,
        dict_processes: list[dict],
    ) -> list[Process]:
        """Loads the set of collisions from the list of processes.

        Loads a list of dictionaries containing processes.

        Parameters
        ----------
        dict_processes : List of dictionary or dictionary-like elements.
           The processes to add to this solver class.
           See :method:`solver.add_process` for the required fields
           of each of the dictionaries.

        Returns
        -------
        processes : list[Process]
           A list of all added processes, as :class:`process.Process` instances.

        See Also
        --------
        add_process : Add a single process, with its cross-sections, to this
           solver.

        """
        plist = [self.add_process(**p) for p in dict_processes]

        for p in dict_processes:
            if p["kind"] == "EXCITATION" and self.superelastic:
                try:
                    weight_ratio = p["weight_ratio"]
                except KeyError:
                    weight_ratio = 1.0
                p_super = self.create_super(p, weight_ratio)
                plist.append(self.add_process(**p_super))

        # We make sure that all targets have their elastic cross-sections
        # in the form of ELASTIC cross sections (not EFFECTIVE / MOMENTUM)
        for key, item in self.target.items():
            item.ensure_elastic()

        return plist

    def add_process(self, **kwargs) -> Process:
        """Adds a new process to the solver.

        Adds a new process to the solver.  The process data is passed with
        keyword arguments.

        Parameters
        ----------
        type : string
           one of "EFFECTIVE", "MOMENTUM", "EXCITATION", "IONIZATION"
           or "ATTACHMENT".
        target : string
           the target species of the process (e.g. "O", "O2"...).
        ratio : float
           the ratio of the electron mass to the mass of the target
           (for elastic/momentum reactions only).
        threshold : float
           the energy threshold of the process in eV (only for
           inelastic reactions).
        data : array or array-like
           cross-section of the process array with two columns: column
           0 must contain energies in eV, column 1 contains the
           cross-section in square meters for each of these energies.

        Returns
        -------
        process : :class:`process.Process`
           The process that has been added.

        Examples
        --------
        >>> import numpy as np
        >>> from bolos import solver, grid
        >>> grid.LinearGrid(0, 60., 400)
        >>> solver = BoltzmannSolver(grid)
        >>> # This is an example cross-section that decays exponentially
        >>> energy = np.linspace(0, 10)
        >>> cross_section = 1e-20 * np.exp(-energy)
        >>> solver.add_process(type="EXCITATION", target="Kriptonite",
        >>>                    ratio=1e-5, threshold=10,
        >>>                    data=np.c_[energy, cross_section])

        See Also
        --------
        load_collisions : Add a set of collisions.

        """
        proc = Process(**kwargs)
        try:
            target = self.target[proc.target_name]
        except KeyError:
            target = Target(proc.target_name)
            self.target[proc.target_name] = target

        target.add_process(proc)

        return proc

    def search(
        self,
        signature: str,
        product: str = None,
        first: bool = True,
    ) -> Process | list[Process]:
        """Search for a process or a number of processes within the solver.

        Parameters
        ----------
        signature : string
           Signature of the process to search for.  It must be in the form
           "TARGET -> RESULT [+ RESULT2]...".
        product : string
           If present, the first parameter is interpreted as TARGET and the
           second parameter is the PRODUCT.
        first : boolean
           If true returns only the first process matching the search; if
           false returns a list of them, even if there is only one result.

        Returns
        -------
        processes : list or :class:`process.Process` instance.
           If ``first`` was true, returns the first process matching the
           search.  Otherwise returns a (possibly empty) list of matches.

        Examples
        --------
        >>> ionization = solver.search("N2 -> N2^+")[0]
        >>> ionization = solver.search("N2", "N2^+", first=True)

        """
        if product is not None:
            l = self.target[signature].by_product[product]
            if not l:
                raise KeyError("Process %s not found" % signature)

            return l[0] if first else l

        t, p = [x.strip() for x in signature.split("->")]
        return self.search(t, p, first=first)

    def iter_elastic(self) -> Iterator[Tuple[Target, Process]]:
        """Iterates over all elastic processes.

        Parameters
        ----------

        Returns
        -------
        An iterator over (target, process) tuples.
        """

        for target in self.target.values():
            if target.density > 0:
                for process in target.elastic:
                    yield target, process

    def iter_inelastic(self) -> Iterator[Tuple[Target, Process]]:
        """Iterates over all inelastic processes.

        Parameters
        ----------

        Returns
        -------
        An iterator over (target, process) tuples."""

        for target in self.target.values():
            if target.density > 0:
                for process in target.inelastic:
                    yield target, process

    def iter_growth(self) -> Iterator[Tuple[Target, Process]]:
        """Iterates over all processes that affect the growth
        of electron density, i.e. ionization and attachment.

        Parameters
        ----------

        Returns
        -------
        An iterator over (target, process) tuples.

        """
        for target in self.target.values():
            if target.density > 0:
                for process in target.ionization:
                    yield target, process

                for process in target.attachment:
                    yield target, process

    def iter_all(self) -> Iterator[Tuple[Target, Process]]:
        """Iterates over all processes.

        Parameters
        ----------

        Returns
        -------
        An iterator over (target, process) tuples.

        """
        for t, k in self.iter_elastic():
            yield t, k

        for t, k in self.iter_inelastic():
            yield t, k

    def iter_momentum(self) -> Iterator[Tuple[Target, Process]]:
        return self.iter_all()

    def iter_coulomb(
        self,
    ) -> Iterator[Tuple[Target, Process]]:
        """Iterates over all Coulomb processes.

        Parameters
        ----------

        Returns
        -------
        An iterator over (target, process) tuples.

        """
        for target in self.target.values():
            if target.density > 0:
                for process in target.coulomb:
                    yield target, process

    def init(self) -> None:
        """Initializes the solver with given conditions and densities of the
        target species.

        This method does all the work previous to the actual iterations.
        It has to be called whenever the densities, the gas temperature
        or the electric field are changed.

        Notes
        -----
        The most expensive calculations in this method are cached so they are
        not repeated in each call.  Therefore the execution time may vary
        wildly in different calls.  It takes very long whenever you change
        the solver's grid; therefore is is strongly recommended not to
        change the grid if is not strictly neccesary.

        """

        # self.sigma_eps_no_coulomb = np.zeros_like(self.benergy)
        # self.sigma_m_no_coulomb = np.zeros_like(self.benergy)
        # self.sigma_eps = np.empty_like(self.benergy)
        # self.sigma_m = np.empty_like(self.benergy)
        self.sigma_eps = np.zeros_like(self.benergy)
        self.sigma_m = np.zeros_like(self.benergy)

        for target, process in self.iter_elastic():
            s = target.density * process.interp(self.benergy)
            # self.sigma_eps_no_coulomb += 2 * target.mass_ratio * s
            # self.sigma_m_no_coulomb += s
            self.sigma_eps += 2 * target.mass_ratio * s
            self.sigma_m += s
            process.set_grid_cache(self.grid)

        for target, process in self.iter_inelastic():
            self.sigma_m += target.density * process.interp(self.benergy)
            process.set_grid_cache(self.grid)

        logging.info("Solver succesfully initialized/updated")

    def update(
        self,
        kT: float,
        EN: float,
        ne: float = None,
        ionization_degree: float = None,
    ) -> None:
        r"""Updates the solver with new values of the gas temperature and the
        reduced electric field.

        This method is a shortcut to update the solver with new values of the
        gas temperature and the reduced electric field.  It calls :func:`init`
        internally, so it is not recommended to use it if you are changing
        other parameters such as densities or the grid.

        Parameters
        ----------
        kT : float
           New value of the gas temperature in eV.
        EN : float
           New value of the reduced electric field in V m^2.
        ne : float, optional
            New value of the electron density in m^-3.  If not provided, the previous value is kept.
        ionization_degree : float, optional
            New value of the ionization degree (dimensionless).  If not provided, the previous value is kept.

        Updates
        -------
        `BoltzmannSolver.kT`, `BoltzmannSolver.EN`
        """
        self.kT = kT
        self.EN = EN
        if ne is not None:
            self.electron_density = ne
        if ionization_degree is not None:
            self.ionization_degree = ionization_degree
        if (self.electron_density is not None and self.ionization_degree is None) or (
            self.electron_density is None and self.ionization_degree is not None
        ):
            logging.warning(
                "Electron density or ionization degree not set. "
                "Coulomb logarithm will not be computed."
            )
        return

    def _correct_eedf(
        self,
        f0: npt.NDArray[np.float64],
    ) -> npt.NDArray[np.float64]:
        """Corrects the EEDF to satisfy the continuity equation.

        Parameters
        ----------
        f0 : 1D-array of floats
           The EEDF to correct.

        Returns
        -------
        1D-array of floats
           The corrected EEDF.

        Details
        -------
        This method will:
        1. Replace all non-positive values of the EEDF with a small random value.
        2. Identify regions of the EEDF that are constant and replace them with a linear interpolation.
        3. Normalize the EEDF to ensure that it integrates to 1.
        """
        f1 = np.copy(f0)
        min_F1 = np.min(f1[f1 > 0.0]) if np.any(f1 > 0.0) else EEDF_MIN
        f1[f1 <= 0.0] = min_F1 * 0.1 * np.random.rand(len(f1[f1 <= 0.0]))
        coefs = np.random.random(len(f1[f1 <= EEDF_MIN]))
        coefs.sort()
        f1[f1 <= EEDF_MIN] = EEDF_MIN * (1 - 0.1 * coefs)

        const_regions = []
        start_idx = 0
        for i in range(1, len(f1)):
            if f1[i] != f1[start_idx]:
                if i - start_idx > 1:
                    const_regions.append((start_idx, i - 1))
                start_idx = i

        if len(f1) - start_idx > 1:
            const_regions.append((start_idx, len(f1) - 1))

        for start, end in const_regions:

            if start == 0:
                f1[: end + 2] = np.linspace(f1[0], f1[end + 1], end + 2)
            elif end == len(f1) - 1:
                f1[start - 1 :] = np.linspace(
                    f1[start - 1], f1[-1], len(f1) - start + 1
                )
            else:
                f1[start : end + 1] = interp1d(
                    x=[start - 1, (end + start) // 2, end + 1],
                    y=[f1[start - 1], f1[start], f1[end + 1]],
                    kind="linear",
                )(np.arange(start, end + 1))

        f2 = self._normalized(f1)
        Fp = np.r_[f2[0], f2, f2[-1]]
        idx_problem = np.where(Fp[2:] == Fp[:-2])[0]
        if len(idx_problem) > 0:
            random_values = np.random.random(len(idx_problem))
            for i in range(len(idx_problem)):
                f2[idx_problem[i] + 1] *= 1 + 0.1 * random_values[i]

        return f2

    def _getCoulombLogarithm(
        self,
        f0: npt.NDArray[np.float64],
    ) -> Tuple[float, float]:
        """Computes the Coulomb logarithm for electron-electron and electron-ion collisions in the first order approximation,
        as detailed in Eqs. (8.2) - (8.8) of [1]_.

        This function returns :math:`\ln \Lambda_{ee}` and :math:`\ln \Lambda_{ei}` for electron-electron and electron-ion collisions, respectively.

        Parameters
        ----------
        f0 : 1D-array of floats
           The EEDF to compute the electron-electron collision term for.

        Returns
        -------
        float
            The Coulomb logarithm for electron-electron collisions.
        float
            The Coulomb logarithm for electron-ion collisions.

        References
        ----------
        .. [1] Mitchner, Morton, and Charles H. Kruger. Partially Ionized Gases. Wiley Series in Plasma Physics. Wiley, 1973.
        """
        bf0_ = np.r_[f0[0], 0.5 * (f0[1:] + f0[:-1]), f0[-1]]
        kTe = 2.0 / 3.0 * simpson(self.benergy**1.5 * bf0_, self.benergy) * co.e  # J
        kTe = max(300.0 * co.k, kTe)
        kTe_ei = max(kTe, self.kT)
        coulomb_param_ee = (
            12.0
            * np.pi
            * (co.epsilon_0 * kTe) ** 1.5
            / co.e**3
            / np.sqrt(self.electron_density)
        )
        coulomb_param_ei = (
            12.0
            * np.pi
            * (co.epsilon_0 * kTe_ei) ** 1.5
            / co.e**3
            / np.sqrt(self.electron_density)
        )
        return np.log(coulomb_param_ee), np.log(coulomb_param_ei)

    def _getCoulombLogarithmKraphak(
        self,
        f0: npt.NDArray[np.float64],
    ) -> Tuple[float, float]:
        """Computes the Coulomb logarithm for electron-electron and electron-ion collisions.
        We use the Kraphak correction formula` [1]_ to compute the Coulomb logarithm, where one should note that the original
        formula is in CGS units.

        Parameters
        ----------
        f0 : 1D-array of floats
           The EEDF to compute the electron-electron collision term for.

        Returns
        -------
        float
            The Coulomb logarithm for electron-electron collisions.
        float
            The Coulomb logarithm for electron-ion collisions.

        References
        ----------
        .. [1] Khrapak, Sergey A. “Effective Coulomb Logarithm for One Component Plasma.” Physics of Plasmas 20, no. 5 (2013): 054501. https://doi.org/10.1063/1.4804341.
        """
        bf0_ = np.r_[f0[0], 0.5 * (f0[1:] + f0[:-1]), f0[-1]]
        kTe = 2.0 / 3.0 * simpson(self.benergy**1.5 * bf0_, self.benergy) * co.e  # J
        kTe = max(300.0 * co.k, kTe)  # J
        kTe_ei = max(kTe, self.kT * co.e)  # J
        gammas = [
            co.e**2
            / co.epsilon_0
            / kTe
            * np.power(4.0 / 3.0 * np.pi * self.electron_density, 1.0 / 3.0),
            co.e**2
            / co.epsilon_0
            / kTe_ei
            * np.power(4.0 / 3.0 * np.pi * self.electron_density, 1.0 / 3.0),
        ]
        coulomb_params = [
            0.5
            * np.log(
                1
                + np.power(
                    np.power(1 + np.power(3 * gamma, 1.5), 1.0 / 3.0) - 1.0,
                    -2,
                )
            )
            for gamma in gammas
        ]
        return coulomb_params[0], coulomb_params[1]

    def _ee_collisions_(
        self,
        coulomb_param: float,
        f0: npt.NDArray[np.float64],
    ) -> None:
        """Computes the electron-electron collision term.


        Parameters
        ----------
        coulomb_param : float
            The Coulomb logarithm for electron-electron collisions.
        f0 : 1D-array of floats
           The EEDF to compute the electron-electron collision term for.

        Updates
        -------
        self.WC : 1D-array of floats
           The collision term updated with the electron-electron collision term.
        self.DC : 1D-array of floats
           The diffusion term updated with the electron-electron diffusion term.

        References
        ----------
        .. [1] Khrapak, Sergey A. “Effective Coulomb Logarithm for One Component Plasma.” Physics of Plasmas 20, no. 5 (2013): 054501. https://doi.org/10.1063/1.4804341.
        """
        bf0_ = np.r_[f0[0], 0.5 * (f0[1:] + f0[:-1]), f0[-1]]
        a = (
            co.elementary_charge**2
            * GAMMA
            / 24.0
            / np.pi
            / co.epsilon_0**2
            * coulomb_param
        )
        # A_tmp = np.sqrt(self.benergy) * bf0_
        # A1 = np.array([simpson(A_tmp[:i+1], self.benergy[:i+1]) for i in range(self.n + 1)])
        # A_tmp = self.benergy**1.5 * bf0_
        # A2 = np.array([simpson(A_tmp[:i+1], self.benergy[:i+1]) for i in range(self.n + 1)])
        # A3 = np.array([simpson(bf0_[i:], x=self.benergy[i:]) for i in range(self.n + 1)])

        A1, A2, A3 = _ee_collision_term_numba(self.benergy, bf0_)

        self.WC = -3.0 * a * self.ionization_degree * A1
        self.DC = 2.0 * a * self.ionization_degree * (A2 + self.benergy**1.5 * A3)

    def _ei_collisions_(
        self,
        coulomb_param: float,
    ) -> None:
        r"""Update the mean cross-sections with electron-ion coulomb collisions.

        Updates
        -------
        self.sigma_eps : 1D-array of floats
           The mean energy loss cross-section updated with the electron-ion coulomb collision term.
        self.sigma_m : 1D-array of floats
            The mean momentum transfer cross-section updated with the electron-ion coulomb collision term.
        """
        sigma_eps = np.copy(self.sigma_eps_no_coulomb)
        sigma_m = np.copy(self.sigma_m_no_coulomb)
        for target, process in self.iter_coulomb():
            # s_ = 1e-4 * 2.87e-14 * process.squared_charge * self.ln_coulomb_ei / self.benergy**2
            s = (
                target.density
                * 4.0
                / 9.0
                / np.pi
                * co.e**4
                / self.benergy**2
                * process.squared_charge
                * coulomb_param
            )
            sigma_eps += 2 * target.mass_ratio * s
            sigma_m += s
        self.sigma_eps = sigma_eps
        self.sigma_m = sigma_m

    ##
    # Here are the functions that depend on F0 and are therefore
    # called in each iteration.  These are all pure-functions without
    # side-effects and without changing the state of self
    def maxwell(
        self,
        kT: float,
    ) -> npt.NDArray[np.float64]:
        """Calculates a Maxwell-Boltzmann distribution function.

        Parameters
        ----------
        kT : float
           The electron temperature in eV.

        Returns
        -------
        f : array of floats
           A normalized Boltzmann-Maxwell EEDF with the given temperature.

        Notes
        -----
        This is often useful to give a starting value for the EEDF.
        """

        return 2 * np.sqrt(1 / np.pi) * kT ** (-3.0 / 2.0) * np.exp(-self.cenergy / kT)

    def iterate(
        self,
        f0: npt.NDArray[np.float64],
        delta: float = 1e14,
    ) -> npt.NDArray[np.float64]:
        """Iterates once the EEDF.

        Parameters
        ----------
        f0 : array of floats
           The previous EEDF
        delta : float
           The convergence parameter.  Generally a larger delta leads to faster
           convergence but a too large value may lead to instabilities or
           slower convergence.

        Returns
        -------
        f1 : array of floats
           A new value of the distribution function.

        Notes
        -----
        This is a low-level routine not intended for normal uses.  The
        standard entry point for the iterative solution of the EEDF is
        the :func:`BoltzmannSolver.converge` method.
        """

        A, Q = self._linsystem(f0)

        f1 = spsolve(
            sparse.eye(self.n) + delta * A - delta * Q,
            f0,
        )

        return self._normalized(f1)

    def converge(
        self,
        f0: npt.NDArray[np.float64],
        maxn: int = 100,
        rtol: float = 1e-5,
        delta0: float = 1e14,
        m: float = 4.0,
        full: bool = False,
        **kwargs,
    ) -> (
        npt.NDArray[np.float64]
        | tuple[npt.NDArray[np.float64], int, float]
        | ConvergenceError
    ):
        """Iterates and attempted EEDF until convergence is reached.

        Parameters
        ----------
        f0 : array of floats
           Initial EEDF.
        maxn : int
           Maximum number of iteration until the convergence is declared as
           failed (default: 100).
        rtol : float
           Target tolerance for the convergence.  The iteration is stopped
           when the difference between EEDFs is smaller than rtol in L1
           norm (default: 1e-5).
        delta0 : float
           Initial value of the iteration parameter.  This parameter
           is adapted in succesive iterations to improve convergence.
           (default: 1e14)
        m : float
           Attempted reduction in the error for each iteration.  The Richardson
           extrapolation attempts to reduce the error by a factor m in each
           iteration.  Larger m means faster convergence but also possible
           instabilities and non-decreasing errors. (default: 4)
        full : boolean
           If true returns convergence information besides the EEDF.

        Returns
        -------
        f1 : array of floats
           Final EEDF
        iters : int (returned only if ``full`` is True)
           Number of iterations required to reach convergence.
        err : float (returned only if ``full`` is True)
           Final error estimation of the EEDF (must me smaller than ``rtol``).

        Notes
        -----
        If convergence is not achieved after ``maxn`` iterations, an exception
        of type ``ConvergenceError`` is raised.
        """

        err0 = err1 = 0
        delta = delta0

        for i in range(maxn):
            # If we have already two error estimations we use Richardson
            # extrapolation to obtain a new delta and speed up convergence.
            if 0 < err1 < err0:
                # Linear extrapolation
                # delta = delta * err1 / (err0 - err1)

                # Log extrapolation attempting to reduce the error a factor m
                delta = delta * np.log(m) / (np.log(err0) - np.log(err1))

            f1 = self.iterate(f0, delta=delta, **kwargs)
            f1 = self._correct_eedf(f1)
            err0 = err1
            err1 = self._norm(abs(f0 - f1))

            logging.debug(
                "After iteration %3d, err = %g (target: %g)" % (i + 1, err1, rtol)
            )
            if err1 < rtol:
                logging.info(
                    "Convergence achieved after %d iterations. "
                    "err = %g" % (i + 1, err1)
                )
                if full:
                    return f1, i + 1, err1

                return f1
            f0 = f1

        logging.error("Convergence failed")

        raise ConvergenceError()

    def _linsystem(
        self,
        F: npt.NDArray[np.float64],
    ) -> tuple[sparse.dia_matrix, sparse.csr_matrix]:
        Q = self._PQ(F)

        # Useful for debugging but wasteful in normal times.
        # if np.any(np.isnan(Q.todense())):
        #     raise ValueError("NaN found in Q")

        nu = np.sum(Q.dot(F))

        if self.ee_collisions or self.ei_collisions:
            if self.electron_density is None or self.ionization_degree is None:
                raise ValueError(
                    "Electron density and ionization degree must be set for Coulomb collisions."
                )
            if self.kraphak_correction:
                ln_ee, ln_ei = self._getCoulombLogarithmKraphak(F)
            else:
                ln_ee, ln_ei = self._getCoulombLogarithm(F)

            if self.ee_collisions:
                self._ee_collisions_(ln_ee, F)
            if self.ei_collisions:
                self._ei_collisions_(ln_ei)
            # else:
            #     np.copyto(self.sigma_eps, self.sigma_eps_no_coulomb)
            #     np.copyto(self.sigma_m, self.sigma_m_no_coulomb)

        sigma_tilde = self.sigma_m + nu / np.sqrt(self.benergy) / GAMMA

        self.W = -GAMMA * self.benergy**2 * self.sigma_eps

        # This is the coeff of sigma_tilde
        self.DA = GAMMA / 3.0 * self.EN**2 * self.benergy

        # This is the independent term
        self.DB = GAMMA * self.kT * self.benergy**2 * self.sigma_eps

        # The R (G) term, which we add to A.
        G = 2 * self.denergy32 * nu / 3

        A = self._scharf_gummel(sigma_tilde, G)

        # if np.any(np.isnan(A.todense())):
        #     raise ValueError("NaN found in A")

        return A, Q

    def _norm(self, f: npt.NDArray[np.float64]) -> float:
        return simpson(f * np.sqrt(self.cenergy), x=self.cenergy)

    def _normalized(self, f: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        N = self._norm(f)
        return f / N

    def _scharf_gummel(
        self, sigma_tilde: npt.NDArray[np.float64], G: float = 0
    ) -> sparse.dia_matrix:
        D = (
            self.DA
            / (sigma_tilde)
            / (1.0 + self.FN**2 / (sigma_tilde**2 * GAMMA**2 * self.benergy))
            + self.DB
        )

        # Due to the zero flux b.c. the values of z[0] and z[-1] are never used.
        # To make sure, we set is a nan so it will taint everything if ever
        # used.
        # TODO: Perhaps it would be easier simply to set the appropriate
        # values here to satisfy the b.c.
        if self.ee_collisions:
            D += self.DC
            z = (self.W + self.WC) * np.r_[np.nan, np.diff(self.cenergy), np.nan] / D
            a0 = (self.W + self.WC) / (1 - np.exp(-z))
            a1 = (self.W + self.WC) / (1 - np.exp(z))
        else:
            z = self.W * np.r_[np.nan, np.diff(self.cenergy), np.nan] / D
            a0 = self.W / (1 - np.exp(-z))
            a1 = self.W / (1 - np.exp(z))

        diags = np.zeros((3, self.n))

        # No flux at the energy = 0 boundary
        diags[0, 0] = a0[1]

        diags[0, 1:] = a0[2:] - a1[1:-1]
        diags[1, :] = a1[:-1]
        diags[2, :] = -a0[1:]

        # F[n+1] = 2 * F[n] + F[n-1] b.c.
        # diags[2, -2] -= a1[-1]
        # diags[0, -1] += 2 * a1[-1]

        # F[n+1] = F[n] b.c.
        # diags[0, -1] += a1[-1]

        # zero flux b.c.
        diags[2, -2] = -a0[-2]
        diags[0, -1] = -a1[-2]

        diags[0, :] += G

        A = sparse.dia_matrix((diags, [0, 1, -1]), shape=(self.n, self.n))

        return A

    def _g(self, F0: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        Fp = np.r_[F0[0], F0, F0[-1]]
        cenergyp = np.r_[self.cenergy[0], self.cenergy, self.cenergy[-1]]
        g = np.log(Fp[2:] / Fp[:-2]) / (cenergyp[2:] - cenergyp[:-2])

        return g

    def _PQ(
        self,
        F0: npt.NDArray[np.float64],
        reactions: list[Tuple[Target, Process]] = None,
    ) -> sparse.csr_matrix:
        PQ = sparse.csr_matrix((self.n, self.n))

        g = self._g(F0)
        if reactions is None:
            reactions = list(self.iter_inelastic())

        data = []
        rows = []
        cols = []

        for t, k in reactions:
            r = t.density * GAMMA * k.scatterings(g, self.cenergy)
            in_factor = k.in_factor

            data.extend([in_factor * r, -r])
            rows.extend([k.i, k.j])
            cols.extend([k.j, k.j])

        data, rows, cols = (np.hstack(x) for x in (data, rows, cols))
        PQ = sparse.coo_matrix((data, (rows, cols)), shape=(self.n, self.n))

        return PQ

    ##
    # Now some functions to calculate rates transport parameters from the
    # converged F0
    def rate(
        self,
        F0: npt.NDArray[np.float64],
        k: Process | str,
        weighted: bool = False,
    ) -> float:
        """Calculates the rate of a process from a (usually converged) EEDF.

        Parameters
        ----------
        F0 : array of floats
           Distribution function.
        k : :class:`process.Process` or string
           The process whose rate we want to calculate.  If `k` is a string,
           it is passed to :func:`search` to obtain a process instance.
        weighted : boolean, optional
           If true, the rate is multiplied by the density of the target.

        Returns
        -------
        rate : float
           The rate of the given process according to `F0`.

        Examples
        --------
        >>> k_ionization = bsolver.rate(F0, "N2 -> N2^+")


        See Also
        --------
        search : Find a process that matches a given signature.

        """
        g = self._g(F0)

        if isinstance(k, str):
            k = self.search(k)

        k.set_grid_cache(self.grid)

        r = k.scatterings(g, self.cenergy)

        P = sparse.coo_matrix(
            (GAMMA * r, (k.j, np.zeros(r.shape))), shape=(self.n, 1)
        ).todense()

        P = np.squeeze(np.array(P))

        rate = F0.dot(P)
        if weighted:
            rate *= k.target.density

        return rate

    def mobility(
        self,
        F0: npt.NDArray[np.float64],
    ) -> float:
        """Calculates the reduced mobility (mobility * N) from the EEDF.

        Parameters
        ----------
        F0 : array of floats
           The EEDF used to compute the mobility.

        Returns
        -------
        mun : float
           The reduced mobility (mu * n) of the electrons in SI
           units (V / m / s).

        Examples
        --------
        >>> mun = bsolver.mobility(F0)

        See Also
        --------
        diffusion : Find the reduced diffusion rate from the EEDF.
        """

        DF0 = np.r_[0.0, np.diff(F0) / np.diff(self.cenergy), 0.0]
        Q = self._PQ(F0, reactions=self.iter_growth())

        nu = np.sum(Q.dot(F0)) / GAMMA
        sigma_tilde = self.sigma_m + nu / np.sqrt(self.benergy)

        y = DF0 * self.benergy / sigma_tilde
        y[0] = 0

        return -(GAMMA / 3) * simpson(y, x=self.benergy)

    def diffusion(
        self,
        F0: npt.NDArray[np.float64],
    ) -> float:
        """Calculates the diffusion coefficient from a
        distribution function.

        Parameters
        ----------
        F0 : array of floats
           The EEDF used to compute the diffusion coefficient.

        Returns
        -------
        diffn : float
           The reduced diffusion coefficient of electrons in SI units..

        See Also
        --------
        mobility : Find the reduced mobility from the EEDF.

        """

        Q = self._PQ(F0, reactions=self.iter_growth())

        nu = np.sum(Q.dot(F0)) / GAMMA

        sigma_m = np.zeros_like(self.cenergy)
        for target, process in self.iter_momentum():
            s = target.density * process.interp(self.cenergy)
            sigma_m += s

        sigma_tilde = sigma_m + nu / np.sqrt(self.cenergy)

        y = F0 * self.cenergy / sigma_tilde

        return (GAMMA / 3) * simpson(y, x=self.cenergy)

    def mean_energy(
        self,
        F0: npt.NDArray[np.float64],
    ) -> float:
        """Calculates the mean energy from a distribution function.

        Parameters
        ----------
        F0 : array of floats
           The EEDF used to compute the diffusion coefficient.

        Returns
        -------
        energy : float
           The mean energy of electrons in the EEDF.

        """

        de52 = np.diff(self.benergy**2.5)
        return np.sum(0.4 * F0 * de52)

    def electron_temperature(
        self,
        F0: npt.NDArray[np.float64],
    ) -> float:
        """Calculate electron temperature base on mean enable_energy.

        Parameters
        ----------
        F0 : array of floats
           The EEDF used to compute the diffusion coefficient.

        Returns
        -------
        Temperature : float
           The electron temperature [K].

        """
        return 2.0 / 3.0 * self.mean_energy(F0) * ELECTRONVOLT / KB

    def normalized_inelastic_energy_loss(
        self,
        F0: npt.NDArray[np.float64],
        target_name: str = None,
    ) -> float:
        r"""Calculates the normalized energy loss due to inelastic processes given a distribution function.

        Parameters
        ----------
        F0 : npt.NDArray[np.float64]
            The EEDF used to compute the energy loss.
        target_name : str, optional
            The name of the target species for which to compute the energy loss, by default None

        Returns
        -------
        float
            The normalized energy loss due to inelastic processes.
        """
        energy = 0.0
        if target_name is None:
            for _, process in self.iter_inelastic():
                energy += process.threshold * self.rate(F0, process, weighted=True)
        else:
            for _, process in self.iter_inelastic():
                if process.target_name == target_name:
                    energy += process.threshold * self.rate(F0, process)
        return energy

    def normalized_elastic_energy_loss(
        self,
        F0: npt.NDArray[np.float64],
    ) -> float:
        r"""Calculates the normalized energy loss due to elastic processes given a distribution function.

        Parameters
        ----------
        F0 : npt.NDArray[np.float64]
            The EEDF used to compute the energy loss.

        Returns
        -------
        float
            The normalized energy loss due to elastic processes.
        """
        energy = 0.0
        DF0 = np.r_[0.0, np.diff(F0) / np.diff(self.cenergy), 0.0]
        for target, process in self.iter_elastic():
            y1 = process.interp(self.cenergy) * self.cenergy * self.cenergy * F0
            y2 = process.interp(self.benergy) * self.kT * DF0
            energy += (
                target.density
                * 2
                * target.mass_ratio
                * (simpson(y1, x=self.cenergy) + simpson(y2, x=self.benergy))
            )
        return GAMMA * energy

    def normalized_total_energy_loss(
        self,
        F0: npt.NDArray[np.float64],
    ) -> float:
        r"""Calculates the normalized total energy loss given a distribution function.

        Parameters
        ----------
        F0 : npt.NDArray[np.float64]
            The EEDF used to compute the energy loss.

        Returns
        -------
        float
            The normalized total energy loss.
        """
        return self.normalized_elastic_energy_loss(
            F0
        ) + self.normalized_inelastic_energy_loss(F0)

    def normalized_total_power(self, F0: npt.NDArray[np.float64]) -> float:
        r"""Calculates the normalized total power given a distribution function.

        Parameters
        ----------
        F0 : npt.NDArray[np.float64]
            The EEDF used to compute the total power.

        Returns
        -------
        float
            The normalized total power.
        """
        return self.mobility(F0) * self.EN * self.EN
