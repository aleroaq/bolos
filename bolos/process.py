import numpy as np
import numpy.typing as npt
from numba import njit, prange, float64
from scipy.interpolate import interp1d
from bolos.grid import Grid
from typing import Tuple


# @njit(
#     inline='always',
#     cache=True,
#     fastmath=True,
# )
# def _build_xy(
#     data: npt.NDArray[np.float64],
# ) -> Tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
#     r"""From a 2D array `data`, where the first column is the nodes and the second column is the values of the function at these nodes, 
#     returns the two columns as separate arrays. At each side of the data, we add an extra point: 
#     the first point is (0, data[0, 1]), and the last point is (1e8, data[-1, 1]). 
    
#     Parameters
#     ----------
#     data: array-like
#         The data of the interpolation, with shape (N, 2) where the first column is the nodes and the second column is the values of the function at these nodes.
    
#     Returns
#     -------
#     tuple of array-like
#         The first element is the array of nodes, and the second element is the array of values of the function at these nodes, with extrapolation at the beginning and end.
#     """
#     cond = data[0, 0] > 0
#     x = np.where(
#         cond,
#         np.concatenate((np.array([0.0]), data[:, 0], np.array([1e8]))), 
#                 #   np.r_[0.0       , data[:, 0], 1e8],
#         np.concatenate((data[:, 0], np.array([1e8, 1e8]))),
#                 #   np.r_[data[:, 0], 1e8       , 1e8]
#         )
#     y = np.where(
#         cond,
#         np.concatenate((np.array([data[0, 1]]), data[:, 1], np.array([data[-1, 1]]))),
#                 #   np.r_[data[0, 1], data[:, 1] , data[-1, 1]],
#         np.concatenate((data[:, 1], np.array([data[-1, 1], data[-1, 1]]))),
#                 #   np.r_[data[:, 1], data[-1, 1], data[-1, 1]]
#         )
#     return x, y


# @njit(
#     inline='always',
#     cache=True,
#     fastmath=True,
# )
# def _interp_numba(
#     nodes: npt.NDArray[np.float64],
#     data: npt.NDArray[np.float64],
# ) -> npt.NDArray[np.float64]:
#     r""" Interpolates from data `nodes` but adds elements at the beginning and end
#     to extrapolate cross-sections.

#     Parameters  
#     ----------
#     nodes: array-like
#         The nodes of the interpolation
#     data: array-like
#         The data of the interpolation, with shape (N, 2) where the first column is the nodes and the second column is the values of the function at these nodes.

#     Returns    
#     -------
#     array-like
#         The values of the interpolation at `nodes` with extrapolation at the beginning and end.
#     """
#     x, y = _build_xy(data)
#     return np.interp(nodes, x, y)


@njit(
    float64[:](float64[:], float64[:], float64[:], float64[:], float64[:], float64[:]),
    cache=True,
    fastmath=True,
)
def _int_linexp0_numba(
    a: npt.NDArray[np.float64], 
    b: npt.NDArray[np.float64], 
    u0: npt.NDArray[np.float64], 
    u1: npt.NDArray[np.float64], 
    g: npt.NDArray[np.float64], 
    x0: npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]:
    """ This is the integral in [a, b] of u(x) * exp(g * (x0 - x)) * x 
    assuming that
    u is linear with u({a, b}) = {u0, u1}."""

    # Since u(x) is linear, we calculate separately the coefficients
    # of degree 0 and 1 which, after multiplying by the x in the integrand
    # correspond to 1 and 2

    # The expressions involve the following exponentials that are problematic:
    # expa = np.exp(g * (-a + x0))
    # expb = np.exp(g * (-b + x0))
    # The problems come with small g: in that case, the exp() rounds to 1
    # and neglects the order 1 and 2 terms that are required to cancel the
    # 1/g**2 and 1/g**3 below.  The solution is to rewrite the expressions
    # as functions of expm1(x) = exp(x) - 1, which is guaranteed to be accurate
    # even for small x.
    expm1a = np.expm1(g * (-a + x0))
    expm1b = np.expm1(g * (-b + x0))

    ag = a * g
    bg = b * g

    ag1 = ag + 1
    bg1 = bg + 1

    g2 = g * g
    g3 = g2 * g

    A1 = (  expm1a * ag1 + ag
          - expm1b * bg1 - bg) / g2

    A2 = (expm1a * (2 * ag1 + ag * ag) + ag * (ag + 2) - 
          expm1b * (2 * bg1 + bg * bg) - bg * (bg + 2)) / g3

    c0 = (a * u1 - b * u0) / (a - b)
    c1 = (u0 - u1) / (a - b)

    r = c0 * A1 + c1 * A2

    return r


# @njit(
#     cache=True,
#     fastmath=True,
# )
# def _set_grid_cache_numba(
#     grid_b: npt.NDArray[np.float64],
#     x: npt.NDArray[np.float64],
#     y: npt.NDArray[np.float64],
#     shift_factor: float,
#     threshold: float,   
# ) -> Tuple[npt.NDArray[np.int64], npt.NDArray[np.int64], npt.NDArray[np.float64], npt.NDArray[np.float64]]: 
#     eps1 = shift_factor * grid_b + threshold
#     eps1[:] = np.maximum(eps1, grid_b[0] + 1e-9)
#     eps1[:] = np.minimum(eps1, grid_b[-1] - 1e-9)

#     fltb = np.logical_and(grid_b >= eps1[0], grid_b <= eps1[-1])
#     fltx = np.logical_and(x >= eps1[0], x <= eps1[-1])
#     # nodes = np.unique(np.r_[eps1, grid_b[fltb], x[fltx]])
#     nodes = np.unique(np.concatenate((eps1, grid_b[fltb], x[fltx])))

    
#     # sigma0 = _interp_numba(nodes, np.c_[x, y])
#     sigma0 = _interp_numba(nodes, np.column_stack((x, y)))
    
#     j = np.searchsorted(grid_b, nodes[1:]) - 1
#     i = np.searchsorted(eps1, nodes[1:]) - 1
#     sigma = np.column_stack((sigma0[:-1], sigma0[1:]))
#     # sigma = np.c_[sigma0[:-1], sigma0[1:]]
#     eps = np.column_stack((nodes[:-1], nodes[1:]))
#     # eps   = np.c_[nodes[:-1], nodes[1:]]
#     return i, j, sigma, eps
    

class Process(object):
    # The factor of in-scatering.  
    IN_FACTOR = {'EXCITATION': 1,
                 'IONIZATION': 2,
                 'ATTACHMENT': 0,
                 'ELASTIC': 1,
                 'MOMENTUM': 1,
                 'EFFECTIVE': 1,
                 'COULOMB': 1,
    }

    # The shift factor for inelastic collisions. 
    SHIFT_FACTOR = {'EXCITATION': 1,
                    'IONIZATION': 2,
                    'ATTACHMENT': 1,
                    'ELASTIC': 1,
                    'MOMENTUM': 1,
                    'EFFECTIVE': 1,
                    'COULOMB': 1,
    }

                 
    def __init__(
        self, 
        target: str = None, 
        kind: str = None, 
        data: npt.NDArray[np.float64] = None,
        comment: str = '', 
        mass_ratio: float = None,
        product: str = None, 
        threshold: float = 0, 
        weight_ratio: float = None,
        extrapolate_tail: bool = True,
        super: bool = False,
        squared_charge: bool = False,
    ) -> None:
        r"""Initializes a process with the given parameters.

        Parameters
        ----------
        target_name: str
            The name of the target of the process.
        kind: str
            The kind of the process, e.g. 'ELASTIC', 'INELASTIC', 'EFFECTIVE', etc.
        data: array-like
            The data of the process, with shape (N, 2) where the first column is the energy and the second column is the cross-section.
        comment: str
            A comment about the process.
        mass_ratio: float
            The mass ratio of the process, i.e. the mass of the product divided by the mass of the target.
        product: str
            The name of the product of the process, e.g. 'p', 'He', etc.
        threshold: float
            The energy threshold of the process, i.e. the minimum energy required for the process to occur.
        weight_ratio: float
            The weight ratio of the process, i.e. the ratio of the weight of the product to the weight of the target.
        extrapolate_tail: bool
            If true, the tail of the cross-section data will be extrapolated to high energies assuming that the cross-section decreases as log(eps)/eps. 
            This is only done if the last point of the data is not zero. The extrapolation is done by increasing the energy by a factor of 1.5 until it is higher than 10 keV.
        super: bool
            If true, the process is a superelastic process.
        squared_charge: bool
            If true, the process is a squared charge process.
        """
        self.target_name = target

        # We will link this later
        self.target = None

        self.kind = kind

        if extrapolate_tail and not kind=='COULOMB':
            data = self.extrapolate_tail(np.asarray(data))

        self.data = np.array(data)

        self.x = self.data[:, 0]
        self.y = self.data[:, 1]

        self.comment = comment
        self.mass_ratio = mass_ratio
        self.product = product
        self.threshold = threshold
        self.weight_ratio = weight_ratio
        self.super = super
        self.squared_charge = squared_charge
        self.interp = padinterp(self.data)
        self.isnull = False

        
        self.in_factor = self.IN_FACTOR.get(self.kind, None)
        self.shift_factor = self.SHIFT_FACTOR.get(self.kind, None)
        
        if np.amin(self.data[:, 0]) < 0:
            raise ValueError("Negative energy in the cross section %s"
                             % str(self))
 
        if np.amin(self.data[:, 1]) < 0:
            raise ValueError("Negative cross section for %s"
                             % str(self))
       
        self.cached_grid = None


    @staticmethod
    def extrapolate_tail(
        data: npt.NDArray[np.float64],
        factor: float = 1.3,
    ) -> npt.NDArray[np.float64]:
        r"""Extrapolates the tail of the cross-section data to high energies assuming that the cross-section decreases as log(eps)/eps.

        This is only done if the last point of the data is not zero. The extrapolation is done by increasing the energy by a factor of `factor` until it is higher than 10 keV.

        Parameters
        ----------
        data: array-like
            The data of the process, with shape (N, 2) where the first column is the energy and the second column is the cross-section.
        factor: float, optional
            The factor by which to increase the energy during extrapolation. By default 1.3.

        Returns
        -------
        array-like
            The data of the process with the extrapolated tail, with shape (M, 2) where M >= N.
        """
        if data[-1, 1] == 0 or data[-1, 0] >= 1e4:
            return data
        else:
            new_data = [data]
            while new_data[-1][-1, 0] < 1e4:
                last_point = new_data[-1][-1]
                new_energy = last_point[0] * factor
                new_cross_section = last_point[1] / (np.log(last_point[0]) / last_point[0]) * np.log(new_energy) / new_energy
                new_data.append(np.array([[new_energy, new_cross_section]]))
            return np.vstack(new_data)


    def scatterings(
        self, 
        g: npt.NDArray[np.float64], 
        eps: npt.NDArray[np.float64],
    ) -> npt.NDArray[np.float64]:
        if len(self.j) == 0:
            # When we do not have inelastic collisions or when the grid is
            # smaller than the thresholds, we still return an empty array
            # and thus avoid exceptions in g[self.j]
            return np.array([], dtype='f')

        gj = g[self.j]
        epsj = eps[self.j]
        # r = int_linexp0(self.eps[:, 0], self.eps[:, 1], 
        #                 self.sigma[:, 0], self.sigma[:, 1],
        #                 gj, epsj)
        r = _int_linexp0_numba(
            self.eps[:, 0], self.eps[:, 1],
            self.sigma[:, 0], self.sigma[:, 1],
            gj, epsj,
        )
        r = np.where(np.isnan(r), 0.0, r)
        return r
        
        
    def set_grid_cache(
        self, 
        grid: Grid,
    ) -> None:
        """ Sets a grid cache of the intersections between grid cell j and grid
        cell i shifted. 
        """

        # We will create an arras with matching 
        # rows ([i], [j], [eps1, eps2], [sigma1, sigma2])
        # that contain the overlap between the shifted cell i and cell j.
        # However we may have more than one row for a given i, j if 
        # an interpolation point for sigma falls inside the interval.

        if self.cached_grid is grid:
            # We only have to redo all these computations when the grid changes
            # so we store the grid for which this has been already calculated.
            return

        self.cached_grid = grid

        eps1 = self.shift_factor * grid.b + self.threshold
        eps1[:] = np.maximum(eps1, grid.b[0] + 1e-9)
        eps1[:] = np.minimum(eps1, grid.b[-1] - 1e-9)

        fltb = np.logical_and(grid.b >= eps1[0], grid.b <= eps1[-1])
        fltx = np.logical_and(self.x >= eps1[0], self.x <= eps1[-1])
        nodes = np.unique(np.r_[eps1, grid.b[fltb], self.x[fltx]])


        sigma0 = self.interp(nodes)
        
        self.j = np.searchsorted(grid.b, nodes[1:]) - 1
        self.i = np.searchsorted(eps1, nodes[1:]) - 1
        self.sigma = np.c_[sigma0[:-1], sigma0[1:]]
        self.eps   = np.c_[nodes[:-1], nodes[1:]]

        # self.i, self.j, self.sigma, self.eps = _set_grid_cache_numba(
        #     grid.b, self.x, self.y, self.shift_factor, self.threshold
        # )

        # print("self.i == i is: ", np.all(self.i == i))
        # print("self.j == j is: ", np.all(self.j == j))
        # print("self.sigma == sigma is: ", np.all(self.sigma == sigma))
        # print("self.eps == eps is: ", np.all(self.eps == eps))
        # raise

    def __str__(self) -> str:
        return "{%s: %s %s}" % (self.kind, self.target_name, 
                                "-> " + self.product if self.product else "")


class NullProcess(Process):
    """ This is a null process with a 0 cross section it is useful 
    when we reduce other processes. """
    def __init__(self, target: str, kind: str):
        self.data = np.empty((0, 2))
        self.interp = lambda x: np.zeros_like(x)
        self.target_name = target
        self.kind = kind
        self.isnull = True

        self.comment = None
        self.mass_ratio = None
        self.product = None
        self.threshold = None
        self.weight_ratio = None

        self.x = np.array([])
        self.y = np.array([])

    def __str__(self) -> str:
        return "{NULL}"


def padinterp(data: npt.NDArray[np.float64]) -> interp1d:
    """ Interpolates from data but adds elements at the beginning and end
    to extrapolate cross-sections. """
    if data[0, 0] > 0:
        x = np.r_[0.0, data[:, 0], 1e8]
        y = np.r_[data[0, 1], data[:, 1], data[-1, 1]]
    else:
        x = np.r_[data[:, 0], 1e8]
        y = np.r_[data[:, 1], data[-1, 1]]

    return interp1d(x, y, kind='linear')


def int_linexp0(a: float, b: float, u0: float, u1: float, g: float, x0: float) -> npt.NDArray[np.float64]:
    """ This is the integral in [a, b] of u(x) * exp(g * (x0 - x)) * x 
    assuming that
    u is linear with u({a, b}) = {u0, u1}."""

    # Since u(x) is linear, we calculate separately the coefficients
    # of degree 0 and 1 which, after multiplying by the x in the integrand
    # correspond to 1 and 2

    # The expressions involve the following exponentials that are problematic:
    # expa = np.exp(g * (-a + x0))
    # expb = np.exp(g * (-b + x0))
    # The problems come with small g: in that case, the exp() rounds to 1
    # and neglects the order 1 and 2 terms that are required to cancel the
    # 1/g**2 and 1/g**3 below.  The solution is to rewrite the expressions
    # as functions of expm1(x) = exp(x) - 1, which is guaranteed to be accurate
    # even for small x.
    expm1a = np.expm1(g * (-a + x0))
    expm1b = np.expm1(g * (-b + x0))

    ag = a * g
    bg = b * g

    ag1 = ag + 1
    bg1 = bg + 1

    g2 = g * g
    g3 = g2 * g

    # These are the expressions as functions of expa/expb
    # A1 = (  expa * ag1
    #        - expb * bg1) / g2

    # A2 = (expa * (2 * ag1 + ag * ag) - 
    #       expb * (2 * bg1 + bg * bg)) / g3

    A1 = (  expm1a * ag1 + ag
          - expm1b * bg1 - bg) / g2

    A2 = (expm1a * (2 * ag1 + ag * ag) + ag * (ag + 2) - 
          expm1b * (2 * bg1 + bg * bg) - bg * (bg + 2)) / g3

    # The factors multiplying each coefficient can be obtained by
    # the interpolation formula of u(x) = c0 + c1 * x
    c0 = (a * u1 - b * u0) / (a - b)
    c1 = (u0 - u1) / (a - b)

    r = c0 * A1 + c1 * A2

    # Where either F0 or F1 is 0 we return 0
    return np.where(np.isnan(r), 0.0, r)

