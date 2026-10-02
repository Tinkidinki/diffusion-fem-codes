"""Shared finite-element helpers for the eigenvector-convergence notebooks."""

import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.linalg import eigh_tridiagonal
from scipy.sparse.linalg import eigsh
from skfem import Basis, BilinearForm, MeshHex, MeshLine, MeshQuad, asm, condense
from skfem.element import ElementHex1, ElementLineP1, ElementQuad1
from skfem.helpers import dot, grad
from scipy.interpolate import RegularGridInterpolator


def _validate_dimension(dim, available_dimensions=None):
    if not isinstance(dim, (int, np.integer)) or isinstance(dim, bool):
        raise TypeError("dim must be an integer.")
    if dim < 1:
        raise ValueError("dim must be at least one.")
    if available_dimensions is not None and available_dimensions < dim:
        raise ValueError(
            f"dim={dim} was requested, but only {available_dimensions} "
            "coordinate dimensions are available."
        )
    return int(dim)


def _material_cell_indices(x, mat_r, dim):
    """Return material-cell indices for quadrature coordinates, x, given 
    the material refinement level mat_r and the spatial dimension dim."""
    x = np.asarray(x)
    available_dimensions = x.shape[0] if x.ndim >= 1 else 0
    dim = _validate_dimension(dim, available_dimensions)
    cells_per_axis = int(2**mat_r)
    if cells_per_axis < 1:
        raise ValueError("mat_r must define at least one cell per axis.")
    indices = np.floor(x[:dim] * cells_per_axis).astype(np.int64)
    return np.clip(indices, 0, cells_per_axis - 1)


def _checkerboard_coefficient(x, r, mat_r, dim, high, low):
    """Evaluate a two-valued Cartesian checkerboard coefficient."""
    cell_indices = _material_cell_indices(x, mat_r, dim)
    if r < mat_r:
        return (low + high) / 2
    checkerboard_parity = np.sum(cell_indices, axis=0) % 2
    return np.where(checkerboard_parity == 0, high, low)


def D_checkerboard(x, r, mat_r=2, dim=2, D_max=1.0, D_min=1.0):
    """Evaluate a checkerboard diffusion coefficient in ``dim`` dimensions."""
    return _checkerboard_coefficient(
        x, r, mat_r, dim, high=D_max, low=D_min
    )


def D_random(x, r, mat_r=2, dim=2, D_max=1.0, D_min=1.0):
    """Evaluate a reproducible random field in ``dim`` dimensions."""
    cell_indices = _material_cell_indices(x, mat_r, dim)
    if r < mat_r:
        return (D_min + D_max) / 2
    cells_per_axis = int(2**mat_r)
    rng = np.random.default_rng(seed=93101)
    coefficients = rng.uniform(
        D_min, D_max, size=(cells_per_axis,) * dim
    )
    return coefficients[tuple(cell_indices[axis] for axis in range(dim))]


def Absorption_checkerboard(
    x, r, mat_r=2, dim=2, sigma_max=1.0, sigma_min=1.0
):
    """Evaluate a checkerboard absorption XS in ``dim`` dimensions."""
    return _checkerboard_coefficient(
        x, r, mat_r, dim, high=sigma_max, low=sigma_min
    )


def Fission_checkerboard(
    x, r, mat_r=2, dim=2, sigma_f_max=1.0, sigma_f_min=1.0
):
    """Evaluate a checkerboard fission XS in ``dim`` dimensions."""
    return _checkerboard_coefficient(
        x, r, mat_r, dim, high=sigma_f_max, low=sigma_f_min
    )


def _defined_material(x, mat_r, values, dim=None):
    if isinstance(values, tuple) and len(values) == 1:
        values = values[0]
    values = np.asarray(values)
    if dim is None:
        dim = values.ndim
    dim = _validate_dimension(dim)
    cells_per_axis = int(2**mat_r)
    if values.ndim != dim or any(
        size < cells_per_axis for size in values.shape
    ):
        raise ValueError(
            f"Material array must be {dim}-dimensional with at least "
            f"{cells_per_axis} entries per axis; got shape {values.shape}."
        )
    cell_indices = _material_cell_indices(x, mat_r, dim)
    return values[tuple(cell_indices[axis] for axis in range(dim))]


def D_defined(x, mat_r, D_mat, dim=None):
    return _defined_material(x, mat_r, D_mat, dim=dim)


def absorption_defined(x, mat_r, sigma_a_mat, dim=None):
    return _defined_material(x, mat_r, sigma_a_mat, dim=dim)


def fission_defined(x, mat_r, sigma_f_mat, dim=None):
    return _defined_material(x, mat_r, sigma_f_mat, dim=dim)


@BilinearForm
def a_checkerboard(u, v, w):
    return D_checkerboard(
        x=w.x,
        r=w.r,
        mat_r=w.mat_r,
        dim=w.x.shape[0],
        D_max=w.D_max,
        D_min=w.D_min,
    ) * dot(grad(u), grad(v)) + Absorption_checkerboard(
        x=w.x,
        r=w.r,
        mat_r=w.mat_r,
        dim=w.x.shape[0],
        sigma_max=w.sigma_a_max,
        sigma_min=w.sigma_a_min,
    ) * u * v


@BilinearForm
def a_random(u, v, w):
    return D_random(
        x=w.x,
        r=w.r,
        mat_r=w.mat_r,
        dim=w.x.shape[0],
        D_max=w.D_max,
        D_min=w.D_min,
    ) * dot(grad(u), grad(v)) + u * v


@BilinearForm
def b_checkerboard(u, v, w):
    return Fission_checkerboard(
        x=w.x,
        r=w.r,
        mat_r=w.mat_r,
        dim=w.x.shape[0],
        sigma_f_max=w.sigma_f_max,
        sigma_f_min=w.sigma_f_min,
    ) * u * v


@BilinearForm
def mass(u, v, w):
    return u * v


@BilinearForm
def a_defined(u, v, w):
    dim = w.x.shape[0]
    return D_defined(w.x, w.mat_r, w.D_mat, dim=dim) * dot(
        grad(u), grad(v)
    ) + absorption_defined(
        w.x, w.mat_r, w.sigma_a_mat, dim=dim
    ) * u * v


@BilinearForm
def b_defined(u, v, w):
    return fission_defined(
        w.x,
        w.mat_r,
        w.nu_sigma_f_mat,
        dim=w.x.shape[0],
    ) * u * v


def fix_sign(u):
    """Choose a deterministic global sign for a real eigenvector."""
    u = np.asarray(u)
    if u.size == 0:
        raise ValueError("Cannot fix the sign of an empty array.")
    j = np.argmax(np.abs(u))
    return u if u.reshape(-1)[j] >= 0 else -u


def plot_bilinear_hat_solution(U, *coordinates, plot_type="contour"):
    """Plot a 1D Q1 solution or a central 2D slice of higher-dimensional data."""
    U = np.asarray(U)
    coordinates = tuple(np.asarray(axis) for axis in coordinates)
    if len(coordinates) != U.ndim:
        raise ValueError(
            f"Expected {U.ndim} coordinate arrays for U.shape={U.shape}; "
            f"got {len(coordinates)}."
        )
    expected_shape = tuple(len(axis) for axis in coordinates)
    if U.shape != expected_shape:
        raise ValueError(
            f"Expected U.shape == {expected_shape}, but got {U.shape}."
        )

    if U.ndim == 1:
        plt.figure(figsize=(6, 4))
        plt.plot(coordinates[0], U)
        plt.xlabel("x")
        plt.ylabel("u(x)")
        plt.title("Linear FEM solution")
        plt.tight_layout()
        plt.show()
        return

    slice_indices = [slice(None), slice(None)] + [
        len(axis) // 2 for axis in coordinates[2:]
    ]
    U_slice = U[tuple(slice_indices)]
    x, y = coordinates[:2]

    X, Y = np.meshgrid(x, y, indexing="ij")
    plt.figure(figsize=(6, 5))
    if plot_type == "contour":
        plt.contourf(X, Y, U_slice, levels=50)
        plt.colorbar(label="u(x, y)")
    elif plot_type == "surface":
        ax = plt.axes(projection="3d")
        ax.plot_surface(X, Y, U_slice, cmap="viridis")
    elif plot_type == "heatmap":
        plt.imshow(
            U_slice,
            extent=(x.min(), x.max(), y.min(), y.max()),
            origin="lower",
            aspect="equal",
        )
        plt.colorbar(label="u(x, y)")
    else:
        raise ValueError(f"Unknown plot_type: {plot_type}")

    plt.xlabel("x")
    plt.ylabel("y")
    title = "Bilinear FEM solution"
    if U.ndim > 2:
        title += f" (central {U.ndim}D slice)"
    plt.title(title)
    if plot_type != "surface":
        plt.gca().set_aspect("equal")
    plt.tight_layout()
    plt.show()


def _solve_smallest_eigenpair(A, B, boundary_dofs):
    A_c, B_c, x0, interior = condense(A, B, D=boundary_dofs)
    if A_c.shape[0] == 1:
        eigenvalue = A_c[0, 0] / B_c[0, 0]
        u_free = np.array([1.0])
    else:
        values, vectors = eigsh(A_c, k=1, M=B_c, sigma=0.0, which="LM")
        eigenvalue = values[0]
        u_free = vectors[:, 0]

    u_full = x0.copy()
    u_full[interior] = u_free
    return eigenvalue, u_full


def _tensor_product_basis(r, dim, intorder=2):
    """Build a uniform Q1 tensor-product FEM space in 1D, 2D, or 3D."""
    dim = _validate_dimension(dim)
    if dim > 3:
        raise ValueError(
            "scikit-fem tensor-product meshes are supported here only for "
            "dim=1, dim=2, or dim=3."
        )

    grid = np.linspace(0.0, 1.0, int(2**r + 1))
    if dim == 1:
        mesh = MeshLine.init_tensor(grid)
        element = ElementLineP1()
    elif dim == 2:
        mesh = MeshQuad.init_tensor(grid, grid)
        element = ElementQuad1()
    else:
        mesh = MeshHex.init_tensor(grid, grid, grid)
        element = ElementHex1()
    return grid, mesh, Basis(mesh, element, intorder=intorder)


def _plot_mesh_and_solution(mesh, u_full, r, dim=None):
    if dim is None:
        dim = mesh.p.shape[0]
    dim = _validate_dimension(dim, mesh.p.shape[0])
    if dim <= 2:
        ax = mesh.draw()
        if dim == 2:
            ax.set_aspect("equal")
        plt.show()

    points_per_axis = int(2**r + 1)
    expected_size = points_per_axis**dim
    if np.size(u_full) != expected_size:
        raise ValueError(
            f"Expected {expected_size} nodal coefficients for dim={dim} "
            f"and r={r}; got {np.size(u_full)}."
        )
    signed_solution = fix_sign(u_full).reshape(-1)
    nodal_indices = np.rint(mesh.p * (points_per_axis - 1)).astype(int)
    flux_solution = np.empty((points_per_axis,) * dim, dtype=signed_solution.dtype)
    flux_solution[tuple(nodal_indices[axis] for axis in range(dim))] = (
        signed_solution
    )
    grid = np.linspace(0.0, 1.0, points_per_axis)
    plot_bilinear_hat_solution(flux_solution, *([grid] * dim))


def _sqrt_matrix_action(B, u, rtol=1e-10, maxiter=200):
    """Approximate B**(1/2) @ u by Lanczos for symmetric positive B.

    Keep B sparse and diagonalize only the small Lanczos projection.
    Stop after two successive relative changes below rtol; this is an
    estimated convergence criterion, not a rigorous error bound.
    """
    norm_u = np.linalg.norm(u)
    if norm_u == 0:
        return np.zeros_like(u)
    steps = min(len(u), maxiter)
    Q = np.empty((len(u), steps), dtype=float, order="F")
    diagonal = []
    off_diagonal = []
    q = u / norm_u
    previous = None
    converged_steps = 0
    for j in range(steps):
        Q[:, j] = q
        residual = B @ q
        alpha = float(q @ residual)
        diagonal.append(alpha)
        residual -= alpha * q
        if j:
            residual -= off_diagonal[-1] * Q[:, j - 1]
        # Reorthogonalize to avoid repeated Ritz values from roundoff.
        basis = Q[:, :j + 1]
        for _ in range(2):
            residual -= basis @ (basis.T @ residual)
        beta = np.linalg.norm(residual)
        values, vectors = eigh_tridiagonal(diagonal, off_diagonal)
        scale = np.max(np.abs(values))
        if np.min(values) < -100 * np.finfo(float).eps * scale:
            raise ValueError("B must be positive semidefinite for its real square root.")
        coefficients = vectors @ (np.sqrt(np.maximum(values, 0)) * vectors[0])
        if previous is not None:
            change = coefficients.copy()
            change[:-1] -= previous
            if np.linalg.norm(change) <= rtol * np.linalg.norm(coefficients):
                converged_steps += 1
            else:
                converged_steps = 0
        if (converged_steps >= 2 or j + 1 == len(u)
                or beta <= 100 * np.finfo(float).eps * scale):
            return norm_u * (basis @ coefficients)
        previous = coefficients
        off_diagonal.append(beta)
        q = residual / beta
    raise RuntimeError(
        f"B square-root action did not converge within {maxiter} Lanczos steps."
    )


def eigvecs_at_level(
    mat_r,
    r,
    D_max=1.0,
    D_min=1.0,
    sigma_a_max=1.0,
    sigma_a_min=1.0,
    sigma_f_max=1.0,
    sigma_f_min=1.0,
    do_plot=False,
    geometry="checkerboard",
    dim=2,
    apply_b_sqrt=False,
):
    """Solve a 1D, 2D, or 3D checkerboard/random eigenproblem.

    If ``apply_b_sqrt`` is True, return ``(eigenvalue, B**(1/2) @ u_full)``
    using a sparse Lanczos approximation to the square-root action of the
    full assembled B (relative convergence tolerance 1e-10, at most 200
    steps).  Plots show the original u_full.
    """
    _, mesh, basis = _tensor_product_basis(r, dim, intorder=2)

    form_parameters = dict(
        mat_r=mat_r,
        r=r,
        D_max=D_max,
        D_min=D_min,
        sigma_a_max=sigma_a_max,
        sigma_a_min=sigma_a_min,
    )
    if geometry == "checkerboard":
        A = asm(a_checkerboard, basis, **form_parameters)
    elif geometry == "random":
        A = asm(a_random, basis, **form_parameters)
    else:
        raise ValueError("geometry must be 'checkerboard' or 'random'")

    B = asm(
        b_checkerboard,
        basis,
        mat_r=mat_r,
        r=r,
        sigma_f_max=sigma_f_max,
        sigma_f_min=sigma_f_min,
    )
    eigenvalue, u_full = _solve_smallest_eigenpair(
        A, B, basis.get_dofs().all()
    )
    if do_plot:
        _plot_mesh_and_solution(mesh, u_full, r, dim=dim)
    if apply_b_sqrt:
        u_full = _sqrt_matrix_action(B, u_full)
    return eigenvalue, u_full


def eigvecs_at_level_defined(
    mat_r,
    r,
    D_mat,
    sigma_a_mat,
    nu_sigma_f_mat,
    do_plot=False,
    dim=None,
    apply_b_sqrt=False,
):
    """Solve a 1D, 2D, or 3D matrix-defined material eigenproblem."""
    D_mat = np.asarray(D_mat)
    sigma_a_mat = np.asarray(sigma_a_mat)
    nu_sigma_f_mat = np.asarray(nu_sigma_f_mat)
    if dim is None:
        dim = D_mat.ndim
    dim = _validate_dimension(dim)
    shapes = {D_mat.shape, sigma_a_mat.shape, nu_sigma_f_mat.shape}
    if len(shapes) != 1:
        raise ValueError(
            "D_mat, sigma_a_mat, and nu_sigma_f_mat must have the same "
            "shape."
        )
    if D_mat.ndim != dim:
        raise ValueError(
            f"Material arrays must have dim={dim} axes; got shape "
            f"{D_mat.shape}."
        )

    _, mesh, basis = _tensor_product_basis(r, dim, intorder=2)
    parameters = dict(
        mat_r=mat_r,
        # Tuples keep raw material tensors from being interpreted as nodal
        # coefficient vectors by scikit-fem's assembly argument normalizer.
        D_mat=(D_mat,),
        sigma_a_mat=(sigma_a_mat,),
        nu_sigma_f_mat=(nu_sigma_f_mat,),
    )
    A = asm(a_defined, basis, **parameters)
    B = asm(b_defined, basis, **parameters)
    #A_test = A.toarray()
    #B_test = B.toarray()
    eigenvalue, u_full = _solve_smallest_eigenpair(
        A, B, basis.get_dofs().all()
    )

    if do_plot:
        _plot_mesh_and_solution(mesh, u_full, r, dim=dim)
    if apply_b_sqrt:
            u_full = _sqrt_matrix_action(B, u_full)
    return eigenvalue, u_full


# helper of get_uniform_superposition_overlaps
def _validate_analysis_dimension(dim):
    if not isinstance(dim, (int, np.integer)) or isinstance(dim, bool):
        raise TypeError("dim must be an integer.")
    if dim not in (1, 2, 3):
        raise ValueError("The scikit-fem solvers support dim=1, 2, or 3.")
    return int(dim)

# helper of get_uniform_superposition_overlaps
def _grid_points_and_interior_mask(r, dim):
    dim = _validate_analysis_dimension(dim)
    grid = np.linspace(0.0, 1.0, 2**r + 1)
    coordinate_arrays = np.meshgrid(*([grid] * dim), indexing="ij")
    points = np.column_stack([axis.ravel() for axis in coordinate_arrays])
    interior_mask = np.all((points > 0.0) & (points < 1.0), axis=1)
    return grid, points, interior_mask

# helper of get_uniform_superposition_overlaps
def _solution_tensor(solution, r, dim):
    dim = _validate_analysis_dimension(dim)
    points_per_axis = 2**r + 1
    solution = np.asarray(solution)
    expected_size = points_per_axis**dim
    if solution.size != expected_size:
        raise ValueError(
            f"Expected {expected_size} coefficients for r={r}, dim={dim}; "
            f"got {solution.size}."
        )
    values = solution.reshape((points_per_axis,) * dim, order="F")
    if dim >= 2:
        values = np.swapaxes(values, 0, 1)
    return values



# Compute the overlaps and L2 errors of fine grid eigenvectors with respect to a coarse reference solution.
# material data: an iteraterable object containing tuples of the form (mat_r, D_min, D_max, sigma_a_min, sigma_a_max, sigma_f_min, sigma_f_max)
# min_r: minimum refinement level for the fine grid
# max_refinement: maximum refinement level for the fine grid
# coarse_refinement: refinement level for the coarse reference solution
def get_uniform_superposition_overlaps(material_data, min_r, max_refinement, geometry="checkerboard", dim=2, apply_b_sqrt=True):
    dim = _validate_analysis_dimension(dim)
    fine_sol_overlap_list = [] # coarse-to-fine overlaps for each material-coefficient tuple
    successive_fine_overlap_list = [] # overlaps between fine solutions at levels r and r+1
    eigenvector_l2_error_list = [] # list of L2 errors between coarse and reference solutions for each D_max
    fine_l1_norm_list = [] # continuous L1 norms for each material-coefficient tuple
    fine_l2_norm_list = [] # continuous L2 norms for each material-coefficient tuple
    for mat_r, D_min, D_max, sigma_a_min, sigma_a_max, sigma_f_min, sigma_f_max in material_data:
        Data_dict_dmax_1 = {"levels": [], "lambda1": [], "full_size": [], "free_size": [], "u_full": [], "elements": [], "chi": []}

        # store overlap information for each fine refinement level
        fine_sol_overlaps = []
        successive_fine_overlaps = []
        eigenvector_l2_errors = []
        fine_l1_norms = []
        fine_l2_norms = []
        previous_fine_solution = None

        # loop through the fine refinement levels and compute the corresponding eigenvectors
        for r in range(min_r, max_refinement+1): 
            lambda1, fine_solution = eigvecs_at_level(mat_r=mat_r, r=r, D_max=D_max, D_min=D_min, 
                                                      sigma_a_max=sigma_a_max, sigma_a_min=sigma_a_min, 
                                                      sigma_f_max=sigma_f_max, sigma_f_min=sigma_f_min, 
                                                      do_plot=False, geometry=geometry, dim=dim, apply_b_sqrt=apply_b_sqrt)
            fine_solution = fix_sign(fine_solution)
            Data_dict_dmax_1["levels"].append(r)
            Data_dict_dmax_1["lambda1"].append(float(lambda1))
            Data_dict_dmax_1["full_size"].append(len(fine_solution)) 
            points_per_axis = 2**r + 1
            Data_dict_dmax_1["free_size"].append((points_per_axis - 2) ** dim)
            Data_dict_dmax_1["u_full"].append(fix_sign(fine_solution))
            print(f"mat_r: {mat_r}, D_min: {D_min}, D_max: {D_max}, sigma_a_min: {sigma_a_min}, sigma_a_max: {sigma_a_max}, sigma_f_min: {sigma_f_min}, sigma_f_max: {sigma_f_max}, level: {r}, lambda1: {lambda1}, full_size: {len(fine_solution)}")

            fine_points_per_axis = 2**r + 1
            fine_grid, fine_points, interior_mask = (
                _grid_points_and_interior_mask(r, dim)
            )
            fine_values = _solution_tensor(fine_solution, r, dim)
            fine_interior = fine_values.ravel()[interior_mask]
            fine_interior_normalized = fine_interior / np.linalg.norm(fine_interior)

            # Compare the previous fine solution (level r-1) with the current
            # fine solution after interpolating it onto the current grid.
            if previous_fine_solution is not None:
                previous_r = r - 1
                previous_grid, _, _ = _grid_points_and_interior_mask(
                    previous_r, dim
                )
                previous_values = _solution_tensor(
                    previous_fine_solution, previous_r, dim
                )
                previous_interpolator = RegularGridInterpolator(
                    (previous_grid,) * dim, previous_values, method="linear"
                )
                previous_interpolated = previous_interpolator(fine_points)
                previous_interpolated_interior = previous_interpolated[interior_mask]
                previous_interpolated_interior /= np.linalg.norm(
                    previous_interpolated_interior
                )
                successive_fine_overlaps.append(
                    abs(np.dot(previous_interpolated_interior, fine_interior_normalized))
                )

            previous_fine_solution = fine_solution.copy()

            # Integrate the absolute value of the continuous Q1 FEM function.
            # Boundary coefficients in fine_solution are already zero due
            # to the homogeneous Dirichlet boundary condition.
            _, _, fine_basis = _tensor_product_basis(
                r, dim, intorder=4
            )
            fine_field = fine_basis.interpolate(fine_solution)
            fine_l1_norms.append(
                float(np.sum(np.abs(fine_field.value) * fine_basis.dx))
            )
            fine_l2_norms.append(
                float(np.sqrt(np.sum(
                    np.abs(fine_field.value) ** 2 * fine_basis.dx
                )))
            )

            # interpolate the coarse solution onto the fine grid and find overlap to fine solution
            uniform_interior_solution = np.ones_like(fine_interior)
            uniform_interior_normalized = (
                uniform_interior_solution / np.linalg.norm(uniform_interior_solution)
            )

            # write results to lists
            eigenvector_l2_errors.append(
                np.linalg.norm(uniform_interior_normalized - fine_interior_normalized)
            )
            fine_sol_overlaps.append(
                np.dot(uniform_interior_normalized, fine_interior_normalized)
            )


        # compute the overlap of the interpolated coarse solutions with the reference solution
        #coarse_sol_overlaps = [np.dot(Data_dict_dmax_1["interpolated_interior_coarse_normalized_solutions"][r], Data_dict_dmax_1["fine_interior_normalized"]) for r in range(0, max_refinement-min_r+1)]
        fine_sol_overlap_list.append(fine_sol_overlaps)
        successive_fine_overlap_list.append(successive_fine_overlaps)
        eigenvector_l2_error_list.append(eigenvector_l2_errors)
        fine_l1_norm_list.append(fine_l1_norms)
        fine_l2_norm_list.append(fine_l2_norms)

        print("Refinement values: ", list(range(min_r, max_refinement+1)))
        #print(r'\hat{u}_c', Data_dict_dmax_1["interpolated_interior_coarse_normalized_solutions"])
        #print(r'\hat{u}_f', Data_dict_dmax_1["reference_interior_normalized"])
        print(r'|| s_{uniform} - \hat{u}_f || :', eigenvector_l2_errors)
        print(r'< s_{uniform} | \hat{u}_f> :', fine_sol_overlaps)
        print(r'< \hat{u}_r | \hat{u}_{r+1}> :', successive_fine_overlaps)
        print(r'||u_h||_{L^1(\Omega)} :', fine_l1_norms)
        print(r'||u_h||_{L^2(\Omega)} :', fine_l2_norms)

    return (
        fine_sol_overlap_list,
        successive_fine_overlap_list,
        fine_l1_norm_list,
        fine_l2_norm_list,
    )


def load_and_collapse_c5g7_full_core(data_directory=None):
    """Load quarter-core C5G7 data and return collapsed full-core grids.

    The input files ``flux.npy``, ``total.npy``, ``absorption.npy``, and
    ``nu_fission.npy`` must each have shape ``(7, 51, 51)``.  Cross sections
    are collapsed independently in each spatial cell using that cell's
    seven-group flux spectrum.  The collapsed quarter-core arrays are then
    reflected across both spatial axes to produce ``(102, 102)`` full-core
    arrays and padded by 13 cells on every side.  Padding repeats the nearest
    boundary cross section, producing ``(128, 128)`` arrays.

    Parameters
    ----------
    data_directory : path-like, optional
        Directory containing the four ``.npy`` files.  By default, the files
        are loaded from the directory containing this module.

    Returns
    -------
    total, absorption, nu_fission : ndarray
        Flux-weighted one-group padded full-core cross-section grids, each
        with shape ``(128, 128)``.
    """
    if data_directory is None:
        data_directory = Path(__file__).resolve().parent
    else:
        data_directory = Path(data_directory)

    expected_shape = (7, 51, 51)

    def load_grid(filename):
        path = data_directory / filename
        try:
            values = np.load(path, allow_pickle=False)
        except (OSError, ValueError) as error:
            raise ValueError(
                f"Could not load {path} as a NumPy .npy array."
            ) from error
        if values.shape != expected_shape:
            raise ValueError(
                f"{path} must have shape {expected_shape}; got "
                f"{values.shape}."
            )
        values = np.asarray(values, dtype=float)
        if not np.all(np.isfinite(values)):
            raise ValueError(f"{path} contains non-finite values.")
        return values

    flux = load_grid("flux.npy")
    total = load_grid("total.npy")
    fission = load_grid("fission.npy")
    absorption = total - fission
    nu_fission = load_grid("nu_fission.npy")

    total_test = total[0]
    fission_test = fission[0]
    absorption_test = absorption[0]
    nu_fission_test = nu_fission[0]

    if np.any(flux < 0.0):
        raise ValueError("flux.npy contains negative flux values.")
    cell_flux = np.sum(flux, axis=0)
    if np.any(cell_flux <= 0.0):
        zero_cells = np.argwhere(cell_flux <= 0.0)
        raise ValueError(
            "Every spatial cell must have positive total flux; found "
            f"{len(zero_cells)} cell(s) with zero total flux."
        )

    def collapse(multigroup_xs):
        return np.sum(flux * multigroup_xs, axis=0) / cell_flux

    def reflect_quarter_core(quarter_core):
        negative_x = np.flip(quarter_core, axis=0)
        full_x = np.concatenate((negative_x, quarter_core), axis=0)
        return np.concatenate((np.flip(full_x, axis=1), full_x), axis=1)

    def pad_full_core(full_core):
        return np.pad(full_core, pad_width=13, mode="edge")

    return tuple(
        pad_full_core(reflect_quarter_core(collapse(multigroup_xs)))
        for multigroup_xs in (total, absorption, nu_fission)
    )
