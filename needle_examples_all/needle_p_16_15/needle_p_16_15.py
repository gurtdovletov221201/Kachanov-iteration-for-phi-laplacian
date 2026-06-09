import firedrake
from firedrake import (
    Constant, DirichletBC, Function, FunctionSpace, SpatialCoordinate,
    TestFunction, TrialFunction, assemble, conditional, div, dx, dS, grad,
    inner, solve, sqrt, max_value, min_value, CellDiameter, FacetArea,
    jump, avg, AdaptiveMeshHierarchy, AdaptiveTransferManager
)
import matplotlib.pyplot as plt
import numpy as np
from netgen.geom2d import SplineGeometry

# Parameters 
N, p = 4, 16/15
max_iters, tol_energy = 85, 1e-7
dorfler_theta = 0.2
max_refinements = 50

# Relaxation & estimator updates
eps_minus, eps_plus = 1.0, 1.0
theta_minus, theta_plus = 0.5, 1.25
delta_factor_minus, delta_factor_plus = 1e-6, 1e6

atm = AdaptiveTransferManager()


def make_initial_mesh(N):
    geo = SplineGeometry()
    geo.AddRectangle(p1=(-1, -1), p2=(1, 1), bc="outer", leftdomain=1, rightdomain=0)
    return firedrake.Mesh(geo.GenerateMesh(maxh=2/N))

def build_problem(mesh):
    V = FunctionSpace(mesh, "CG", 1)
    X, Y = SpatialCoordinate(mesh)
    r = sqrt(X**2 + Y**2)
    alpha = 1 - 1/p
    u_exact_expr = r**alpha - 1
    u_exact = Function(V, name="u_exact").interpolate(u_exact_expr)
    grad_e = grad(u_exact_expr)
    f_expr = -div((inner(grad_e, grad_e)**(p/2 - 1)) * grad_e)
    
    return V, u_exact_expr, u_exact, f_expr


def grad_norm(w): return sqrt(inner(grad(w), grad(w)))
def clip(t, e_m, e_p): return max_value(min_value(t, e_p), e_m)

def kappa_eps(t, e_m, e_p):
    c = clip(t, e_m, e_p)
    return 0.5 * c**(p-2) * (t**2) + (1/p - 0.5) * c**p

def phi_eps_dual(t, e_m, e_p):
    return conditional(
        t <= e_m**(p - 1),
        0.5 * e_m**(2 - p) * (t**2),
        conditional(
            t <= e_p**(p - 1),
            ((p - 1)/p) * (t**(p/(p - 1))) + (1/p - 0.5) * (e_m**p),
            0.5 * e_p**(2 - p) * (t**2) - (1/p - 0.5) * (e_p**p - e_m**p)
        )
    )

#  Energies & Estimators 
def calc_energies(w, f_expr, e_m, e_p):
    g = grad_norm(w)
    J_exact = assemble((1/p) * g**p * dx - f_expr * w * dx)
    J_rel = assemble(kappa_eps(g, e_m, e_p) * dx - f_expr * w * dx)
    return J_exact, J_rel

def calc_eta_pm(w, e_m, e_p, is_plus=False):
    d_m, d_p = delta_factor_minus * e_m, delta_factor_plus * e_p
    g = grad_norm(w)
    cond = g > e_p if is_plus else g < e_m
    integrand = conditional(cond, kappa_eps(g, e_m, e_p) - kappa_eps(g, d_m, d_p), 0)
    return assemble(integrand * dx) * (10**3 if is_plus else 1.0)

def eta_kac_squared(u_old, u_new, e_m, e_p):
    alpha = clip(grad_norm(u_old), e_m, e_p)
    argument = alpha**(p-2) * grad_norm(u_old - u_new)
    return assemble(phi_eps_dual(argument, clip(alpha, e_m, e_p), e_p) * dx)

def eta_h_squared(w, f_expr, e_m, e_p):
    mesh = w.function_space().mesh()
    DG0 = FunctionSpace(mesh, "DG", 0)
    q = TestFunction(DG0)
    
    alpha = grad_norm(w)
    vol_density = phi_eps_dual(CellDiameter(mesh) * sqrt(f_expr**2), clip(alpha, e_m, e_p), e_p)
    
    Veps = clip(alpha, e_m, e_p)**(p/2 - 1) * grad(w)
    jump_density = avg(FacetArea(mesh)) * inner(jump(Veps), jump(Veps))
    
    eta2_T = Function(DG0)
    eta2_T.dat.data[:] = assemble(vol_density * q * dx).dat.data_ro[:]
    eta2_T.dat.data[:] += assemble(jump_density * (q("+") + q("-")) * dS).dat.data_ro[:]
    eta2_T.dat.data[:] = np.maximum(eta2_T.dat.data[:], 0.0) * 1e-3
    
    return float(np.sum(eta2_T.dat.data_ro)), eta2_T

#  Dörfler Marking 
def dorfler_marking(eta2_T, theta):
    eta2 = np.maximum(eta2_T.dat.data_ro.copy(), 0.0)
    total = np.sum(eta2)
    marker = Function(eta2_T.function_space())
    marker.assign(0.0)
    
    if total <= 0.0: return marker, 0.0, total
    
    order = np.argsort(-eta2)
    accum = np.cumsum(eta2[order])
    cutoff = np.searchsorted(accum, theta * total)
    
    marked = order[:cutoff + 1]
    marker.dat.data[marked] = 1.0
    return marker, accum[cutoff], total

# Solver
mesh = make_initial_mesh(N)
amh = AdaptiveMeshHierarchy(mesh)
V, u_exact_expr, u_exact, f_expr = build_problem(mesh)

u_old = Function(V, name="u_0").assign(0)
bc = DirichletBC(V, u_exact_expr, "on_boundary")
bc.apply(u_old)
history, cost, num_refinements = [], 0, 0
J_exact_ref, _ = calc_energies(u_exact, f_expr, eps_minus, eps_plus)

for n in range(max_iters):
    # Kachanov step
    u_new, u_trial, v_test = Function(V), TrialFunction(V), TestFunction(V)
    coeff = clip(grad_norm(u_old), eps_minus, eps_plus)**(p-2)
    
    solve(
        inner(coeff * grad(u_trial), grad(v_test)) * dx == f_expr * v_test * dx,
        u_new, bcs=bc,
        solver_parameters={"pc_type": "lu", "pc_factor_mat_solver_type": "mumps"}
    )
    
    # Compute Estimators
    cost += V.dim()
    eta_m2 = calc_eta_pm(u_new, eps_minus, eps_plus, is_plus=False)
    eta_p2 = calc_eta_pm(u_new, eps_minus, eps_plus, is_plus=True)
    eta_k2 = eta_kac_squared(u_old, u_new, eps_minus, eps_plus)
    eta_h2, eta_h2_cell = eta_h_squared(u_new, f_expr, eps_minus, eps_plus)
    _, J_rel = calc_energies(u_new, f_expr, eps_minus, eps_plus)
    
    err_val = J_rel - J_exact_ref
    
    print(f"n={n:2d} | cost={cost:6d} | err={err_val:8.2e} | "
          f"η_m²={eta_m2:8.2e} η_p²={eta_p2:8.2e} η_k²={eta_k2:8.2e} η_h²={eta_h2:8.2e} | "
          f"ε-={eps_minus:8.2e} ε+={eps_plus:8.2e} | refs={num_refinements}")

    history.append((cost, err_val, eta_k2, eta_m2, eta_p2, eta_h2, eps_minus, eps_plus))
    if abs(err_val) < tol_energy:
        print(f"Converged in {n} iterations.")
        break

    errors = {"m": float(eta_m2), "p": float(eta_p2), "k": float(eta_k2), "h": float(eta_h2)}
    largest = max(errors, key=errors.get)

    if largest == "m":
        eps_minus *= theta_minus
        u_old.assign(u_new)
    elif largest == "p":
        eps_plus *= theta_plus
        u_old.assign(u_new)
    elif largest == "h":
        marker, marked_sum, total_sum = dorfler_marking(eta_h2_cell, dorfler_theta)
        
        if num_refinements < max_refinements:
            mesh = mesh.refine_marked_elements(marker)
            amh.add_mesh(mesh)
            num_refinements += 1
            
            V, u_exact_expr, u_exact, f_expr = build_problem(mesh)
            bc = DirichletBC(V, u_exact_expr, "on_boundary")
            J_exact_ref, _ = calc_energies(u_exact, f_expr, eps_minus, eps_plus)
            
            u_old = Function(V)
            atm.prolong(u_new, u_old)
            bc.apply(u_old)
        else:
            u_old.assign(u_new)
    else:  # largest == "k"
        u_old.assign(u_new)

# Visualization 
history = np.array(history)
costs = history[:, 0]
ref_line = 30 * costs**(-1.0)

plt.figure(figsize=(10, 6))
plt.loglog(costs, np.abs(history[:, 1]), "ko-", label="energy_err")
plt.loglog(costs, history[:, 2], "go-", label="eta_k2")
plt.loglog(costs, history[:, 3], "ys-", label="eta_m2")
plt.loglog(costs, history[:, 4], "bs-", label="eta_p2")
plt.loglog(costs, history[:, 5], "ro-", label="eta_h2")
plt.loglog(costs, history[:, 6], "k:", label="e_m")
plt.loglog(costs, history[:, 7], "k--", label="e_p")
plt.loglog(costs, ref_line, "k-", label=r"$30 \times \text{cost}^{-1}$")

plt.xlabel("cost")
plt.legend(loc="lower left")
plt.savefig("needle_p_16_15.png", dpi=200)