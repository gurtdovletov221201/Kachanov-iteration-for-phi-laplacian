import firedrake
from firedrake import (
    Constant, DirichletBC, Function, FunctionSpace, SpatialCoordinate,
    TestFunction, TrialFunction, assemble, conditional, div, dx, dS, grad,
    inner, solve, sqrt, max_value, min_value, CellDiameter, FacetArea,
    jump, avg, AdaptiveMeshHierarchy, AdaptiveTransferManager, ln, exp
)
import matplotlib.pyplot as plt
import numpy as np
from netgen.geom2d import SplineGeometry
import os
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.tri as mtri

# Parameters 
N = 4
p = 3/2
max_iters, tol_energy = 70, 1e-7
dorfler_theta = 0.2
max_refinements = 50

# Relaxation & estimator updates
eps_minus, eps_plus = 1.0, 1.0
theta_minus, theta_plus = 0.5, 1.25
delta_factor_minus, delta_factor_plus = 1e-6, 1e6

atm = AdaptiveTransferManager()



def grad_norm(w): return sqrt(inner(grad(w), grad(w)))

def clip(t, e_m, e_p): return max_value(min_value(t, e_p), e_m)

def phi(t):
    return (1 + t) * ln(1 + t) - t


def Phi(t):
    return phi(sqrt(t))

def mu(t):
    return t**(-1/2) * ln(1 + sqrt(t))


def kappa_eps(t, e_m, e_p):
    c = clip(t, e_m, e_p)
    return 0.5 * mu(c**2) * t**2 + Phi(c**2) - 0.5 * mu(c**2) * c**2

def phi_eps(t, e_m, e_p):
    return kappa_eps(t, e_m, e_p) - kappa_eps(0, e_m, e_p)

# where shift a>0 this true
def phi_eps_shift(t, e_m, e_p, a):
    return phi_eps(t, clip(a, e_m, e_p), e_p)

# antiderivative of phi' inverse 
def anti_der_phi_der_inv_raw(t):
    return exp(t) - t
def anti_der_phi_der_inv(t):
    return anti_der_phi_der_inv_raw(t) - anti_der_phi_der_inv_raw(0)

# phi eps dual
def phi_eps_dual(t, e_m, e_p):
    mu_m = mu(e_m**2)
    mu_p = mu(e_p**2)

    A = mu_m * e_m
    B = mu_p * e_p

    first = 0.5 * t**2 / mu_m

    middle = (
        0.5 * A**2 / mu_m
        + anti_der_phi_der_inv(t)
        - anti_der_phi_der_inv(A)
    )

    last = (
        0.5 * A**2 / mu_m
        + anti_der_phi_der_inv(B)
        - anti_der_phi_der_inv(A)
        + 0.5 * (t**2 - B**2) / mu_p
    )

    return conditional(
        t <= A,
        first,
        conditional(t <= B, middle, last)
    )

# phi eps shift dual (it is proven already)
def phi_eps_shift_dual(t, e_m, e_p, a):
    return phi_eps_dual(t, clip(a, e_m, e_p), e_p)


os.makedirs("refinement_output", exist_ok=True)

def save_mesh_and_solution(mesh, u, step=0, tag=""):
    plt.figure(figsize=(6,6))
    firedrake.triplot(mesh)
    plt.gca().set_aspect("equal")
    plt.title(f"Mesh {tag}")
    plt.savefig(
        f"refinement_output/mesh_{step:03d}_{tag}.png",
        dpi=200
    )
    plt.close()
    coords = mesh.coordinates.dat.data_ro
    x = coords[:, 0]
    y = coords[:, 1]
    z = u.dat.data_ro
    triang = mtri.Triangulation(
        x,
        y,
        mesh.coordinates.cell_node_map().values
    )
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection="3d")
    ax.plot_trisurf(
        triang,
        z,
        cmap="viridis",
        linewidth=0.0
    )
    ax.set_title(f"Solution {tag}")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("u")
    plt.tight_layout()
    plt.savefig(
        f"refinement_output/solution3d_{step:03d}_{tag}.png",
        dpi=200
    )

    plt.close()




def make_initial_mesh(N):
    geo = SplineGeometry()

    p0 = geo.AppendPoint(-1, -1)
    p1 = geo.AppendPoint(1, -1)
    p2 = geo.AppendPoint(1, 1)
    p3 = geo.AppendPoint(0, 1)
    p4 = geo.AppendPoint(0, 0)
    p5 = geo.AppendPoint(-1, 0)

    geo.Append(["line", p0, p1], bc="outer")
    geo.Append(["line", p1, p2], bc="outer")
    geo.Append(["line", p2, p3], bc="outer")
    geo.Append(["line", p3, p4], bc="outer")
    geo.Append(["line", p4, p5], bc="outer")
    geo.Append(["line", p5, p0], bc="outer")

    ngmesh = geo.GenerateMesh(maxh=2.0 / N)
    return firedrake.Mesh(ngmesh)

def build_problem(mesh):
    V = FunctionSpace(mesh, "CG", 1)
    X, Y = SpatialCoordinate(mesh)
    r = sqrt(X**2 + Y**2)
    alpha = 1 - 1/p
    u_exact_expr = r**alpha - 1
    u_exact = Function(V, name="u_exact").interpolate(u_exact_expr)
    grad_e = grad(u_exact_expr)
    f_expr = -div(mu(inner(grad_e, grad_e)) * grad_e)
    
    return V, u_exact_expr, u_exact, f_expr



#  Energies & Estimators 
def calc_energies(w, f_expr, e_m, e_p):
    g = grad_norm(w)
    J_exact = assemble(phi(g) * dx - f_expr * w * dx)
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
    argument = mu(alpha**2) * grad_norm(u_old - u_new)
    return assemble(phi_eps_shift_dual(argument, e_m, e_p, grad_norm(u_old)) * dx)

def eta_h_squared(w, f_expr, e_m, e_p):
    mesh = w.function_space().mesh()
    DG0 = FunctionSpace(mesh, "DG", 0)
    q = TestFunction(DG0)
    
    alpha = grad_norm(w)
    vol_density = phi_eps_shift_dual(CellDiameter(mesh) * sqrt(f_expr**2), e_m, e_p, alpha)
    
    Veps = sqrt(mu(clip(alpha, e_m, e_p)**2)) * grad(w)
    jump_density = avg(FacetArea(mesh)) * inner(jump(Veps), jump(Veps))
    
    eta2_T = Function(DG0)
    eta2_T.dat.data[:] = assemble(vol_density * q * dx).dat.data_ro[:]
    eta2_T.dat.data[:] += assemble(jump_density * (q("+") + q("-")) * dS).dat.data_ro[:]
    # We have added everyhting however need some clean up here because negative values might appear in practically small negatives we will use this
    # And paper multiplied 1e-3 we have to do also
    eta2_T.dat.data[:] = np.maximum(eta2_T.dat.data[:], 0.0) * 1e-3
    
    return float(np.sum(eta2_T.dat.data_ro)), eta2_T

# Dörfler Marking 
def dorfler_marking(eta2_T, theta):
    eta2 = np.maximum(eta2_T.dat.data_ro.copy(), 0.0)
    total = np.sum(eta2)
    marker = Function(eta2_T.function_space())
    marker.assign(0.0)
    
    if total <= 0.0: return marker, 0.0, total
    # index sorting from largest error to the smallest indecies
    order = np.argsort(-eta2)
    # cumelative sum
    accum = np.cumsum(eta2[order])
    # index of when accum reacs the target 
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
    coeff = mu(clip(grad_norm(u_old), eps_minus, eps_plus)**2)
    
    solve(
        inner(coeff * grad(u_trial), grad(v_test)) * dx == f_expr * v_test * dx,
        u_new, bcs=DirichletBC(V, u_exact_expr, "on_boundary"),
        solver_parameters={"pc_type": "lu", "pc_factor_mat_solver_type": "mumps"}
    )

    # Save current mesh and solution
    save_mesh_and_solution(
        mesh,
        u_new,
        step=n,
        tag=f"iter_{n}_refs_{num_refinements}"
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
            
            u_old = Function(V, name=f"u_after_ref_{num_refinements}")
            atm.prolong(u_new, u_old)
            bc.apply(u_old)

            # save refined mesh and prolonged solution
            save_mesh_and_solution(
                mesh,
                u_old,
                step=n,
                tag=f"iter_{n}_refs_{num_refinements}"
            )

        else:
            u_old.assign(u_new)
    else:  # largest == "k"
        u_old.assign(u_new)

# --- Visualization ---
history = np.array(history)
costs = history[:, 0]
ref_line = 25 * costs**(-1.0)

plt.figure(figsize=(10, 6))
plt.loglog(costs, np.abs(history[:, 1]), "ko-", label="energy_err")
plt.loglog(costs, history[:, 2], "go-", label="eta_k2")
plt.loglog(costs, history[:, 3], "ys-", label="eta_m2")
plt.loglog(costs, history[:, 4], "bs-", label="eta_p2")
plt.loglog(costs, history[:, 5], "ro-", label="eta_h2")
plt.loglog(costs, history[:, 6], "k:", label="e_m")
plt.loglog(costs, history[:, 7], "k--", label="e_p")
plt.loglog(costs, ref_line, "k-", label=r"$25 \times \text{cost}^{-1}$")

plt.xlabel("cost")
plt.legend(loc="lower left")
plt.tight_layout()
plt.savefig("main_orlicz_function.png", dpi=200)