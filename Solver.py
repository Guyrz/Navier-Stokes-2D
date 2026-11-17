"""
2D incompressible Navier-Stokes solver.

MAC staggered grid + Chorin projection method, explicit time stepping
with an adaptive (CFL + diffusion-limited) time step.

Problems:
  poiseuille  Pressure-driven channel flow, periodic in x.
              Validated against the analytical Poiseuille solution.
  cavity      Lid-driven cavity. Experimental, not validated (see README).

Usage:
  python Solver.py poiseuille
  python Solver.py cavity --t-end 20
  python Solver.py --help
"""

import argparse
import sys
from dataclasses import dataclass

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import factorized

# Fixed domain parameters
LX = 2.0
LY = 1.0
RHO = 1.0


@dataclass
class Config:
    problem: str          # "poiseuille" or "cavity"
    nx: int
    ny: int
    nu: float
    G: float = 0.0        # pressure gradient (Poiseuille forcing)
    u_lid: float = 0.0    # lid velocity (cavity)
    t_end: float = 150.0
    lx: float = LX
    ly: float = LY
    rho: float = RHO

    @property
    def dx(self):
        return self.lx / self.nx

    @property
    def dy(self):
        return self.ly / self.ny


# ── Grid and arrays ───────────────────────────────────────────────
def make_fields(cfg):
    """MAC staggered grid:
    u (nx+1, ny)  horizontal velocity on vertical cell edges
    v (nx, ny+1)  vertical velocity on horizontal cell edges
    p (nx, ny)    pressure at cell centres
    """
    u = np.zeros((cfg.nx + 1, cfg.ny))
    v = np.zeros((cfg.nx, cfg.ny + 1))
    p = np.zeros((cfg.nx, cfg.ny))
    apply_boundary_conditions(u, v, cfg)
    return u, v, p


# ── Boundary conditions ───────────────────────────────────────────
def apply_boundary_conditions(u, v, cfg):
    # No-slip top and bottom. The walls are imposed on the first and
    # last rows of u, so the wall-to-wall height is (ny - 1) * dy.
    u[:, 0] = 0.0
    u[:, -1] = 0.0
    v[:, 0] = 0.0
    v[:, -1] = 0.0

    if cfg.problem == "poiseuille":
        # Periodic in x
        u[0, :] = u[-2, :]
        u[-1, :] = u[1, :]
        v[0, :] = v[-2, :]
        v[-1, :] = v[1, :]
    elif cfg.problem == "cavity":
        # No-slip left and right
        u[0, :] = 0.0
        u[-1, :] = 0.0
        v[0, :] = 0.0
        v[-1, :] = 0.0
        # Moving lid on the top wall
        u[:, -1] = cfg.u_lid


# ── Diffusion ─────────────────────────────────────────────────────
def diffuse(u, v, nu, dt, dx, dy):
    u_new = u.copy()
    u_new[1:-1, 1:-1] = u[1:-1, 1:-1] + nu * dt * (
        (u[2:, 1:-1] - 2 * u[1:-1, 1:-1] + u[:-2, 1:-1]) / dx**2
        + (u[1:-1, 2:] - 2 * u[1:-1, 1:-1] + u[1:-1, :-2]) / dy**2
    )
    v_new = v.copy()
    v_new[1:-1, 1:-1] = v[1:-1, 1:-1] + nu * dt * (
        (v[2:, 1:-1] - 2 * v[1:-1, 1:-1] + v[:-2, 1:-1]) / dx**2
        + (v[1:-1, 2:] - 2 * v[1:-1, 1:-1] + v[1:-1, :-2]) / dy**2
    )
    return u_new, v_new


# ── Advection (first-order upwind) ────────────────────────────────
def advect(u, v, dt, dx, dy):
    u_new = u.copy()
    v_new = v.copy()

    # Interpolate the other velocity component to each u / v location
    v_at_u = 0.25 * (v[:-1, 1:-2] + v[:-1, 2:-1] + v[1:, 1:-2] + v[1:, 2:-1])
    u_at_v = 0.25 * (u[1:-2, :-1] + u[1:-2, 1:] + u[2:-1, :-1] + u[2:-1, 1:])

    dudx_b = (u[1:-1, 1:-1] - u[:-2, 1:-1]) / dx
    dudx_f = (u[2:, 1:-1] - u[1:-1, 1:-1]) / dx
    dudy_b = (u[1:-1, 1:-1] - u[1:-1, :-2]) / dy
    dudy_f = (u[1:-1, 2:] - u[1:-1, 1:-1]) / dy
    dudx = np.where(u[1:-1, 1:-1] > 0, dudx_b, dudx_f)
    dudy = np.where(v_at_u > 0, dudy_b, dudy_f)
    u_new[1:-1, 1:-1] -= dt * (u[1:-1, 1:-1] * dudx + v_at_u * dudy)

    dvdx_b = (v[1:-1, 1:-1] - v[:-2, 1:-1]) / dx
    dvdx_f = (v[2:, 1:-1] - v[1:-1, 1:-1]) / dx
    dvdy_b = (v[1:-1, 1:-1] - v[1:-1, :-2]) / dy
    dvdy_f = (v[1:-1, 2:] - v[1:-1, 1:-1]) / dy
    dvdx = np.where(u_at_v > 0, dvdx_b, dvdx_f)
    dvdy = np.where(v[1:-1, 1:-1] > 0, dvdy_b, dvdy_f)
    v_new[1:-1, 1:-1] -= dt * (u_at_v * dvdx + v[1:-1, 1:-1] * dvdy)

    return u_new, v_new


# ── Pressure Poisson equation ─────────────────────────────────────
def build_poisson_matrix(nx, ny, dx, dy):
    """5-point Laplacian on cell centres, Neumann BCs on all sides,
    pressure pinned at cell (0, 0) for a unique solution."""
    n = nx * ny
    A = sparse.lil_matrix((n, n))

    def idx(i, j):
        return j * nx + i

    for j in range(ny):
        for i in range(nx):
            k = idx(i, j)
            A[k, k] = -2 / dx**2 - 2 / dy**2

            if i > 0:
                A[k, idx(i - 1, j)] = 1 / dx**2
            else:
                A[k, k] += 1 / dx**2      # Neumann: left
            if i < nx - 1:
                A[k, idx(i + 1, j)] = 1 / dx**2
            else:
                A[k, k] += 1 / dx**2      # Neumann: right
            if j > 0:
                A[k, idx(i, j - 1)] = 1 / dy**2
            else:
                A[k, k] += 1 / dy**2      # Neumann: bottom
            if j < ny - 1:
                A[k, idx(i, j + 1)] = 1 / dy**2
            else:
                A[k, k] += 1 / dy**2      # Neumann: top

    A[0, :] = 0
    A[0, 0] = 1
    return A.tocsc()


def make_pressure_solver(cfg):
    """Factorise the Poisson matrix once; it does not change between steps."""
    return factorized(build_poisson_matrix(cfg.nx, cfg.ny, cfg.dx, cfg.dy))


def divergence(u, v, dx, dy):
    return (u[1:, :] - u[:-1, :]) / dx + (v[:, 1:] - v[:, :-1]) / dy


def solve_pressure(solve, u, v, rho, dt, dx, dy):
    nx, ny = u.shape[0] - 1, u.shape[1]
    b = (rho / dt) * divergence(u, v, dx, dy).flatten(order="F")  # matches idx(i,j)=j*nx+i
    b[0] = 0
    return solve(b).reshape((nx, ny), order="F")


# ── Projection ────────────────────────────────────────────────────
def project(u, v, p, rho, dt, dx, dy):
    """Subtract the pressure gradient so the velocity is divergence-free."""
    u_new = u.copy()
    v_new = v.copy()
    u_new[1:-1, :] -= (dt / rho) * (p[1:, :] - p[:-1, :]) / dx
    v_new[:, 1:-1] -= (dt / rho) * (p[:, 1:] - p[:, :-1]) / dy
    return u_new, v_new


# ── Time stepping ─────────────────────────────────────────────────
def compute_dt(u, v, nu, dx, dy, safety=0.4):
    h = min(dx, dy)
    max_vel = max(np.abs(u).max(), np.abs(v).max(), 1e-6)
    dt_adv = h / max_vel
    dt_diff = h**2 / (4 * nu)
    return safety * min(dt_adv, dt_diff)


def step(u, v, cfg, dt, solve):
    if cfg.problem == "cavity":
        u_star, v_star = advect(u, v, dt, cfg.dx, cfg.dy)
    else:
        # Fully developed Poiseuille flow: the advection term is zero
        u_star, v_star = u.copy(), v.copy()
    u_star, v_star = diffuse(u_star, v_star, cfg.nu, dt, cfg.dx, cfg.dy)
    u_star[1:-1, :] += dt * cfg.G / cfg.rho
    apply_boundary_conditions(u_star, v_star, cfg)
    p_new = solve_pressure(solve, u_star, v_star, cfg.rho, dt, cfg.dx, cfg.dy)
    u_new, v_new = project(u_star, v_star, p_new, cfg.rho, dt, cfg.dx, cfg.dy)
    apply_boundary_conditions(u_new, v_new, cfg)
    return u_new, v_new, p_new


def run(cfg, snap_every=50, log_every=1000, verbose=True):
    """Run to t_end. Returns final (u, v, p) and lists of snapshots."""
    solve = make_pressure_solver(cfg)
    u, v, p = make_fields(cfg)
    t, n = 0.0, 0
    snaps_u, snaps_v, snap_t = [u.copy()], [v.copy()], [0.0]

    while t < cfg.t_end:
        dt = compute_dt(u, v, cfg.nu, cfg.dx, cfg.dy)
        u, v, p = step(u, v, cfg, dt, solve)
        t += dt
        n += 1

        if np.isnan(u).any() or np.abs(u).max() > 1e6:
            raise RuntimeError(
                f"Solution diverged at t={t:.3f}, step={n}. "
                "Try lower G / U, higher nu, or a finer grid.")

        if n % snap_every == 0:
            snaps_u.append(u.copy())
            snaps_v.append(v.copy())
            snap_t.append(t)

        if verbose and n % log_every == 0:
            div_max = np.abs(divergence(u, v, cfg.dx, cfg.dy)[1:-1, 1:-1]).max()
            print(f"  t={t:8.3f}  step={n:6d}  dt={dt:.5f}  "
                  f"u_max={u.max():.4f}  div={div_max:.2e}")

    if verbose:
        print(f"Done: {n} steps, t={t:.3f}")
    return u, v, p, snaps_u, snaps_v, snap_t


# ── Validation ────────────────────────────────────────────────────
def poiseuille_exact(G, nu, ny, dy):
    """Analytical Poiseuille profile u(y) = G/(2 nu) * y * (H - y),
    sampled at y = j*dy, with the walls where the solver imposes them
    (first and last u rows), i.e. channel height H = (ny - 1) * dy."""
    j = np.arange(ny)
    return G * dy**2 / (2 * nu) * j * (ny - 1 - j)


def poiseuille_error(u, cfg):
    u_num = u[cfg.nx // 2, :]
    u_ex = poiseuille_exact(cfg.G, cfg.nu, cfg.ny, cfg.dy)
    return np.abs(u_num - u_ex).max() / u_ex.max(), u_num, u_ex


def reynolds(cfg):
    if cfg.problem == "poiseuille":
        H = (cfg.ny - 1) * cfg.dy
        u_max = cfg.G * H**2 / (8 * cfg.nu)
        return u_max * H / cfg.nu
    return cfg.u_lid * cfg.ly / cfg.nu


# ── Visualisation ─────────────────────────────────────────────────
def visualise(cfg, snaps_u, snaps_v, snap_t, u_ex, rel_error, save_dir=None, gif=False, show=True):
    import matplotlib
    if not show:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.animation as animation
    import os

    nx, ny, dx, dy = cfg.nx, cfg.ny, cfg.dx, cfg.dy
    # u rows sit at y = j*dy measured from the bottom wall row
    y_u = np.arange(ny) * dy
    x_c = (np.arange(nx) + 0.5) * dx
    H = (ny - 1) * dy

    def cc_u(u):
        return 0.5 * (u[:-1, :] + u[1:, :])

    def cc_v(v):
        return 0.5 * (v[:, :-1] + v[:, 1:])

    title = ("2D Poiseuille Flow — Development" if cfg.problem == "poiseuille"
             else "2D Lid-Driven Cavity — Development (experimental)")
    fig, (ax_vel, ax_prof) = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle(title, fontsize=13)

    u_vmax = u_ex.max() if u_ex is not None else max(np.abs(s).max() for s in snaps_u) * 1.05
    skip = 3
    Xq, Yq = np.meshgrid(x_c[::skip], y_u[::skip], indexing="ij")

    im = ax_vel.imshow(cc_u(snaps_u[0]).T, origin="lower", extent=[0, cfg.lx, 0, H],
                       aspect="auto", cmap="viridis", interpolation="bilinear",
                       vmin=min(0.0, min(s.min() for s in snaps_u)), vmax=u_vmax)
    quiv = ax_vel.quiver(Xq, Yq, cc_u(snaps_u[0])[::skip, ::skip].T,
                         cc_v(snaps_v[0])[::skip, ::skip].T,
                         color="white", scale=50, width=0.003, alpha=0.6)
    ax_vel.set_xlabel("x")
    ax_vel.set_ylabel("y (from bottom wall)")
    ax_vel.set_title("velocity field u")
    plt.colorbar(im, ax=ax_vel, label="u")

    if u_ex is not None:
        ax_prof.fill_betweenx(y_u, 0, u_ex, alpha=0.15, color="red", label="exact envelope")
        ax_prof.plot(u_ex, y_u, "r--", lw=1.5, alpha=0.5, label="exact")
    line_num, = ax_prof.plot(snaps_u[0][nx // 2, :], y_u, "b-", lw=2, label="numerical")
    ax_prof.set_xlabel("u")
    ax_prof.set_ylabel("y (from bottom wall)")
    ax_prof.set_title("cross-section at x = L/2")
    ax_prof.set_xlim(min(0, min(s.min() for s in snaps_u)) * 1.15, u_vmax * 1.15)
    ax_prof.set_ylim(0, H)
    ax_prof.legend(loc="upper right", fontsize=9)
    time_text = ax_prof.text(0.05, 0.92, "", transform=ax_prof.transAxes, fontsize=10, va="top")
    err_text = ax_prof.text(0.05, 0.84, "", transform=ax_prof.transAxes, fontsize=9,
                            color="gray", va="top")

    def update(k):
        u_cc = cc_u(snaps_u[k])
        im.set_data(u_cc.T)
        quiv.set_UVC(u_cc[::skip, ::skip].T, cc_v(snaps_v[k])[::skip, ::skip].T)
        prof = snaps_u[k][nx // 2, :]
        line_num.set_xdata(prof)
        time_text.set_text(f"t = {snap_t[k]:.1f}")
        if u_ex is not None:
            err_text.set_text(f"rel err = {np.abs(prof - u_ex).max() / u_ex.max():.4f}")
        return im, quiv, line_num, time_text, err_text

    # At most ~150 frames, so the animation and GIF stay light
    frames = np.unique(np.linspace(0, len(snaps_u) - 1, 150).astype(int))
    ani = animation.FuncAnimation(fig, update, frames=frames, interval=50,
                                  blit=True, repeat=True)
    fig.tight_layout()

    fig2 = None
    if u_ex is not None:
        fig2, ax2 = plt.subplots(figsize=(6, 5))
        ax2.fill_betweenx(y_u, 0, u_ex, alpha=0.15, color="red", label="exact envelope")
        ax2.plot(u_ex, y_u, "r--", lw=2, alpha=0.6, label="exact")
        ax2.plot(snaps_u[-1][nx // 2, :], y_u, "b-", lw=2,
                 label=f"numerical (t = {snap_t[-1]:.0f})")
        ax2.set_xlabel("u")
        ax2.set_ylabel("y (from bottom wall)")
        ax2.set_title("Poiseuille Validation — Numerical vs Exact")
        ax2.legend()
        ax2.text(0.55, 0.06, f"max rel error = {rel_error:.1e}", transform=ax2.transAxes,
                 fontsize=10, bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))
        fig2.tight_layout()

    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        update(len(snaps_u) - 1)
        fig.savefig(os.path.join(save_dir, f"{cfg.problem}_final.png"), dpi=120)
        if fig2 is not None:
            fig2.savefig(os.path.join(save_dir, "poiseuille_validation.png"), dpi=120)
        if gif:
            ani.save(os.path.join(save_dir, f"{cfg.problem}.gif"),
                     writer=animation.PillowWriter(fps=20), dpi=60)
        print(f"Figures saved to {save_dir}/")

    if show:
        plt.show()


# ── Command line ──────────────────────────────────────────────────
def parse_args(argv=None):
    ap = argparse.ArgumentParser(description="2D incompressible Navier-Stokes solver (MAC grid, projection).")
    ap.add_argument("problem", choices=["poiseuille", "cavity"])
    ap.add_argument("--G", type=float, default=1.0, help="pressure gradient (poiseuille), default 1.0")
    ap.add_argument("--u-lid", type=float, default=1.0, help="lid velocity (cavity), default 1.0")
    ap.add_argument("--nu", type=float, default=0.01, help="kinematic viscosity, default 0.01")
    ap.add_argument("--nx", type=int, default=40, help="cells in x, default 40")
    ap.add_argument("--ny", type=int, default=None, help="cells in y, default 20 (poiseuille) / 40 (cavity)")
    ap.add_argument("--t-end", type=float, default=None, help="end time, default 150 (poiseuille) / 50 (cavity)")
    ap.add_argument("--save", metavar="DIR", help="save figures to DIR")
    ap.add_argument("--gif", action="store_true", help="also save an animated GIF (needs --save)")
    ap.add_argument("--no-show", action="store_true", help="do not open plot windows")
    return ap.parse_args(argv)


def main(argv=None):
    a = parse_args(argv)
    if a.problem == "poiseuille":
        cfg = Config("poiseuille", a.nx, a.ny or 20, a.nu, G=a.G, t_end=a.t_end or 150.0)
    else:
        cfg = Config("cavity", a.nx, a.ny or 40, a.nu, u_lid=a.u_lid, t_end=a.t_end or 50.0)

    if cfg.nu <= 0:
        sys.exit("Error: nu must be positive.")
    if cfg.problem == "poiseuille" and cfg.G <= 0:
        sys.exit("Error: G must be positive.")
    if cfg.problem == "cavity" and cfg.u_lid <= 0:
        sys.exit("Error: u-lid must be positive.")

    Re = reynolds(cfg)
    print(f"Problem: {cfg.problem}   grid {cfg.nx}x{cfg.ny}   dx={cfg.dx:.4f} dy={cfg.dy:.4f}")
    print(f"Re = {Re:.1f}   running to t = {cfg.t_end}")
    if cfg.problem == "poiseuille":
        H = (cfg.ny - 1) * cfg.dy
        u_ex_max = poiseuille_exact(cfg.G, cfg.nu, cfg.ny, cfg.dy).max()
        print(f"Channel height (wall to wall) H = (ny-1)*dy = {H:.4f}")
        print(f"Expected steady u_max on the grid rows = {u_ex_max:.4f}")
    else:
        print("Note: lid-driven cavity is experimental and not validated.")

    u, v, p, snaps_u, snaps_v, snap_t = run(cfg)

    u_ex, rel_error = None, None
    if cfg.problem == "poiseuille":
        rel_error, u_num, u_ex = poiseuille_error(u, cfg)
        print("\nPoiseuille validation (centre-line profile at x = L/2):")
        print(f"  u_max numerical: {u_num.max():.6f}")
        print(f"  u_max exact:     {u_ex.max():.6f}")
        print(f"  max rel error:   {rel_error:.2e}   (target < 1e-3)")
        print("  PASSED" if rel_error < 1e-3 else "  FAILED")

    if a.save or not a.no_show:
        visualise(cfg, snaps_u, snaps_v, snap_t, u_ex, rel_error,
                  save_dir=a.save, gif=a.gif, show=not a.no_show)


if __name__ == "__main__":
    main()
