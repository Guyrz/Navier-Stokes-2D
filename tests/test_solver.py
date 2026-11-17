import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from Solver import (Config, advect, diffuse, divergence,
                    make_pressure_solver, poiseuille_error, project, run, solve_pressure)


def small_cfg(problem="poiseuille"):
    return Config(problem, nx=10, ny=10, nu=0.01, G=1.0, u_lid=1.0, t_end=150.0)


def test_diffusion_spreads_and_conserves():
    cfg = small_cfg()
    u = np.zeros((cfg.nx + 1, cfg.ny))
    v = np.zeros((cfg.nx, cfg.ny + 1))
    u[cfg.nx // 2, cfg.ny // 2] = 1.0
    dt = 0.25 * min(cfg.dx, cfg.dy) ** 2 / cfg.nu
    u2, _ = diffuse(u, v, cfg.nu, dt, cfg.dx, cfg.dy)
    assert u2.max() < 1.0
    assert np.isclose(u2.sum(), u.sum())


def test_uniform_flow_is_unchanged_by_advection():
    cfg = small_cfg()
    u = np.full((cfg.nx + 1, cfg.ny), 0.5)
    v = np.zeros((cfg.nx, cfg.ny + 1))
    u2, v2 = advect(u, v, 0.1, cfg.dx, cfg.dy)
    assert np.abs(u2 - u).max() < 1e-12
    assert np.abs(v2 - v).max() < 1e-12


def test_projection_removes_divergence():
    cfg = small_cfg("cavity")
    rng = np.random.default_rng(0)
    u = rng.normal(size=(cfg.nx + 1, cfg.ny))
    v = rng.normal(size=(cfg.nx, cfg.ny + 1))
    u[0, :] = u[-1, :] = 0.0
    v[:, 0] = v[:, -1] = 0.0
    solve = make_pressure_solver(cfg)
    dt = 0.01
    p = solve_pressure(solve, u, v, cfg.rho, dt, cfg.dx, cfg.dy)
    u2, v2 = project(u, v, p, cfg.rho, dt, cfg.dx, cfg.dy)
    div = divergence(u2, v2, cfg.dx, cfg.dy)
    # cell (0, 0) holds the pressure pin, so its equation is not enforced
    assert np.abs(div.flatten(order="F")[1:]).max() < 1e-8


def test_poiseuille_matches_analytical_solution():
    cfg = small_cfg()
    u, *_ = run(cfg, verbose=False)
    rel_error, _, _ = poiseuille_error(u, cfg)
    assert rel_error < 1e-3


def test_cavity_runs_without_diverging():
    cfg = small_cfg("cavity")
    cfg.t_end = 2.0
    u, v, *_ = run(cfg, verbose=False)
    assert np.isfinite(u).all() and np.isfinite(v).all()
    assert np.allclose(u[:, -1], cfg.u_lid)
