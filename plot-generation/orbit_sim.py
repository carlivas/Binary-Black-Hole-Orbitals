"""
orbit_sim.py — generate binary black-hole orbital data from a config file.

Usage:
    python orbit_sim.py                 # uses config.json, saves data/<name>.npz
    python orbit_sim.py --preview       # also pop up the orbit in a window
    python orbit_sim.py --name myrun    # override the output name
    python orbit_sim.py --model 1PM     # override the model
    python orbit_sim.py --res 10000     # override the resolution (faster)
    python orbit_sim.py --config other.json

Edit config.json to set the initial conditions (masses, SMA, alpha/eccentricity,
years) and the gravitational model. The post-Minkowskian models (1PM/2PM) are the
slow ones — use a lower --res while iterating.
"""
import argparse
import json
import os
import time

import numpy as np

import physics

HERE = os.path.dirname(os.path.abspath(__file__))


def load_config(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def orbital_period(mass_bin, r, v):
    """Keplerian orbital period from (r, v) via vis-viva (G=1 geometric units).

    Returns (T, a): the period and the true semi-major axis (a<=0 if unbound).
    For an eccentric orbit the true a differs from the initial separation r, so
    using a (not r) is what makes 'years' correspond to actual orbit counts.
    """
    a = 1.0 / (2.0 / r - v**2 / mass_bin)  # semi-major axis; a<=0 => unbound
    if a > 0:
        T = 2 * np.pi * np.sqrt(a**3 / mass_bin)
    else:  # unbound/hyperbolic: fall back to a circular-period window
        T = 2 * np.pi * np.sqrt(r**3 / mass_bin)
    return T, a


def build_initial_conditions(cfg):
    """COM-frame initial conditions.

    Two input modes (mutually exclusive; rel_ic wins if present):
      A) SMA_bin + alpha  -> circular/eccentric orbit, same as Cases_All.ipynb
      B) rel_ic           -> explicit relative separation r_vec and relative
                             velocity v_vec in the COM frame
    Returns (s, s_cm, R_S, t_max, T) where s = (x1, x2, v1, v2, m1, m2) and
    T is the single-orbit period (t_max = T * years).
    """
    m1, m2 = cfg["m1"], cfg["m2"]
    years = cfg.get("years", 1)

    mass_bin = m1 + m2
    R_S = 2 * mass_bin

    rel_ic = cfg.get("rel_ic")
    if rel_ic is not None:
        r_vec = np.array([rel_ic["r_x"], rel_ic["r_y"]], dtype=float)
        v_vec = np.array([rel_ic["v_x"], rel_ic["v_y"]], dtype=float)

        x_1 = (m2 / mass_bin) * r_vec
        x_2 = -(m1 / mass_bin) * r_vec
        v_1 = (m2 / mass_bin) * v_vec
        v_2 = -(m1 / mass_bin) * v_vec

        s = x_1, x_2, v_1, v_2, m1, m2
        s_cm = physics.sToCOM(s)

        r = np.linalg.norm(r_vec)
        v = np.linalg.norm(v_vec)
        T, a = orbital_period(mass_bin, r, v)
        if a <= 0:
            print("warning: rel_ic looks unbound (a<=0); using circular-period estimate for t_max")
        return s, s_cm, R_S, T * years, T

    SMA = cfg["SMA_bin"]
    alpha = cfg.get("alpha", 1.0)
    v_redmass = alpha * np.sqrt(mass_bin / SMA)

    x_1 = np.array([(m2 / mass_bin) * SMA, 0.0])
    x_2 = np.array([-(m1 / mass_bin) * SMA, 0.0])
    v_1 = np.array([0.0, (m2 / mass_bin) * v_redmass])
    v_2 = np.array([0.0, -(m1 / mass_bin) * v_redmass])

    s = x_1, x_2, v_1, v_2, m1, m2
    s_cm = physics.sToCOM(s)

    # True period from the ICs (r=SMA, v=v_redmass), not from SMA directly
    T, a = orbital_period(mass_bin, SMA, v_redmass)
    if a <= 0:
        print("warning: alpha >= sqrt(2) is unbound; using circular-period estimate for t_max")
    return s, s_cm, R_S, T * years, T


def main():
    ap = argparse.ArgumentParser(description="Generate orbital data from a config file.")
    ap.add_argument("--config", default=os.path.join(HERE, "config.json"))
    ap.add_argument("--name", default=None, help="output name (overrides config 'name')")
    ap.add_argument("--model", default=None, help="override model (N, N_rel, EIH, 1PN, 1PM, 2PM)")
    ap.add_argument("--res", type=int, default=None, help="override time resolution")
    ap.add_argument("--preview", action="store_true", help="pop up the orbit in a window")
    ap.add_argument("-f", "--force", action="store_true", help="overwrite an existing data file without prompting")
    args = ap.parse_args()

    cfg = load_config(args.config)
    if args.name is not None:
        cfg["name"] = args.name
    if args.model is not None:
        cfg["model"] = args.model
    if args.res is not None:
        cfg["res"] = args.res

    if cfg.get("rel_ic") is None and "SMA_bin" not in cfg:
        raise SystemExit("config must define either 'rel_ic' (relative ICs) or 'SMA_bin' (+ 'alpha').")

    name = cfg["name"]
    model = cfg["model"]
    res = int(cfg.get("res", 1e5))
    DL = bool(cfg.get("DL", True))

    s, s_cm, R_S, t_max, T = build_initial_conditions(cfg)
    dt = t_max / res

    print(f"case   : {name}")
    print(f"model  : {model}")
    if cfg.get("rel_ic") is not None:
        r_ = cfg["rel_ic"]
        print(f"m1, m2 : {cfg['m1']}, {cfg['m2']}   rel_ic r=({r_['r_x']}, {r_['r_y']})  v=({r_['v_x']}, {r_['v_y']})")
    else:
        print(f"m1, m2 : {cfg['m1']}, {cfg['m2']}   SMA = {cfg['SMA_bin']}   alpha = {cfg.get('alpha', 1.0)}")
    print(f"period : {T:.4g}   t_max = {t_max:.4g} (~{t_max / T:.2f} orbits)   res = {res}   dt = {dt:.4g}")

    outdir = os.path.join(HERE, "data")
    npz_path = os.path.join(outdir, f"{name}.npz")
    if os.path.exists(npz_path) and not args.force:
        try:
            ans = input(f"  {npz_path} already exists. Overwrite? [y/N] ").strip().lower()
        except EOFError:
            ans = "n"
        if ans not in ("y", "yes"):
            print("aborted - nothing written")
            return

    t0 = time.time()
    out = physics.run_model(s_cm, cfg["m1"], cfg["m2"], t_max=t_max, dt=dt, R_S=R_S, d=5, mode=model, DL=DL)
    if out is None:
        raise SystemExit(f"run_model failed for mode={model!r}")
    t, positions, velocities, r, t_max_achieved, sol = out
    wall = time.time() - t0
    print(f"done in {wall:.2f}s   -> {len(t)} points, t_max_achieved = {t_max_achieved:.4g}")

    npz, jpath = physics.save_orbits(name, outdir, cfg, t, positions, velocities, r, t_max_achieved)
    print(f"saved  : {npz}")
    print(f"        {jpath}")

    if args.preview:
        physics.plot_orbit(positions, pretty=True, show=True, title=f"{name}  ({model})")


if __name__ == "__main__":
    main()
