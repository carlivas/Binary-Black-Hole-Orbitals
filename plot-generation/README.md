# plot-generation

A small, self-contained tool for generating **binary black-hole orbital data** and
rendering the resulting **orbital trajectories as SVG** plots.

This is the "bi-product" of the bachelor project: the post-Minkowskian (and
companion) relativistic two-body simulations produce beautiful orbital curves.
This folder isolates just the parts needed to (1) generate the data and (2)
export beautiful SVGs, so you can iterate on plots without re-deriving anything.

The physics is ported verbatim from `05 POST MINKOWSKIAN EXPANSION/Functions_All.ipynb`.

## Setup

```bash
pip install -r requirements.txt
```

Dependencies: `numpy`, `scipy`, `matplotlib`, `astropy`.

## Workflow

1. **Edit `config.json`** to set the initial conditions and the model.
2. **Generate data** (and optionally preview it):
   ```bash
   python orbit_sim.py --preview
   ```
   This runs the simulation and saves `data/<name>.npz` (+ a `.json` sidecar with
   the parameters used). `--preview` pops up the orbit in a window so you can
   eyeball it and iterate on `config.json`.
3. **Export to SVG** once you're happy with the data:
   ```bash
   python export_svg.py
   ```
   This reads the most recent data file and writes `plots/<name>.svg`.

Because the post-Minkowskian models are computationally heavy, the slow
**data generation** is kept separate from the fast **SVG rendering** — re-export
a plot (or change its style) without re-running the simulation.

## Config (`config.json`)

| field     | meaning                                                        | example |
|-----------|----------------------------------------------------------------|---------|
| `name`    | output name (`data/<name>.npz`, `plots/<name>.svg`)            | `"testbody_circular_2PM"` |
| `model`   | gravitational model: `N`, `N_rel`, `EIH`, `1PN`, `1PM`, `2PM`  | `"2PM"` |
| `m1`      | mass of body 1 `[M_sun]`                                       | `1e-4`  |
| `m2`      | mass of body 2 `[M_sun]`                                       | `1`     |
| `SMA_bin` | semi-major axis of the relative orbit `[G M_sun / c^2]` (mode A) | `1000`  |
| `alpha`   | velocity scaling; `1` = circular, `<1` = eccentric (mode A)     | `1.0`   |
| `rel_ic`  | explicit relative ICs (mode B): `{r_x, r_y, v_x, v_y}`          | see below |
| `years`   | number of orbital periods to simulate                          | `5`     |
| `res`     | time resolution (number of steps); lower = faster              | `100000`|
| `DL`      | `true` = geometrized units `c = G = 1` (all pretty plots)      | `true`  |

There are two ways to set the initial conditions (both in the center-of-mass
frame). If `rel_ic` is present it wins; otherwise `SMA_bin` + `alpha` are used.

- **Mode A — `SMA_bin` + `alpha`** (convenient): built exactly as in
  `Cases_All.ipynb`: bodies start on the x-axis with purely tangential velocity
  `alpha * sqrt(M/SMA)`, so `alpha = 1` is a circular orbit and `alpha < 1` is
  eccentric.
- **Mode B — `rel_ic`** (explicit): give the relative separation and relative
  velocity of the two bodies directly:
  ```json
  "rel_ic": { "r_x": 1000.0, "r_y": 0.0, "v_x": 0.0, "v_y": 0.0207846 }
  ```

In **both** modes the integration window is `years` × the true orbital period,
where the period is computed from the ICs via vis-viva using the **true
semi-major axis** `a` (not the initial separation). This matters for eccentric
orbits: the initial separation is the apoapsis, which is larger than `a`, so
using it would make "1 year" span several orbits. With the true `a`, `years = n`
gives roughly `n` orbits. Unbound ICs (`a <= 0`) fall back to a circular-period
window with a warning. The console output reports the period and orbit count.

## CLI

`orbit_sim.py`:
```
python orbit_sim.py [--config PATH] [--name NAME] [--model MODE] [--res N] [--preview] [-f|--force]
```
- If `data/<name>.npz` already exists you're prompted `Overwrite? [y/N]` **before**
  the (slow) simulation runs; answer anything other than `y`/`yes` (or run
  non-interactively) to abort. Use `-f`/`--force` to overwrite without prompting.

`export_svg.py`:
```
python export_svg.py [--name NAME] [--style pretty|grid] [--output PATH] [--overlay NAME ...] [-f|--force]
```
- `--style pretty` (default): clean, no axes/ticks — the "beautiful" look.
- `--style grid`: grid + axis labels + legend (like the original `orbPlotter`).
- `--overlay a b c`: draw several saved models on one SVG (handy for comparing
  periastron precession across N / 1PN / 1PM / 2PM).
- If the output SVG already exists you're prompted `Overwrite? [y/N]`; answer
  anything other than `y`/`yes` (or run non-interactively) to abort. Use
  `-f`/`--force` to overwrite without prompting.

## Notes

- **Units:** with `DL = true` the system is in geometrized units (`c = G = 1`),
  so distances are in `G M_sun / c^2` and times in `G M_sun / c^3`.
- **Runtime:** `1PM`/`2PM` with the DOP853 integrator are the slow cases
  (~1 min at `res = 1e5`, ~10 min at `res = 1e6` in the original notes). Use a
  lower `res` (e.g. `1e4`) while iterating, then bump it up for the final export.
- **Events:** the integrator stops if the bodies plunge to `R_S`, escape to
  `5 * r0`, or go superluminal. The saved data reflects whatever was actually
  integrated.
- Generated artifacts (`data/`, `plots/`) are git-ignored.
