"""
export_svg.py — render a saved orbit (data/<name>.npz) to an SVG.

Usage:
    python export_svg.py                            # most recent data file, pretty style
    python export_svg.py --name testbody_circular_2PM
    python export_svg.py --style grid               # grid + labels + legend
    python export_svg.py --output plots/my.svg
    python export_svg.py --name 2PM --overlay 1PM N # overlay several models
"""
import argparse
import glob
import os

import matplotlib
matplotlib.use("Agg")  # headless-safe; we only write files, never show
import matplotlib.pyplot as plt  # noqa: E402  (import after backend selection)

import physics  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))


def latest_data(indir):
    files = glob.glob(os.path.join(indir, "*.npz"))
    if not files:
        raise SystemExit(f"No .npz files found in {indir}. Run orbit_sim.py first.")
    return max(files, key=os.path.getmtime)


def main():
    ap = argparse.ArgumentParser(description="Render a saved orbit to SVG.")
    ap.add_argument("--name", default=None, help="data name (data/<name>.npz); default = most recent")
    ap.add_argument("--indir", default=os.path.join(HERE, "data"))
    ap.add_argument("--output", default=None, help="output SVG path (default plots/<name>.svg)")
    ap.add_argument("--style", choices=["pretty", "grid"], default="pretty")
    ap.add_argument("--overlay", nargs="*", default=[], help="extra data names to draw on the same figure")
    ap.add_argument("-f", "--force", action="store_true", help="overwrite an existing SVG without prompting")
    args = ap.parse_args()

    indir = args.indir
    if args.name is None:
        name = os.path.splitext(os.path.basename(latest_data(indir)))[0]
    else:
        name = args.name

    pretty = args.style == "pretty"
    orbit = physics.load_orbits(name, indir)
    positions = orbit["positions"]
    title = orbit["case"].get("name", name)

    if args.overlay:
        fig, ax = plt.subplots(figsize=(11, 11))
        physics.plot_orbit(positions, ax=ax, pretty=pretty, title=title)
        for extra in args.overlay:
            o = physics.load_orbits(extra, indir)
            physics.plot_orbit(o["positions"], ax=ax, pretty=pretty)
    else:
        fig = physics.plot_orbit(positions, pretty=pretty, title=title)

    outdir = os.path.join(HERE, "plots")
    os.makedirs(outdir, exist_ok=True)
    out = args.output or os.path.join(outdir, f"{name}.svg")

    if os.path.exists(out) and not args.force:
        try:
            ans = input(f"  {out} already exists. Overwrite? [y/N] ").strip().lower()
        except EOFError:
            ans = "n"
        if ans not in ("y", "yes"):
            print("aborted - nothing written")
            return

    fig.savefig(out, format="svg")
    print(f"saved SVG: {out}")


if __name__ == "__main__":
    main()
