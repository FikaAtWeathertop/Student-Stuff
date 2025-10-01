#
# analyze_vec.py
# Compute IGGs (Tx gaps) and IPGs (Rx gaps) from OMNeT++ .vec files.

import argparse
import math
import os
from pathlib import Path
from collections import defaultdict
from statistics import mean, pstdev  # population std; switch to stdev if you prefer sample std

def parse_vec_file(path: Path):
    """
    Returns a dict: vector_id -> list of (event, time, value).
    Accepts two common data layouts:
      1) Four columns:  vector  event  time  value
      2) Three columns after a 'vector <id> ...' header:  event  time  value
    Ignores lines starting with 'attr', 'vector' (but remembers last id), '#', or blank lines.
    """
    data = defaultdict(list)
    current_vector = None

    with path.open("r", encoding="utf-8", errors="ignore") as f:
        for raw in f:
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            # vector header
            if line.startswith("vector "):
                # format: "vector <id> ..."  -> we remember the id
                parts = line.split()
                try:
                    current_vector = int(parts[1])
                except Exception:
                    current_vector = None
                continue
            # fluff/meta
            if line.startswith("attr "):
                continue

            # Try numeric rows (tab or space separated)
            parts = line.split()
            # Only accept rows with 3 or 4 numeric fields
            if len(parts) not in (3, 4):
                continue
            try:
                nums = [float(x) if i in (2, 3) else int(x)  # event=int, time=float, value=float
                        for i, x in enumerate(parts)]
            except ValueError:
                # Not purely numeric -> skip
                continue

            if len(nums) == 4:
                vec, ev, t, val = nums
                vec = int(vec)
            else:  # 3 columns; need a prior "vector <id>" header
                if current_vector is None:
                    # Malformed row for our purposes
                    continue
                ev, t, val = nums
                vec = int(current_vector)

            data[vec].append((int(ev), float(t), float(val)))

    # Sort each vector by time (stable)
    for v in data:
        data[v].sort(key=lambda x: x[1])
    return data

def diffs(times):
    """Consecutive differences of a sorted time list."""
    return [t2 - t1 for t1, t2 in zip(times, times[1:]) if t2 >= t1]

def node_and_signal_from_vector(vec_id: int):
    """
    We assume the mapping:
      even vec -> Transmit, odd vec -> Receive
      node index = vec_id // 2 + 1
    e.g., 0->(node1, Tx), 1->(node1, Rx), 2->(node2, Tx), 3->(node2, Rx), ...
    """
    node = vec_id // 2 + 1
    sig = "tx" if vec_id % 2 == 0 else "rx"
    return node, sig

def summarize(gaps):
    """Return (count, mean, std) where std is population std; zero if <2 samples."""
    n = len(gaps)
    if n == 0:
        return (0, float("nan"), float("nan"))
    if n == 1:
        return (1, gaps[0], 0.0)
    # population stddev (pstdev). Change to statistics.stdev for sample std if desired.
    return (n, mean(gaps), pstdev(gaps))

def analyze_file(path: Path):
    """
    Returns:
      per_node = {
         node_id: {
             'IGG': [gaps...],   # from Tx vectors
             'IPG': [gaps...]    # from Rx vectors
         }, ...
      }
    """
    vecs = parse_vec_file(path)
    per_node = defaultdict(lambda: {"IGG": [], "IPG": []})

    for vec_id, rows in vecs.items():
        node, sig = node_and_signal_from_vector(vec_id)
        times = [t for (_, t, _) in rows]
        gaps = diffs(times)
        if sig == "tx":
            per_node[node]["IGG"].extend(gaps)
        else:
            per_node[node]["IPG"].extend(gaps)

    return per_node

def format_stats_line(label, n, avg, sd, width=10):
    def f(x):
        if isinstance(x, float) and (math.isnan(x) or math.isinf(x)):
            return "—"
        return f"{x:.6g}" if isinstance(x, float) else str(x)
    return f"{label:<16} {f(n):>{width}} {f(avg):>{width}} {f(sd):>{width}}"

def main():
    ap = argparse.ArgumentParser(description="Compute IGGs and IPGs from OMNeT++ .vec files.")
    ap.add_argument("root", help="Root directory containing .vec files (searched recursively).")
    ap.add_argument("--sample-std", action="store_true",
                    help="Use sample standard deviation instead of population (default is population).")
    args = ap.parse_args()

    root = Path(args.root)
    files = sorted(p for p in root.rglob("*.vec") if p.is_file())
    if not files:
        print(f"No .vec files found under: {root}")
        return

    # Per-file reporting and overall pooling
    overall = defaultdict(lambda: {"IGG": [], "IPG": []})  # node -> metric -> pooled gaps

    print("\n=== Per-file statistics ===\n")
    for fp in files:
        per_node = analyze_file(fp)

        print(f"[{fp}]")
        print(f"{'Metric (Node)':<16} {'N':>10} {'Mean':>10} {'StdDev':>10}")

        all_nodes = sorted(per_node.keys())
        if not all_nodes:
            print("  (no usable vector data found)\n")
            continue

        for node in all_nodes:
            for metric in ("IGG", "IPG"):
                gaps = per_node[node][metric]
                # pool into overall
                overall[node][metric].extend(gaps)

                # summarize
                n, avg, sd = summarize(gaps)
                if args.sample_std and n >= 2:
                    from statistics import stdev
                    sd = stdev(gaps)
                label = f"{metric} (N{node})"
                print(format_stats_line(label, n, avg, sd))
        print()

    # Overall summary
    print("=== Overall (pooled across files) ===\n")
    print(f"{'Metric (Node)':<16} {'N':>10} {'Mean':>10} {'StdDev':>10}")
    for node in sorted(overall.keys()):
        for metric in ("IGG", "IPG"):
            gaps = overall[node][metric]
            n, avg, sd = summarize(gaps)
            if args.sample_std and n >= 2:
                from statistics import stdev
                sd = stdev(gaps)
            label = f"{metric} (N{node})"
            print(format_stats_line(label, n, avg, sd))
    print()

if __name__ == "__main__":
    main()
