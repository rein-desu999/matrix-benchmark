"""
Benchmark: sequential vs 2 process parallel matrix multiplication.

Run this on each machine. It writes CSVs and PNG graphs to ./results/.

"""

import csv
import os
import platform
import statistics
import time
from multiprocessing import Process, Queue

import numpy as np
import psutil

# --- Config ---

MATRIX_SIZES = [200, 400, 800]
TRIALS = 5
NUM_WORKERS = 2
SEED = 42

RESULTS_DIR = "results"


# --- Helpers ---

def get_cpu_name():
    """
    psutil gives us core count but not the CPU model on Windows reliably,
    so pull it from the registry-ish / platform bits where possible.
    """
    name = platform.processor()
    if not name:
        name = platform.machine()
    # Windows often returns something generic; fall back to env var if present
    name = os.environ.get("PROCESSOR_IDENTIFIER", name)
    return name.strip() or "unknown"


def make_matrix(n, rng):
    return rng.standard_normal((n, n))


def multiply_sequential(a, b):
    return a @ b


def worker_worker(a, b, start, end, out_queue):
    """
    Multiply rows [start:end) of a by b, ship the block back through the queue.
    Kept as a top-level function so Windows spawn pickles it correctly.
    """
    block = a[start:end] @ b
    out_queue.put((start, block))


def multiply_parallel(a, b, n_workers):
    n = a.shape[0]
    chunk = n // n_workers
    q = Queue()
    procs = []

    for i in range(n_workers):
        start = i * chunk
        # last worker grabs the remainder so odd sizes don't drop rows
        end = n if i == n_workers - 1 else (i + 1) * chunk
        p = Process(target=worker_worker, args=(a, b, start, end, q))
        procs.append(p)
        p.start()

    result = np.empty((n, n), dtype=a.dtype)
    for _ in procs:
        start, block = q.get()
        result[start:start + block.shape[0], :] = block

    for p in procs:
        p.join()

    return result


def peak_mem_mb():
    """RSS in MB for this process. Cheap enough to call around a run."""
    return psutil.Process(os.getpid()).memory_info().rss / (1024 * 1024)


# --- 1 Trial ---

def run_trial(a, b, mode, n_workers):
    """
    Returns (elapsed_ms, peak_mem_mb, output_matrix_checksum).
    checksum is just for sanity matching between modes, not perf.
    """
    mem_before = peak_mem_mb()

    t0 = time.perf_counter()
    if mode == "seq":
        c = multiply_sequential(a, b)
    else:
        c = multiply_parallel(a, b, n_workers)
    elapsed = (time.perf_counter() - t0) * 1000.0

    mem_after = peak_mem_mb()
    peak = max(mem_before, mem_after)
    checksum = float(np.sum(c))
    return elapsed, peak, checksum


# --- Main Loop ---

def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)

    cpu_name = get_cpu_name()
    phys = psutil.cpu_count(logical=False)
    logical = psutil.cpu_count(logical=True)
    print(f"CPU: {cpu_name}")
    print(f"Physical cores: {phys} | Logical cores: {logical}")
    print(f"Python: {platform.python_version()} | NumPy: {np.__version__}")
    print("-" * 60)

    rows = []

    for n in MATRIX_SIZES:
        rng = np.random.default_rng(SEED)
        a = make_matrix(n, rng)
        b = make_matrix(n, rng)

        # ref result so we can verify the parallel version matches
        reference = multiply_sequential(a, b)

        print(f"Matrix size {n}x{n}")

        for mode in ("seq", "par"):
            times = []
            peaks = []
            for t in range(TRIALS):
                elapsed, peak, checksum = run_trial(a, b, mode, NUM_WORKERS)

                # additional check: parallel output must match sequential
                if mode == "par":
                    ok = np.allclose(checksum, float(np.sum(reference)), rtol=1e-5)
                    if not ok:
                        print(f"  !! parallel result mismatch at trial {t+1}")

                times.append(elapsed)
                peaks.append(peak)
                print(f"  {mode} trial {t+1}/{TRIALS}: {elapsed:.1f} ms")

            mean_t = statistics.mean(times)
            std_t = statistics.stdev(times) if len(times) > 1 else 0.0
            throughput = 1000.0 / mean_t  # matrix ops per second
            peak_mem = max(peaks)

            rows.append({
                "cpu": cpu_name,
                "matrix_size": n,
                "mode": "sequential" if mode == "seq" else "parallel",
                "workers": 1 if mode == "seq" else NUM_WORKERS,
                "trials": TRIALS,
                "mean_ms": round(mean_t, 3),
                "std_ms": round(std_t, 3),
                "throughput_ops_s": round(throughput, 4),
                "peak_mem_mb": round(peak_mem, 2),
            })

    # --- Speedup ---

    by_size = {}
    for r in rows:
        by_size.setdefault(r["matrix_size"], {})[r["mode"]] = r

    for n, modes in by_size.items():
        seq = modes["sequential"]["mean_ms"]
        par = modes["parallel"]["mean_ms"]
        speedup = seq / par if par else 0.0
        for r in rows:
            if r["matrix_size"] == n:
                r["speedup"] = round(speedup, 3)

    # --- write CSV ---

    csv_path = os.path.join(RESULTS_DIR, "results.csv")
    fields = [
        "cpu", "matrix_size", "mode", "workers", "trials",
        "mean_ms", "std_ms", "throughput_ops_s", "peak_mem_mb", "speedup",
    ]
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"\nWrote {csv_path}")

    # --- Graphs ---

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed, skipping graphs")
        return

    sizes = MATRIX_SIZES
    seq_ms = [by_size[n]["sequential"]["mean_ms"] for n in sizes]
    par_ms = [by_size[n]["parallel"]["mean_ms"] for n in sizes]
    seq_std = [by_size[n]["sequential"]["std_ms"] for n in sizes]
    par_std = [by_size[n]["parallel"]["std_ms"] for n in sizes]
    speedups = [by_size[n]["sequential"]["mean_ms"] / by_size[n]["parallel"]["mean_ms"] for n in sizes]

    # Graph 1: runtime
    x = np.arange(len(sizes))
    width = 0.35
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(x - width/2, seq_ms, width, yerr=seq_std, capsize=4, label="Sequential")
    ax.bar(x + width/2, par_ms, width, yerr=par_std, capsize=4, label=f"Parallel ({NUM_WORKERS} workers)")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{s}x{s}" for s in sizes])
    ax.set_ylabel("Execution time (ms)")
    ax.set_title("Runtime: sequential vs parallel")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(RESULTS_DIR, "runtime.png"), dpi=150)
    plt.close(fig)

    # Graph 2: speedup
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot([f"{s}x{s}" for s in sizes], speedups, marker="o", label="Measured speedup")
    ax.axhline(1.0, color="gray", linestyle="--", label="No speedup")
    ax.axhline(2.0, color="red", linestyle=":", label="Ideal (2 workers)")
    ax.set_ylabel("Speedup (sequential / parallel)")
    ax.set_title("Speedup by matrix size")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(RESULTS_DIR, "speedup.png"), dpi=150)
    plt.close(fig)

    print(f"Wrote {RESULTS_DIR}/runtime.png and {RESULTS_DIR}/speedup.png")


if __name__ == "__main__":
    main()