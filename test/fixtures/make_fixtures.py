#!/usr/bin/env python3
"""Generate the C++ parity fixture for HNNG.jl (spec section 10, test/parity_cpp.jl).

Run once from the repository root:  python3 test/fixtures/make_fixtures.py
The JSON output is committed; the C++ repository is only read.
"""
import base64
import json
import os
import struct
import sys
import tempfile

import h5py
import numpy as np

sys.path.insert(0, "/home/coral/working/hnng/python_bindings")
import hnnglib  # noqa: E402

DATA = "/home/coral/working/benchmarks/data/random-xs-20-euclidean.hdf5"
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cpp_parity_random_xs_20.json")
N_TRAIN, N_TEST, K = 2000, 100, 10


def cpp_level_counts(path):
    """Read maxlevel and per-level cluster counts from a C++ save file (hnnalg.h:2471-2526)."""
    with open(path, "rb") as f:
        buf = f.read()
    off = 0

    def take(fmt):
        nonlocal off
        vals = struct.unpack_from("<" + fmt, buf, off)
        off += struct.calcsize("<" + fmt)
        return vals

    _max_el, n, _data_size, size_per = take("QQQQ")
    (maxlevel,) = take("i")
    take("I")                                   # enterpoint
    off += n * size_per                         # level-0 block
    levels = take(f"{n}i")                      # element_levels
    off += n                                    # deleted flags, one byte each
    for _ in range(n):                          # level-0 inlinks
        (c,) = take("I")
        off += 4 * c
    (nlev,) = take("i")
    for lvl in range(nlev):                     # upper-level links
        for i in range(n):
            if levels[i] >= lvl + 1:
                take("I")
                take("f")
                (c,) = take("I")
                off += 4 * c
    (ncl,) = take("i")
    counts = []
    for _ in range(ncl):
        (nc,) = take("I")
        for _ in range(nc):
            take("I")                           # rep
            take("f")                           # radius
            take("I")                           # parent
            take("B")                           # c_status
            (nm,) = take("I")
            off += 4 * nm
        counts.append(nc)
    assert off == len(buf), f"parsed {off} of {len(buf)} bytes"
    return maxlevel + 1, counts[: maxlevel + 1]


def main():
    with h5py.File(DATA, "r") as f:
        train = np.ascontiguousarray(f["train"][:N_TRAIN], dtype="<f4")
        test = np.ascontiguousarray(f["test"][:N_TEST], dtype="<f4")
    dim = train.shape[1]

    idx = hnnglib.Index(space="l2", dim=dim)
    idx.init_index(max_elements=N_TRAIN)
    idx.add_items(train, num_threads=1)       # 2000 rows < 4096: incremental addPoint path
    idx.set_num_threads(1)
    labels, sq = idx.knn_query(test, k=K, num_threads=1)

    t64, q64 = train.astype(np.float64), test.astype(np.float64)
    bf = np.sqrt(((q64[:, None, :] - t64[None, :, :]) ** 2).sum(-1))
    bf_sorted = np.sort(bf, axis=1)[:, : K + 1]

    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "cpp.idx")
        idx.save_index(path)
        nlev, ncl = cpp_level_counts(path)

    fx = {
        "dataset": "random-xs-20-euclidean",
        "n_train": N_TRAIN,
        "n_test": N_TEST,
        "dim": dim,
        "k": K,
        "train_f32_b64": base64.b64encode(train.tobytes()).decode("ascii"),
        "test_f32_b64": base64.b64encode(test.tobytes()).decode("ascii"),
        "cpp_ids": labels.astype(np.int64).tolist(),
        "cpp_sqdists": [[float(v) for v in row] for row in sq],
        "bf_dists": bf_sorted.tolist(),
        "cpp_nlevels": nlev,
        "cpp_nclusters": ncl,
    }
    with open(OUT, "w") as f:
        json.dump(fx, f)
    print(f"wrote {OUT}: {N_TRAIN} train, {N_TEST} queries, C++ levels {nlev}, clusters {ncl}")


if __name__ == "__main__":
    main()
