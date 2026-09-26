#!/usr/bin/env python3
"""Generate degraded handcrafted features for the permuted lot order.

MD3-RS is defined on a feature-bagged ensemble, so it needs the handcrafted
feature matrix, not the CNN's probabilities. The permutation changes which
lots fall after the change point, so those features cannot be reused from the
natural-order run. Resumable: each change point is cached separately.
"""
from wmmon.paths import WMMON_HOME
import argparse, time
from pathlib import Path
import numpy as np
from wmmon import classifier, data, drift
from wmmon import features as feat

CACHE = Path(f"{WMMON_HOME}/cache"); CLASSES = list(data.CLASSES)
ap = argparse.ArgumentParser()
ap.add_argument("--mode", default="defect_remix")
ap.add_argument("--confine-to-defective", action="store_true")
ap.add_argument("--stream-lots", type=int, default=1200)
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--budget-seconds", type=int, default=240)
a = ap.parse_args(); t0 = time.time()

df, _ = data.load_lswmd(f"{WMMON_HOME}/data/subset.pkl")
train, _, stream = data.split_labelled(df, seed=a.seed)
stream = stream.reset_index(drop=True)
model = classifier.train_inspection_model(
    np.load(CACHE / "train.npz")["x"], train["label"].tolist(), CLASSES, seed=a.seed)

sequence = data.permuted_lot_order(stream, seed=a.seed)[: a.stream_lots]
keep = set(sequence)
lots_all = stream["lot"].to_numpy(dtype=object)
rows = np.flatnonzero(np.array([str(l) in keep for l in lots_all]))
lots = lots_all[rows]
position = np.array([{l: i for i, l in enumerate(sequence)}[str(l)] for l in lots])
maps = [np.asarray(m) for m in stream["wafer_map"].to_numpy()[rows]]

blocks = []
for p in sorted(CACHE.glob("stream_part_*.npz")):
    with np.load(p) as z: blocks.append((int(z["lo"]), z["x"]))
blocks.sort(key=lambda t: t[0])
x_clean = np.vstack([b for _, b in blocks])[rows]
proba_clean = model.predict_proba(x_clean)
confine = "_conf" if a.confine_to_defective else ""

for cp in [450, 500, 550, 600, 650, 700, 750, 800, 850, 900, 950, 1000]:
    path = CACHE / f"deg_{a.mode}{confine}_perm_s1.0_r0_cp{cp}_seed{a.seed}.npz"
    if path.exists():
        continue
    if time.time() - t0 > a.budget_seconds:
        print("budget reached"); break
    post = position >= cp
    maps_post = [maps[i] for i in np.flatnonzero(post)]
    deg = drift.apply_degradation(maps_post, lots[post], sequence, a.mode,
                                  change_point=cp, ramp_lots=0, seed=a.seed)
    if a.confine_to_defective:
        normal = proba_clean[post].argmax(axis=1) == CLASSES.index("none")
        for i in np.flatnonzero(normal):
            deg[i] = maps_post[i]
    np.savez_compressed(path, x=feat.extract_batch(deg, progress_every=0, n_jobs=1))
    print(f"cp {cp} done [{time.time()-t0:.0f}s]", flush=True)
