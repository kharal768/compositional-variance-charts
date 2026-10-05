#!/usr/bin/env python3
"""Train the CNN inspection module, resumably, and report held-out quality."""
from wmmon.paths import WMMON_HOME
import argparse, json, time
from pathlib import Path
import numpy as np
from wmmon import cnn, data

CACHE = Path(f"{WMMON_HOME}/cache")
ap = argparse.ArgumentParser()
ap.add_argument("--epochs", type=int, default=30)
ap.add_argument("--resume-from", type=int, default=0)
args = ap.parse_args()

t0 = time.time()
df, _ = data.load_lswmd(f"{WMMON_HOME}/data/subset.pkl")
train, test, _ = data.split_labelled(df, seed=0)

def cached(name, maps):
    p = CACHE / name
    if p.exists():
        return np.load(p)["x"]
    x = cnn.encode_batch(maps)
    np.savez_compressed(p, x=x)
    return x

xtr = cached("cnn_train_images.npz", train["wafer_map"].tolist())
xte = cached("cnn_test_images.npz", test["wafer_map"].tolist())
print(f"train {xtr.shape} | test {xte.shape} | {time.time()-t0:.0f}s", flush=True)

model = cnn.train_cnn(xtr, train["label"].tolist(), list(data.CLASSES),
                      epochs=args.epochs, seed=0,
                      checkpoint=str(CACHE / "cnn_ckpt.pt"),
                      resume_from=args.resume_from)

q = cnn.evaluate(model, xte, test["label"].tolist())
print(f"\nCNN macro-F1 {q['macro_f1']:.3f} | weighted-F1 {q['weighted_f1']:.3f}")
print(q["report"])
Path(f"{WMMON_HOME}/results").mkdir(exist_ok=True)
json.dump({k: v for k, v in q.items() if k != "report"},
          open(f"{WMMON_HOME}/results/cnn_quality.json", "w"), indent=2)
print(f"total {time.time()-t0:.0f}s")
