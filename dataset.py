"""Parse Orbit Wars replays into the on-disk feature cache that training reads.

Split out from train_sl.py so parsing/caching and training are separate steps:
build the cache once here, then train_sl.py just loads it.

  python dataset.py                 # parse data/*.json + data/zips/*.zip
  python dataset.py --rebuild       # force re-parse
  python dataset.py --workers 8     # parse replays across 8 processes
  python dataset.py --limit 100     # use only the latest 100 episodes (default: all)
  python dataset.py --top 50        # only games with a top-50 (by rank) leaderboard player
  python dataset.py --top-pct 10    # ...or a top-10% leaderboard player
  python dataset.py a.json b.zip    # parse specific files (.json and/or .zip)

--top N (count) / --top-pct P (percent) read the leaderboard CSV
(data/orbit-wars-publicleaderboard-*.csv) and keep an episode only if at least
one of its players is among the top N teams / top P% by rank.

.zip archives of replays are read in-memory one episode at a time (never
extracted to disk), so a tight disk only needs room for the zips + the cache.
"""

import csv
import glob
import hashlib
import itertools
import json
import os
import random
import shutil
import sys
import zipfile

import numpy as np
import torch
from torch.utils.data import Dataset
from tqdm import tqdm

import features as F
from replay import parse_episode, parse_episode_data

VAL_FRACTION = 0.1        # held-out share of replays for validation
SPLIT_SEED = 0            # deterministic train/val split
CACHE_DIR = os.environ.get("ORBIT_CACHE_DIR", "data/parsed_cache")
# ^ disk store (per-key .dat + meta + signature). Override via ORBIT_CACHE_DIR to
#   put it on a drive with room -- the materialized feature store is ~20+ GB.
KEYS = ["feats", "valid", "mine", "launch", "target", "ships", "has_ship"]  # model-facing
PARSER_VERSION = 16   # bump when replay.py/features.py/encoding change -> drops cache

# On disk we don't store the materialized [MAX_PLANETS, FEATURE_DIM] float feats.
# The temporal block is most of the bytes and its owner part is a one-hot over
# OWNER_CH channels (mine / neutral / enemy_1..3, all-zero = gone), so collapsing
# those into one int8 owner code is the main win. The remaining temporal channels
# (log-scaled garrison + x,y position) stay float16 -- stored directly.
#   static [MAX_PLANETS, STATIC_DIM]                 float16  (instantaneous block)
#   owner  [MAX_PLANETS, HORIZON]                    int8     0=gone, else 1..OWNER_CH
#   tempf  [MAX_PLANETS, HORIZON, TEMP_CH-OWNER_CH]  float16  per turn: [shipslog, x, y]
STORE_KEYS = ["static", "owner", "tempf", "valid", "mine", "launch", "target", "ships", "has_ship"]
_LOG1000 = F._LOG1000


def encode_feats(feats):
    """[n, MAX_PLANETS, FEATURE_DIM] float feats -> compact (static, owner, tempf)."""
    n = feats.shape[0]
    static = feats[:, :, :F.STATIC_DIM].astype(np.float16)
    temp = feats[:, :, F.STATIC_DIM:].reshape(n, F.MAX_PLANETS, F.HORIZON, F.TEMP_CH)
    owner_oh = temp[..., :F.OWNER_CH]                       # [n,MAX_PLANETS,H,OWNER_CH]
    present = owner_oh.sum(axis=-1) > 0.5
    owner = (np.argmax(owner_oh, axis=-1) + 1).astype(np.int8)
    owner[~present] = 0                                     # gone/absent
    tempf = temp[..., F.OWNER_CH:].astype(np.float16)       # [shipslog, x, y]; 0 where gone
    return static, owner, tempf


def decode_feats(static, owner, tempf):
    """Rebuild one example's [MAX_PLANETS, FEATURE_DIM] float32 feats from compact form."""
    feats = np.zeros((F.MAX_PLANETS, F.FEATURE_DIM), dtype=np.float32)
    feats[:, :F.STATIC_DIM] = static
    temp = feats[:, F.STATIC_DIM:].reshape(F.MAX_PLANETS, F.HORIZON, F.TEMP_CH)
    for c in range(F.OWNER_CH):
        temp[..., c] = (owner == c + 1)
    temp[..., F.OWNER_CH:] = tempf
    return feats


def signature(paths, limit=None, filter_tag=""):
    """Fingerprint of the input files + parser version + split params + episode
    limit + leaderboard filter, so the cache is invalidated whenever any change."""
    h = hashlib.md5()
    h.update(f"v{PARSER_VERSION}|val{VAL_FRACTION}|seed{SPLIT_SEED}"
             f"|lim{limit}|filt{filter_tag}".encode())
    for p in sorted(paths):
        h.update(p.encode())
        h.update(str(os.path.getsize(p)).encode())
    return h.hexdigest()


# Leaderboard filter: keep only episodes with >=1 player in the top fraction of
# the leaderboard. The set is installed per worker (so it survives spawn) and
# read inside _parse_one. None = no filtering.
_TOP_TEAMS = None


def _set_top_teams(top):
    global _TOP_TEAMS
    _TOP_TEAMS = top


def load_top_teams(csv_path, n_top=None, frac=None):
    """Set of the highest-ranked team names from the leaderboard CSV. Give either
    `n_top` (an absolute count of teams) or `frac` (a fraction, e.g. 0.5 for top
    half). Episodes whose info.TeamNames intersect this set are kept; others
    dropped."""
    with open(csv_path, encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    rows.sort(key=lambda r: int(r["Rank"]))
    if n_top is None:
        n_top = int(len(rows) * frac)
    n_top = max(1, min(n_top, len(rows)))
    return {r["TeamName"] for r in rows[:n_top]}


def _item_key(item):
    """Stable string key for a work item (a .json path or a (zip, member) pair),
    so a mixed list of files and zip members still sorts deterministically."""
    return item if isinstance(item, str) else f"{item[0]}::{item[1]}"


def expand_items(inputs):
    """Flatten a list of inputs into per-episode work items. A .json path stays
    as-is; each .zip is expanded into (zip_path, member) tuples for its .json
    members. Only the zip's central directory is read here (member names) -- the
    episodes themselves are read in-memory one at a time at parse time, never
    extracted to disk, which is the whole point on a tight disk."""
    items = []
    for p in inputs:
        if p.lower().endswith(".zip"):
            with zipfile.ZipFile(p) as zf:
                items += [(p, n) for n in zf.namelist() if n.lower().endswith(".json")]
        else:
            items.append(p)
    return items


def split_paths(items):
    """Split episodes into (train, val) at the *episode* level, so adjacent frames
    of one game never straddle the split (which would leak). Each item is one
    game (a file or a single zip member), so item-level == game-level."""
    shuffled = sorted(items, key=_item_key)
    random.Random(SPLIT_SEED).shuffle(shuffled)
    n_val = int(len(shuffled) * VAL_FRACTION) if len(shuffled) > 1 else 0
    return shuffled[n_val:], shuffled[:n_val]


def _parse_one(item):
    """Parse a single episode in a worker process. `item` is either a .json path
    or a (zip_path, member) tuple read in-memory. With a leaderboard filter set
    (_TOP_TEAMS), episodes whose players are all below the cut are skipped (empty
    result). Returns (item, examples, error_msg)."""
    try:
        if isinstance(item, tuple):
            zip_path, member = item
            with zipfile.ZipFile(zip_path) as zf:
                d = json.loads(zf.read(member))
        else:
            with open(item, "r", encoding="utf-8") as f:
                d = json.load(f)
        if _TOP_TEAMS is not None:
            teams = (d.get("info") or {}).get("TeamNames") or []
            if not any(t in _TOP_TEAMS for t in teams):
                return item, [], None          # no top player in this game -> skip
        return item, parse_episode_data(d, perspective="winner"), None
    except Exception as e:
        return item, [], str(e)


def _iter_parsed(paths, workers, desc, top_teams=None):
    """Yield (path, examples, error) for each replay, fanning parsing out across
    `workers` processes. Each replay is handled independently, so nothing but the
    one replay being processed is ever held by a given worker. `top_teams` (the
    leaderboard filter) is installed in each worker so it survives spawn."""
    workers = max(1, min(workers, len(paths)))
    if workers > 1:
        from concurrent.futures import ProcessPoolExecutor
        with ProcessPoolExecutor(max_workers=workers,
                                 initializer=_set_top_teams,
                                 initargs=(top_teams,)) as pool:
            for item in tqdm(pool.map(_parse_one, paths), total=len(paths),
                             desc=f"{desc} ({workers}w)", unit="ep"):
                yield item
    else:
        _set_top_teams(top_teams)
        for p in tqdm(paths, desc=desc, unit="ep"):
            yield _parse_one(p)


def _scan_teams(item):
    """Read only an episode's info.TeamNames -- cheap (no feature parsing) -- so the
    leaderboard filter can be decided up front. Returns (item, teams, error)."""
    try:
        if isinstance(item, tuple):
            zip_path, member = item
            with zipfile.ZipFile(zip_path) as zf:
                d = json.loads(zf.read(member))
        else:
            with open(item, "r", encoding="utf-8") as f:
                d = json.load(f)
        return item, (d.get("info") or {}).get("TeamNames") or [], None
    except Exception as e:
        return item, None, str(e)


def filter_top_items(items, top_teams, workers, limit=None,
                     desc="scanning leaderboard"):
    """Keep items (episodes) with >=1 player in `top_teams`, decided up front by
    reading each episode's TeamNames. Items are scanned newest first and, when
    `limit` is set, scanning stops as soon as `limit` are kept -- so a small
    --limit never reads the whole dataset. Episodes that can't be read are dropped
    (and reported), same as at parse time. Returns (kept_items, n_scanned)."""
    ordered = sorted(items, key=_item_key, reverse=True)   # newest first
    workers = max(1, min(workers, len(ordered)))
    kept, scanned = [], 0

    def take(item, teams, err):
        nonlocal scanned
        scanned += 1
        if err:
            print(f"  skip {_item_key(item)}: {err}")
        elif any(t in top_teams for t in teams):
            kept.append(item)

    enough = lambda: limit is not None and len(kept) >= limit
    if workers > 1:
        from concurrent.futures import ProcessPoolExecutor
        chunk = workers * 8                                # bound early-stop overshoot
        it = iter(ordered)
        with ProcessPoolExecutor(max_workers=workers) as pool, \
                tqdm(total=len(ordered), desc=f"{desc} ({workers}w)", unit="ep") as bar:
            while not enough():
                batch = list(itertools.islice(it, chunk))
                if not batch:
                    break
                for item, teams, err in pool.map(_scan_teams, batch):
                    take(item, teams, err)
                    bar.update(1)
                    bar.set_postfix(found=len(kept), refresh=False)
                    if enough():
                        break
    else:
        with tqdm(total=len(ordered), desc=desc, unit="ep") as bar:
            for item in ordered:
                if enough():
                    break
                take(*_scan_teams(item))
                bar.update(1)
                bar.set_postfix(found=len(kept), refresh=False)
    return kept, scanned


def build_store(paths, out_dir, workers=1, desc="parsing", top_teams=None):
    """Parse every replay and stream its step-examples straight to disk, in a
    single pass, never holding the whole dataset in RAM *or* a second copy on
    disk.

    Each key gets one append-only `<key>.dat` raw binary file: as a replay is
    parsed we stack its (small) arrays and append their bytes, then drop them.
    A `meta.json` records the example count and each key's dtype/item-shape so the
    flat files can be reopened as memmaps. Peak memory is ~one replay; peak disk
    is exactly the final store (no shard/consolidate doubling).

    Returns the number of step-examples written (0 if none).
    """
    os.makedirs(out_dir, exist_ok=True)
    files = {k: open(os.path.join(out_dir, f"{k}.dat"), "wb") for k in STORE_KEYS}
    spec = None
    total = 0
    try:
        for path, ex, err in _iter_parsed(paths, workers, desc, top_teams):
            if err:
                print(f"  skip {path}: {err}")
            if not ex:
                continue
            feats = np.stack([e["feats"] for e in ex])              # [n,MAX_PLANETS,FEATURE_DIM] f32
            static, owner, tempf = encode_feats(feats)
            blk = {"static": static, "owner": owner, "tempf": tempf,
                   "valid": np.stack([e["valid"] for e in ex]),
                   "mine": np.stack([e["mine"] for e in ex]),
                   "launch": np.stack([e["launch"] for e in ex]),
                   "target": np.stack([e["target"] for e in ex]),
                   "ships": np.stack([e["ships"] for e in ex]).astype(np.float32),  # ratio [0,1]
                   "has_ship": np.stack([e["has_ship"] for e in ex])}
            if spec is None:
                spec = {k: (np.ascontiguousarray(blk[k]).dtype.str,
                            list(blk[k].shape[1:])) for k in STORE_KEYS}
            for k in STORE_KEYS:
                files[k].write(np.ascontiguousarray(blk[k]).tobytes())
            total += static.shape[0]
    finally:
        for f in files.values():
            f.close()

    if total == 0:
        shutil.rmtree(out_dir)
        return 0

    with open(os.path.join(out_dir, "meta.json"), "w", encoding="utf-8") as f:
        json.dump({"count": total, "spec": spec}, f)
    return total


class DiskDataset(Dataset):
    """Map-style dataset backed by the flat per-key `.dat` store on disk. Each
    __getitem__ pulls one step-example's rows straight from disk via memmap (the
    OS page cache keeps hot pages resident), so a DataLoader streams batches
    without ever loading the full dataset into RAM. The compact (static, owner,
    garrison) encoding is expanded back to the model's float32 feats here, so the
    network is unaffected."""

    def __init__(self, store_dir):
        self.dir = store_dir
        with open(os.path.join(store_dir, "meta.json"), encoding="utf-8") as f:
            meta = json.load(f)
        self._len = int(meta["count"])
        self._spec = meta["spec"]          # key -> (dtype_str, item_shape)
        self._mm = None                    # opened lazily (cheap to construct/fork)

    def _maps(self):
        if self._mm is None:
            self._mm = {
                k: np.memmap(os.path.join(self.dir, f"{k}.dat"),
                             dtype=np.dtype(dt), mode="r",
                             shape=(self._len, *item))
                for k, (dt, item) in self._spec.items()
            }
        return self._mm

    def __len__(self):
        return self._len

    def __getitem__(self, i):
        m = self._maps()
        feats = decode_feats(np.asarray(m["static"][i]),
                             np.asarray(m["owner"][i]),
                             np.asarray(m["tempf"][i]))
        # np.array(..) forces a writable copy (memmap rows are read-only, which
        # torch.from_numpy would warn about).
        return {
            "feats": torch.from_numpy(feats),
            "valid": torch.from_numpy(np.array(m["valid"][i])),
            "mine": torch.from_numpy(np.array(m["mine"][i])),
            "launch": torch.from_numpy(np.array(m["launch"][i])),
            "target": torch.from_numpy(np.array(m["target"][i])),
            "ships": torch.from_numpy(np.array(m["ships"][i])),
            "has_ship": torch.from_numpy(np.array(m["has_ship"][i])),
        }


def load_cached(cache_dir=CACHE_DIR):
    """Open the existing on-disk cache as (train_ds, val_ds) WITHOUT parsing.
    Used by train_sl.py (pure training). Raises if no cache has been built yet."""
    train_dir = os.path.join(cache_dir, "train")
    val_dir = os.path.join(cache_dir, "val")
    if not os.path.exists(os.path.join(train_dir, "meta.json")):
        raise FileNotFoundError(
            f"no parsed cache at {cache_dir!r}. Build it first, e.g.:\n"
            f"    python dataset.py               # parse data/*.json + data/zips/*.zip\n"
            f"    python dataset.py --limit 200   # quick subset")
    train = DiskDataset(train_dir)
    val = DiskDataset(val_dir) if os.path.exists(os.path.join(val_dir, "meta.json")) else None
    return train, val


def load_or_build(paths, rebuild=False, workers=1, limit=None,
                  top_teams=None, filter_tag=""):
    """Parse (or reuse the on-disk store of) the replays, returning (train_ds,
    val_ds) as DiskDatasets. `paths` are input files (.json replays and/or .zip
    archives of replays); zips are expanded into per-episode items that are read
    in-memory at parse time (never extracted). `limit` caps the number of episodes
    used (default: all). `top_teams` (with `filter_tag` for the signature) keeps
    only games with a top-leaderboard player. val_ds is None when there are too
    few episodes to split. The store is reused when its signature matches the
    input files + parser + limit + filter."""
    sig = signature(paths, limit, filter_tag)   # fingerprint inputs + filter
    sig_file = os.path.join(CACHE_DIR, "signature.txt")
    train_dir = os.path.join(CACHE_DIR, "train")
    val_dir = os.path.join(CACHE_DIR, "val")
    have_train = os.path.exists(os.path.join(train_dir, "meta.json"))

    if (not rebuild and have_train and os.path.exists(sig_file)
            and open(sig_file, encoding="utf-8").read() == sig):
        train = DiskDataset(train_dir)
        val = DiskDataset(val_dir) if os.path.exists(os.path.join(val_dir, "meta.json")) else None
        print(f"loaded cache {CACHE_DIR} (train {len(train)}, "
              f"val {len(val) if val else 0})")
        return train, val
    if have_train:
        print("cache stale (replay set / parser changed) -> re-parsing")

    if os.path.exists(CACHE_DIR):
        shutil.rmtree(CACHE_DIR)
    os.makedirs(CACHE_DIR, exist_ok=True)

    items = expand_items(paths)             # zips -> per-episode (zip, member) items
    if limit is not None:
        # newest first (filenames are ISO-dated, so reverse sort == newest date),
        # matching download_latest.sh which fetches the latest days first.
        items = sorted(items, key=_item_key, reverse=True)[:limit]
    train_paths, val_paths = split_paths(items)
    # No upfront leaderboard scan: the top_teams filter (if any) is applied
    # per-episode during parsing (_parse_one), so parsing starts immediately.
    n_train = build_store(train_paths, train_dir, workers=workers,
                          desc="parsing train", top_teams=top_teams)
    if n_train == 0:
        return None, None
    n_val = (build_store(val_paths, val_dir, workers=workers,
                         desc="parsing val", top_teams=top_teams)
             if val_paths else 0)

    with open(sig_file, "w", encoding="utf-8") as f:
        f.write(sig)
    print(f"stored train {n_train}, val {n_val} -> {CACHE_DIR}")
    return DiskDataset(train_dir), (DiskDataset(val_dir) if n_val else None)


def main():
    rebuild = "--rebuild" in sys.argv
    workers = min(8, os.cpu_count() or 1)   # parse replays in parallel by default
    limit = None                            # cap on #episodes (default: use all)
    top_n = None                            # keep only games with a top-N team player
    top_pct = None                          # ...or a top-N% team player
    paths = []
    argv = iter(sys.argv[1:])
    for a in argv:
        if a == "--rebuild":
            continue
        if a == "--workers":
            workers = int(next(argv))
        elif a.startswith("--workers="):
            workers = int(a.split("=", 1)[1])
        elif a == "--limit":
            limit = int(next(argv))
        elif a.startswith("--limit="):
            limit = int(a.split("=", 1)[1])
        elif a == "--top":
            top_n = int(next(argv))
        elif a.startswith("--top="):
            top_n = int(a.split("=", 1)[1])
        elif a == "--top-pct":
            top_pct = float(next(argv))
        elif a.startswith("--top-pct="):
            top_pct = float(a.split("=", 1)[1])
        else:
            paths.append(a)
    # Default: all loose .json replays plus any downloaded .zip archives. Zips are
    # read in-memory at parse time (never extracted), so they cost no extra disk.
    paths = paths or sorted(glob.glob("data/*.json") + glob.glob("data/zips/*.zip"))
    paths = [p for p in paths if "kaggle" not in os.path.basename(p).lower()]
    print(f"input files: {len(paths)} "
          f"({sum(p.lower().endswith('.zip') for p in paths)} zip)"
          + (f"  --limit {limit} episodes" if limit is not None else "")
          + f"  (parse workers: {workers})")

    # Optional leaderboard filter: keep games with >=1 player among the top teams,
    # given either as a count (--top N) or a percentage (--top-pct P).
    top_teams, filter_tag = None, ""
    if top_n is not None or top_pct is not None:
        lbs = sorted(glob.glob("data/orbit-wars-publicleaderboard-*.csv"))
        if not lbs:
            print("--top/--top-pct given but no leaderboard CSV found "
                  "(data/orbit-wars-publicleaderboard-*.csv)")
            return
        lb_path = lbs[-1]                    # latest snapshot
        if top_n is not None:
            top_teams = load_top_teams(lb_path, n_top=top_n)
            filter_tag = f"topN{top_n}:{os.path.basename(lb_path)}"
            cut = f"top {top_n} teams"
        else:
            top_teams = load_top_teams(lb_path, frac=top_pct / 100.0)
            filter_tag = f"top{top_pct}:{os.path.basename(lb_path)}"
            cut = f"top {top_pct}%"
        print(f"leaderboard filter: {cut} -> {len(top_teams)} teams "
              f"(from {os.path.basename(lb_path)})")

    train_ds, val_ds = load_or_build(paths, rebuild=rebuild, workers=workers,
                                     limit=limit, top_teams=top_teams,
                                     filter_tag=filter_tag)
    if train_ds is None:
        print("no examples parsed.")
        return
    print(f"cache ready -> {CACHE_DIR}  "
          f"(train {len(train_ds)}, val {len(val_ds) if val_ds else 0} steps)")


if __name__ == "__main__":
    main()
