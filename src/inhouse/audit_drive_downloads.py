"""Audit of data/recordings/drive_2026-10-19/ (Drive folder 1wS9USehoXPs37GDOZKEbiNTCKUq29yo_ downloaded 2026-10-19). Read-only: zips are streamed, never extracted.
Writes results/inhouse_audit/drive_2026-10-19_audit.json. Inputs are untrusted files -> run with `python3 -I`."""
import hashlib, io, json, os, struct, zipfile
import numpy as np, pandas as pd
D = "data/recordings/drive_2026-10-19"; HOME = os.path.expanduser("~")
def _prefix(a, b):
    while True:
        c = a.read(1 << 20)
        if not c: return True
        if c != b.read(len(c)): return False
def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""): h.update(c)
    return h.hexdigest()
out = {"folder_id": "1wS9USehoXPs37GDOZKEbiNTCKUq29yo_", "dest": D, "files": {}}
local = {"MukuBreathHold": "data/recordings/breath_hold/%s", "MithinBreathHold": HOME + "/Desktop/Mithin-BreathHold/%s"}
for n in sorted(os.listdir(D)):
    p = f"{D}/{n}"; r = dict(size=os.path.getsize(p), sha256=sha(p))
    if n.endswith(".csv"):
        loc = local["MukuBreathHold" if "Muku" in n else "MithinBreathHold"] % n; r["local_counterpart"] = loc
        if os.path.exists(loc):
            r["local_size"] = os.path.getsize(loc); r["identical_to_local"] = sha(loc) == r["sha256"]
            if not r["identical_to_local"]:
                with open(p, "rb") as a, open(loc, "rb") as b:
                    r["is_byte_prefix_of_local"] = r["size"] < r["local_size"] and all((c := a.read(1 << 20)) == b.read(len(c)) for _ in range(r["size"] // (1 << 20) + 1) if True) if False else _prefix(a, b)
        d = pd.read_csv(p); trunc = int(d.isna().any(axis=1).sum()); d = d.dropna(); t = d["time[us]"].values / 1e6
        q = d["ah_qvar[LSB]"].values.astype(float)
        r.update(rows=len(d), truncated_rows=trunc, duration_s=float(t[-1] - t[0]), fs_mean=float((len(t) - 1) / (t[-1] - t[0])), max_gap_s=float(np.diff(t).max()),
                 qvar_std=float(q.std()), qvar_unique=int(len(np.unique(q))), size_mod_4096=r["size"] % 4096)
    elif n.endswith(".zip"):
        z = zipfile.ZipFile(p); assert z.testzip() is None; ent = z.infolist(); r["entries"] = [dict(name=e.filename, size=e.file_size) for e in ent]
        with z.open(ent[0]) as f: d = pd.read_csv(io.BytesIO(f.read()), low_memory=False)
        ts = pd.to_datetime(d["TimeStamp"], errors="coerce"); raw = d[["RAW_TP9", "RAW_AF7", "RAW_AF8", "RAW_TP10"]]
        el = d["Elements"].dropna().unique().tolist()[:20] if "Elements" in d else None
        r.update(rows=len(d), columns_n=len(d.columns), columns_head=list(d.columns[:6]), has_elements_col="Elements" in d, element_values=el,
                 first_ts=str(ts.min()), last_ts=str(ts.max()), span_s=float((ts.max() - ts.min()).total_seconds()),
                 raw_nonnull_rows=int(raw.notna().all(axis=1).sum()), has_ppg_or_hr=[c for c in d.columns if any(k in c.upper() for k in ("PPG", "HR", "HEART", "SPO2"))],
                 heart_rate_bpm_min_med_max=[float(d.Heart_Rate.min()), float(d.Heart_Rate.median()), float(d.Heart_Rate.max())] if "Heart_Rate" in d else None,
                 element_counts=d["Elements"].dropna().value_counts().to_dict() if "Elements" in d else None,
                 median_dt_raw_rows_s=float(ts[raw.notna().all(axis=1)].diff().dt.total_seconds().median()),
                 accel_cols=[c for c in d.columns if c.startswith(("Accelerometer", "Gyro"))])
    elif n.endswith(".mp4"):
        b = open(p, "rb").read(); r["ftyp"] = b[8:12].decode(errors="ignore")
        def boxes(buf, s=0, e=None):
            e = len(buf) if e is None else e
            while s + 8 <= e:
                sz, typ = struct.unpack(">I4s", buf[s:s + 8]); h = 8
                if sz == 1: sz = struct.unpack(">Q", buf[s + 8:s + 16])[0]; h = 16
                if sz == 0: sz = e - s
                yield typ.decode("latin1"), s + h, s + sz; s += sz
        def find(buf, path, s=0, e=None):
            for t, a, z in boxes(buf, s, e):
                if t == path[0]: return (a, z) if len(path) == 1 else find(buf, path[1:], a, z)
        mv = find(b, ["moov", "mvhd"])
        if mv:
            a = mv[0]; ver = b[a]; o = a + (20 if ver == 1 else 12); ts_, du = struct.unpack(">II", b[o:o + 8]) if ver == 0 else (struct.unpack(">I", b[o:o + 4])[0], struct.unpack(">Q", b[o + 4:o + 12])[0])
            r["duration_s"] = du / ts_
        r["has_moov"] = find(b, ["moov"]) is not None; r["tracks"] = [t for t, _, _ in boxes(b, *find(b, ["moov"]))].count("trak") if find(b, ["moov"]) else 0
        tr = []
        mo = find(b, ["moov"])
        for t, a_, z_ in boxes(b, *mo):
            if t == "trak":
                hd = find(b, ["mdia", "hdlr"], a_, z_); k = b[hd[0] + 8:hd[0] + 12].decode("latin1") if hd else "?"
                st = find(b, ["mdia", "minf", "stbl", "stsd"], a_, z_); cod = b[st[0] + 12:st[0] + 16].decode("latin1") if st else "?"
                tk = find(b, ["tkhd"], a_, z_); w = h_ = None
                if tk and k == "vide": w, h_ = struct.unpack(">II", b[tk[1] - 8:tk[1]]); w, h_ = w >> 16, h_ >> 16
                tr.append(dict(kind=k, codec=cod, width=w, height=h_))
        r["track_info"] = tr
        r["frame_extraction"] = "NOT DONE: no ffmpeg/ffprobe/cv2/av/imageio installed"
    out["files"][n] = r
json.dump(out, open("results/inhouse_audit/drive_2026-10-19_audit.json", "w"), indent=1, default=str)
for n, r in out["files"].items(): print(n, {k: v for k, v in r.items() if k not in ("sha256", "columns_head", "entries", "local_counterpart")})
