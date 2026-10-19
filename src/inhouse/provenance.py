"""Write results/inhouse_audit/provenance.json: path, size, sha256 and identity facts for the six breath-hold logs + Drive retrieval status.
Identity facts come only from filenames and file contents; participant identity and sensor placement are NOT recorded in the files."""
import hashlib, json, os, datetime
B = "data/recordings/breath_hold/user_MukuBreathHold_%s_07_10_2026.csv"
M = os.path.expanduser("~/Desktop/Mithin-BreathHold/user_MithinBreathHold_%s_000.csv")
def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""): h.update(c)
    return h.hexdigest()
files = []
for sess, tpl, where in (("S1", B, "project: data/recordings/breath_hold/ (copied 2026-10-08; original acquisition path not recorded)"),
                         ("S2", M, "NOT in project: ~/Desktop/Mithin-BreathHold/ (read in place, unmodified)")):
    for tile in ("ECG", "EMG", "EOG"):
        p = tpl % tile
        files.append(dict(session=sess, tile=tile, name=os.path.basename(p), location=where, size_bytes=os.path.getsize(p), sha256=sha(p)))
assert len({f["sha256"] for f in files}) == 6, "duplicate files"
out = dict(generated=datetime.datetime.now().isoformat(timespec="seconds"), files=files, all_distinct=True,
           drive=dict(folder_id="1wS9USehoXPs37GDOZKEbiNTCKUq29yo", status="NOT RETRIEVED",
                      detail="gdown (unauthenticated) -> HTTP 404 (folder not public; editor sharing does not make it public). claude.ai Google Drive connector requires OAuth: "
                             "run /mcp and authorise 'claude.ai Google Drive'. Until then it is unknown whether the Desktop 'Mithin' files are the Drive files.",
                      checked=datetime.date.today().isoformat()),
           identity_notes=["File names suggest two different volunteers ('Muku' vs 'Mithin') - this is an inference from names only; no participant metadata exists in the files.",
                           "Each session has three independent SensorTile logs (ECG/EMG/EOG label = intended electrode wiring). Each tile has its own clock starting near 0 us; no shared timestamps.",
                           "Session-2 logs end in one truncated row each (dropped, <=5-row policy in src/ingest/sensortile.py). Raw files untouched."])
json.dump(out, open("results/inhouse_audit/provenance.json", "w"), indent=1); print("ok", len(files))
