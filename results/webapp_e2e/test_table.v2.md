| Test | Expected | Actual | Status |
|---|---|---|---|
| P1 Overview | 200; headline MESA result + historical CNN card | HTTP 200; missing=[] | PASS |
| P2 In-house | 200; inventory with 6 logs, Drive status, provenance | HTTP 200; missing=[] | PASS |
| P3 Models | 200; historical table; controlled section present | HTTP 200; missing=[] | PASS |
| P4 MESA | 200; six configurations listed | HTTP 200; missing=[] | PASS |
| P5 Methods | 200; leakage + limitations + checkpoint hashes | HTTP 200; missing=[] | PASS |
| P6 Detect form | 200; upload form | HTTP 200; missing=[] | PASS |
| P7 Unknown page | HTTP 404 with message | HTTP 404 | PASS |
| P8 Artifact whitelist | png under results/inhouse_audit ok (200); traversal and non-png refused (404) | png 200, traversal 404, json 404 | PASS |
| U1 upload valid recording (session 1 ECG tile) | 200; metadata 1796 s; plots; CNN/RNN/LSTM + effort + report link | HTTP 200; 3+ plots=True; duration 1796s shown=True; CNN/RNN/LSTM rows=True; effort section=True; report link=True | PASS |
| U2 upload different valid recording (session 2 ECG tile) | 200; duration 1574 s (differs from U1 => result tied to upload) | HTTP 200; 3+ plots=True; duration 1574s shown=True; CNN/RNN/LSTM rows=True; effort section=True; report link=True | PASS |
| U3 report export | 200 attachment HTML containing metadata, model SHA, limitations | HTTP 200; disposition=attachment; filename="apnea_screening_re | PASS |
| U3b results are computed from each upload | ECG-branch numbers differ between the two uploads; each report carries its own recording duration | ecg sections differ=True (len 721/720) | PASS |
| U4 unknown report id | HTTP 404 | HTTP 404 | PASS |
| E1 wrong file type (.pdf) | HTTP 400, 'Unsupported file type' | HTTP 400 | PASS |
| E2 garbage .csv | HTTP 400, 'Could not parse' | HTTP 400: linical datasets (PhysioNet Apnea-ECG, MESA); in-house recordings are feasibility data from healthy volunteers. Not clinically validated. 2026-10-19.  | PASS |
| E3 valid format but too short (~2 min) | HTTP 400, 'at least 600 s' | HTTP 400: ESA); in-house recordings are feasibility data from healthy volunteers. Not clinically validated. 2026-10-19.  | PASS |
| E4 empty upload | HTTP 400, 'No file uploaded' | HTTP 400 | PASS |
| E5 session-1 EMG-wired tile | HTTP 400 'no clear cardiac rhythm' (not accepted as ECG) | HTTP 400: Could not analyse this file. The QVAR channel shows no clear cardiac rhythm (spectral prominence 6.6 &lt; 20). This ECG  | PASS |
| E6 session-1 EOG-wired tile | HTTP 400 'no clear cardiac rhythm' (not accepted as ECG) | HTTP 400: Could not analyse this file. The QVAR channel shows no clear cardiac rhythm (spectral prominence 6.1 &lt; 20). This ECG  | PASS |
| E7 session-2 EMG-wired tile | HTTP 400 'no clear cardiac rhythm' (not accepted as ECG) | HTTP 400: Could not analyse this file. The QVAR channel shows no clear cardiac rhythm (spectral prominence 9.2 &lt; 20). This ECG  | PASS |
| E8 IMU-only log (QVAR flat) | HTTP 400, flat QVAR rejected | HTTP 400: Could not analyse this file. QVAR channel is flat or accelerometer contains non-finite values; check the sensor and the  | PASS |
| E9 QVAR = noise (no cardiac rhythm) | HTTP 400, cardiac-rhythm gate | HTTP 400 | PASS |
| R1 restart server (documented command) + GET / | HTTP 200 after restart | HTTP 200; missing=[] | PASS |
| R2 report from before restart is gone (in-memory) | HTTP 404 (documented behaviour) | HTTP 404 | PASS |
| R3 server healthy after errors (demo button) | HTTP 200, effort + ECG sections | HTTP 200 | PASS |
