| Test | Expected | Actual | Status |
|---|---|---|---|
| T1 start + GET / | HTTP 200, upload form | HTTP 200 | PASS |
| T2 upload valid recording (session 1) | 200; metadata duration 1796 s; plots; CNN/RNN/LSTM + effort results | HTTP 200; 3+ plots=True; duration 1796s shown=True; CNN/RNN/LSTM rows=True; effort section=True | PASS |
| T3 upload different valid recording (session 2) | 200; metadata duration 1574 s (differs from T2 => result tied to upload) | HTTP 200; 3+ plots=True; duration 1574s shown=True; CNN/RNN/LSTM rows=True; effort section=True | PASS |
| T4 wrong file type (.pdf) | HTTP 400, 'Unsupported file type' | HTTP 400 | PASS |
| T5 garbage .csv | HTTP 400, 'Could not parse' | HTTP 400: rse as a SensorTile recording: /tmp/tmp8dwzo4f2/x.csv: could not find a header row (looked for &#39;Timestamp&#39; or &#39;time[us]&#39;) in the first 20 lines  | PASS |
| T6 valid format but too short (~2 min) | HTTP 400, 'at least 600 s' | HTTP 400: e longer) Error: Recording is 123 s; at least 600 s (10 min) is required (the effort models use 150 s context windows).  | PASS |
| T7 empty upload | HTTP 400, 'No file uploaded' | HTTP 400 | PASS |
| T8 restart server (documented command) + GET / | HTTP 200 after restart | HTTP 200 | PASS |
| T9 server still healthy after errors (demo button) | HTTP 200, effort + ECG sections | HTTP 200 | PASS |
