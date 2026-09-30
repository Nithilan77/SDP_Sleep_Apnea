# MESA (NSRR) — dataset notes for Track B

## Cohort
- MESA Sleep ancillary study, Exam 5 (2010-2013), home-based PSG (Compumedics Somte).
- 2237 enrolled; 2166 PSGs performed; ~2060 passed minimal quality. ~2000 usable full PSGs.
- Age 54-93 (mean ~68). NOTE: elderly cohort — OPPOSITE of our SensorTile subjects (19-20yo).
  This age/physiology gap is a stated limitation (cf. in-house track: PhysioNet model misread
  young 45bpm bradycardia). MESA is for proving the fusion method + event-type labels, not for
  claiming transfer to our young population.

## Channels we use (exact EDF labels + sampling rates)
- ECG            = EKG   @ 256 Hz  (patch electrodes)
- Thoracic effort= Thor  @ 32 Hz   (Compumedics inductive respiratory band, RIP)
- Abdominal eff. = Abdo  @ 32 Hz   (Compumedics inductive respiratory band, RIP)
- SpO2           = SpO2  @ 1 Hz     (Nonin 8000)
- (ignore HR/DHR derived channels; detect our own R-peaks from EKG, as everywhere else.)
- CAVEAT: Thor/Abdo are RIP belts (circumference change), NOT accelerometers. They are a PROXY
  for our SensorTile IMU chest-motion effort branch, not an identical signal. Effort-branch
  signal processing may need re-tuning belt-vs-accelerometer.

## Apnea/hypopnea event scoring (MESA SRC Scoring Manual, sec 6.2.7)
- Obstructive apnea: airflow amplitude <=10% of baseline for >=10 s.
- Central apnea: apnea with NO displacement on BOTH Thor and Abdo bands (effort stops).
  Cannot be designated if either band is missing/uninterpretable. => obstructive-vs-central
  comes ENTIRELY from the effort belts (this is the core motivation for CANet's dual stream).
- Hypopnea (AASM Alternative): >=50% amplitude reduction on the belts, >=10 s.
- Hypopnea (AASM Recommended): >=30% reduction thorax+abdomen, >=10 s.
- CRITICAL: events are marked INDEPENDENT of desaturation. Desaturation (>=2/3/4%) is applied
  AFTERWARD only to compute the different summary AHI thresholds. => we CAN build a valid
  apnea/hypopnea label set from airflow+effort ALONE. SpO2 is available in MESA to additionally
  build desaturation-linked AHI if wanted. The no-SpO2 gap is a limitation of OUR hardware, not
  of what MESA lets us learn/validate.
- Epochs are 30 s (AASM). Sleep staged per AASM (W, N1, N2, N3, R).

## Labelling plan for our models
- Parse the NSRR event-annotation XML into (event_type, start, duration).
- Build a per-epoch (30 s) or per-minute apnea/normal label grid; choose one and state it.
  (Per-minute keeps continuity with the PhysioNet pipeline; 30 s matches MESA-native scoring.)
- Optionally build event-type labels (obstructive / central / hypopnea) for the stretch goal.

## Access / files
- Download via the `nsrr` Ruby gem using NSRR_TOKEN. Data is gated (approved).
- PSG EDFs and event-annotation XML live under the mesa polysomnography/ tree (confirm exact
  subfolder names by listing the tree — edfs/ and annotations-events-*/).
- Subject key = mesaid.
- data/mesa/ is human-subject data: gitignored, NEVER committed (same rule as data/recordings/).
