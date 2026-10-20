# CNN-LSTM vs baselines -- identical epochs, grouped 5-fold CV, paired subject bootstrap (200 reps)

| model                           |   auroc |   auprc |   sens |   spec |    f1 | d_auroc_vs_baseline   | d_auprc_vs_baseline   |
|:--------------------------------|--------:|--------:|-------:|-------:|------:|:----------------------|:----------------------|
| ECG-only baseline (CardiacCNN)  |   0.609 |   0.335 |  0.622 |  0.535 | 0.416 |                       |                       |
| CANet ECG+effort                |   0.780 |   0.490 |  0.822 |  0.613 | 0.555 | [+0.151, +0.191]      | [+0.132, +0.184]      |
| CANet ECG+effort+SpO2 (ceiling) |   0.813 |   0.541 |  0.853 |  0.641 | 0.587 | [+0.183, +0.224]      | [+0.181, +0.227]      |
| CNN-LSTM ecg_resp_spo2_lstm     |   0.798 |   0.511 |  0.841 |  0.640 | 0.580 | [+0.163, +0.214]      | [+0.142, +0.212]      |
| CNN-LSTM ecg_spo2_lstm          |   0.766 |   0.465 |  0.786 |  0.639 | 0.552 | [+0.135, +0.178]      | [+0.099, +0.159]      |
