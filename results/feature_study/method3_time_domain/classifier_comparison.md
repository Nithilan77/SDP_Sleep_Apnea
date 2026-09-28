# Time-domain HRV bank (method 3) -- classifier comparison

Feature columns: ['sdnn_ms', 'rmssd_ms', 'pnn50', 'mean_hr', 'hr_range_bpm', 'hrv_triangular_index']
Minutes evaluated: 16949 (subject-independent, 35-fold LOSO)

| classifier | accuracy | sensitivity | specificity | precision | f1 | record-acc |
|---|---|---|---|---|---|---|
| cnn | 0.633 | 0.675 | 0.607 | 0.518 | 0.586 | 0.714 |
| logreg **<- winner** | 0.702 | 0.620 | 0.754 | 0.611 | 0.615 | 0.743 |
| mlp | 0.676 | 0.688 | 0.669 | 0.564 | 0.620 | 0.714 |
