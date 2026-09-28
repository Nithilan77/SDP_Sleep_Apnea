# Nonlinear/Poincare HRV bank (method 5) -- classifier comparison

Feature columns: ['sd1_ms', 'sd2_ms', 'sd1_sd2_ratio', 'sampen']
Minutes evaluated: 16207 (subject-independent, 35-fold LOSO)

| classifier | accuracy | sensitivity | specificity | precision | f1 | record-acc |
|---|---|---|---|---|---|---|
| cnn | 0.688 | 0.733 | 0.658 | 0.582 | 0.649 | 0.743 |
| logreg **<- winner** | 0.707 | 0.673 | 0.728 | 0.617 | 0.644 | 0.743 |
| mlp | 0.699 | 0.712 | 0.692 | 0.600 | 0.651 | 0.743 |
