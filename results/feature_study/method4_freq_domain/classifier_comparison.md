# Frequency-domain HRV bank (method 4) -- classifier comparison

Feature columns: ['vlf_power', 'lf_power', 'hf_power', 'lf_hf_ratio']
Minutes evaluated: 16946 (subject-independent, 35-fold LOSO)

| classifier | accuracy | sensitivity | specificity | precision | f1 | record-acc |
|---|---|---|---|---|---|---|
| cnn | 0.450 | 0.701 | 0.293 | 0.382 | 0.495 | 0.714 |
| logreg **<- winner** | 0.672 | 0.513 | 0.772 | 0.584 | 0.546 | 0.743 |
| mlp | 0.651 | 0.647 | 0.654 | 0.539 | 0.588 | 0.714 |
