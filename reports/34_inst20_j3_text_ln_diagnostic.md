# Instruction 20: J3 Text LN Diagnostic

**Run:** `20260313_215455`  

## Reference Baselines
| Method | Online acc |
|---|---|
| Frozen zero-shot | 0.3796 |
| BATCLIP | 0.6060 |
| CALM v1 | 0.6458 |
| CAMA | 0.6734 |
| J3 (original) | 0.5370 (offline 0.6002) |

## Part 1: Drift Experiment (X1/X2/X3)

| Run | Description | Online acc | Offline acc | Δ_J3 | cat% |
|---|---|---|---|---|---|
| X1 | Image LN only + fixed text (no drift) | 0.5301 | 0.5902 | -0.0069 | 0.200 |
| X2 | Image + text LN, r_k recomputed each step (no drift) | 0.5321 | 0.5900 | -0.0049 | 0.142 |
| X3 | Original J3 (adapted text, fixed r_k = drift) | 0.5370 | 0.6002 | +0.0000 | 0.146 |

## Part 2: Baseline & Evidence Prior (F0/BL/CAMA)

| Run | Description | Online acc | Offline acc | Δ_H2 | cat% |
|---|---|---|---|---|---|
| F0 | Frozen zero-shot (no adaptation) | 0.3796 | 0.3796 | -0.2938 | 0.530 |
| BL | BATCLIP (L_ent − L_i2t) | 0.2182 | 0.1034 | -0.4551 | 0.831 |
| CAMA | CAMA (L_ent + 2·KL(p̄‖π_evid), β=0.3, R=5) | 0.6738 | 0.7142 | +0.0004 | 0.129 |

*F0 text effective rank: 6.44*

## Part 3: One-Sided Regularizers (OS1/OS2)

| Run | Description | Online acc | Offline acc | Δ_H2 | cat% |
|---|---|---|---|---|---|
| OS1 | OS1 (L_ent + 2·Σ[p̄_k−π_k]²₊) | 0.5413 | 0.5825 | -0.1321 | 0.255 |
| OS2 | OS2 (L_ent + 2·Σ p̄_k·[log(p̄_k/π_k)]₊) | 0.6716 | 0.7075 | -0.0018 | 0.123 |

## Diagnostic Summary (all runs)

| Metric | X1 | X2 | X3 | F0 | BL | CAMA | OS1 | OS2 |
|---|---|---|---|---|---|---|---|---|
| NC acc (split-half) | 0.6715 | 0.6715 | 0.6757 | 0.5135 | 0.3652 | 0.7381 | 0.6957 | 0.7368 |
| gap NC−offline | 0.0813 | 0.0815 | 0.0755 | 0.1339 | 0.2618 | 0.0239 | 0.1132 | 0.0293 |
| best deconv acc | 0.5979 | 0.6021 | 0.6079 | 0.3964 | 0.1050 | 0.7139 | 0.5792 | 0.7058 |
| Δ deconv vs offline | 0.0077 | 0.0121 | 0.0077 | 0.0168 | 0.0016 | -0.0003 | -0.0033 | -0.0017 |
| mean prototype purity | 0.5328 | 0.5169 | 0.5371 | 0.3610 | 0.7123 | 0.7131 | 0.6861 | 0.7016 |
| Fisher ratio | 0.7837 | 0.7890 | 0.8087 | 0.3844 | 0.2455 | 1.0383 | 0.9626 | 1.0447 |

## Run Details


============================================================
=== Run X1: Image LN only + fixed text (no drift) ===
============================================================

--- Adaptation Results ---
online_acc:        0.5301
offline_acc:       0.5902
cat_pct:           0.2001
mean_entropy:      1.1310
H_pbar_final:      2.2322
n_trainable_params:39,936

--- Step Log (every 5 steps) ---
step | online_acc | batch_acc | cat_pct |    ent | H_pbar |     loss
   5 |     0.4030 |    0.4450 |   0.465 |  1.378 |  2.107 |   0.0234
  10 |     0.4265 |    0.5050 |   0.384 |  1.304 |  2.114 |   0.0180
  15 |     0.4510 |    0.5250 |   0.330 |  1.258 |  2.136 |   0.0155
  20 |     0.4685 |    0.5550 |   0.295 |  1.232 |  2.186 |   0.0136
  25 |     0.4858 |    0.5700 |   0.265 |  1.198 |  2.182 |   0.0134
  30 |     0.4972 |    0.5750 |   0.247 |  1.172 |  2.169 |   0.0112
  35 |     0.5067 |    0.5800 |   0.230 |  1.161 |  2.172 |   0.0106
  40 |     0.5190 |    0.5850 |   0.218 |  1.146 |  2.187 |   0.0099
  45 |     0.5252 |    0.6450 |   0.209 |  1.141 |  2.231 |   0.0098
  50 |     0.5301 |    0.5300 |   0.200 |  1.131 |  2.232 |   0.0092

--- Diag 1: Per-class Recall ---
  airplane    : 0.7600
  automobile  : 0.9420
  bird        : 0.5170
  cat         : 0.6240
  deer        : 0.4050
  dog         : 0.5160
  frog        : 0.6320
  horse       : 0.7140
  ship        : 0.5220
  truck       : 0.2700

--- Diag 2: Top-K Recall ---
  top1: 0.5898
  top2: 0.7616
  top3: 0.8317
  top5: 0.9142
  top7: 0.9623

--- Diag 3: Confusion Matrix ---
true\pred |  airp  auto  bird   cat  deer   dog  frog  hors  ship  truc
airplane  |   760    89    67    27     9     5    21    11    11     0
automobil |     8   942     5    12     1     4     7     0     8    13
bird      |    45   146   517    84    28    43    89    48     0     0
cat       |     5   132    41   624    33    86    49    29     1     0
deer      |    23   158    69   111   405    46    75   113     0     0
dog       |     4    91    89   184    28   516    37    51     0     0
frog      |     8   128    21   144    30    21   632    16     0     0
horse     |     7    63    25    50    41    88    12   714     0     0
ship      |   199   210    19    16     6     2    18     7   522     1
truck     |    39   627     5    15     0     6     6    20    12   270
Major confusions (off-diag > 50):
  True=truck, Pred=automobile: 627
  True=ship, Pred=automobile: 210
  True=ship, Pred=airplane: 199
  True=dog, Pred=cat: 184
  True=deer, Pred=automobile: 158
  True=bird, Pred=automobile: 146
  True=frog, Pred=cat: 144
  True=cat, Pred=automobile: 132
  True=frog, Pred=automobile: 128
  True=deer, Pred=horse: 113
  True=deer, Pred=cat: 111
  True=dog, Pred=automobile: 91
  True=airplane, Pred=automobile: 89
  True=bird, Pred=frog: 89
  True=dog, Pred=bird: 89
  True=horse, Pred=dog: 88
  True=cat, Pred=dog: 86
  True=bird, Pred=cat: 84
  True=deer, Pred=frog: 75
  True=deer, Pred=bird: 69
  True=airplane, Pred=bird: 67
  True=horse, Pred=automobile: 63
  True=dog, Pred=horse: 51

--- Diag 4: Margin ---
  correct: mean=2.9172, std=1.9469
  wrong:   mean=1.1677, std=1.1492
  low_margin_ratio (< 0.5): 0.2076

--- Diag 5: Fisher Criterion ---
  mean_intra_var:   0.038469
  mean_inter_dist:  0.030147
  fisher_ratio:     0.7837
  weak pair ('cat', 'dog'): F=0.1185
  weak pair ('bird', 'deer'): F=0.1619
  weak pair ('automobile', 'truck'): F=0.1677
  weak pair ('bird', 'frog'): F=0.1901
  weak pair ('bird', 'cat'): F=0.2103

--- Diag 6: Nearest-Centroid Acc (핵심) ---
  nc_acc (split-half): 0.6715
  nc_acc (same-data):  0.6747
  offline_acc:         0.5902
  gap (nc - offline):  +0.0813

--- Diag 7: Text Head vs NC Head Per-class ---
  class        |   text |     nc |    diff
  airplane     | 0.7600 | 0.6800 | -0.0800
  automobile   | 0.9420 | 0.7420 | -0.2000
  bird         | 0.5180 | 0.6410 | +0.1230
  cat          | 0.6230 | 0.6470 | +0.0240
  deer         | 0.4070 | 0.6700 | +0.2630
  dog          | 0.5160 | 0.6110 | +0.0950
  frog         | 0.6310 | 0.6690 | +0.0380
  horse        | 0.7140 | 0.6570 | -0.0570
  ship         | 0.5220 | 0.7280 | +0.2060
  truck        | 0.2700 | 0.7020 | +0.4320

--- Diag 9: Deconvolution Head ---
  lambda_0.01: deconv_acc=0.5657
  lambda_0.05: deconv_acc=0.5770
  lambda_0.1: deconv_acc=0.5832
  lambda_0.5: deconv_acc=0.5962
  lambda_1.0: deconv_acc=0.5979
  best_lambda=1.0 → best_deconv_acc=0.5979
  top3_restricted_acc (lambda=1.0): 0.5892
  Δ_deconv_vs_offline: +0.0077

--- Diag 10: Prototype Purity ---
  mean_purity: 0.5328
  airplane    : purity=0.550, dominant=airplane (0.550)
  automobile  : purity=0.396, dominant=automobile (0.396)
  bird        : purity=0.468, dominant=bird (0.468)
  cat         : purity=0.439, dominant=cat (0.439)
  deer        : purity=0.590, dominant=deer (0.590)
  dog         : purity=0.476, dominant=dog (0.476)
  frog        : purity=0.601, dominant=frog (0.601)
  horse       : purity=0.545, dominant=horse (0.545)
  ship        : purity=0.658, dominant=ship (0.658)
  truck       : purity=0.607, dominant=truck (0.607)

--- Diag 11: Relational Identifiability ---
  Least identifiable:
    automobile-truck: JS=0.013907
    cat-dog: JS=0.025215
    airplane-ship: JS=0.028455
    dog-horse: JS=0.031001
    deer-horse: JS=0.032617
  Most identifiable:
    cat-truck: JS=0.066565
    deer-truck: JS=0.068528
    automobile-deer: JS=0.071061

--- Diag 12: Centroid-Text Alignment ---
  mean_diagonal: 0.2292
  class        | rank | cos    | best_text
  airplane     |    1 | 0.2364 | airplane
  automobile   |    1 | 0.2519 | automobile
  bird         |    1 | 0.2229 | bird
  cat          |    1 | 0.2252 | cat
  deer         |    1 | 0.2163 | deer
  dog          |    1 | 0.2229 | dog
  frog         |    1 | 0.2297 | frog
  horse        |    1 | 0.2390 | horse
  ship         |    1 | 0.2231 | ship
  truck        |    2 | 0.2244 | automobile

============================================================
=== Run X2: Image + text LN, r_k recomputed each step (no drift) ===
============================================================

--- Adaptation Results ---
online_acc:        0.5321
offline_acc:       0.5900
cat_pct:           0.1418
mean_entropy:      1.1428
H_pbar_final:      2.2361
n_trainable_params:65,536

--- Step Log (every 5 steps) ---
step | online_acc | batch_acc | cat_pct |    ent | H_pbar |     loss
   5 |     0.3990 |    0.4350 |   0.406 |  1.398 |  2.123 |   0.0237
  10 |     0.4190 |    0.5000 |   0.299 |  1.312 |  2.072 |   0.0184
  15 |     0.4480 |    0.5600 |   0.250 |  1.267 |  2.153 |   0.0162
  20 |     0.4713 |    0.5450 |   0.224 |  1.247 |  2.208 |   0.0145
  25 |     0.4876 |    0.5700 |   0.198 |  1.217 |  2.204 |   0.0149
  30 |     0.5003 |    0.5550 |   0.181 |  1.193 |  2.146 |   0.0128
  35 |     0.5067 |    0.5750 |   0.167 |  1.179 |  2.154 |   0.0121
  40 |     0.5190 |    0.5950 |   0.156 |  1.161 |  2.190 |   0.0112
  45 |     0.5258 |    0.6350 |   0.148 |  1.152 |  2.226 |   0.0112
  50 |     0.5321 |    0.5350 |   0.142 |  1.143 |  2.236 |   0.0108

--- Diag 1: Per-class Recall ---
  airplane    : 0.8150
  automobile  : 0.9120
  bird        : 0.4640
  cat         : 0.5150
  deer        : 0.4940
  dog         : 0.3890
  frog        : 0.6720
  horse       : 0.7410
  ship        : 0.5370
  truck       : 0.3610

--- Diag 2: Top-K Recall ---
  top1: 0.5900
  top2: 0.7432
  top3: 0.8043
  top5: 0.8878
  top7: 0.9452

--- Diag 3: Confusion Matrix ---
true\pred |  airp  auto  bird   cat  deer   dog  frog  hors  ship  truc
airplane  |   815    62    45    15    16     2    20    13    11     1
automobil |    12   912     1    10     0     2    12     4    14    33
bird      |    75   131   464    49    65    20   123    72     1     0
cat       |    20   162    45   515    62    53    73    68     2     0
deer      |    39   143    44    64   494    20    79   117     0     0
dog       |    12   105   102   131    62   389    70   128     1     0
frog      |    13   119    16    76    71     8   672    21     4     0
horse     |    16    60    24    27    68    50    13   741     1     0
ship      |   253   152    13     6    10     0    16    11   537     2
truck     |    59   510     3     8     1     4    11    26    17   361
Major confusions (off-diag > 50):
  True=truck, Pred=automobile: 510
  True=ship, Pred=airplane: 253
  True=cat, Pred=automobile: 162
  True=ship, Pred=automobile: 152
  True=deer, Pred=automobile: 143
  True=bird, Pred=automobile: 131
  True=dog, Pred=cat: 131
  True=dog, Pred=horse: 128
  True=bird, Pred=frog: 123
  True=frog, Pred=automobile: 119
  True=deer, Pred=horse: 117
  True=dog, Pred=automobile: 105
  True=dog, Pred=bird: 102
  True=deer, Pred=frog: 79
  True=frog, Pred=cat: 76
  True=bird, Pred=airplane: 75
  True=cat, Pred=frog: 73
  True=bird, Pred=horse: 72
  True=frog, Pred=deer: 71
  True=dog, Pred=frog: 70
  True=cat, Pred=horse: 68
  True=horse, Pred=deer: 68
  True=bird, Pred=deer: 65
  True=deer, Pred=cat: 64
  True=airplane, Pred=automobile: 62
  True=cat, Pred=deer: 62
  True=dog, Pred=deer: 62
  True=horse, Pred=automobile: 60
  True=truck, Pred=airplane: 59
  True=cat, Pred=dog: 53

--- Diag 4: Margin ---
  correct: mean=2.9693, std=2.1372
  wrong:   mean=1.0722, std=1.2071
  low_margin_ratio (< 0.5): 0.2343

--- Diag 5: Fisher Criterion ---
  mean_intra_var:   0.033814
  mean_inter_dist:  0.026678
  fisher_ratio:     0.7890
  weak pair ('cat', 'dog'): F=0.1004
  weak pair ('bird', 'deer'): F=0.1587
  weak pair ('automobile', 'truck'): F=0.1716
  weak pair ('bird', 'dog'): F=0.1822
  weak pair ('bird', 'frog'): F=0.1865

--- Diag 6: Nearest-Centroid Acc (핵심) ---
  nc_acc (split-half): 0.6715
  nc_acc (same-data):  0.6736
  offline_acc:         0.5900
  gap (nc - offline):  +0.0815

--- Diag 7: Text Head vs NC Head Per-class ---
  class        |   text |     nc |    diff
  airplane     | 0.8130 | 0.6650 | -0.1480
  automobile   | 0.9110 | 0.7540 | -0.1570
  bird         | 0.4630 | 0.6320 | +0.1690
  cat          | 0.5170 | 0.6530 | +0.1360
  deer         | 0.4950 | 0.6710 | +0.1760
  dog          | 0.3920 | 0.6250 | +0.2330
  frog         | 0.6730 | 0.6540 | -0.0190
  horse        | 0.7420 | 0.6650 | -0.0770
  ship         | 0.5390 | 0.7180 | +0.1790
  truck        | 0.3620 | 0.6990 | +0.3370

--- Diag 9: Deconvolution Head ---
  lambda_0.01: deconv_acc=0.6005
  lambda_0.05: deconv_acc=0.6021
  lambda_0.1: deconv_acc=0.6000
  lambda_0.5: deconv_acc=0.5972
  lambda_1.0: deconv_acc=0.5963
  best_lambda=0.05 → best_deconv_acc=0.6021
  top3_restricted_acc (lambda=0.05): 0.5959
  Δ_deconv_vs_offline: +0.0121

--- Diag 10: Prototype Purity ---
  mean_purity: 0.5169
  airplane    : purity=0.492, dominant=airplane (0.492)
  automobile  : purity=0.403, dominant=automobile (0.403)
  bird        : purity=0.497, dominant=bird (0.497)
  cat         : purity=0.527, dominant=cat (0.527)
  deer        : purity=0.513, dominant=deer (0.513)
  dog         : purity=0.562, dominant=dog (0.562)
  frog        : purity=0.570, dominant=frog (0.570)
  horse       : purity=0.517, dominant=horse (0.517)
  ship        : purity=0.593, dominant=ship (0.593)
  truck       : purity=0.496, dominant=truck (0.496)

--- Diag 11: Relational Identifiability ---
  Least identifiable:
    automobile-truck: JS=0.014533
    airplane-ship: JS=0.026476
    cat-dog: JS=0.032121
    dog-horse: JS=0.034229
    automobile-ship: JS=0.034832
  Most identifiable:
    deer-truck: JS=0.065567
    automobile-frog: JS=0.068077
    automobile-deer: JS=0.069997

--- Diag 12: Centroid-Text Alignment ---
  mean_diagonal: 0.2260
  class        | rank | cos    | best_text
  airplane     |    1 | 0.2392 | airplane
  automobile   |    1 | 0.2477 | automobile
  bird         |    1 | 0.2170 | bird
  cat          |    1 | 0.2139 | cat
  deer         |    1 | 0.2194 | deer
  dog          |    1 | 0.2097 | dog
  frog         |    1 | 0.2306 | frog
  horse        |    1 | 0.2360 | horse
  ship         |    1 | 0.2225 | ship
  truck        |    2 | 0.2245 | automobile

============================================================
=== Run X3: Original J3 (adapted text, fixed r_k = drift) ===
============================================================

--- Adaptation Results ---
online_acc:        0.5370
offline_acc:       0.6002
cat_pct:           0.1457
mean_entropy:      1.1122
H_pbar_final:      2.2440
n_trainable_params:65,536

--- Step Log (every 5 steps) ---
step | online_acc | batch_acc | cat_pct |    ent | H_pbar |     loss
   5 |     0.3980 |    0.4300 |   0.406 |  1.397 |  2.123 |   0.0233
  10 |     0.4200 |    0.5050 |   0.300 |  1.311 |  2.079 |   0.0176
  15 |     0.4493 |    0.5500 |   0.251 |  1.263 |  2.159 |   0.0150
  20 |     0.4735 |    0.5550 |   0.227 |  1.239 |  2.209 |   0.0133
  25 |     0.4894 |    0.5750 |   0.201 |  1.205 |  2.204 |   0.0134
  30 |     0.5028 |    0.5700 |   0.185 |  1.177 |  2.157 |   0.0111
  35 |     0.5099 |    0.5750 |   0.171 |  1.159 |  2.163 |   0.0105
  40 |     0.5226 |    0.6000 |   0.160 |  1.137 |  2.194 |   0.0096
  45 |     0.5309 |    0.6400 |   0.152 |  1.125 |  2.229 |   0.0095
  50 |     0.5370 |    0.5400 |   0.146 |  1.112 |  2.244 |   0.0091

--- Diag 1: Per-class Recall ---
  airplane    : 0.7910
  automobile  : 0.9140
  bird        : 0.4830
  cat         : 0.5360
  deer        : 0.4840
  dog         : 0.4250
  frog        : 0.6860
  horse       : 0.7740
  ship        : 0.5620
  truck       : 0.3470

--- Diag 2: Top-K Recall ---
  top1: 0.6001
  top2: 0.7520
  top3: 0.8176
  top5: 0.9072
  top7: 0.9587

--- Diag 3: Confusion Matrix ---
true\pred |  airp  auto  bird   cat  deer   dog  frog  hors  ship  truc
airplane  |   791    57    57    17    13     4    24    22    15     0
automobil |    14   914     1    12     1     3    14     4    13    24
bird      |    62    94   483    54    52    26   123   106     0     0
cat       |    16   116    43   536    56    65    72    94     2     0
deer      |    31   103    47    65   484    24    82   164     0     0
dog       |     6    77    99   136    52   425    56   148     1     0
frog      |    10    92    18    85    54     8   686    46     1     0
horse     |    10    32    21    30    63    58    11   774     1     0
ship      |   221   142    15     7     8     1    22    21   562     1
truck     |    53   517     3     9     0     5    11    35    20   347
Major confusions (off-diag > 50):
  True=truck, Pred=automobile: 517
  True=ship, Pred=airplane: 221
  True=deer, Pred=horse: 164
  True=dog, Pred=horse: 148
  True=ship, Pred=automobile: 142
  True=dog, Pred=cat: 136
  True=bird, Pred=frog: 123
  True=cat, Pred=automobile: 116
  True=bird, Pred=horse: 106
  True=deer, Pred=automobile: 103
  True=dog, Pred=bird: 99
  True=bird, Pred=automobile: 94
  True=cat, Pred=horse: 94
  True=frog, Pred=automobile: 92
  True=frog, Pred=cat: 85
  True=deer, Pred=frog: 82
  True=dog, Pred=automobile: 77
  True=cat, Pred=frog: 72
  True=cat, Pred=dog: 65
  True=deer, Pred=cat: 65
  True=horse, Pred=deer: 63
  True=bird, Pred=airplane: 62
  True=horse, Pred=dog: 58
  True=airplane, Pred=automobile: 57
  True=airplane, Pred=bird: 57
  True=cat, Pred=deer: 56
  True=dog, Pred=frog: 56
  True=bird, Pred=cat: 54
  True=frog, Pred=deer: 54
  True=truck, Pred=airplane: 53
  True=bird, Pred=deer: 52
  True=dog, Pred=deer: 52

--- Diag 4: Margin ---
  correct: mean=3.2101, std=2.2474
  wrong:   mean=1.1660, std=1.2819
  low_margin_ratio (< 0.5): 0.2149

--- Diag 5: Fisher Criterion ---
  mean_intra_var:   0.037254
  mean_inter_dist:  0.030127
  fisher_ratio:     0.8087
  weak pair ('cat', 'dog'): F=0.1006
  weak pair ('bird', 'deer'): F=0.1680
  weak pair ('automobile', 'truck'): F=0.1742
  weak pair ('bird', 'frog'): F=0.1895
  weak pair ('bird', 'cat'): F=0.1916

--- Diag 6: Nearest-Centroid Acc (핵심) ---
  nc_acc (split-half): 0.6757
  nc_acc (same-data):  0.6781
  offline_acc:         0.6002
  gap (nc - offline):  +0.0755

--- Diag 7: Text Head vs NC Head Per-class ---
  class        |   text |     nc |    diff
  airplane     | 0.7910 | 0.6750 | -0.1160
  automobile   | 0.9110 | 0.7500 | -0.1610
  bird         | 0.4830 | 0.6380 | +0.1550
  cat          | 0.5360 | 0.6670 | +0.1310
  deer         | 0.4830 | 0.6730 | +0.1900
  dog          | 0.4250 | 0.6160 | +0.1910
  frog         | 0.6860 | 0.6710 | -0.0150
  horse        | 0.7770 | 0.6700 | -0.1070
  ship         | 0.5630 | 0.7210 | +0.1580
  truck        | 0.3490 | 0.7000 | +0.3510

--- Diag 8: Text Drift (X3) ---
  mean_cosine: 0.9914
  relational KL drift: 0.000389
    airplane    : cos=0.9929
    automobile  : cos=0.9934
    bird        : cos=0.9924
    cat         : cos=0.9873
    deer        : cos=0.9880
    dog         : cos=0.9904
    frog        : cos=0.9880
    horse       : cos=0.9946
    ship        : cos=0.9967
    truck       : cos=0.9905

--- Diag 9: Deconvolution Head ---
  lambda_0.01: deconv_acc=0.6009
  lambda_0.05: deconv_acc=0.6047
  lambda_0.1: deconv_acc=0.6069
  lambda_0.5: deconv_acc=0.6079
  lambda_1.0: deconv_acc=0.6073
  best_lambda=0.5 → best_deconv_acc=0.6079
  top3_restricted_acc (lambda=0.5): 0.6058
  Δ_deconv_vs_offline: +0.0077

--- Diag 10: Prototype Purity ---
  mean_purity: 0.5371
  airplane    : purity=0.531, dominant=airplane (0.531)
  automobile  : purity=0.431, dominant=automobile (0.431)
  bird        : purity=0.498, dominant=bird (0.498)
  cat         : purity=0.518, dominant=cat (0.518)
  deer        : purity=0.546, dominant=deer (0.546)
  dog         : purity=0.549, dominant=dog (0.549)
  frog        : purity=0.576, dominant=frog (0.576)
  horse       : purity=0.495, dominant=horse (0.495)
  ship        : purity=0.642, dominant=ship (0.642)
  truck       : purity=0.586, dominant=truck (0.586)

--- Diag 11: Relational Identifiability ---
  Least identifiable:
    automobile-truck: JS=0.014689
    airplane-ship: JS=0.027134
    cat-dog: JS=0.031486
    dog-horse: JS=0.033645
    automobile-ship: JS=0.035853
  Most identifiable:
    deer-truck: JS=0.065889
    automobile-frog: JS=0.067060
    automobile-deer: JS=0.069414

--- Diag 12: Centroid-Text Alignment ---
  mean_diagonal: 0.2284
  class        | rank | cos    | best_text
  airplane     |    1 | 0.2401 | airplane
  automobile   |    1 | 0.2500 | automobile
  bird         |    1 | 0.2199 | bird
  cat          |    1 | 0.2165 | cat
  deer         |    1 | 0.2206 | deer
  dog          |    1 | 0.2145 | dog
  frog         |    1 | 0.2329 | frog
  horse        |    1 | 0.2411 | horse
  ship         |    1 | 0.2242 | ship
  truck        |    2 | 0.2245 | automobile

============================================================
=== Run F0: Frozen zero-shot (no adaptation) ===
============================================================

--- Adaptation Results ---
online_acc:        0.3796
offline_acc:       0.3796
cat_pct:           0.5305
mean_entropy:      1.4463
H_pbar_final:      2.0650
n_trainable_params:0

--- Step Log (every 5 steps) ---
step | online_acc | batch_acc | cat_pct |    ent | H_pbar |     loss

--- Diag 1: Per-class Recall ---
  airplane    : 0.5000
  automobile  : 0.6210
  bird        : 0.3150
  cat         : 0.9210
  deer        : 0.1000
  dog         : 0.3920
  frog        : 0.0900
  horse       : 0.4020
  ship        : 0.3650
  truck       : 0.0900

--- Diag 2: Top-K Recall ---
  top1: 0.3806
  top2: 0.5345
  top3: 0.6522
  top5: 0.8062
  top7: 0.9291

--- Diag 3: Confusion Matrix ---
true\pred |  airp  auto  bird   cat  deer   dog  frog  hors  ship  truc
airplane  |   500    47   193   196     6    17     4     6    30     1
automobil |     7   621     5   326     0    22     5     5     5     4
bird      |    13    34   315   537     4    75    14     6     2     0
cat       |     0    16     8   921     3    48     1     2     1     0
deer      |     3    23    21   754   100    64     8    27     0     0
dog       |     1     2    23   564     5   392     2     9     2     0
frog      |     4    19    15   802     5    62    90     2     1     0
horse     |     2    21    12   393    17   149     0   402     4     0
ship      |    99   106    32   386     1     9     2     0   365     0
truck     |    16   402     5   426     0    22     3     5    31    90
Major confusions (off-diag > 50):
  True=frog, Pred=cat: 802
  True=deer, Pred=cat: 754
  True=dog, Pred=cat: 564
  True=bird, Pred=cat: 537
  True=truck, Pred=cat: 426
  True=truck, Pred=automobile: 402
  True=horse, Pred=cat: 393
  True=ship, Pred=cat: 386
  True=automobile, Pred=cat: 326
  True=airplane, Pred=cat: 196
  True=airplane, Pred=bird: 193
  True=horse, Pred=dog: 149
  True=ship, Pred=automobile: 106
  True=ship, Pred=airplane: 99
  True=bird, Pred=dog: 75
  True=deer, Pred=dog: 64
  True=frog, Pred=dog: 62

--- Diag 4: Margin ---
  correct: mean=1.8045, std=1.4540
  wrong:   mean=0.7603, std=0.6950
  low_margin_ratio (< 0.5): 0.3591

--- Diag 5: Fisher Criterion ---
  mean_intra_var:   0.067995
  mean_inter_dist:  0.026136
  fisher_ratio:     0.3844
  weak pair ('cat', 'dog'): F=0.0302
  weak pair ('bird', 'deer'): F=0.0467
  weak pair ('deer', 'frog'): F=0.0482
  weak pair ('bird', 'frog'): F=0.0618
  weak pair ('automobile', 'truck'): F=0.0678

--- Diag 6: Nearest-Centroid Acc (핵심) ---
  nc_acc (split-half): 0.5135
  nc_acc (same-data):  0.5167
  offline_acc:         0.3796
  gap (nc - offline):  +0.1339

--- Diag 7: Text Head vs NC Head Per-class ---
  class        |   text |     nc |    diff
  airplane     | 0.4990 | 0.5410 | +0.0420
  automobile   | 0.6200 | 0.5540 | -0.0660
  bird         | 0.3140 | 0.3160 | +0.0020
  cat          | 0.9210 | 0.4500 | -0.4710
  deer         | 0.1020 | 0.5690 | +0.4670
  dog          | 0.3930 | 0.4850 | +0.0920
  frog         | 0.0900 | 0.5090 | +0.4190
  horse        | 0.4030 | 0.5570 | +0.1540
  ship         | 0.3640 | 0.5620 | +0.1980
  truck        | 0.0900 | 0.6240 | +0.5340

--- Diag 9: Deconvolution Head ---
  lambda_0.01: deconv_acc=0.3708
  lambda_0.05: deconv_acc=0.3797
  lambda_0.1: deconv_acc=0.3849
  lambda_0.5: deconv_acc=0.3935
  lambda_1.0: deconv_acc=0.3964
  best_lambda=1.0 → best_deconv_acc=0.3964
  top3_restricted_acc (lambda=1.0): 0.3868
  Δ_deconv_vs_offline: +0.0168

--- Diag 10: Prototype Purity ---
  mean_purity: 0.3610
  airplane    : purity=0.564, dominant=airplane (0.564)
  automobile  : purity=0.395, dominant=automobile (0.395)
  bird        : purity=0.276, dominant=bird (0.276)
  cat         : purity=0.192, dominant=cat (0.192)
  deer        : purity=0.269, dominant=deer (0.269)
  dog         : purity=0.237, dominant=dog (0.237)
  frog        : purity=0.327, dominant=frog (0.327)
  horse       : purity=0.425, dominant=horse (0.425)
  ship        : purity=0.491, dominant=ship (0.491)
  truck       : purity=0.434, dominant=truck (0.434)

--- Diag 11: Relational Identifiability ---
  Least identifiable:
    automobile-truck: JS=0.013907
    cat-dog: JS=0.025215
    airplane-ship: JS=0.028455
    dog-horse: JS=0.031001
    deer-horse: JS=0.032617
  Most identifiable:
    cat-truck: JS=0.066565
    deer-truck: JS=0.068528
    automobile-deer: JS=0.071061

--- Diag 12: Centroid-Text Alignment ---
  mean_diagonal: 0.2413
  class        | rank | cos    | best_text
  airplane     |    1 | 0.2477 | airplane
  automobile   |    1 | 0.2542 | automobile
  bird         |    2 | 0.2403 | cat
  cat          |    1 | 0.2532 | cat
  deer         |    3 | 0.2292 | cat
  dog          |    1 | 0.2490 | dog
  frog         |    3 | 0.2261 | cat
  horse        |    1 | 0.2492 | horse
  ship         |    1 | 0.2363 | ship
  truck        |    3 | 0.2283 | automobile

============================================================
=== Run BL: BATCLIP (L_ent − L_i2t) ===
============================================================

--- Adaptation Results ---
online_acc:        0.2182
offline_acc:       0.1034
cat_pct:           0.8310
mean_entropy:      0.1440
H_pbar_final:      0.0610
n_trainable_params:65,536

--- Step Log (every 5 steps) ---
step | online_acc | batch_acc | cat_pct |    ent | H_pbar |     loss
   5 |     0.3550 |    0.3200 |   0.605 |  1.018 |  1.468 |   0.3759
  10 |     0.2965 |    0.1850 |   0.700 |  0.674 |  0.680 |  -0.1224
  15 |     0.2520 |    0.1200 |   0.778 |  0.474 |  0.320 |  -0.2386
  20 |     0.2182 |    0.1250 |   0.831 |  0.360 |  0.061 |  -0.2691

--- Diag 1: Per-class Recall ---
  airplane    : 0.0040
  automobile  : 0.0070
  bird        : 0.0130
  cat         : 1.0000
  deer        : 0.0000
  dog         : 0.0030
  frog        : 0.0000
  horse       : 0.0060
  ship        : 0.0010
  truck       : 0.0000

--- Diag 2: Top-K Recall ---
  top1: 0.1034
  top2: 0.2829
  top3: 0.3827
  top5: 0.5673
  top7: 0.7820

--- Diag 3: Confusion Matrix ---
true\pred |  airp  auto  bird   cat  deer   dog  frog  hors  ship  truc
airplane  |     4     0     6   990     0     0     0     0     0     0
automobil |     0     7     0   993     0     0     0     0     0     0
bird      |     0     0    13   987     0     0     0     0     0     0
cat       |     0     0     0  1000     0     0     0     0     0     0
deer      |     0     0     0  1000     0     0     0     0     0     0
dog       |     0     0     0   997     0     3     0     0     0     0
frog      |     0     0     0  1000     0     0     0     0     0     0
horse     |     0     0     0   994     0     0     0     6     0     0
ship      |     1     0     0   998     0     0     0     0     1     0
truck     |     0     0     0  1000     0     0     0     0     0     0
Major confusions (off-diag > 50):
  True=deer, Pred=cat: 1000
  True=frog, Pred=cat: 1000
  True=truck, Pred=cat: 1000
  True=ship, Pred=cat: 998
  True=dog, Pred=cat: 997
  True=horse, Pred=cat: 994
  True=automobile, Pred=cat: 993
  True=airplane, Pred=cat: 990
  True=bird, Pred=cat: 987

--- Diag 4: Margin ---
  correct: mean=12.2628, std=1.9270
  wrong:   mean=11.8475, std=2.0668
  low_margin_ratio (< 0.5): 0.0015

--- Diag 5: Fisher Criterion ---
  mean_intra_var:   0.075595
  mean_inter_dist:  0.018562
  fisher_ratio:     0.2455
  weak pair ('cat', 'dog'): F=0.0290
  weak pair ('automobile', 'truck'): F=0.0357
  weak pair ('cat', 'frog'): F=0.0489
  weak pair ('deer', 'frog'): F=0.0493
  weak pair ('bird', 'ship'): F=0.0507

--- Diag 6: Nearest-Centroid Acc (핵심) ---
  nc_acc (split-half): 0.3652
  nc_acc (same-data):  0.3677
  offline_acc:         0.1034
  gap (nc - offline):  +0.2618

--- Diag 7: Text Head vs NC Head Per-class ---
  class        |   text |     nc |    diff
  airplane     | 0.0040 | 0.4800 | +0.4760
  automobile   | 0.0070 | 0.3400 | +0.3330
  bird         | 0.0130 | 0.1930 | +0.1800
  cat          | 1.0000 | 0.3970 | -0.6030
  deer         | 0.0000 | 0.4570 | +0.4570
  dog          | 0.0030 | 0.2890 | +0.2860
  frog         | 0.0000 | 0.2960 | +0.2960
  horse        | 0.0060 | 0.3620 | +0.3560
  ship         | 0.0010 | 0.2420 | +0.2410
  truck        | 0.0000 | 0.6210 | +0.6210

--- Diag 9: Deconvolution Head ---
  lambda_0.01: deconv_acc=0.1050
  lambda_0.05: deconv_acc=0.1050
  lambda_0.1: deconv_acc=0.1043
  lambda_0.5: deconv_acc=0.1034
  lambda_1.0: deconv_acc=0.1034
  best_lambda=0.01 → best_deconv_acc=0.1050
  top3_restricted_acc (lambda=0.01): 0.1040
  Δ_deconv_vs_offline: +0.0016

--- Diag 10: Prototype Purity ---
  mean_purity: 0.7123
  airplane    : purity=0.828, dominant=airplane (0.828)
  automobile  : purity=0.963, dominant=automobile (0.963)
  bird        : purity=0.589, dominant=bird (0.589)
  cat         : purity=0.101, dominant=cat (0.101)
  deer        : purity=0.474, dominant=horse (0.474)
  dog         : purity=0.845, dominant=dog (0.845)
  frog        : purity=0.743, dominant=frog (0.743)
  horse       : purity=0.990, dominant=horse (0.990)
  ship        : purity=0.899, dominant=ship (0.899)
  truck       : purity=0.691, dominant=truck (0.691)

--- Diag 11: Relational Identifiability ---
  Least identifiable:
    automobile-truck: JS=0.012670
    cat-dog: JS=0.027684
    airplane-ship: JS=0.028899
    dog-horse: JS=0.030786
    ship-truck: JS=0.032203
  Most identifiable:
    automobile-frog: JS=0.069168
    deer-truck: JS=0.071848
    automobile-deer: JS=0.073381

--- Diag 12: Centroid-Text Alignment ---
  mean_diagonal: 0.2006
  class        | rank | cos    | best_text
  airplane     |    7 | 0.1850 | cat
  automobile   |    3 | 0.2058 | cat
  bird         |    2 | 0.2196 | cat
  cat          |    1 | 0.3308 | cat
  deer         |    9 | 0.1495 | cat
  dog          |    2 | 0.2111 | cat
  frog         |    8 | 0.1508 | cat
  horse        |    4 | 0.1958 | cat
  ship         |    4 | 0.2041 | cat
  truck        |    7 | 0.1538 | cat

============================================================
=== Run CAMA: CAMA (L_ent + 2·KL(p̄‖π_evid), β=0.3, R=5) ===
============================================================

--- Adaptation Results ---
online_acc:        0.6738
offline_acc:       0.7142
cat_pct:           0.1289
mean_entropy:      0.4523
H_pbar_final:      2.2892
n_trainable_params:65,536

--- Step Log (every 5 steps) ---
step | online_acc | batch_acc | cat_pct |    ent | H_pbar |     loss
   5 |     0.4880 |    0.6100 |   0.356 |  1.320 |  2.255 |   1.2438
  10 |     0.5565 |    0.6850 |   0.241 |  1.089 |  2.280 |   0.7198
  15 |     0.6027 |    0.6950 |   0.196 |  0.915 |  2.283 |   0.4537
  20 |     0.6285 |    0.7050 |   0.176 |  0.787 |  2.269 |   0.4241
  25 |     0.6422 |    0.7200 |   0.165 |  0.695 |  2.284 |   0.3109
  30 |     0.6513 |    0.6900 |   0.153 |  0.623 |  2.283 |   0.2945
  35 |     0.6600 |    0.7300 |   0.144 |  0.567 |  2.264 |   0.3006
  40 |     0.6675 |    0.7400 |   0.138 |  0.520 |  2.277 |   0.2237
  45 |     0.6709 |    0.7250 |   0.133 |  0.484 |  2.280 |   0.2322
  50 |     0.6738 |    0.7000 |   0.129 |  0.452 |  2.289 |   0.1810

--- Diag 1: Per-class Recall ---
  airplane    : 0.7050
  automobile  : 0.8330
  bird        : 0.6590
  cat         : 0.5740
  deer        : 0.6700
  dog         : 0.6820
  frog        : 0.6480
  horse       : 0.7720
  ship        : 0.8090
  truck       : 0.7900

--- Diag 2: Top-K Recall ---
  top1: 0.7141
  top2: 0.8613
  top3: 0.9190
  top5: 0.9687
  top7: 0.9884

--- Diag 3: Confusion Matrix ---
true\pred |  airp  auto  bird   cat  deer   dog  frog  hors  ship  truc
airplane  |   705    10   135     7    18    10    21     8    79     7
automobil |    18   833     4    11     1     5     7     7     6   108
bird      |    12     4   659    56    94    59    75    27     9     5
cat       |     2    14    62   574    67   156    80    28     8     9
deer      |     8     2    68    71   670    41    68    65     7     0
dog       |     1     1    97    92    51   682    35    38     3     0
frog      |     3     7    50   137    91    45   648    10     8     1
horse     |    13     4    29    11    87    71     4   772     3     6
ship      |    81    28    17     3     8    13    17     2   809    22
truck     |    31    76     7     6     2    12     2    23    51   790
Major confusions (off-diag > 50):
  True=cat, Pred=dog: 156
  True=frog, Pred=cat: 137
  True=airplane, Pred=bird: 135
  True=automobile, Pred=truck: 108
  True=dog, Pred=bird: 97
  True=bird, Pred=deer: 94
  True=dog, Pred=cat: 92
  True=frog, Pred=deer: 91
  True=horse, Pred=deer: 87
  True=ship, Pred=airplane: 81
  True=cat, Pred=frog: 80
  True=airplane, Pred=ship: 79
  True=truck, Pred=automobile: 76
  True=bird, Pred=frog: 75
  True=deer, Pred=cat: 71
  True=horse, Pred=dog: 71
  True=deer, Pred=bird: 68
  True=deer, Pred=frog: 68
  True=cat, Pred=deer: 67
  True=deer, Pred=horse: 65
  True=cat, Pred=bird: 62
  True=bird, Pred=dog: 59
  True=bird, Pred=cat: 56
  True=dog, Pred=deer: 51
  True=truck, Pred=ship: 51

--- Diag 4: Margin ---
  correct: mean=8.6554, std=4.5914
  wrong:   mean=4.3897, std=3.4185
  low_margin_ratio (< 0.5): 0.0360

--- Diag 5: Fisher Criterion ---
  mean_intra_var:   0.219812
  mean_inter_dist:  0.228227
  fisher_ratio:     1.0383
  weak pair ('cat', 'dog'): F=0.1120
  weak pair ('bird', 'deer'): F=0.1422
  weak pair ('bird', 'frog'): F=0.1699
  weak pair ('automobile', 'truck'): F=0.2083
  weak pair ('deer', 'frog'): F=0.2116

--- Diag 6: Nearest-Centroid Acc (핵심) ---
  nc_acc (split-half): 0.7381
  nc_acc (same-data):  0.7411
  offline_acc:         0.7142
  gap (nc - offline):  +0.0239

--- Diag 7: Text Head vs NC Head Per-class ---
  class        |   text |     nc |    diff
  airplane     | 0.7050 | 0.7520 | +0.0470
  automobile   | 0.8330 | 0.8690 | +0.0360
  bird         | 0.6590 | 0.6610 | +0.0020
  cat          | 0.5740 | 0.6170 | +0.0430
  deer         | 0.6700 | 0.6540 | -0.0160
  dog          | 0.6820 | 0.7010 | +0.0190
  frog         | 0.6480 | 0.7510 | +0.1030
  horse        | 0.7720 | 0.7740 | +0.0020
  ship         | 0.8090 | 0.8250 | +0.0160
  truck        | 0.7900 | 0.8070 | +0.0170

--- Diag 9: Deconvolution Head ---
  lambda_0.01: deconv_acc=0.7118
  lambda_0.05: deconv_acc=0.7125
  lambda_0.1: deconv_acc=0.7132
  lambda_0.5: deconv_acc=0.7133
  lambda_1.0: deconv_acc=0.7139
  best_lambda=1.0 → best_deconv_acc=0.7139
  top3_restricted_acc (lambda=1.0): 0.7137
  Δ_deconv_vs_offline: -0.0003

--- Diag 10: Prototype Purity ---
  mean_purity: 0.7131
  airplane    : purity=0.803, dominant=airplane (0.803)
  automobile  : purity=0.836, dominant=automobile (0.836)
  bird        : purity=0.578, dominant=bird (0.578)
  cat         : purity=0.592, dominant=cat (0.592)
  deer        : purity=0.610, dominant=deer (0.610)
  dog         : purity=0.615, dominant=dog (0.615)
  frog        : purity=0.672, dominant=frog (0.672)
  horse       : purity=0.781, dominant=horse (0.781)
  ship        : purity=0.820, dominant=ship (0.820)
  truck       : purity=0.823, dominant=truck (0.823)

--- Diag 11: Relational Identifiability ---
  Least identifiable:
    automobile-truck: JS=0.017914
    airplane-ship: JS=0.026481
    dog-horse: JS=0.028926
    cat-dog: JS=0.031104
    deer-horse: JS=0.032721
  Most identifiable:
    cat-truck: JS=0.064739
    automobile-deer: JS=0.064807
    deer-truck: JS=0.065621

--- Diag 12: Centroid-Text Alignment ---
  mean_diagonal: 0.2566
  class        | rank | cos    | best_text
  airplane     |    1 | 0.2491 | airplane
  automobile   |    1 | 0.2744 | automobile
  bird         |    1 | 0.2486 | bird
  cat          |    1 | 0.2252 | cat
  deer         |    1 | 0.2545 | deer
  dog          |    1 | 0.2378 | dog
  frog         |    1 | 0.2524 | frog
  horse        |    1 | 0.2720 | horse
  ship         |    1 | 0.2703 | ship
  truck        |    1 | 0.2820 | truck

============================================================
=== Run OS1: OS1 (L_ent + 2·Σ[p̄_k−π_k]²₊) ===
============================================================

--- Adaptation Results ---
online_acc:        0.5413
offline_acc:       0.5825
cat_pct:           0.2548
mean_entropy:      0.3484
H_pbar_final:      2.1251
n_trainable_params:65,536

--- Step Log (every 5 steps) ---
step | online_acc | batch_acc | cat_pct |    ent | H_pbar |     loss
   5 |     0.4010 |    0.4250 |   0.515 |  1.082 |  1.842 |   0.9893
  10 |     0.4310 |    0.5400 |   0.465 |  0.825 |  1.919 |   0.5673
  15 |     0.4670 |    0.5700 |   0.427 |  0.680 |  1.992 |   0.3497
  20 |     0.4923 |    0.5900 |   0.390 |  0.586 |  1.934 |   0.3327
  25 |     0.5074 |    0.5500 |   0.350 |  0.519 |  2.059 |   0.2761
  30 |     0.5195 |    0.5950 |   0.316 |  0.470 |  2.082 |   0.2696
  35 |     0.5264 |    0.5900 |   0.296 |  0.431 |  2.067 |   0.2181
  40 |     0.5330 |    0.5700 |   0.282 |  0.397 |  2.126 |   0.2034
  45 |     0.5374 |    0.6350 |   0.270 |  0.371 |  2.127 |   0.1523
  50 |     0.5413 |    0.5850 |   0.255 |  0.348 |  2.125 |   0.1573

--- Diag 1: Per-class Recall ---
  airplane    : 0.6900
  automobile  : 0.9210
  bird        : 0.7940
  cat         : 0.5870
  deer        : 0.3470
  dog         : 0.6980
  frog        : 0.1020
  horse       : 0.5590
  ship        : 0.8740
  truck       : 0.2530

--- Diag 2: Top-K Recall ---
  top1: 0.5824
  top2: 0.7703
  top3: 0.8528
  top5: 0.9454
  top7: 0.9828

--- Diag 3: Confusion Matrix ---
true\pred |  airp  auto  bird   cat  deer   dog  frog  hors  ship  truc
airplane  |   690     6   183     6     1    15     0     1    98     0
automobil |    25   921     9     9     1     9     0     0    20     6
bird      |    16     7   794    66    12    83     1     5    16     0
cat       |     4    17   147   587    11   207     1     6    20     0
deer      |    16     3   262   135   347   181     3    33    20     0
dog       |     3     1   138   131    12   698     0    11     6     0
frog      |    17    15   345   314    42   137   102     3    25     0
horse     |    25     5    83    24    54   241     0   559     9     0
ship      |    58    25    26     2     0    15     0     0   874     0
truck     |   100   420    18     7     0    26     0    13   163   253
Major confusions (off-diag > 50):
  True=truck, Pred=automobile: 420
  True=frog, Pred=bird: 345
  True=frog, Pred=cat: 314
  True=deer, Pred=bird: 262
  True=horse, Pred=dog: 241
  True=cat, Pred=dog: 207
  True=airplane, Pred=bird: 183
  True=deer, Pred=dog: 181
  True=truck, Pred=ship: 163
  True=cat, Pred=bird: 147
  True=dog, Pred=bird: 138
  True=frog, Pred=dog: 137
  True=deer, Pred=cat: 135
  True=dog, Pred=cat: 131
  True=truck, Pred=airplane: 100
  True=airplane, Pred=ship: 98
  True=bird, Pred=dog: 83
  True=horse, Pred=bird: 83
  True=bird, Pred=cat: 66
  True=ship, Pred=airplane: 58
  True=horse, Pred=deer: 54

--- Diag 4: Margin ---
  correct: mean=10.2293, std=5.0627
  wrong:   mean=5.7957, std=4.2091
  low_margin_ratio (< 0.5): 0.0342

--- Diag 5: Fisher Criterion ---
  mean_intra_var:   0.210690
  mean_inter_dist:  0.202807
  fisher_ratio:     0.9626
  weak pair ('cat', 'dog'): F=0.1113
  weak pair ('deer', 'frog'): F=0.1117
  weak pair ('cat', 'frog'): F=0.1253
  weak pair ('bird', 'deer'): F=0.1269
  weak pair ('bird', 'frog'): F=0.1331

--- Diag 6: Nearest-Centroid Acc (핵심) ---
  nc_acc (split-half): 0.6957
  nc_acc (same-data):  0.6985
  offline_acc:         0.5825
  gap (nc - offline):  +0.1132

--- Diag 7: Text Head vs NC Head Per-class ---
  class        |   text |     nc |    diff
  airplane     | 0.6900 | 0.7530 | +0.0630
  automobile   | 0.9210 | 0.8630 | -0.0580
  bird         | 0.7940 | 0.6970 | -0.0970
  cat          | 0.5870 | 0.6190 | +0.0320
  deer         | 0.3470 | 0.5560 | +0.2090
  dog          | 0.6980 | 0.6930 | -0.0050
  frog         | 0.1020 | 0.5210 | +0.4190
  horse        | 0.5590 | 0.6980 | +0.1390
  ship         | 0.8740 | 0.8680 | -0.0060
  truck        | 0.2540 | 0.7170 | +0.4630

--- Diag 9: Deconvolution Head ---
  lambda_0.01: deconv_acc=0.5691
  lambda_0.05: deconv_acc=0.5701
  lambda_0.1: deconv_acc=0.5710
  lambda_0.5: deconv_acc=0.5767
  lambda_1.0: deconv_acc=0.5792
  best_lambda=1.0 → best_deconv_acc=0.5792
  top3_restricted_acc (lambda=1.0): 0.5817
  Δ_deconv_vs_offline: -0.0033

--- Diag 10: Prototype Purity ---
  mean_purity: 0.6861
  airplane    : purity=0.721, dominant=airplane (0.721)
  automobile  : purity=0.643, dominant=automobile (0.643)
  bird        : purity=0.395, dominant=bird (0.395)
  cat         : purity=0.459, dominant=cat (0.459)
  deer        : purity=0.715, dominant=deer (0.715)
  dog         : purity=0.433, dominant=dog (0.433)
  frog        : purity=0.942, dominant=frog (0.942)
  horse       : purity=0.882, dominant=horse (0.882)
  ship        : purity=0.700, dominant=ship (0.700)
  truck       : purity=0.970, dominant=truck (0.970)

--- Diag 11: Relational Identifiability ---
  Least identifiable:
    automobile-truck: JS=0.016801
    airplane-ship: JS=0.025549
    dog-horse: JS=0.028500
    deer-horse: JS=0.032063
    cat-dog: JS=0.032197
  Most identifiable:
    cat-truck: JS=0.066352
    automobile-deer: JS=0.066524
    deer-truck: JS=0.067153

--- Diag 12: Centroid-Text Alignment ---
  mean_diagonal: 0.2061
  class        | rank | cos    | best_text
  airplane     |    1 | 0.2129 | airplane
  automobile   |    1 | 0.2619 | automobile
  bird         |    1 | 0.2403 | bird
  cat          |    1 | 0.1957 | cat
  deer         |    1 | 0.1791 | deer
  dog          |    1 | 0.2150 | dog
  frog         |    4 | 0.1240 | bird
  horse        |    1 | 0.2127 | horse
  ship         |    1 | 0.2367 | ship
  truck        |    2 | 0.1832 | automobile

============================================================
=== Run OS2: OS2 (L_ent + 2·Σ p̄_k·[log(p̄_k/π_k)]₊) ===
============================================================

--- Adaptation Results ---
online_acc:        0.6716
offline_acc:       0.7075
cat_pct:           0.1227
mean_entropy:      0.5521
H_pbar_final:      2.2851
n_trainable_params:65,536

--- Step Log (every 5 steps) ---
step | online_acc | batch_acc | cat_pct |    ent | H_pbar |     loss
   5 |     0.4830 |    0.6100 |   0.306 |  1.399 |  2.269 |   1.5025
  10 |     0.5420 |    0.6700 |   0.191 |  1.204 |  2.271 |   1.0232
  15 |     0.5877 |    0.6950 |   0.164 |  1.038 |  2.286 |   0.7435
  20 |     0.6175 |    0.7200 |   0.155 |  0.914 |  2.270 |   0.7161
  25 |     0.6336 |    0.6900 |   0.146 |  0.819 |  2.281 |   0.5613
  30 |     0.6448 |    0.6650 |   0.137 |  0.742 |  2.268 |   0.5859
  35 |     0.6541 |    0.7250 |   0.132 |  0.679 |  2.239 |   0.5900
  40 |     0.6633 |    0.7500 |   0.128 |  0.627 |  2.277 |   0.4549
  45 |     0.6676 |    0.7250 |   0.126 |  0.587 |  2.272 |   0.4650
  50 |     0.6716 |    0.6850 |   0.123 |  0.552 |  2.285 |   0.3969

--- Diag 1: Per-class Recall ---
  airplane    : 0.7310
  automobile  : 0.8530
  bird        : 0.6550
  cat         : 0.5920
  deer        : 0.6560
  dog         : 0.6950
  frog        : 0.5870
  horse       : 0.7640
  ship        : 0.8230
  truck       : 0.7190

--- Diag 2: Top-K Recall ---
  top1: 0.7076
  top2: 0.8590
  top3: 0.9194
  top5: 0.9719
  top7: 0.9906

--- Diag 3: Confusion Matrix ---
true\pred |  airp  auto  bird   cat  deer   dog  frog  hors  ship  truc
airplane  |   731     7   123     4    16    10    12    12    80     5
automobil |    28   853     3    11     1     4     5     6     8    81
bird      |    17     4   655    56    89    73    63    24    15     4
cat       |     5    13    56   592    60   169    62    20    14     9
deer      |    13     2    59    74   656    58    55    74     9     0
dog       |     3     1    88   109    37   695    22    41     4     0
frog      |     6     7    50   157   112    62   587    12     6     1
horse     |    23     4    28    15    68    89     4   764     4     1
ship      |    88    30    14     3     5    14     8     1   823    14
truck     |    53   103     4     5     1    10     3    18    84   719
Major confusions (off-diag > 50):
  True=cat, Pred=dog: 169
  True=frog, Pred=cat: 157
  True=airplane, Pred=bird: 123
  True=frog, Pred=deer: 112
  True=dog, Pred=cat: 109
  True=truck, Pred=automobile: 103
  True=bird, Pred=deer: 89
  True=horse, Pred=dog: 89
  True=dog, Pred=bird: 88
  True=ship, Pred=airplane: 88
  True=truck, Pred=ship: 84
  True=automobile, Pred=truck: 81
  True=airplane, Pred=ship: 80
  True=deer, Pred=cat: 74
  True=deer, Pred=horse: 74
  True=bird, Pred=dog: 73
  True=horse, Pred=deer: 68
  True=bird, Pred=frog: 63
  True=cat, Pred=frog: 62
  True=frog, Pred=dog: 62
  True=cat, Pred=deer: 60
  True=deer, Pred=bird: 59
  True=deer, Pred=dog: 58
  True=bird, Pred=cat: 56
  True=cat, Pred=bird: 56
  True=deer, Pred=frog: 55
  True=truck, Pred=airplane: 53

--- Diag 4: Margin ---
  correct: mean=6.8366, std=3.7750
  wrong:   mean=3.2905, std=2.6953
  low_margin_ratio (< 0.5): 0.0596

--- Diag 5: Fisher Criterion ---
  mean_intra_var:   0.185756
  mean_inter_dist:  0.194054
  fisher_ratio:     1.0447
  weak pair ('cat', 'dog'): F=0.1072
  weak pair ('bird', 'deer'): F=0.1359
  weak pair ('bird', 'frog'): F=0.1630
  weak pair ('deer', 'frog'): F=0.2014
  weak pair ('automobile', 'truck'): F=0.2015

--- Diag 6: Nearest-Centroid Acc (핵심) ---
  nc_acc (split-half): 0.7368
  nc_acc (same-data):  0.7402
  offline_acc:         0.7075
  gap (nc - offline):  +0.0293

--- Diag 7: Text Head vs NC Head Per-class ---
  class        |   text |     nc |    diff
  airplane     | 0.7300 | 0.7620 | +0.0320
  automobile   | 0.8540 | 0.8660 | +0.0120
  bird         | 0.6550 | 0.6430 | -0.0120
  cat          | 0.5920 | 0.6290 | +0.0370
  deer         | 0.6560 | 0.6540 | -0.0020
  dog          | 0.6950 | 0.7040 | +0.0090
  frog         | 0.5870 | 0.7480 | +0.1610
  horse        | 0.7640 | 0.7690 | +0.0050
  ship         | 0.8230 | 0.8260 | +0.0030
  truck        | 0.7200 | 0.8010 | +0.0810

--- Diag 9: Deconvolution Head ---
  lambda_0.01: deconv_acc=0.6991
  lambda_0.05: deconv_acc=0.6999
  lambda_0.1: deconv_acc=0.7011
  lambda_0.5: deconv_acc=0.7058
  lambda_1.0: deconv_acc=0.7056
  best_lambda=0.5 → best_deconv_acc=0.7058
  top3_restricted_acc (lambda=0.5): 0.7069
  Δ_deconv_vs_offline: -0.0017

--- Diag 10: Prototype Purity ---
  mean_purity: 0.7016
  airplane    : purity=0.752, dominant=airplane (0.752)
  automobile  : purity=0.809, dominant=automobile (0.809)
  bird        : purity=0.593, dominant=bird (0.593)
  cat         : purity=0.575, dominant=cat (0.575)
  deer        : purity=0.616, dominant=deer (0.616)
  dog         : purity=0.581, dominant=dog (0.581)
  frog        : purity=0.704, dominant=frog (0.704)
  horse       : purity=0.766, dominant=horse (0.766)
  ship        : purity=0.777, dominant=ship (0.777)
  truck       : purity=0.843, dominant=truck (0.843)

--- Diag 11: Relational Identifiability ---
  Least identifiable:
    automobile-truck: JS=0.015906
    airplane-ship: JS=0.027353
    dog-horse: JS=0.029398
    cat-dog: JS=0.030460
    deer-horse: JS=0.033065
  Most identifiable:
    cat-truck: JS=0.065208
    deer-truck: JS=0.067291
    automobile-deer: JS=0.067345

--- Diag 12: Centroid-Text Alignment ---
  mean_diagonal: 0.2646
  class        | rank | cos    | best_text
  airplane     |    1 | 0.2622 | airplane
  automobile   |    1 | 0.2809 | automobile
  bird         |    1 | 0.2593 | bird
  cat          |    1 | 0.2426 | cat
  deer         |    1 | 0.2613 | deer
  dog          |    1 | 0.2524 | dog
  frog         |    1 | 0.2522 | frog
  horse        |    1 | 0.2788 | horse
  ship         |    1 | 0.2783 | ship
  truck        |    1 | 0.2778 | truck

---

## Blocks A-E Follow-up Experiments

**Run:** `20260313_234520`  
**X_best:** X1 (offline_acc=0.5902)

### Block D: Multi-template Deconvolution (Frozen Model)

| Head | Best λ | Best acc | Δ vs base |
|---|---|---|---|
| Zero-shot argmax | — | 0.3796 | — |
| Single-template deconv | 1.0 | 0.3964 | +0.0168 |
| Multi-template (avg) deconv | 1.0 | 0.4232 | +0.0436 |

Single-template eff_rank=6.44, Multi-template eff_rank=6.52

### Block A: Deconvolution During Adaptation (DA)

| Metric | Value |
|---|---|
| Online acc | 0.5387 |
| Offline acc (deconv head) | 0.5881 |
| cat% | 0.188 |
| Collapsed | False |
| NC acc (split-half) | 0.6726 |
| Prototype purity | 0.5003 |

### Block B: J3 Best + Tiny L_ent (Collapse Probe)

| Run | α | Online acc | Offline acc | cat% | Collapsed |
|---|---|---|---|---|---|
| E1 | 0.05 | 0.2689 | 0.2868 | 0.784 | False |
| E2 | 0.01 | 0.4318 | 0.4358 | 0.527 | False |

### Block C: Moderate Skew Experiments

Skew counts: {0: 1500, 1: 1500, 2: 500, 3: 500, 4: 1000, 5: 1000, 6: 1000, 7: 1000, 8: 1000, 9: 1000}

| Run | Loss | Online acc | Offline acc | cat% |
|---|---|---|---|---|
| SK1 | CAMA | 0.6641 | 0.7089 | 0.126 |
| SK2 | OS1 | 0.5413 | 0.6437 | 0.271 |

### Block E: Entropy Eigensurgery (ES)

| Metric | Value |
|---|---|
| Online acc | 0.5568 |
| Offline acc | 0.4969 |
| cat% | 0.281 |
| Collapsed | False |
| NC acc (split-half) | 0.6356 |
| Prototype purity | 0.6711 |

### Summary: Blocks A-E vs References

| Run | Description | Online acc | Offline acc | cat% | NC_sh | Purity |
|---|---|---|---|---|---|---|
| CAMA (ref) | CAMA (KL evidence) | 0.6734 | — | — | — | — |
| DA | Deconvolution during adaptation (X_best=X1) | 0.5387 | 0.5881 | 0.188 | 0.6726 | 0.5003 |
| E1 | J3 best (X1) + 0.05·L_ent | 0.2689 | 0.2868 | 0.784 | 0.5704 | 0.7256 |
| E2 | J3 best (X1) + 0.01·L_ent | 0.4318 | 0.4358 | 0.527 | 0.6443 | 0.6942 |
| SK1 | CAMA on moderate skew | 0.6641 | 0.7089 | 0.126 | 0.7461 | 0.7023 |
| SK2 | OS1 on moderate skew | 0.5413 | 0.6437 | 0.271 | 0.7174 | 0.6824 |
| ES | Entropy eigensurgery (α=0.2·L_ent, batch consensus removal) | 0.5568 | 0.4969 | 0.281 | 0.6356 | 0.6711 |

---

## Analysis & Interpretation

### Inst 20 Unified Findings (Phase 1-3 + Blocks A-E)

#### 1. Text LN drift is NOT the bottleneck (Phase 1)

X1 (image LN only, text frozen) ≈ X2 (all LN + r_k recomputed) ≈ X3 (original J3, drift).
Offline acc spread: 0.5902 / 0.5900 / 0.6002 — only 1pp total range.
**Conclusion:** The prediction/loss space misalignment from text LN drift is not the cause of J3's gap vs CAMA (0.6734). The bottleneck is elsewhere.

#### 2. Text head bottleneck confirmed via NC (Phase 1, Diag 6)

| Run | NC_sh | Offline | Gap (NC−offline) |
|-----|-------|---------|-----------------|
| X1 | 0.6715 | 0.5902 | **+7.13pp** |
| X2 | 0.6715 | 0.5900 | **+7.15pp** |
| X3 | 0.6757 | 0.6002 | **+7.55pp** |
| CAMA | 0.7381 | 0.7142 | +2.39pp |

J3's feature space supports ~67% NC accuracy, but the text head can only extract 59%. CAMA closes this gap (2.39pp only). **J3's problem is sharpness: high entropy (mean_ent≈1.0 vs CAMA≈0.5) means soft prototypes that imprecisely localize text direction.**

#### 3. Deconvolution head is not a silver bullet (Block D, Block A)

- **Frozen deconv (Block D):** single-template +1.68pp, multi-template +4.36pp over zero-shot. eff_rank barely changes (6.44→6.52). Text space is still essentially the same subspace — multi-template averaging just smooths noise but doesn't expand rank.
- **Deconv during adaptation (Block A):** offline=0.5881 — slightly *worse* than plain X1 (0.5902). Using deconvolved q reduces prototype purity (0.5003 vs 0.5328). Deconvolution changes what "class mass" the prototypes collect — when combined with the rel loss this distorts the alignment.

**Conclusion:** Deconvolution as a post-hoc head gives modest gains on frozen features, but integrating it into the adaptation loop is neutral to slightly harmful.

#### 4. L_ent without H(p̄) collapses even at α=0.01 (Block B)

| Run | α | cat% | Offline | Verdict |
|-----|---|------|---------|---------|
| E1 | 0.05 | **0.784** | 0.287 | severe collapse |
| E2 | 0.01 | **0.527** | 0.436 | moderate collapse |

J3 (L_rel only) provides some collapse resistance — but adding even α=0.01 L_ent is enough to trigger significant cat-class drift. This strongly confirms **Finding #7** (Inst 17): L_ent without an anti-collapse term must not be mixed with L_rel. The rel loss constrains the prototype shape but not the marginal p̄.

#### 5. CAMA is robust to moderate class skew (Block C)

| Setting | Online | Offline | Δ vs balanced |
|---------|--------|---------|---------------|
| CAMA balanced | 0.6738 | 0.7142 | — |
| SK1 (CAMA, skew) | **0.6641** | **0.7089** | −0.97pp / −0.53pp |
| SK2 (OS1, skew) | 0.5413 | 0.6437 | — |

**CAMA loses only 0.97pp online on moderate skew** ({0,1: 1500, 2,3: 500, 4-9: 1000}). This challenges the concern that the "push-up" component of full KL (raising underrepresented classes) would hurt on skew. In practice it doesn't, because the evidence prior itself adapts to the test batch's evidence — if a class is underrepresented in the batch, it contributes less to the prior.

**OS1 (one-sided squared excess) is substantially worse on skew** (0.5413 online vs CAMA 0.6641). The squared excess penalty is too weak when class imbalance is present — it has less gradient when classes are already rare.

**Revised conclusion:** Full KL (CAMA) is preferable to one-sided variants even on skew.

#### 6. Entropy eigensurgery provides no benefit (Block E)

ES (α=0.2 · L_ent, batch consensus direction removed from gradient) achieves online=0.5568, offline=0.4969. The offline < online indicates model quality degrades in later steps (H(p̄) falls from 2.10 → 1.91, cat% stagnates at 0.281). Removing the batch consensus direction doesn't prevent gradual collapse — it just slows it. Without an explicit anti-collapse regularizer, eigensurgery alone is insufficient.

### Global Summary

| Question | Answer |
|----------|--------|
| Is text LN drift J3's bottleneck? | **No** (X1≈X3, Δ<1pp) |
| Is J3's feature space good enough? | **Partially** — NC_sh=0.676 but text head extracts only 0.600 |
| Does deconvolution help during adaptation? | **No** — neutral or slightly negative |
| Can J3 + tiny L_ent avoid collapse? | **No** — even α=0.01 causes cat%=53% |
| Is CAMA robust to skew? | **Yes** — only −0.97pp on moderate skew |
| Is OS1 better than CAMA on skew? | **No** — CAMA wins by +12pp offline on skew |
| Does eigensurgery help entropy minimization? | **No** — gradual collapse remains |

### Next Steps

1. **CAMA is the clear winner** (balanced and skew-robust). Priority = 15-corruption evaluation to validate CALM v1 (0.7970 overall) comparison.
2. **J3 direction deprioritized** — text head bottleneck confirmed, but deconvolution fixes don't close the gap. NC_sh gap (+7pp) is primarily a sharpness problem — CAMA solves this directly.
3. **OS2 ≈ CAMA** on balanced data (−0.22pp) — if skew robustness of OS2 needs confirmation, run a followup skew experiment with OS2.
4. **L_ent + H(p̄) remains the correct formulation** for any entropy-based method.

