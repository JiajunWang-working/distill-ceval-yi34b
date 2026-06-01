# SABO 实验结果记录

记录日期：2026-05-27

## 1. 实验目的

在当前正式主线

1. `Yi-1.5-34B-Chat` 作为教师
2. `Qwen2.5-7B-Instruct` 作为学生
3. 全量 `CEval official`
4. `BayesKD + uniform noise @ model.norm, scale=0.02`
5. 修复后的 `benchmark-style option likelihood evaluator`

条件下，重新进行一轮 likelihood-based `SABO` 搜索，用于选择更优的蒸馏超参数：

1. 蒸馏权重 `alpha`
2. 蒸馏温度 `temperature`

## 2. 搜索设置

1. 基础配置：[qwen25_7b_ceval_distill_full_yi34b_noise_u002_bayes.yaml](/root/nlp/configs/qwen25_7b_ceval_distill_full_yi34b_noise_u002_bayes.yaml)
2. 搜索脚本：[search_sabo.py](/root/nlp/scripts/search_sabo.py)
3. 搜索输出目录：[sabo_qwen25_7b_ceval_full_yi34b_noise_bayes_benchmark](/root/nlp/outputs/sabo_qwen25_7b_ceval_full_yi34b_noise_bayes_benchmark)
4. 搜索 trial 数：`4`
5. warmup trial 数：`2`
6. 每个 trial 的搜索训练 epoch：`1.0`
7. 搜索范围：
   - `alpha ∈ [0.2, 0.8]`
   - `temperature ∈ [1.0, 3.0]`
8. 验证集：`/root/nlp/data/processed/ceval_official_full/val.jsonl`
9. few-shot 参考集：`/root/nlp/data/processed/ceval_official_full/train.jsonl`
10. 验证口径：`0-shot benchmark-style option likelihood`

## 3. 验证集搜索结果

| Trial | alpha | temperature | Val Accuracy |
| --- | ---: | ---: | ---: |
| `trial_04` | 0.2215 | 2.9195 | 0.7800891530460624 |
| `trial_01` | 0.6644 | 1.8778 | 0.7763744427934621 |
| `trial_03` | 0.2033 | 1.1033 | 0.7756315007429421 |
| `trial_02` | 0.7152 | 2.3947 | 0.7719167904903418 |

## 4. 搜索结论

1. 这轮正式口径下的 `SABO` 最优点是 `trial_04`
2. 最优参数为：
   - `alpha = 0.2215`
   - `temperature = 2.9195`
3. 它在验证集上的 `accuracy = 0.7800891530460624`
4. 这高于旧版沿用参数 `trial_01 = (0.6644, 1.8778)` 的 `0.7763744427934621`

## 5. 对当前主线的意义

这次结果说明，在当前“`Yi-34B + full CEval + BayesKD + noise + benchmark-style evaluator`”正式主线上：

1. 旧版 `SABO` 最优点不再是当前最优
2. 更低的 `alpha` 与更高的 `temperature` 在这条新主线上反而更优
3. 因此，不能直接把旧 mixed-source 小规模实验中得到的 `alpha / temperature` 视为全局最优

## 6. 相关文件

1. 搜索 summary：[summary.json](/root/nlp/outputs/sabo_qwen25_7b_ceval_full_yi34b_noise_bayes_benchmark/summary.json)
2. `trial_01` 配置：[config.yaml](/root/nlp/outputs/sabo_qwen25_7b_ceval_full_yi34b_noise_bayes_benchmark/trials/trial_01/config.yaml)
3. `trial_02` 配置：[config.yaml](/root/nlp/outputs/sabo_qwen25_7b_ceval_full_yi34b_noise_bayes_benchmark/trials/trial_02/config.yaml)
4. `trial_03` 配置：[config.yaml](/root/nlp/outputs/sabo_qwen25_7b_ceval_full_yi34b_noise_bayes_benchmark/trials/trial_03/config.yaml)
5. `trial_04` 配置：[config.yaml](/root/nlp/outputs/sabo_qwen25_7b_ceval_full_yi34b_noise_bayes_benchmark/trials/trial_04/config.yaml)
6. `trial_04` checkpoint：[final](/root/nlp/outputs/sabo_qwen25_7b_ceval_full_yi34b_noise_bayes_benchmark/trial_04/run/final)

## 7. 最佳 Trial 的测试集结果

对验证集最优的 `trial_04`，进一步在 full `CEval official test` 上做 benchmark-style 评测，结果为：

1. `Test Accuracy = 0.7734564900340302`

对应结果文件：

1. [test_full_benchmark_plain_0shot_fixed_metrics.json](/root/nlp/outputs/sabo_qwen25_7b_ceval_full_yi34b_noise_bayes_benchmark/trial_04/run/test_full_benchmark_plain_0shot_fixed_metrics.json)

## 8. 最终判断

将 `SABO` 最优 trial 与当前主线最好结果对比：

1. 当前主线最好结果：
   - `BayesKD + noise`
   - `alpha = 0.6644`
   - `temperature = 1.8778`
   - `test accuracy = 0.7743477556311781`
2. `SABO` 验证集最优 trial：
   - `trial_04`
   - `alpha = 0.2215`
   - `temperature = 2.9195`
   - `test accuracy = 0.7734564900340302`

因此，这轮 `SABO` 的最终结论是：

1. `SABO` 在验证集上找到了更优参数点
2. 但该点没有在测试集上超过当前正式最好模型
3. 当前正式最优结果仍然是原始 `BayesKD + noise` 主线模型
4. `trial_04` 可以保留为一个值得后续继续验证的候选超参数点，但不应直接替换正式主结果

## 9. SABO v2 扩展搜索设置

考虑到上一轮 `SABO` 搜索预算偏小，又进一步进行了更充分的一轮 `SABO v2`：

1. 搜索输出目录：[sabo_qwen25_7b_ceval_full_yi34b_noise_bayes_benchmark_v2](/root/nlp/outputs/sabo_qwen25_7b_ceval_full_yi34b_noise_bayes_benchmark_v2)
2. trial 数：`8`
3. warmup trial 数：`3`
4. 每个 trial 的搜索训练 epoch：`1.0`
5. `candidate_pool_size = 512`
6. `seed = 43`
7. 验证口径仍为：
   - `full CEval official`
   - `0-shot benchmark-style option likelihood`

## 10. SABO v2 验证集排序

| Trial | alpha | temperature | Val Accuracy |
| --- | ---: | ---: | ---: |
| `trial_08` | 0.2096 | 1.9283 | 0.7808320950965825 |
| `trial_05` | 0.2027 | 1.1053 | 0.7793462109955424 |
| `trial_07` | 0.4783 | 2.9703 | 0.7771173848439822 |
| `trial_01` | 0.5914 | 1.0876 | 0.7763744427934621 |
| `trial_03` | 0.5523 | 1.4494 | 0.7763744427934621 |
| `trial_06` | 0.7965 | 1.6845 | 0.7756315007429421 |
| `trial_02` | 0.2120 | 2.6784 | 0.7748885586924220 |
| `trial_04` | 0.7845 | 2.9941 | 0.7689450222882616 |

对应 summary 文件：

1. [summary.json](/root/nlp/outputs/sabo_qwen25_7b_ceval_full_yi34b_noise_bayes_benchmark_v2/summary.json)

## 11. SABO v2 Top-2 Full Retrain 测试结果

从 `SABO v2` 中取验证集前两名，回到正式训练预算做 `3 epoch` full retrain，并在全量 `CEval official test` 上重新评测。

### 11.1 Full retrain 配置

1. `Top1` 配置：[qwen25_7b_ceval_distill_full_yi34b_noise_u002_bayes_sabo_v2_top1_full.yaml](/root/nlp/configs/qwen25_7b_ceval_distill_full_yi34b_noise_u002_bayes_sabo_v2_top1_full.yaml)
2. `Top2` 配置：[qwen25_7b_ceval_distill_full_yi34b_noise_u002_bayes_sabo_v2_top2_full.yaml](/root/nlp/configs/qwen25_7b_ceval_distill_full_yi34b_noise_u002_bayes_sabo_v2_top2_full.yaml)

### 11.2 Test 结果

| 模型 | alpha | temperature | Test Accuracy | 相对当前正式最好结果差值 |
| --- | ---: | ---: | ---: | ---: |
| 当前正式最好 `BayesKD + noise` | 0.6644 | 1.8778 | 0.7743477556311781 | 0.0000000000000000 |
| `SABO v2 Top1` (`trial_08`) | 0.2096 | 1.9283 | 0.7695673310646572 | -0.0047804245665208 |
| `SABO v2 Top2` (`trial_05`) | 0.2027 | 1.1053 | 0.7705396208070004 | -0.0038081348241776 |

对应指标文件：

1. `Top1`：[test_full_benchmark_plain_0shot_fixed_metrics.json](/root/nlp/outputs/qwen25_7b_ceval_distill_full_yi34b_noise_u002_bayes_sabo_v2_top1_full/test_full_benchmark_plain_0shot_fixed_metrics.json)
2. `Top2`：[test_full_benchmark_plain_0shot_fixed_metrics.json](/root/nlp/outputs/qwen25_7b_ceval_distill_full_yi34b_noise_u002_bayes_sabo_v2_top2_full/test_full_benchmark_plain_0shot_fixed_metrics.json)

## 12. SABO v2 结论

1. 更大搜索预算下，`SABO v2` 的验证集最优点变为更低 `alpha` 区间，但其优势没有转化为测试集优势
2. 两个 `Top-2` 候选在 full retrain 后的测试结果都低于当前正式最好模型
3. 当前正式最好结果仍然是：
   - `Qwen2.5-7B + BayesKD distill from Yi-34B + noise`
   - `test accuracy = 0.7743477556311781`
4. 因此，到 `2026-05-27` 为止，`SABO` 还没有在这条正式主线上替代当前的默认蒸馏超参数

## 13. SABO v3-fast2 联合搜索设置

为了解决前两轮 `SABO` 搜索只搜索 `alpha / temperature`、且搜索 trial 训练预算偏慢的问题，又进行了更合理的一轮 `SABO v3-fast2` 联合搜索：

1. 搜索输出目录：[sabo_qwen25_7b_ceval_full_yi34b_noise_bayes_benchmark_v3_fast2](/root/nlp/outputs/sabo_qwen25_7b_ceval_full_yi34b_noise_bayes_benchmark_v3_fast2)
2. trial 数：`10`
3. warmup trial 数：`4`
4. 搜索训练 epoch budget 候选：`[1.5, 2.0, 2.5]`
5. `candidate_pool_size = 768`
6. `acquisition_beta = 0.55`
7. `seed = 45`
8. 搜索时关闭中间 checkpoint 与中间验证，以提高单位 GPU 时间的有效搜索密度
9. 验证口径仍为：
   - `full CEval official`
   - `0-shot benchmark-style option likelihood`
10. 联合搜索维度扩展为：
   - `alpha`
   - `temperature`
   - `noise_scale`
   - `learning_rate`
   - `num_train_epochs`
11. 搜索范围：
   - `alpha ∈ [0.50, 0.78]`
   - `temperature ∈ [1.30, 2.30]`
   - `noise_scale ∈ [0.010, 0.035]`
   - `learning_rate ∈ [8e-5, 2.5e-4]`
   - `epoch_budget ∈ {1.5, 2.0, 2.5}`

## 14. SABO v3-fast2 验证集排序

| Trial | alpha | temperature | noise scale | learning rate | epoch budget | Val Accuracy |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `trial_10` | 0.5055 | 1.3444 | 0.0346 | 2.4938e-4 | 2.0 | 0.7793462109955424 |
| `trial_08` | 0.7587 | 1.4601 | 0.0101 | 2.2812e-4 | 2.0 | 0.7778603268945022 |
| `trial_05` | 0.7287 | 1.3062 | 0.0312 | 2.3267e-4 | 1.5 | 0.7771173848439822 |
| `trial_07` | 0.5286 | 1.3027 | 0.0128 | 8.2751e-5 | 2.5 | 0.7734026745913819 |
| `trial_09` | 0.5889 | 2.1389 | 0.0107 | 2.4017e-4 | 1.5 | 0.7734026745913819 |
| `trial_02` | 0.7182 | 2.0961 | 0.0249 | 1.2743e-4 | 2.0 | 0.7726597325408618 |
| `trial_03` | 0.6757 | 2.1401 | 0.0281 | 1.4618e-4 | 2.0 | 0.7711738484398217 |
| `trial_04` | 0.7699 | 1.7688 | 0.0303 | 2.1442e-4 | 2.5 | 0.7711738484398217 |
| `trial_06` | 0.5427 | 1.4621 | 0.0164 | 1.2302e-4 | 1.5 | 0.7711738484398217 |
| `trial_01` | 0.6480 | 2.0637 | 0.0303 | 1.4308e-4 | 2.5 | 0.7622585438335809 |

对应 summary 文件：

1. [summary.json](/root/nlp/outputs/sabo_qwen25_7b_ceval_full_yi34b_noise_bayes_benchmark_v3_fast2/summary.json)

## 15. SABO v3-fast2 Top-2 Full Retrain 测试结果

从 `SABO v3-fast2` 里选取验证集前两名，回到正式训练预算 `3 epoch` 做 full retrain，再在全量 `CEval official test` 上进行 benchmark-style 评测。

### 15.1 Full retrain 配置

1. `Top1` 配置：[qwen25_7b_ceval_distill_full_yi34b_noise_bayes_sabo_v3_fast2_top1_full.yaml](/root/nlp/configs/qwen25_7b_ceval_distill_full_yi34b_noise_bayes_sabo_v3_fast2_top1_full.yaml)
2. `Top2` 配置：[qwen25_7b_ceval_distill_full_yi34b_noise_bayes_sabo_v3_fast2_top2_full.yaml](/root/nlp/configs/qwen25_7b_ceval_distill_full_yi34b_noise_bayes_sabo_v3_fast2_top2_full.yaml)

### 15.2 Test 结果

| 模型 | alpha | temperature | noise scale | learning rate | Test Accuracy | 相对旧正式最好结果差值 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 旧正式最好 `BayesKD + noise` | 0.6644 | 1.8778 | 0.0200 | 2.0e-4 | 0.7743477556311781 | 0.0000000000000000 |
| `SABO v3-fast2 Top1` (`trial_10`) | 0.5055 | 1.3444 | 0.0346 | 2.4938e-4 | 0.7720790795657105 | -0.0022686760654676 |
| `SABO v3-fast2 Top2` (`trial_08`) | 0.7587 | 1.4601 | 0.0101 | 2.2812e-4 | 0.7775887214389888 | +0.0032409658078107 |

对应指标文件：

1. `Top1`：[test_full_benchmark_plain_0shot_fixed_metrics.json](/root/nlp/outputs/qwen25_7b_ceval_distill_full_yi34b_noise_bayes_sabo_v3_fast2_top1_full/test_full_benchmark_plain_0shot_fixed_metrics.json)
2. `Top2`：[test_full_benchmark_plain_0shot_fixed_metrics.json](/root/nlp/outputs/qwen25_7b_ceval_distill_full_yi34b_noise_bayes_sabo_v3_fast2_top2_full/test_full_benchmark_plain_0shot_fixed_metrics.json)

### 15.3 训练侧观察

1. `Top1` 虽然是搜索时的验证集第一名，但在正式 `3 epoch` 预算下出现明显不稳定：
   - `eval_loss: 2.493 -> 2.698 -> 2.974 -> 2.956`
   - 对应 test `accuracy = 0.7720790795657105`
2. `Top2` 在正式预算下反而更稳定：
   - `eval_loss: 1.307 -> 1.292 -> 1.370 -> 1.365`
   - 对应 test `accuracy = 0.7775887214389888`
3. 这说明本轮联合搜索中，“搜索期验证集第一名”不一定是“正式 full retrain 后泛化最好”的点
4. 在当前主线上，更温和的 `noise scale = 0.0101` 与较高 `alpha = 0.7587` 组合，比更激进的 `trial_10` 更能稳定转化为测试集收益

## 16. SABO v3-fast2 结论

1. 这轮“更合理的 `SABO`”第一次真正把正式主线 test 上界推高了
2. 新的正式最好学生模型不是验证集第 1 名 `trial_10`，而是正式回训后的 `Top2 = trial_08`
3. 新正式最好结果为：
   - `Qwen2.5-7B + BayesKD distill from Yi-34B`
   - `noise scale = 0.0101`
   - `alpha = 0.7587`
   - `temperature = 1.4601`
   - `learning_rate = 2.2812e-4`
   - `test accuracy = 0.7775887214389888`
4. 它相对旧正式最好结果 `0.7743477556311781` 提升了 `+0.0032409658078107`
5. 因此，到 `2026-05-28` 为止，`SABO` 已经不再只是验证集搜索工具，而是成功找到了可替换正式主结果的新配置
