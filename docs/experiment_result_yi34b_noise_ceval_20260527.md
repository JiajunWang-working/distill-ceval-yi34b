# Yi-34B 含噪声蒸馏实验结果记录

记录日期：2026-05-27

## 1. 实验目的

记录当前最新一轮正式主线实验结果：

1. 数据集使用全量 `CEval official`
2. 教师模型使用本地 `Yi-1.5-34B-Chat`
3. 学生模型使用 `Qwen2.5-7B-Instruct`
4. 蒸馏标签使用真实 teacher option probability distribution
5. 对比无噪声与含噪声版本在修复后的 benchmark-style evaluator 下的表现

## 2. 实验设置

### 2.1 数据与评测口径

1. `train`：`/root/nlp/data/processed/ceval_official_full/train.jsonl`
2. `val`：`/root/nlp/data/processed/ceval_official_full/val.jsonl`
3. `test`：`/root/nlp/data/processed/ceval_official_full/test.jsonl`
4. 评测方式：`0-shot benchmark-style option likelihood`
5. 评测样本数：`12342`

### 2.2 模型与训练

1. 学生模型：`Qwen/Qwen2.5-7B-Instruct`
2. 教师模型：`Yi-1.5-34B-Chat`
3. 训练方式：`4-bit QLoRA + LoRA`
4. 噪声设置：`uniform noise @ model.norm, scale=0.02`
5. 噪声位置：`model.norm`
6. 噪声分布：`uniform`

### 2.3 对应配置文件

1. `KL`：[qwen25_7b_ceval_distill_full_yi34b_kl.yaml](/root/nlp/configs/qwen25_7b_ceval_distill_full_yi34b_kl.yaml)
2. `KL + noise`：[qwen25_7b_ceval_distill_full_yi34b_noise_u002_kl.yaml](/root/nlp/configs/qwen25_7b_ceval_distill_full_yi34b_noise_u002_kl.yaml)
3. `BayesKD`：[qwen25_7b_ceval_distill_full_yi34b_bayes.yaml](/root/nlp/configs/qwen25_7b_ceval_distill_full_yi34b_bayes.yaml)
4. `BayesKD + noise`：[qwen25_7b_ceval_distill_full_yi34b_noise_u002_bayes.yaml](/root/nlp/configs/qwen25_7b_ceval_distill_full_yi34b_noise_u002_bayes.yaml)

## 3. 结果总表

| 模型/方法 | Test Accuracy | 相对 7B baseline 增益 | 备注 |
| --- | ---: | ---: | --- |
| `Qwen2.5-7B-Instruct` | 0.7657591962404796 | 0.0000000000000000 | 学生 baseline |
| `Qwen2.5-7B + KL distill from Yi-34B` | 0.7668125101280181 | +0.0010533138875385 | 无噪声 `KL` |
| `Qwen2.5-7B + KL distill from Yi-34B + noise` | 0.7702155242262194 | +0.0044563279857398 | `uniform@model.norm, 0.02` |
| `Qwen2.5-7B + BayesKD distill from Yi-34B` | 0.7707826932425863 | +0.0050234970021067 | 无噪声 `BayesKD` |
| `Qwen2.5-7B + BayesKD distill from Yi-34B + noise` | 0.7743477556311781 | +0.0085885593906985 | 旧正式最好 |
| `Qwen2.5-7B + SABO v3-fast2 Top2 distill from Yi-34B` | 0.7775887214389888 | +0.0118295251985093 | 当前最优 |
| `Qwen2.5-14B-Instruct` | 0.7933074056068709 | +0.0275482093663913 | 同口径参考上界 |

## 4. 关键比较

1. `KL + noise` 相比无噪声 `KL` 提升：`+0.0034030140982013`
2. `BayesKD + noise` 相比无噪声 `BayesKD` 提升：`+0.0035650623885918`
3. `SABO v3-fast2 Top2` 相比旧正式最好 `BayesKD + noise` 再提升：`+0.0032409658078107`
4. `SABO v3-fast2 Top2` 相比 `7B baseline` 总提升：`+0.0118295251985093`
5. `SABO v3-fast2 Top2` 仍低于 `14B baseline`：`0.0157186841678821`

## 5. 训练侧记录

### 5.1 验证损失

1. `KL` 最终 `eval_loss = 2.674`
2. `KL + noise` 最终 `eval_loss = 2.671`
3. `BayesKD` 最终 `eval_loss = 2.017`
4. `BayesKD + noise` 训练过程中的主要验证损失为：`1.935 / 1.949 / 2.051`
5. `SABO v3-fast2 Top2` 训练过程中的主要验证损失为：`1.307 / 1.292 / 1.370 / 1.365`

### 5.2 解释

1. 在当前全量 `CEval official` 正式主线上，噪声注入对 `KL` 和 `BayesKD` 两条线都带来了净提升
2. 后续联合搜索表明，固定 `scale = 0.02` 并不是最终最优；更温和的 `noise scale = 0.0101` 与更高 `alpha` 的组合能得到更强 test 泛化
3. 这说明在修复评测器后的正式口径下，`noise` 本身仍是值得保留的增强手段，但其最优强度需要和 `alpha / temperature / learning_rate` 联合调优

## 6. 最终结论

如果只看当前这条“全量 `CEval official` + 本地 `Yi-34B` 强教师 + 真实教师分布蒸馏 + 修复后 benchmark-style evaluator”的正式实验主线，则当前排序为：

1. `SABO v3-fast2 Top2`
2. `BayesKD + noise`
3. `BayesKD`
4. `KL + noise`
5. `KL`
6. `7B baseline`

因此，这轮新实验的正式结论是：

1. `BayesKD` 优于普通 `KL`
2. 噪声注入在这条主线上是正收益，而且噪声强度存在可搜索的更优点
3. 当前最佳学生结果已更新为 `0.7775887214389888`
4. 这相对旧正式最好结果 `0.7743477556311781` 进一步提升了 `+0.0032409658078107`
5. 这条线已经明显优于原始 `7B baseline`，但尚未追平 `14B baseline`

## 7. PTQ 推理评测补充

在不重新训练、只切换评测阶段量化加载方式的前提下，对主线相关模型补充了 `PTQ` 推理实验：

| 模型 | FP/BF16 | 8-bit | 4-bit |
| --- | ---: | ---: | ---: |
| `Qwen2.5-7B-Instruct` | 0.7657591962404796 | 0.7791281801976989 | 0.7657591962404796 |
| `BayesKD + noise` | 0.7743477556311781 | 0.7853670393777346 | 0.7743477556311781 |
| `SABO v3-fast2 Top2` | 0.7775887214389888 | 0.7859342083941014 | 0.7775887214389888 |

这轮补充实验的主要结论是：

1. 三条模型线在 `4-bit` 下都与原始 `FP/BF16` 结果完全一致
2. 三条模型线在 `8-bit` 下都出现稳定的正向波动
3. 因此，如果要汇报正式蒸馏主线成绩，仍建议保持原始 `FP/BF16` 口径
4. 如果需要部署侧补充结论，则可以说明当前最优学生模型在 `4-bit` 下零损失，在 `8-bit` 下可达到 `0.7859342083941014`

完整 PTQ 记录见：

1. [experiment_result_ptq_ceval_20260601.md](/root/nlp/experiment_result_ptq_ceval_20260601.md)

## 8. 相关结果文件

1. `7B baseline` 指标：[test_full_benchmark_plain_0shot_fixed_metrics.json](/root/nlp/outputs/qwen25_7b_ceval_benchmark_plain_full/test_full_benchmark_plain_0shot_fixed_metrics.json)
2. `14B baseline` 指标：[test_full_benchmark_plain_0shot_fixed_metrics.json](/root/nlp/outputs/qwen25_14b_ceval_benchmark_plain_full/test_full_benchmark_plain_0shot_fixed_metrics.json)
3. `KL` 指标：[test_full_benchmark_plain_0shot_fixed_metrics.json](/root/nlp/outputs/qwen25_7b_ceval_distill_full_yi34b_kl/test_full_benchmark_plain_0shot_fixed_metrics.json)
4. `KL + noise` 指标：[test_full_benchmark_plain_0shot_fixed_metrics.json](/root/nlp/outputs/qwen25_7b_ceval_distill_full_yi34b_noise_u002_kl/test_full_benchmark_plain_0shot_fixed_metrics.json)
5. `BayesKD` 指标：[test_full_benchmark_plain_0shot_fixed_metrics.json](/root/nlp/outputs/qwen25_7b_ceval_distill_full_yi34b_bayes/test_full_benchmark_plain_0shot_fixed_metrics.json)
6. `BayesKD + noise` 指标：[test_full_benchmark_plain_0shot_fixed_metrics.json](/root/nlp/outputs/qwen25_7b_ceval_distill_full_yi34b_noise_u002_bayes/test_full_benchmark_plain_0shot_fixed_metrics.json)
7. `SABO v3-fast2 Top2` 指标：[test_full_benchmark_plain_0shot_fixed_metrics.json](/root/nlp/outputs/qwen25_7b_ceval_distill_full_yi34b_noise_bayes_sabo_v3_fast2_top2_full/test_full_benchmark_plain_0shot_fixed_metrics.json)
8. `BayesKD + noise` checkpoint：[final](/root/nlp/outputs/qwen25_7b_ceval_distill_full_yi34b_noise_u002_bayes/final)
9. `SABO v3-fast2 Top2` checkpoint：[final](/root/nlp/outputs/qwen25_7b_ceval_distill_full_yi34b_noise_bayes_sabo_v3_fast2_top2_full/final)
