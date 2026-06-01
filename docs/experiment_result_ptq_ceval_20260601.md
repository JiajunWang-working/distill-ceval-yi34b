# CEval PTQ 推理实验结果记录

记录日期：2026-06-01

## 1. 实验目的

在当前正式主线的 `CEval official` benchmark-style evaluator 上，补充一轮 `PTQ` 推理侧实验，验证当前学生模型在低比特加载下的精度稳定性。

本轮实验中的 `PTQ` 采用仓库现有评测路径：

1. 不重新训练模型
2. 仅在评测阶段切换量化加载方式
3. 分别测试：
   - 默认 `FP/BF16` 推理
   - `BitsAndBytes 8-bit` 推理
   - `BitsAndBytes 4-bit` 推理

## 2. 实验设置

### 2.1 数据与评测口径

1. 数据集：全量 `CEval official`
2. `test`：`/root/nlp/data/processed/ceval_official_full/test.jsonl`
3. few-shot 参考集：`/root/nlp/data/processed/ceval_official_full/train.jsonl`
4. 评测方式：`0-shot benchmark-style option likelihood`
5. 评测样本数：`12342`

### 2.2 评测脚本

1. 脚本：[evaluate_benchmark_mcq.py](/root/nlp/scripts/evaluate_benchmark_mcq.py)
2. 核心实现：[evaluate_benchmark.py](/root/nlp/src/tight_distill/evaluate_benchmark.py)
3. 量化参数：
   - `--load-in-8bit`
   - `--load-in-4bit`
   - `--quant-compute-dtype bf16`

### 2.3 对比模型

1. `Qwen2.5-7B-Instruct` baseline
2. `Qwen2.5-7B + BayesKD + noise`：
   - 旧正式最好模型
3. `Qwen2.5-7B + SABO v3-fast2 Top2`：
   - 当前正式最好模型

## 3. 总结果表

| 模型 | FP/BF16 | 8-bit | 4-bit | 8-bit 相对 FP/BF16 | 4-bit 相对 FP/BF16 |
| --- | ---: | ---: | ---: | ---: | ---: |
| `Qwen2.5-7B-Instruct` | 0.7657591962404796 | 0.7791281801976989 | 0.7657591962404796 | +0.0133689839572193 | +0.0000000000000000 |
| `BayesKD + noise` | 0.7743477556311781 | 0.7853670393777346 | 0.7743477556311781 | +0.0110192837465565 | +0.0000000000000000 |
| `SABO v3-fast2 Top2` | 0.7775887214389888 | 0.7859342083941014 | 0.7775887214389888 | +0.0083454869551126 | +0.0000000000000000 |

## 4. 逐项观察

### 4.1 4-bit 结果

1. 三条模型线在 `4-bit` 下都与原始 `FP/BF16` 结果完全一致
2. 不只是 accuracy 一致，逐题预测也完全一致：
   - `7B baseline`：`0` 题变化
   - `BayesKD + noise`：`0` 题变化
   - `SABO v3-fast2 Top2`：`0` 题变化
3. 这说明在当前这套 `CEval` 评测口径下，`4-bit` 加载没有带来可见精度损失

### 4.2 8-bit 结果

1. 三条模型线在 `8-bit` 下都出现了比原始 `FP/BF16` 更高的测试分数
2. 这种提升不是单模型偶发，而是在 baseline、旧正式最好、当前正式最好三条线上都复现了
3. 但这不应直接解释为“8-bit 一定更好”，更合理的表述是：
   - `8-bit` 改变了推理数值路径
   - 在当前 evaluator 上产生了稳定的正向波动
   - 因而 `8-bit` 结果可以作为一个有价值的部署观察，但不建议替代正式主结果

## 5. 8-bit 与原始结果的逐题差异

### 5.1 `Qwen2.5-7B-Instruct`

1. 改变预测的题目数：`1109`
2. 从错变对：`520`
3. 从对变错：`355`
4. 净增益：`+165`

### 5.2 `BayesKD + noise`

1. 改变预测的题目数：`1147`
2. 从错变对：`513`
3. 从对变错：`377`
4. 净增益：`+136`

### 5.3 `SABO v3-fast2 Top2`

1. 改变预测的题目数：`1019`
2. 从错变对：`449`
3. 从对变错：`346`
4. 净增益：`+103`

## 6. 结论

如果把这轮 `PTQ` 实验限定为“当前仓库现有推理路径下的低比特评测”来理解，那么结论是：

1. `4-bit` 加载在三条模型线上都没有带来精度退化
2. `8-bit` 加载在三条模型线上都带来了稳定的正向波动
3. 但 `8-bit` 的更高分数更适合被视作一种“量化后推理行为变化”，而不是新的正式主结果
4. 因此，后续若要汇报正式蒸馏主线成绩，仍建议使用默认 `FP/BF16` 口径：
   - baseline：`0.7657591962404796`
   - 旧正式最好：`0.7743477556311781`
   - 当前正式最好：`0.7775887214389888`
5. 若要补充部署侧结论，则可以额外说明：
   - 当前正式最好模型在 `4-bit` 下零损失
   - 在 `8-bit` 下测试分数可达到 `0.7859342083941014`

## 7. 相关结果文件

### 7.1 `Qwen2.5-7B-Instruct`

1. 原始指标：[test_full_benchmark_plain_0shot_fixed_metrics.json](/root/nlp/outputs/qwen25_7b_ceval_benchmark_plain_full/test_full_benchmark_plain_0shot_fixed_metrics.json)
2. `8-bit` 指标：[test_full_benchmark_plain_0shot_fixed_8bit_metrics.json](/root/nlp/outputs/qwen25_7b_ceval_benchmark_plain_full/test_full_benchmark_plain_0shot_fixed_8bit_metrics.json)
3. `4-bit` 指标：[test_full_benchmark_plain_0shot_fixed_4bit_metrics.json](/root/nlp/outputs/qwen25_7b_ceval_benchmark_plain_full/test_full_benchmark_plain_0shot_fixed_4bit_metrics.json)

### 7.2 `BayesKD + noise`

1. 原始指标：[test_full_benchmark_plain_0shot_fixed_metrics.json](/root/nlp/outputs/qwen25_7b_ceval_distill_full_yi34b_noise_u002_bayes/test_full_benchmark_plain_0shot_fixed_metrics.json)
2. `8-bit` 指标：[test_full_benchmark_plain_0shot_fixed_8bit_metrics.json](/root/nlp/outputs/qwen25_7b_ceval_distill_full_yi34b_noise_u002_bayes/test_full_benchmark_plain_0shot_fixed_8bit_metrics.json)
3. `4-bit` 指标：[test_full_benchmark_plain_0shot_fixed_4bit_metrics.json](/root/nlp/outputs/qwen25_7b_ceval_distill_full_yi34b_noise_u002_bayes/test_full_benchmark_plain_0shot_fixed_4bit_metrics.json)

### 7.3 `SABO v3-fast2 Top2`

1. 原始指标：[test_full_benchmark_plain_0shot_fixed_metrics.json](/root/nlp/outputs/qwen25_7b_ceval_distill_full_yi34b_noise_bayes_sabo_v3_fast2_top2_full/test_full_benchmark_plain_0shot_fixed_metrics.json)
2. `8-bit` 指标：[test_full_benchmark_plain_0shot_fixed_8bit_metrics.json](/root/nlp/outputs/qwen25_7b_ceval_distill_full_yi34b_noise_bayes_sabo_v3_fast2_top2_full/test_full_benchmark_plain_0shot_fixed_8bit_metrics.json)
3. `4-bit` 指标：[test_full_benchmark_plain_0shot_fixed_4bit_metrics.json](/root/nlp/outputs/qwen25_7b_ceval_distill_full_yi34b_noise_bayes_sabo_v3_fast2_top2_full/test_full_benchmark_plain_0shot_fixed_4bit_metrics.json)
