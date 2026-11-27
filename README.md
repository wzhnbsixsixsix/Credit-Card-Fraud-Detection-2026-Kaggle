# Credit Card Fraud Detection 2026

Kaggle Playground Series S6E11 信用卡欺诈预测。用固定五折比较模型，以 ROC AUC 选优，输出按样例 ID 排序的 `id,is_fraud` CSV。每日安排见[高分冲刺计划](PROJECT_PLAN.md)。

## 跑出第一份提交

使用 Python 3.11–3.13。在 macOS 上先装 LightGBM 需要的 OpenMP 运行库 `brew install libomp`。

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

加入比赛并接受规则后，从 [Kaggle Data](https://www.kaggle.com/competitions/playground-series-s6e11/data) 下载并解压 `train.csv`、`test.csv` 和 `sample_submission.csv` 到 `data/raw/`。在 Kaggle Notebook 上运行时，把命令中的 `--data-dir` 指向已挂载的比赛数据目录。正式基线使用比赛包里的三份文件。核对过程见[数据审计记录](reports/data_audit.md)。

```bash
python -m src.train audit
python -m src.train fit e01-lgb2 --phase baseline --fold 0
python -m src.train fit e01-lgb2 --phase baseline
python -m src.validate_submission artifacts/baseline/e01-lgb2/submission.csv
```

一折检查只生成该折的预测和指标。完整训练生成 `oof.csv`、`test.csv`、`run.json` 和 `submission.csv`。打开 [Kaggle 提交页](https://www.kaggle.com/competitions/playground-series-s6e11/submissions) 上传完整版本，通过平台评分后把提交 ID、Public AUC 和对应的文件哈希记到[实验记录](reports/experiments.md)。

本地已生成首版正式参赛文件 `artifacts/baseline/e01-lgb2/submission.csv`，五折 OOF ROC AUC 为 **0.947874**。

## 在 Jupyter 中运行基线

在项目根目录激活上述环境，执行 `python -m pip install jupyterlab matplotlib`，再运行 `python -m jupyterlab`。打开 [01_baseline.ipynb](01_baseline.ipynb)，选择 Python 3 内核并依次运行。Notebook 包含 EDA、LightGBM 五折、OOF AUC 和 ROC 曲线，以及提交文件检查。已有基线时会核对数据哈希并读取产物。改用新运行目录即可重新训练。

## 本地运行 Notebook

在项目根目录或 `Code/` 打开 Notebook，选择 **Python (s6e11-notebooks)** 内核。四份 Notebook 共用 `Code/data/raw/` 中的 `train.csv`、`test.csv` 和 `sample_submission.csv`。

需要在另一台机器安装时，在仓库目录运行：

```bash
conda env create -f environment-notebooks.yml
conda run -n s6e11-notebooks python -m ipykernel install --user \
  --name s6e11-notebooks --display-name "Python (s6e11-notebooks)"
```

- `01_baseline.ipynb` 会核对并复用已完成的 LightGBM 基线。首次运行时会训练并生成提交文件。
- `Deep EDA / 9-Model Blend` 使用 `data/reference/credit_card_fraud_2026.csv` 做原始数据对照，最后两节绘制作者记录的模型成绩。 [Kaggle 原始参考数据](https://www.kaggle.com/datasets/uditjain13/credit-card-fraud-detection-2026)
- 中英文 `Beginner to SOTA Guide` 在现有比赛 CSV 上运行分析。复现最终延拓时，将作者提交文件或 Step AG 与 Step AC 的预测 CSV 放入 `data/reference/sota-v18/`。两份 Notebook 在该目录内按原文 SHA-256 查找。没有对应文件时，最后一张图可预览本地 LightGBM 基线，并显示预测来源。 [作者 Notebook](https://www.kaggle.com/code/rishabhbafnaiiitd/s6e11-beginner-to-sota-guide-lb-0-9574)

## 比较模型与融合

```bash
python -m src.train fit e01-lgb2
python -m src.train fit e04-digits
python -m src.ensemble search artifacts/dev/e01-lgb2 artifacts/dev/e04-digits --name sprint
python -m src.ensemble review artifacts/dev/e01-lgb2 artifacts/dev/sprint --name sprint-vs-baseline
```

默认的开发阶段锁定完整五折中的第 0 折，再对其余数据重新做五折。训练时输出开发集 OOF 和锁定集预测。冻结候选后用 `review` 一次对比锁定集 AUC 和配对置信区间。融合搜索比较概率、logit 和排名均值及三档权重，`comparison.json` 同时记录两个单模型、折均增益和改善折数。

| 实验 | 命令中的实验名 |
| --- | --- |
| LightGBM 深度对照 | `e01-lgb1`、`e01-lgb2`、`e01-lgb4` |
| 加性 GLM 与 CatBoost | `e02-glm`、`e03-cat4`、`e03-cat6` |
| 数字位消融与打乱对照 | `e04-digits`、`e04-digits-only`、`e04-digit-control`、`e04-lgb-digits` |
| 交互、折内频数和目标编码 | `e05-interactions`、`e06-freq`、`e06-target` |
| 外部参考数据对照 | `e06-external-count`、`e06-external-risk`、`e06-external-rows` |
| RealMLP 原始及数字位特征 | `e07-realmlp`、`e07-realmlp-digits` |

参数见 [experiments.json](configs/experiments.json)。用 `--threads` 控制每次训练线程数，用 `--model-seed` 和 `--name` 跑多种子对照。RealMLP 需安装 `requirements-neural.txt`，如有可用的 NVIDIA GPU 可加 `--device cuda`。外部数据对照需显式传入 `--external-data 路径`，使用前核对 [比赛规则](https://www.kaggle.com/competitions/playground-series-s6e11/rules) 和参考数据许可。

## 生成最终文件

将通过复核的单模放回完整五折训练。LightGBM、CatBoost 和 RealMLP 使用开发阶段各折最佳轮数的中位数。下面以双模型融合通过复核为例。若复核选中单模，仅运行对应的重训和提交检查。

```bash
python -m src.train fit e01-lgb2 --phase final --from-run artifacts/dev/e01-lgb2
python -m src.train fit e04-digits --phase final --from-run artifacts/dev/e04-digits
python -m src.ensemble apply artifacts/final/e01-lgb2 artifacts/final/e04-digits \
  --recipe artifacts/dev/sprint/recipe.json --name sprint
python -m src.validate_submission artifacts/final/sprint/submission.csv
```

每次运行保留数据、折分配和预测文件的 SHA-256、依赖版本、Git 版本和参数。代码、参数与实验摘要纳入 Git，原始数据、预测和提交文件保存在本地。每个运行目录保留一次完整结果，重跑时用新参数 `--name`，或指定另一个 `--out`。

## 修改记录

- 2026-11-25 · 补充比赛说明和计划入口，便于从仓库首页找到每日任务与实验记录要求。
- 2026-11-25 · 加入数据审计、五折训练、锁定集复核及最终融合的运行说明，链接实测记录。
- 2026-11-25 · 增加正式数据基线的本地文件位置和验证成绩。

- 2026-11-25 · 加入基线 Notebook 及打开方式，复用现有训练和提交检查代码。
- 2026-11-25 · 清理早期参考数据、派生验证目录和产物，统一使用正式比赛数据。
- 2026-11-27 · 记录四份 Notebook 的本地数据目录、独立 Conda 内核和参考预测文件位置。
