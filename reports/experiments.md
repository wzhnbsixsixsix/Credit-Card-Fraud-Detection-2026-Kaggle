# 实验与变更记录

## 2026-11-25 · 训练管线实现

| 变更 | 对应文件和动机 |
| --- | --- |
| 数据、切分和质量检查 | `src/data.py` 核对输入、固化可复用的分层切分、生成数据审计。 |
| E01–E07 | `configs/experiments.json`、`src/features.py`、`src/models.py` 定义浅树、GLM、CatBoost、数字位和交互消融、折内编码、外部参考和 RealMLP。数字位方案使用自有实现，设计参考 [Scale-Aware Digit GLM](https://www.kaggle.com/code/denpugovkin/s6e11-scale-aware-digit-glm)。 |
| 运行与复现 | `src/__init__.py`、`src/train.py`、`src/results.py` 提供训练入口，记录每折指标、环境、配置和产物 SHA-256。 |
| E08 和提交 | `src/ensemble.py`、`src/validate_submission.py` 对齐所有预测，搜索及冻结融合权重，检查双列 CSV，给出锁定集配对复核。 |
| 安装与产物管理 | `requirements.txt`、`requirements-neural.txt`、`.gitignore` 管理基本和神经网络依赖；数据和训练输出留在本地。 |
| 复现指引和回归检查 | `README.md`、`PROJECT_PLAN.md`、`reports/data_audit.md` 记录执行与数据状态；`tests/test_pipeline.py` 检查切分隔离、目标编码、融合和提交对齐。 |

## 2026-11-25 · 清理早期验证数据

删除了参考 CSV 派生的两套本地验证数据及其运行产物，并从实验记录和数据审计中移除对应指标及统计。正式实验以 `data/raw/` 的比赛包为准。

## 正式比赛记录

正式实验运行后记录候选名称、数据和运行 SHA-256、OOF、复核 ΔAUC、采用理由、Git 提交和 Kaggle 提交 ID。逐折和合并 ROC AUC、预测文件哈希与耗时由各运行的 `run.json` 生成。

## 2026-11-25 · Kaggle 公开方案参考

通过 Kaggle Notebook 下载接口取得 [evgendvorkin 的 Deep EDA / 9-Model Blend](https://www.kaggle.com/code/evgendvorkin/s6e11-deep-eda-9-model-blend-oof-0-951881) 第 53 版。原始 Notebook 保存在 `Kaggle_Credict_Card2026_/` 根目录，位于 `Code/` 仓库之外，文件名为 `s6e11-deep-eda-9-model-blend-oof-0-951881.ipynb`。

第 62 节记录 9 个组件及其权重。外部 OOF Bank 的 `gb_bag5` 占 45.00%，其余为 GPT-2 19.59%、RealMLP PyTabKit 8.34%、LightGBM 6.95%、GLM 6.95%、XGBoost 6.25%、RealMLP PyTorch 2.78%、TabM 2.75%、CatBoost 1.39%。

作者报告八模型 Logit-Rank 的五折 OOF ROC AUC 为 0.95161228，九模型为 0.95188091（提升 0.00026863）。用四折选融合权重并在剩余一折上评估的检查结果为 0.95185586，作者报告的公开榜最佳为 0.95204。以上数值来自 Notebook，项目本地未复跑。

## 2026-11-25 · E01 正式数据首版基线

`e01-lgb2` 使用 24 个原始特征和深度 2 的 LightGBM，按种子 42 对 699,960 条官方训练数据做五折分层验证。训练代码版本为 `11135cb`，LightGBM 4.7.0，本地训练耗时 152.6 秒。本次基线使用正式比赛数据。数据版本见[数据审计记录](data_audit.md)。

| 折 | ROC AUC | 最佳迭代轮数 |
| --- | ---: | ---: |
| 1 | 0.947485 | 2,236 |
| 2 | 0.948363 | 1,944 |
| 3 | 0.946865 | 2,385 |
| 4 | 0.951295 | 2,718 |
| 5 | 0.945528 | 2,985 |

合并 **OOF ROC AUC 为 0.947874**，折均 0.947907，折间标准差 0.001929。

五折平均预测写入本地 `artifacts/baseline/e01-lgb2/submission.csv`。299,984 行通过列名、样例 ID 顺序及预测值校验。提交文件 SHA-256 为 `14d577f8ef0b9151a4bf0d7dd7bc5142e8a18e37386f1124ae0a75d156b78130`，完整记录见同目录的 `run.json`。Kaggle 状态为待上传。

## 2026-11-27 · Beginner to SOTA Guide

从 Kaggle 的 Notebook 下载接口取得 rishabhbafnaiiitd 的第 18 版，供后续复现实验参考。文件保存在项目根目录（`Code/` 外）的 `s6e11-beginner-to-sota-guide-lb-0-9574.ipynb`。Jupyter 文件结构校验通过，含 17 个单元，SHA-256 为 `c3dee3db40a21fd4bb8adfe2a8a74c3632d152fb833617f0912ceca46323490c`。 [Kaggle 原始 Notebook](https://www.kaggle.com/code/rishabhbafnaiiitd/s6e11-beginner-to-sota-guide-lb-0-9574)

中文版与原始文件保存在同一目录，文件名为 `s6e11-beginner-to-sota-guide-lb-0-9574-zh.ipynb`。翻译了 7 个说明单元并将成绩演进表改为适合中文阅读的 Markdown 表格，另翻译了 20 条 Python 注释。保留独立公式和可执行代码，为 17 个单元补齐 Jupyter ID。格式和代码等价性校验通过，SHA-256 为 `90d8066edd067257dd01d2150636968f540371d2460069ac51169101e422cade`。

## 2026-11-27 · 四份 Notebook 适配本地数据

- 四份 Notebook 使用 `Code/data/raw/` 与已注册的 **Python (s6e11-notebooks)** 内核。`environment-notebooks.yml` 记录环境，`README.md` 记录运行方法。
- `01_baseline.ipynb` 可从项目根目录或 `Code/` 打开，已在新内核中核对 299,984 行提交和 OOF ROC AUC 0.947874。LightGBM 首折训练入口在 12,000 行正式训练数据样本上单独通过检查。
- `src/notebook_support.py` 提供共享数据定位、测试 ID 对齐和 SOTA 参考预测校验。两份指南均完整运行，零风险规则命中 16,730 条正式测试记录。末图使用已标注来源的本地 LightGBM 基线预览，精确复现延拓所需的作者预测文件应放入 `data/reference/sota-v18/` 目录。缺失、哈希匹配提交与 AG/AC 延拓分支用临时样本检查通过。
- Deep EDA 的 62 个代码单元全部执行成功，读入比赛数据和 20,000 行公开参考 CSV。第 61–62 节绘制原作者记录的成绩。参考 CSV 保存在 `data/reference/credit_card_fraud_2026.csv`，SHA-256 为 `4e32b1ad4880cae452d1f2e5aa53224d38eb913c185455c00a3759df5eb3da61`。 [参考数据来源](https://www.kaggle.com/datasets/uditjain13/credit-card-fraud-detection-2026)

| 项目根目录中的本地 Notebook | SHA-256 |
| --- | --- |
| `s6e11-beginner-to-sota-guide-lb-0-9574.ipynb` | `b8f435ca12019acc1a9002965c797574be569683a603e097d9b055d0a8c9b43d` |
| `s6e11-beginner-to-sota-guide-lb-0-9574-zh.ipynb` | `4f90ce299feac6cad10a059a231ba589ad08c1ba49bcc204929ac54848bff78f` |
| `s6e11-deep-eda-9-model-blend-oof-0-951881.ipynb` | `f1b1609c30a9e509e95c256338ad4e786533250a917738c33500ce6fd98380ca` |

## 2026-11-27 · Notebook 中英文分类

项目根目录的 `CN/` 和 `EN/` 各提供四份 Notebook，分别对应基线、Deep EDA、Beginner to SOTA Guide 和 Top 100。沿用 SOTA 的现成中文版；补充 Deep EDA 与 Top 100 的中文正文，以及仓库基线的英文正文。Deep EDA 的 90 个 Markdown 单元逐节覆盖，长段落和图文卡片整理为中文要点，代码和输出用语沿用源文件。

Top 100 的两个版本接入本地正式数据和独立 Conda 内核，原始预测组合与哈希校验沿用作者代码。完整复现仍需作者的固定预测文件，目录为 `Code/data/reference/top-100/`；未适配前的原稿归档在 `EN/.originals/`。仓库中的 `01_baseline.ipynb` 是两份分类副本的来源。
