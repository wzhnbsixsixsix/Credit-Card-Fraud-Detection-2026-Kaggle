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

## 2026-11-25 · 本地流程验证

以下数值来自原始参考数据的本地分层切分，比赛正式成绩需要使用官方 CSV 重新评估。

| 验证项 | 实测结果 |
| --- | --- |
| LightGBM 完整五折基线 | OOF ROC AUC `0.940016`。模拟提交通过 4,000 行顺序和概率检查，SHA-256 `2935b76cefa74fcefafd3334150dae975b895baf9c0dadef897b9e152643d800`。 |
| 开发集的完整五折 | CatBoost `0.932938`、加性 GLM `0.921562`、数字位 GLM `0.900720`、RealMLP `0.933117`。 |
| 融合与最终重训 | CatBoost 和 GLM 的排名融合开发 OOF 为 `0.941775`。锁定集 AUC 从 `0.930918` 升至 `0.932689`，配对 95% 区间为 `[-0.003399, +0.006335]`，本地复核保留 CatBoost。README 的 LightGBM 与数字位 GLM 路线也完整跑过，锁定集复核保留 LightGBM。两条融合路线都完成固定配置重训、权重重放及 CSV 校验。RealMLP 使用冻结的 9 轮完成重训并通过 CSV 校验。 |
| 特征分支 | 频数、折内目标编码、数字位树模型、打乱对照、显式交互、RealMLP 数字位，以及外部计数、外部风险率、追加外部行都完成单折检查。 |
| 回归测试 | `python -m unittest discover -s tests -v` 通过 4 项。 |

## 正式比赛记录

正式实验运行后记录候选名称、数据和运行 SHA-256、OOF、复核 ΔAUC、采用理由、Git 提交和 Kaggle 提交 ID。逐折和合并 ROC AUC、预测文件哈希与耗时由各运行的 `run.json` 生成。

## 2026-11-25 · Kaggle 公开方案参考

通过 Kaggle Notebook 下载接口取得 [evgendvorkin 的 Deep EDA / 9-Model Blend](https://www.kaggle.com/code/evgendvorkin/s6e11-deep-eda-9-model-blend-oof-0-951881) 第 53 版。原始 Notebook 保存在 `Kaggle_Credict_Card2026_/` 根目录，位于 `Code/` 仓库之外，文件名为 `s6e11-deep-eda-9-model-blend-oof-0-951881.ipynb`。

第 62 节记录 9 个组件及其权重。外部 OOF Bank 的 `gb_bag5` 占 45.00%，其余为 GPT-2 19.59%、RealMLP PyTabKit 8.34%、LightGBM 6.95%、GLM 6.95%、XGBoost 6.25%、RealMLP PyTorch 2.78%、TabM 2.75%、CatBoost 1.39%。

作者报告八模型 Logit-Rank 的五折 OOF ROC AUC 为 0.95161228，九模型为 0.95188091（提升 0.00026863）。用四折选融合权重并在剩余一折上评估的检查结果为 0.95185586，作者报告的公开榜最佳为 0.95204。以上数值来自 Notebook，项目本地未复跑。

## 2026-11-25 · E01 正式数据首版基线

`e01-lgb2` 使用 24 个原始特征和深度 2 的 LightGBM，按种子 42 对 699,960 条官方训练数据做五折分层验证。训练代码版本为 `11135cb`，LightGBM 4.7.0，本地训练耗时 152.6 秒。原始参考数据用于此前的流程验证，本次基线仅使用正式比赛数据。数据版本见[数据审计记录](data_audit.md)。

| 折 | ROC AUC | 最佳迭代轮数 |
| --- | ---: | ---: |
| 1 | 0.947485 | 2,236 |
| 2 | 0.948363 | 1,944 |
| 3 | 0.946865 | 2,385 |
| 4 | 0.951295 | 2,718 |
| 5 | 0.945528 | 2,985 |

合并 **OOF ROC AUC 为 0.947874**，折均 0.947907，折间标准差 0.001929。

五折平均预测写入本地 `artifacts/baseline/e01-lgb2/submission.csv`。299,984 行通过列名、样例 ID 顺序及预测值校验。提交文件 SHA-256 为 `14d577f8ef0b9151a4bf0d7dd7bc5142e8a18e37386f1124ae0a75d156b78130`，完整记录见同目录的 `run.json`。Kaggle 状态为待上传。
