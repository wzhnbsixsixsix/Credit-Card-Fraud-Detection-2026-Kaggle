# 数据审计记录

## 2026-11-25 · 文件核对

首次核对时，根目录的 `archive.zip` 只有一个 `credit_card_fraud_2026.csv`，对应 20,000 条带标签交易，标识列为 `transaction_id`。官方比赛下载需 Kaggle 登录授权，当时工作区尚未取得官方 `train.csv`、`test.csv` 和 `sample_submission.csv`。 [官方比赛数据页](https://www.kaggle.com/competitions/playground-series-s6e11/data)

本地流程验证以这 20,000 条交易做 80/20 分层划分（种子 42），得到 16,000 条本地输入训练样本和 4,000 条模拟测试样本。模拟样例的 ID 顺序特意反转，检查提交程序能否按 ID 对齐。额外的参考表实验使用一次独立子划分，在输入训练行与参考行之间保持互斥。生成文件在 `data/local-check/` 与 `data/local-check-external/`。

官方数据到位后，运行 `python -m src.train audit` 生成带 SHA-256 的实测审计。审计统计开发集类别和数值范围、重复特征组、ID 分段、分箱欺诈率以及测试覆盖率。最终复核集保留到候选冻结。

## 2026-11-25 · 正式比赛包核对

已核验根目录的 `playground-series-s6e11.zip` 并将三份 CSV 解压到 `data/raw/`。CSV、压缩包与模型输出均留在本地。`.gitignore` 加入比赛 ZIP 文件名，避免误提交。

| 文件 | 数据行数 | SHA-256 |
| --- | ---: | --- |
| `train.csv` | 699,960 | `87eaa10b8d17a7e4f657b8da1c66242431338bdd8b6ad45861521ac0c0e9fbd5` |
| `test.csv` | 299,984 | `10f516959020357b565abaaa4fd25baae97fc2f536d0805695bb7cc15c1660f0` |
| `sample_submission.csv` | 299,984 | `13fc6ed8b73de5329884ec490f2a0b750cf29a7cda772b78cfa0cf486f6f35fa` |

正式训练集有 11,689 条欺诈样本（1.670%）。原始参考数据有 339 条欺诈样本（1.695%）。将参考数据的 `transaction_id` 对应到比赛的 `id` 后，两份带标签数据的字段一致，正式训练集含 24 个预测特征。正式测试集的 ID 与样例提交一一对应。

固定切分保留 139,992 行供冲刺复核，开发集 559,968 行。审计在开发集和正式测试集的 24 个特征中未发现缺失或无穷值，开发集无重复特征行，测试行未与开发集发生全特征匹配。完整报告见本地 `artifacts/audit/c648e55d95bc-5-42/audit.json`。
