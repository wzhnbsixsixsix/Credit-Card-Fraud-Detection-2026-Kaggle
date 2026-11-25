# 数据审计记录

## 2026-11-25 · 文件核对

根目录的 `archive.zip` 只有一个 `credit_card_fraud_2026.csv`，对应 20,000 条带标签交易，标识列为 `transaction_id`。官方比赛下载需 Kaggle 登录授权，工作区未取得官方 `train.csv`、`test.csv` 和 `sample_submission.csv`。 [官方比赛数据页](https://www.kaggle.com/competitions/playground-series-s6e11/data)

本地流程验证以这 20,000 条交易做 80/20 分层划分（种子 42），得到 16,000 条本地输入训练样本和 4,000 条模拟测试样本。模拟样例的 ID 顺序特意反转，检查提交程序能否按 ID 对齐。额外的参考表实验使用一次独立子划分，在输入训练行与参考行之间保持互斥。生成文件在 `data/local-check/` 与 `data/local-check-external/`。

官方数据到位后，运行 `python -m src.train audit` 生成带 SHA-256 的实测审计。审计统计开发集类别和数值范围、重复特征组、ID 分段、分箱欺诈率以及测试覆盖率。最终复核集保留到候选冻结。
