# 沪深300 AI 波动率预测研究系统

这是一个可运行、可复现、按真实时间顺序评估的量化金融研究项目：

> 基于市场状态识别的 AI 波动率预测失效诊断与动态组合防御研究

系统使用沪深300公开日线行情，比较 HAR-RV、XGBoost、固定权重组合和
状态感知动态组合，并提供滚动样本外回测、DM/Giacomini-White 检验、近似 MCS、
情景稳健性、研究报告和中文 Streamlit 仪表盘。

## 当前真实结果

最近一次完整运行使用：

- 日线样本：2005-01-04 至 2026-11-10，共 5,306 行；
- 样本外预测：2010-01-04 至 2026-11-09，共 4,090 个逐日预测；
- 模型重训：186 次；XGBoost 调参：16 次；
- 目标：下一交易日 Garman–Klass 日度 OHLC 方差代理。

整体样本外 QLIKE：

| 模型 | QLIKE |
| --- | ---: |
| HAR-RV | 0.69545 |
| XGBoost | **0.32314** |
| 固定组合 | 0.47832 |
| 动态组合 | 0.43724 |

关键发现：

1. XGBoost 在整体、低波动、正常波动和高波动状态中均有最低 QLIKE。
2. 动态组合显著优于固定组合，但在本样本中没有击败纯 XGBoost。
3. 近似 MCS（10% 水平）只保留 XGBoost，说明其他模型差距不是简单随机波动。
4. 高波动状态下，方向约束把 XGBoost 权重降至 0.5；这体现了防御假设，
   但也牺牲了相对 XGBoost 的预测精度。
5. 结论只涉及波动率预测误差，不代表指数涨跌方向、投资收益或可实施交易策略。

完整结果见 [reports/research_report.md](reports/research_report.md)。

## 数据限制

公开接口能稳定获取长期日线，但当前 5 分钟接口只返回约 31 个交易日的滚动历史。
因此：

- 长期主实验使用日线 OHLC 方差代理；
- 近期 5 分钟数据单独计算真实日内 RV；
- 没有把日线代理冒充高频 RV；
- 无法对 2015/2020 做完整历史 5 分钟 RV 复刻；
- 若要继续升级，应导入合法授权的全历史 1/5 分钟数据或交易所授权数据。

详见 [reports/data_source.md](reports/data_source.md)。

## 快速启动

环境要求：Python 3.11+（本项目验证于 Python 3.12.14）、PowerShell、可访问
东方财富公开行情接口的网络。

在项目根目录运行：

```powershell
.\scripts\run_full_research.ps1
```

也可以直接双击项目根目录的 `run_research.cmd`，或运行：

```bat
run_research.cmd
```

脚本会：

1. 若无 `.venv` 则创建隔离环境；
2. 安装锁定版本依赖；
3. 下载并校验数据；
4. 运行自动测试；
5. 执行滚动回测与统计检验；
6. 生成报告和图表。

启动仪表盘：

```powershell
.\scripts\start_dashboard.ps1
```

也可以直接双击 `start_dashboard.cmd`。浏览器通常自动打开
`http://localhost:8501`。

首次发送到另一台电脑时，`start_dashboard.cmd` 会自动创建隔离环境并安装依赖。
更完整的分享说明见 `SHARE_GUIDE.md`。

已有结果时，也可只重启服务：

```powershell
.\.venv\Scripts\streamlit.exe run .\streamlit_app.py
```

## 开源与公网分享

本仓库已按 MIT License 准备为开源项目：

- 代码许可证：[LICENSE](LICENSE)；
- 贡献规则：[CONTRIBUTING.md](CONTRIBUTING.md)；
- 安全规则：[SECURITY.md](SECURITY.md)；
- 研究免责声明：[DISCLAIMER.md](DISCLAIMER.md)；
- GitHub 与 Streamlit Community Cloud 部署步骤：
  [docs/DEPLOYMENT.md](docs/DEPLOYMENT.md)；
- GitHub Actions 自动测试：`.github/workflows/tests.yml`。

`streamlit_app.py` 是公网部署入口。公开 Git 仓库不会提交供应商原始 CSV；
下载脚本、manifest、报告和样本外预测足以支持复现与仪表盘展示。数据权利和
再分发边界见 [data/README.md](data/README.md)。

## 项目结构

```text
Stock_Volatility_Research/
├── streamlit_app.py            # Streamlit Cloud 主入口
├── LICENSE
├── CONTRIBUTING.md
├── SECURITY.md
├── DISCLAIMER.md
├── Dockerfile                  # 容器化公网部署
├── docs/DEPLOYMENT.md          # GitHub 与 Streamlit Cloud 部署
├── .github/workflows/tests.yml # CI
├── AGENTS.md                   # 长期研究规则
├── README.md
├── requirements.txt            # 精确版本
├── .agents/skills/
│   └── quant-research/         # 可复用量化研究 Skill
├── config/default.yaml         # 数据、模型、回测与检验配置
├── data/
│   ├── raw/                    # 原始行情与 manifest
│   └── processed/              # 特征、近期 RV 与 manifest
├── src/
│   ├── data_loader.py
│   ├── feature_engineering.py
│   ├── har_model.py
│   ├── xgboost_model.py
│   ├── regime_detection.py
│   ├── dynamic_ensemble.py
│   ├── backtest.py
│   ├── evaluation.py
│   ├── diagnostics.py
│   └── reporting.py
├── dashboard/app.py
├── experiments/                # 指标、检验、重训日志
├── models/                     # 最终模型对象
├── reports/                    # 报告、预测、图表与数据说明
├── scripts/                    # 数据、研究和启动脚本
└── tests/                      # 自动测试
```

## 研究设计

### 严格时间顺序

- 预测起点 `t` 的特征仅使用截至 `t` 收盘的数据；
- 目标为 `t+1`；
- 当前预测的训练样本必须满足目标日 `<= t`；
- 禁止随机打乱时间序列；
- XGBoost 参数只在训练窗口内部的 `TimeSeriesSplit` 上选择；
- 动态权重只使用已经实现的既往样本外损失。

### 市场状态

以 5 日平均波动代理的滚动分位数划分低/正常/高状态：

- 阈值基于过去 252 个交易日；
- 当日被排除在自身阈值计算之外；
- 至少连续 3 日确认，减少临界值附近频繁切换；
- 事件窗口只用于事后诊断，不进入实时状态。

### 动态权重

- 在当前预测前的 252 个样本外预测中，筛选与当前状态相同的历史记录；
- 至少 30 条记录才启用状态特定权重；
- 使用 HAR/XGBoost 历史 QLIKE 的反比例；
- 向 0.5 收缩 20%；
- 方向约束：低/正常状态 XGB 权重不低于 0.5，高状态不高于 0.5；
- 具体幅度仍由历史误差决定，而不是固定某个“最优”权重。

## 自动检查

```powershell
.\.venv\Scripts\python.exe -m pytest
```

覆盖：

- CSV 数据加载与 OHLC 完整性；
- 5 分钟 RV；
- 特征因果性；
- 目标严格位于未来；
- HAR 和 XGBoost 训练；
- 动态权重及状态方向；
- 市场状态因果性；
- QLIKE、MAE、RMSE、DM、Giacomini-White；
- 短滚动回测；
- 训练与预测时间不重叠。

外部审计预测文件：

```powershell
.\.venv\Scripts\python.exe `
  .\.agents\skills\quant-research\scripts\check_temporal_split.py `
  .\reports\forecasts.csv `
  --origin-column forecast_origin `
  --target-column target_date
```

## 主要输出

- `reports/research_report.md`：实证研究报告；
- `reports/forecasts.csv`：全部逐日样本外预测；
- `reports/latest_forecast.json`：最新状态和预测；
- `experiments/*.csv`：七组实验和统计检验；
- `experiments/har_refits.json`：每次 HAR 重训参数；
- `experiments/xgb_refits.json`：每次 XGBoost 重训与调参记录；
- `experiments/run_metadata.json`：配置、时间范围和运行信息；
- `models/*.joblib`：最新最终模型；
- `data/**/manifest.json`：数据来源、范围和 SHA-256。

## 继续升级

优先顺序建议：

1. 导入全历史 5 分钟或 1 分钟数据，重新定义真实 RV；
2. 增加 VIX 类隐含波动率、宏观信息发布日期和期权信息；
3. 用 DM/GW 的预注册子样本检验决定组合是否需要方向约束；
4. 加入交易成本、预测区间和风险预算，但不要将其与纯预测精度混为一谈；
5. 若扩展深度学习，必须以当前 HAR/XGBoost 为强基线并重复多个随机种子。

本项目不构成投资建议。

