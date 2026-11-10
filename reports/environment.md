# 环境检查记录

检查时间：2026-11-11（Asia/Shanghai）

## 系统默认环境

- `python`：Python 3.5.3，32-bit
- `pip`：20.3.4（旧环境）
- `git`：2.56.0.windows.4
- `jupyter`：4.6.3，绑定旧 Python 3.5 环境
- 默认旧环境缺少 scipy、scikit-learn、xgboost、statsmodels、pytest 等现代依赖

因此没有修改系统 Python，而是创建了项目隔离环境：

```text
Stock_Volatility_Research/.venv
Python 3.12.14
```

## 项目依赖

- numpy 2.5.4
- pandas 3.1.1
- scipy 1.18.1
- scikit-learn 1.9.1
- xgboost 3.4.1
- statsmodels 0.15.0
- matplotlib 3.11.2
- plotly 6.9.0
- streamlit 1.67.0
- pytest 9.2.0
- akshare 1.20.37

精确版本见 `requirements.txt`。

## 联网与数据接口

- 沙箱初始禁用网络；获得临时权限后联网成功。
- PyPI 可访问并完成隔离环境安装。
- 东方财富公开历史行情接口可返回沪深300日线。
- 5 分钟接口可用，但只返回近期滚动窗口，本次为 31 个交易日。
- 未使用需要 API Key 的 Tushare 或学校/商业数据库。

## Skill 检查

项目内 Skill：

```text
.agents/skills/quant-research/
├── SKILL.md
├── agents/openai.yaml
├── references/
│   ├── financial-time-series.md
│   ├── data-quality.md
│   ├── validation-and-evaluation.md
│   └── research-reporting.md
└── scripts/check_temporal_split.py
```

`skill-creator/scripts/quick_validate.py` 验证结果：

```text
Skill is valid!
```

## 测试状态

完整自动测试最后一轮：16 passed（包含 Streamlit `AppTest`）。

真实预测文件时间审计：通过，4,090 行，目标日期严格晚于预测起点。

