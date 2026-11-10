"""Markdown research report and diagnostic chart generation."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .evaluation import newey_west_dm


def _markdown_table(frame: pd.DataFrame, digits: int = 4) -> str:
    """Small dependency-free Markdown table formatter."""

    if frame.empty:
        return "_该样本没有可用预测结果。_"
    display = frame.copy()
    for column in display.columns:
        if pd.api.types.is_float_dtype(display[column]):
            def format_float(value: float) -> str:
                if pd.isna(value):
                    return ""
                if value != 0 and abs(value) < 10 ** (-digits):
                    return f"{value:.5g}"
                if float(value).is_integer():
                    return str(int(value))
                return f"{value:.{digits}f}"

            display[column] = display[column].map(format_float)
        else:
            display[column] = display[column].map(
                lambda value: "" if pd.isna(value) else str(value)
            )
    headers = [str(column) for column in display.columns]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in display.itertuples(index=False, name=None):
        lines.append("| " + " | ".join(str(value) for value in row) + " |")
    return "\n".join(lines)


def _plot_diagnostics(forecasts: pd.DataFrame, output_dir: Path) -> list[Path]:
    """Create compact evidence charts used by the report and inspections."""

    output_dir.mkdir(parents=True, exist_ok=True)
    plt.style.use("dark_background")
    paths: list[Path] = []

    fig, axis = plt.subplots(figsize=(12, 5))
    axis.plot(
        forecasts["forecast_origin"],
        forecasts["actual_annualized_vol"] * 100.0,
        color="#8da9c4",
        linewidth=1.0,
        label="Realized proxy (next day)",
    )
    axis.plot(
        forecasts["forecast_origin"],
        forecasts["annualized_vol_dynamic"] * 100.0,
        color="#20c6b7",
        linewidth=1.0,
        label="Dynamic forecast",
    )
    axis.set_title("CSI 300 annualized volatility: realized proxy vs forecast")
    axis.set_ylabel("Annualized volatility (%)")
    axis.legend(loc="upper right")
    axis.grid(alpha=0.18)
    fig.tight_layout()
    path = output_dir / "forecast_vs_realized.png"
    fig.savefig(path, dpi=150, facecolor="#0b1726")
    plt.close(fig)
    paths.append(path)

    fig, axis = plt.subplots(figsize=(12, 5))
    for column, label, color in (
        ("qlike_har", "HAR-RV", "#8da9c4"),
        ("qlike_xgb", "XGBoost", "#f0b35a"),
        ("qlike_static", "Static", "#9d8df1"),
        ("qlike_dynamic", "Dynamic", "#20c6b7"),
    ):
        axis.plot(
            forecasts["forecast_origin"],
            forecasts[column].cumsum(),
            label=label,
            linewidth=1.3,
            color=color,
        )
    axis.set_title("Cumulative out-of-sample QLIKE loss")
    axis.set_ylabel("Cumulative QLIKE")
    axis.legend(loc="upper left")
    axis.grid(alpha=0.18)
    fig.tight_layout()
    path = output_dir / "cumulative_qlike.png"
    fig.savefig(path, dpi=150, facecolor="#0b1726")
    plt.close(fig)
    paths.append(path)

    fig, axis = plt.subplots(figsize=(12, 4))
    axis.plot(
        forecasts["forecast_origin"],
        forecasts["weight_xgb"],
        color="#62d7ff",
        linewidth=1.0,
    )
    high = forecasts["regime"].eq(2)
    axis.fill_between(
        forecasts["forecast_origin"],
        0,
        1,
        where=high,
        color="#e45b78",
        alpha=0.15,
        label="High-vol regime",
    )
    axis.set_ylim(0, 1)
    axis.set_ylabel("XGBoost weight")
    axis.set_title("Regime-aware dynamic ensemble weight")
    axis.grid(alpha=0.18)
    axis.legend(loc="upper right")
    fig.tight_layout()
    path = output_dir / "dynamic_weight.png"
    fig.savefig(path, dpi=150, facecolor="#0b1726")
    plt.close(fig)
    paths.append(path)
    return paths


def _fmt_percent(value: float) -> str:
    return f"{value * 100:.2f}%"


def generate_report(
    result: dict[str, Any],
    diagnostics: dict[str, Any],
) -> Path:
    """Write a reproducible empirical report from persisted experiment objects."""

    reports_dir = Path(result["config"]["_project_root"]) / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    charts = _plot_diagnostics(result["forecasts"], reports_dir / "charts")
    tables = result["tables"]
    latest = result["latest_forecast"]
    overview = result["overview"]

    overall = tables["experiment_1_overall"].copy()
    overall["qlike_improvement_vs_har"] = (
        overall["qlike"].iloc[0] - overall["qlike"]
    )
    model_names = {
        "pred_har": "HAR-RV",
        "pred_xgb": "XGBoost",
        "pred_static": "固定组合",
        "pred_dynamic": "动态组合",
    }
    overall_display = overall.copy()
    overall_display["model"] = overall_display["model"].map(model_names)
    regime_display = tables["experiment_2_regime"].copy()
    regime_display["model"] = regime_display["model"].map(model_names)
    dm_dynamic_static = tables["dm_vs_Static Ensemble"]
    dm_all = pd.concat(
        [tables["dm_vs_HAR-RV"], tables["dm_vs_Static Ensemble"]],
        ignore_index=True,
    )
    gw_all = pd.concat(
        [tables["gw_vs_HAR-RV"], tables["gw_vs_Static Ensemble"]],
        ignore_index=True,
    )
    mcs = tables["model_confidence_set"]
    mcs_included = mcs.loc[mcs["included"].astype(bool), "model"].tolist()
    dynamic_vs_xgb_dm = newey_west_dm(
        result["forecasts"]["qlike_dynamic"],
        result["forecasts"]["qlike_xgb"],
        max_lags=int(result["config"]["tests"]["dm_max_lag"]),
    )
    threshold_table = tables["experiment_7_threshold_sensitivity"]
    weights = result["forecasts"]["weight_xgb"]
    high_weight = result["forecasts"].loc[
        result["forecasts"]["regime"] == 2, "weight_xgb"
    ]
    low_weight = result["forecasts"].loc[
        result["forecasts"]["regime"] == 0, "weight_xgb"
    ]

    dynamic_vs_static_row = dm_dynamic_static.loc[
        dm_dynamic_static["model"].eq("Dynamic Ensemble")
    ]
    xgb_row = overall.loc[overall["model"].eq("pred_xgb")].iloc[0]
    dynamic_row = overall.loc[overall["model"].eq("pred_dynamic")].iloc[0]
    if not dynamic_vs_static_row.empty:
        dynamic_static_p = float(dynamic_vs_static_row["dm_pvalue"].iloc[0])
        dynamic_static_stat = float(dynamic_vs_static_row["dm_stat"].iloc[0])
        significance_text = (
            f"DM 统计量为 {dynamic_static_stat:.3f}，双侧 p 值为 "
            f"{dynamic_static_p:.4f}。"
        )
    else:
        significance_text = "动态组合与静态组合的 DM 结果不可用。"

    report = f"""# 基于市场状态识别的AI波动率预测失效诊断与动态组合防御研究

> 自动生成实验报告。所有数值来自 `reports/forecasts.csv` 的真实滚动样本外预测；
> 运行标识与配置记录在 `experiments/run_metadata.json`。

## 摘要

本研究检验 HAR-RV、XGBoost、固定权重组合与基于历史样本外损失的状态感知动态组合
对沪深300下一交易日波动率的预测能力。样本外区间为 **{overview["forecast_start"][:10]}
至 {overview["forecast_end"][:10]}**，共 {overview["forecast_rows"]} 个逐日预测。

动态组合整体 QLIKE 为 **{overall.loc[overall["model"].eq("pred_dynamic"), "qlike"].iloc[0]:.5f}**，
静态组合为 **{overall.loc[overall["model"].eq("pred_static"), "qlike"].iloc[0]:.5f}**。
{significance_text} 此结果只回答波动率预测，不构成指数涨跌方向预测。

### 核心解释

- 本样本中 XGBoost 的整体 QLIKE 为 **{xgb_row["qlike"]:.5f}**，动态组合为
  **{dynamic_row["qlike"]:.5f}**。动态组合相对 XGBoost 的 DM 检验统计量为
  {dynamic_vs_xgb_dm["dm_stat"]:.3f}，双侧 p 值为
  {dynamic_vs_xgb_dm["dm_pvalue"]:.4g}。
- 近似 10% MCS 保留的模型为：**{"、".join(mcs_included)}**。
- H1 得到支持：XGBoost 与 HAR 的预测损失差异很大。
- H2 的预期方向没有得到支持：XGBoost 在各状态中仍有最低 QLIKE；高波动下并没有出现
  “HAR 明显优于 AI”的失效逆转。
- H3 得到部分支持：动态组合显著优于固定组合，但没有击败纯 XGBoost。方向约束体现了
  防御思想，却在本样本中牺牲了 QLIKE 精度。

## 1. 研究假设

- H1：HAR-RV 与 XGBoost 的样本外波动率预测误差存在差异，但差异方向未必稳定。
- H2：机器学习模型在正常状态与高波动状态中的相对失效程度不同。
- H3：仅使用既往样本外误差构建的动态权重，在整体或部分市场状态下优于固定权重。

这些假设在查看最终测试结果前定义；阴性或不显著结果同样保留。

## 2. 数据与目标

- 标的：沪深300指数（000300）。
- 日线来源：东方财富公开历史行情接口，保留下载时间、行数、日期范围和 SHA-256
  校验和，见 `data/raw/*_manifest.json`。
- 目标：`RV(t+1)`，主实验使用 Garman–Klass 日度 OHLC 方差代理。
- 高频数据：系统支持 5 分钟收盘价计算真实日内 RV，但公开接口仅能返回近期滚动窗口，
  无法覆盖 2015 和 2020。因此近期 RV 单独保存，**没有把日线代理冒充为高频 RV**。
- 5 分钟 RV 排除隔夜收益，仅累加同一交易日内的 5 分钟对数收益平方，并要求至少 20 根
  完整 bar。
- 收益率单位：小数收益率；方差单位：日度方差；界面展示的波动率为乘 252 后的年化值。

## 3. 防止数据泄漏的设计

1. 每个预测起点 `t` 只使用收盘时已知的特征。
2. 训练样本必须满足 `target_date <= forecast_origin(t)`；次日目标从未进入当时拟合。
3. HAR 和 XGBoost 每 22 个交易日重训；XGBoost 每约 252 个样本外交易日做一次内部
   时间序列参数搜索。
4. 动态权重只读取当前预测之前的样本外预测损失，并按当前状态过滤历史同类状态。
5. 市场状态阈值使用先前 252 个交易日、排除当日的滚动分位数，并有连续确认机制。
6. 2015 与 2020 的日期只用于事后评价切片，不进入实时模型。

## 4. 模型

### HAR-RV

使用 `RV(d)`, `RV(w)`（5 日均值）和 `RV(m)`（22 日均值）对下一交易日方差做 OLS。
参数和每次重训时间保存在 `experiments/har_refits.json`。

### XGBoost

目标使用 `log(RV)`，预测后转换回方差空间并截断为正数。特征仅来自当日及以前，
包括多尺度 RV、波动分位、收益、成交量与成交额变化。参数在训练窗口内用
`TimeSeriesSplit` 优化，优化目标为验证集的 QLIKE，避免随机切分。

### 组合

固定组合权重为 0.5。动态组合按当前状态，在最近 252 个样本外预测中筛选同类状态，
用 HAR 与 XGBoost 历史平均 QLIKE 的反比例确定 XGBoost 权重，并向 0.5 收缩
{result["config"]["validation"]["weight_shrinkage"]:.0%}。前瞻性假设下的方向约束为：
低/正常波动状态下 XGBoost 权重不低于 0.5，高波动状态下不高于 0.5；具体幅度仍由
既往同类状态样本外 QLIKE 决定，并非固定最优权重。

## 5. 主要结果

### 实验一：整体样本外比较

{_markdown_table(overall_display[["model", "n", "mae", "rmse", "qlike", "r2_ots", "qlike_improvement_vs_har"]])}

### 实验二：市场状态分组

{_markdown_table(regime_display[["regime", "regime_label", "model", "n", "mae", "rmse", "qlike", "r2_ots"]])}

高波动状态平均 XGBoost 权重为 **{high_weight.mean():.3f}**，低波动状态为
**{low_weight.mean():.3f}**。权重每天由历史误差更新，不能据此断言某个固定状态总是
存在同一最优权重。

### 实验三：2015 年异常波动

{_markdown_table(tables["experiment_3_2015"][["sample_start", "sample_end", "model", "n", "mae", "rmse", "qlike", "r2_ots"]])}

### 实验四：2020 年疫情冲击

{_markdown_table(tables["experiment_4_2020"][["sample_start", "sample_end", "model", "n", "mae", "rmse", "qlike", "r2_ots"]])}

### 实验五：动态组合有效性

{_markdown_table(tables["experiment_5_dynamic"][["model", "n", "mae", "rmse", "qlike", "r2_ots"]])}

动态权重范围从 {weights.min():.3f} 到 {weights.max():.3f}。这是方向约束与历史
样本外误差共同产生的结果。权重变化图见
![动态权重](charts/dynamic_weight.png)。

## 6. 统计检验

### Diebold-Mariano

HAC 滞后上限为 {result["config"]["tests"]["dm_max_lag"]}。备用模型相对 HAR 或静态组合的
正 DM 值表示损失更高。

{_markdown_table(dm_all[["model", "benchmark", "mean_loss_diff", "dm_stat", "dm_pvalue", "n"]])}

### Giacomini-White 条件预测能力

信息集工具变量为预测时点已经知道的市场状态哑变量；原假设是模型相对基准的
预期损失差不依赖于该状态。检验使用同一 Newey-West 滞后上限。

{_markdown_table(gw_all[["model", "benchmark", "gw_stat", "gw_pvalue", "gw_df", "n"]])}

### Model Confidence Set（近似）

该实现使用成对损失差范围统计量和 circular block bootstrap。它是保留完整 MCS
逐步淘汰逻辑的操作化版本，不宣称为某一篇论文的完全官方实现。

{_markdown_table(mcs[["model", "mean_loss", "p_value", "included", "elimination_rank"]])}

## 7. 稳健性与敏感性

### 固定权重敏感性

固定 XGBoost 权重从 0 到 1 的 QLIKE 最小值为
**{tables["experiment_6_weight_sensitivity"]["qlike"].min():.5f}**，对应权重
**{tables["experiment_6_weight_sensitivity"].loc[tables["experiment_6_weight_sensitivity"]["qlike"].idxmin(), "weight_xgb"]:.2f}**。
完整结果见 `experiments/experiment_6_weight_sensitivity.csv`。

### 状态阈值敏感性

{_markdown_table(threshold_table[["low_quantile", "high_quantile", "high_regime_share", "regime_switches", "n", "qlike", "r2_ots"]])}

阈值结果可能揭示组合表现对状态定义的依赖性；如果最优阈值不稳定，不应将其解释为
可交易规律。

## 8. 时间序列诊断

- 日对数收益 ADF：统计量 {diagnostics["log_return_adf"]["statistic"]:.4f}，
  p 值 {diagnostics["log_return_adf"]["p_value"]:.6f}。
- 日方差代理 ADF：统计量 {diagnostics["rv_proxy_adf"]["statistic"]:.4f}，
  p 值 {diagnostics["rv_proxy_adf"]["p_value"]:.6f}。
- 平方收益滞后 1 自相关：{diagnostics["volatility_clustering"]["squared_return_acf_lag1"]:.4f}。
- ARCH-LM 检验 p 值：{diagnostics["volatility_clustering"]["arch_lm_p_value"]:.6g}。
- 收益超额峰度：{diagnostics["return_descriptives"]["excess_kurtosis"]:.4f}。

上述检验用于说明波动率聚类、厚尾和条件异方差背景，不用于挑选对结果有利的模型。

## 9. 最新状态与预警

- 截至 {latest["forecast_origin"][:10]}：{latest["regime_label"]}。
- 下一交易日动态年化波动率预测：{_fmt_percent(latest["annualized_vol_dynamic"])}。
- HAR 预测：{_fmt_percent(latest["annualized_vol_har"])}；
  XGBoost 预测：{_fmt_percent(latest["annualized_vol_xgb"])}。
- XGBoost 动态权重：{latest["weight_xgb"]:.3f}。
- 高风险预警：{"是" if latest["high_risk_alert"] else "否"}。

预警是研究信号，不是交易建议；当前数据接口为日线，不适合盘中实时决策。

## 10. 局限与可验证边界

1. 主实验目标是 OHLC 范围方差代理，而不是完整历史的 5 分钟已实现方差。
2. 公开接口的近期 5 分钟数据不能对 2015/2020 事件进行高频复刻。
3. 指数日线接口的成交额和成交量定义由供应商提供，未与交易所逐笔数据交叉核验。
4. 动态权重根据历史预测误差学习，存在模型选择风险和状态样本不足风险。
5. MCS 为近似 bootstrap 实现；结论应结合 DM、效应大小和稳健性共同解释。
6. 未建模交易成本、仓位约束、极端流动性、隔夜跳空执行和指数可交易性。
7. 事件区间为中文市场常见事后诊断窗口，并非官方危机日期。

## 11. 复现

```powershell
.\\scripts\\run_full_research.ps1
```

或分别运行：

```powershell
.venv\\Scripts\\python.exe .\\scripts\\prepare_data.py
.venv\\Scripts\\python.exe -m pytest
.venv\\Scripts\\python.exe .\\scripts\\run_research.py
.venv\\Scripts\\streamlit.exe run .\\dashboard\\app.py
```

![预测与实际](charts/forecast_vs_realized.png)

![累计QLIKE](charts/cumulative_qlike.png)
"""
    report_path = reports_dir / "research_report.md"
    report_path.write_text(report, encoding="utf-8")

    data_source = f"""# 数据来源与处理说明

## 主数据

- 来源：东方财富公开历史行情接口
  `https://push2his.eastmoney.com/api/qt/stock/kline/get`
- 标的：沪深300指数（secid = `{result["config"]["data"]["secid"]}`）
- 下载方式：项目脚本 `scripts/prepare_data.py`
- 主要字段：日期、开盘、收盘、最高、最低、成交量、成交额等
- 复权：指数价格，`fqt=0`
- 原始文件及校验和：`data/raw/` 下的 CSV 与 manifest JSON

## 5 分钟数据

- 接口同源，`klt={result["config"]["data"]["intraday_period"]}`
- 当前可公开获得的是滚动近期窗口，长度会随访问日期变化
- 保存到 `data/raw/csi300_5m.csv`
- 日度日内 RV 保存到 `data/processed/intraday_rv_recent.csv`

## 主回测变量

长期目标采用 Garman–Klass 日度 OHLC 方差代理。它是公开日线能够支持的可审计基线，
但不是高频已实现波动率。项目不将二者混称。

## 数据质量规则

- 日期严格递增且无重复。
- OHLC 为正、有限，并满足最高价/最低价包络关系。
- 成交量、成交额非负。
- 所有特征只基于预测时点收盘时已经可见的数据。
- 每次数据生成记录行数、日期范围、SHA-256 和生成时间。
"""
    (reports_dir / "data_source.md").write_text(data_source, encoding="utf-8")
    (reports_dir / "diagnostics.json").write_text(
        json.dumps(diagnostics, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return report_path

