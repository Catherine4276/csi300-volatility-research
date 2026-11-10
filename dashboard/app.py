"""Streamlit research dashboard for CSI 300 volatility forecasts."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots


PROJECT_ROOT = Path(__file__).resolve().parents[1]

st.set_page_config(
    page_title="沪深300波动率研究台",
    page_icon="◈",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    :root { --bg:#07111f; --panel:#0d1d31; --line:#1d3a56; --cyan:#20c6b7;
            --blue:#62d7ff; --amber:#f0b35a; --red:#e45b78; --muted:#8ea7bd; }
    .stApp { background: radial-gradient(circle at 80% 0%, #102a46 0%, #07111f 42%, #050b13 100%); }
    [data-testid="stSidebar"] { background: linear-gradient(180deg,#081525,#0b1b2d); border-right:1px solid #17314c; }
    h1,h2,h3 { letter-spacing:.02em; }
    .hero { padding:1.25rem 1.5rem; border:1px solid rgba(32,198,183,.28);
            border-radius:16px; background:linear-gradient(120deg,rgba(13,29,49,.95),rgba(8,24,39,.78));
            box-shadow:0 16px 50px rgba(0,0,0,.24); margin-bottom:1rem; }
    .eyebrow { color:#20c6b7; font-size:.75rem; letter-spacing:.18em; text-transform:uppercase; }
    .hero-title { font-size:2rem; font-weight:700; margin:.25rem 0; color:#eef7ff; }
    .hero-sub { color:#8ea7bd; font-size:.93rem; }
    .metric-card { min-height:112px; padding:1rem 1.05rem; border-radius:14px;
                  border:1px solid #1d3a56; background:linear-gradient(145deg,#0d1d31,#0a1727);
                  box-shadow:inset 0 1px rgba(255,255,255,.03); }
    .metric-label { color:#8ea7bd; font-size:.76rem; text-transform:uppercase; letter-spacing:.08em; }
    .metric-value { color:#eef7ff; font-size:1.55rem; font-weight:650; margin-top:.32rem; }
    .metric-note { color:#667f95; font-size:.72rem; margin-top:.28rem; }
    .alert-high { padding:.9rem 1rem; border-radius:12px; background:rgba(228,91,120,.12);
                  border:1px solid rgba(228,91,120,.45); color:#ffd1dc; }
    .alert-calm { padding:.9rem 1rem; border-radius:12px; background:rgba(32,198,183,.10);
                  border:1px solid rgba(32,198,183,.38); color:#c9fff8; }
    div[data-testid="stDataFrame"] { border:1px solid #1d3a56; border-radius:12px; overflow:hidden; }
    .small-note { color:#7890a4; font-size:.8rem; }
    </style>
    """,
    unsafe_allow_html=True,
)


def card(label: str, value: str, note: str = "") -> str:
    return (
        f'<div class="metric-card"><div class="metric-label">{label}</div>'
        f'<div class="metric-value">{value}</div><div class="metric-note">{note}</div></div>'
    )


@st.cache_data(show_spinner=False)
def load_csv(path: str) -> pd.DataFrame:
    frame = pd.read_csv(path)
    for column in ("date", "forecast_origin", "target_date"):
        if column in frame:
            frame[column] = pd.to_datetime(frame[column], errors="coerce")
    return frame


@st.cache_data(show_spinner=False)
def load_json(path: str) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


forecast_path = PROJECT_ROOT / "reports" / "forecasts.csv"
dataset_path = PROJECT_ROOT / "data" / "processed" / "modeling_dataset.csv"
latest_path = PROJECT_ROOT / "reports" / "latest_forecast.json"
metric_path = PROJECT_ROOT / "experiments" / "experiment_1_overall.csv"
dm_path = PROJECT_ROOT / "experiments" / "dm_vs_Static Ensemble.csv"
mcs_path = PROJECT_ROOT / "experiments" / "model_confidence_set.csv"

st.markdown(
    """
    <div class="hero">
      <div class="eyebrow">Quantitative Research Console / CSI 300</div>
      <div class="hero-title">波动率预测失效诊断与动态组合防御</div>
      <div class="hero-sub">严格滚动样本外 · 状态识别 · HAR-RV / XGBoost · 可复现实验</div>
    </div>
    """,
    unsafe_allow_html=True,
)

if not forecast_path.exists() or not latest_path.exists():
    st.warning("尚无研究结果。请先运行 `./scripts/run_full_research.ps1`。")
    st.code(
        ".\\.venv\\Scripts\\python.exe .\\scripts\\prepare_data.py\n"
        ".\\.venv\\Scripts\\python.exe .\\scripts\\run_research.py",
        language="powershell",
    )
    st.stop()

forecasts = load_csv(str(forecast_path))
dataset = load_csv(str(dataset_path)) if dataset_path.exists() else None
latest = load_json(str(latest_path))
metrics = load_csv(str(metric_path))

with st.sidebar:
    st.markdown("### 研究控制台")
    st.caption("所有预测均为历史样本外结果；最新预测为下一交易日方差预测。")
    min_date = forecasts["forecast_origin"].min().date()
    max_date = forecasts["forecast_origin"].max().date()
    date_range = st.date_input(
        "样本外区间",
        value=(min_date, max_date),
        min_value=min_date,
        max_value=max_date,
    )
    if isinstance(date_range, tuple) and len(date_range) == 2:
        selected = forecasts.loc[
            forecasts["forecast_origin"].dt.date.between(*date_range)
        ].copy()
    else:
        selected = forecasts.copy()
    model_labels = {
        "HAR-RV": "annualized_vol_har",
        "XGBoost": "annualized_vol_xgb",
        "动态组合": "annualized_vol_dynamic",
    }
    selected_models = st.multiselect(
        "显示模型",
        options=list(model_labels),
        default=list(model_labels),
    )
    st.divider()
    st.markdown("**口径提醒**")
    st.caption("主目标为日线 OHLC 方差代理；近期 5 分钟 RV 不用于长期结论。")

if selected.empty:
    st.info("当前筛选区间没有预测记录。")
    st.stop()

regime_name = latest["regime_label"]
alert_class = "alert-high" if latest["high_risk_alert"] else "alert-calm"
alert_text = (
    "高波动状态或滚动阈值已触发。请检查预测误差、状态稳定性与事件暴露。"
    if latest["high_risk_alert"]
    else "当前未触发高波动预警；状态和误差仍需随每日数据滚动复核。"
)
st.markdown(
    f'<div class="{alert_class}"><strong>风险状态：{regime_name}</strong> · {alert_text}</div>',
    unsafe_allow_html=True,
)
st.write("")

columns = st.columns(4)
dynamic_metrics = metrics.loc[metrics["model"].eq("pred_dynamic")]
qlike_text = (
    f"{dynamic_metrics['qlike'].iloc[0]:.5f}" if not dynamic_metrics.empty else "N/A"
)
values = [
    ("下一交易日动态波动率", f"{latest['annualized_vol_dynamic'] * 100:.2f}%", "年化口径"),
    ("XGBoost 动态权重", f"{latest['weight_xgb']:.1%}", "由既往样本外损失推导"),
    ("当前市场状态", regime_name, f"截至 {latest['forecast_origin'][:10]}"),
    ("动态组合整体 QLIKE", qlike_text, "方差预测损失"),
]
for column, (label, value, note) in zip(columns, values):
    with column:
        st.markdown(card(label, value, note), unsafe_allow_html=True)

st.write("")
market_tab, volatility_tab, ensemble_tab, model_tab, event_tab, method_tab = st.tabs(
    ["市场行情", "波动率预测", "状态与权重", "模型诊断", "事件复盘", "数据与方法"]
)

with market_tab:
    range_start = pd.Timestamp(selected["forecast_origin"].min())
    range_end = pd.Timestamp(selected["forecast_origin"].max())
    if dataset is not None:
        chart_data = dataset.loc[
            dataset["date"].between(range_start - pd.Timedelta(days=40), range_end)
        ]
        figure = go.Figure(
            data=[
                go.Candlestick(
                    x=chart_data["date"],
                    open=chart_data["open"],
                    high=chart_data["high"],
                    low=chart_data["low"],
                    close=chart_data["close"],
                    increasing_line_color="#20c6b7",
                    decreasing_line_color="#e45b78",
                    name="沪深300",
                )
            ]
        )
        figure.update_layout(
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(8,20,34,.65)",
            height=520,
            margin=dict(l=20, r=20, t=55, b=20),
            title="沪深300 OHLC 历史行情",
            xaxis_rangeslider_visible=False,
            hovermode="x unified",
        )
        caption = (
            f"数据区间 {dataset['date'].min().date()} 至 {dataset['date'].max().date()}；"
            "价格为目标指数行情，不做交易执行假设。"
        )
    else:
        # Public repositories exclude provider CSV files. Persisted forecasts
        # still contain a close-price column, so the dashboard remains viewable.
        figure = go.Figure(
            go.Scatter(
                x=selected["forecast_origin"],
                y=selected["close"],
                mode="lines",
                name="沪深300收盘价",
                line=dict(color="#62d7ff", width=1.3),
            )
        )
        figure.update_layout(
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(8,20,34,.65)",
            height=520,
            margin=dict(l=20, r=20, t=55, b=20),
            title="沪深300收盘价（样本外结果归档）",
            hovermode="x unified",
        )
        caption = (
            "公开仓库不包含供应商原始 CSV；此图使用已归档的样本外收盘价。"
            "运行 run_research.cmd 可下载数据并恢复完整 OHLC 图。"
        )
    st.plotly_chart(figure, width="stretch", alt="沪深300指数价格历史图")
    st.caption(caption)

with volatility_tab:
    figure = make_subplots(specs=[[{"secondary_y": True}]])
    figure.add_trace(
        go.Scatter(
            x=selected["forecast_origin"],
            y=selected["actual_annualized_vol"] * 100.0,
            name="次日已实现代理",
            line=dict(color="#607d96", width=1),
        ),
        secondary_y=False,
    )
    colors = {"HAR-RV": "#f0b35a", "XGBoost": "#9d8df1", "动态组合": "#20c6b7"}
    for label in selected_models:
        figure.add_trace(
            go.Scatter(
                x=selected["forecast_origin"],
                y=selected[model_labels[label]] * 100.0,
                name=label,
                line=dict(color=colors[label], width=1.2),
            ),
            secondary_y=False,
        )
    figure.add_trace(
        go.Scatter(
            x=selected["forecast_origin"],
            y=selected["close"],
            name="指数收盘价",
            line=dict(color="#62d7ff", width=1),
            opacity=0.48,
        ),
        secondary_y=True,
    )
    figure.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(8,20,34,.65)",
        height=560,
        title="历史波动率与逐日样本外预测",
        hovermode="x unified",
        legend=dict(orientation="h", y=1.08),
        margin=dict(l=20, r=20, t=70, b=20),
    )
    figure.update_yaxes(title_text="年化波动率 / %", secondary_y=False)
    figure.update_yaxes(title_text="指数点位", secondary_y=True, showgrid=False)
    st.plotly_chart(
        figure,
        width="stretch",
        alt="历史波动率与HAR、XGBoost和动态组合的逐日预测对比图",
    )
    latest_cols = st.columns(3)
    for column, key, label in zip(
        latest_cols,
        ("annualized_vol_har", "annualized_vol_xgb", "annualized_vol_dynamic"),
        ("HAR-RV", "XGBoost", "动态组合"),
    ):
        with column:
            st.markdown(
                card(label, f"{latest[key] * 100:.2f}%", "下一交易日年化预测"),
                unsafe_allow_html=True,
            )

with ensemble_tab:
    figure = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.08)
    figure.add_trace(
        go.Scatter(
            x=selected["forecast_origin"],
            y=selected["weight_xgb"],
            name="XGBoost 权重",
            line=dict(color="#62d7ff", width=1.3),
            fill="tozeroy",
            fillcolor="rgba(98,215,255,.08)",
        ),
        row=1,
        col=1,
    )
    figure.add_trace(
        go.Scatter(
            x=selected["forecast_origin"],
            y=selected["regime"],
            name="市场状态",
            mode="lines",
            line=dict(color="#f0b35a", width=1),
        ),
        row=2,
        col=1,
    )
    figure.update_yaxes(title_text="权重", range=[0, 1], row=1, col=1)
    figure.update_yaxes(
        title_text="状态 0/1/2",
        tickmode="array",
        tickvals=[0, 1, 2],
        ticktext=["低", "正常", "高"],
        row=2,
        col=1,
    )
    figure.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(8,20,34,.65)",
        height=590,
        title="动态模型权重与波动率状态",
        hovermode="x unified",
        margin=dict(l=20, r=20, t=70, b=20),
    )
    st.plotly_chart(
        figure,
        width="stretch",
        alt="动态XGBoost权重与低、正常、高波动市场状态的时间序列图",
    )
    state_counts = (
        selected["regime_label"].value_counts().rename_axis("状态").reset_index(name="天数")
    )
    left, right = st.columns([2, 3])
    with left:
        st.dataframe(
            state_counts,
            width="stretch",
            hide_index=True,
            alt="不同市场状态的样本天数统计表",
        )
    with right:
        st.caption(
            "状态阈值来自先前 252 个交易日滚动分位数，并要求连续确认以减少临界值附近"
            "的频繁切换。权重仅使用当前预测之前已经实现的样本外误差。"
        )

with model_tab:
    display_metrics = metrics.rename(
        columns={
            "model": "模型",
            "n": "样本数",
            "mae": "MAE",
            "rmse": "RMSE",
            "qlike": "QLIKE",
            "r2_ots": "R² OOS",
        }
    ).copy()
    model_name_map = {
        "pred_har": "HAR-RV",
        "pred_xgb": "XGBoost",
        "pred_static": "固定组合",
        "pred_dynamic": "动态组合",
    }
    if "模型" in display_metrics:
        display_metrics["模型"] = display_metrics["模型"].map(model_name_map).fillna(
            display_metrics["模型"]
        )
    figure = go.Figure(
        go.Bar(
            x=display_metrics["模型"],
            y=display_metrics["QLIKE"],
            marker_color=["#f0b35a", "#9d8df1", "#607d96", "#20c6b7"],
            text=display_metrics["QLIKE"].map(lambda value: f"{value:.5f}"),
            textposition="outside",
        )
    )
    figure.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(8,20,34,.65)",
        height=390,
        title="整体样本外 QLIKE（越低越好）",
        margin=dict(l=20, r=20, t=60, b=20),
    )
    st.plotly_chart(
        figure,
        width="stretch",
        alt="HAR、XGBoost、固定组合和动态组合的整体样本外QLIKE柱状图",
    )
    st.dataframe(
        display_metrics.style.format(
            {"MAE": "{:.3e}", "RMSE": "{:.3e}", "QLIKE": "{:.5f}", "R² OOS": "{:.4f}"}
        ),
        width="stretch",
        hide_index=True,
        alt="整体样本外模型误差比较表",
    )
    if dm_path.exists():
        st.markdown("#### Diebold–Mariano：相对固定组合")
        dm = load_csv(str(dm_path))
        st.dataframe(
            dm.style.format(
                {"mean_loss_diff": "{:.5f}", "dm_stat": "{:.3f}", "dm_pvalue": "{:.4f}"}
            ),
            width="stretch",
            hide_index=True,
            alt="动态组合相对固定组合的Diebold-Mariano检验结果表",
        )
    if mcs_path.exists():
        st.markdown("#### 近似模型置信集")
        mcs = load_csv(str(mcs_path))
        st.dataframe(
            mcs,
            width="stretch",
            hide_index=True,
            alt="近似模型置信集检验结果表",
        )
        st.caption("Block bootstrap 操作化实现；与 DM 和效应大小联合解释。")

with event_tab:
    event_specs = [
        ("2015 年异常波动", "2015-06-01", "2015-09-30"),
        ("2020 年疫情冲击", "2020-01-20", "2020-04-30"),
    ]
    for title, start, end in event_specs:
        subset = forecasts.loc[
            forecasts["forecast_origin"].between(pd.Timestamp(start), pd.Timestamp(end))
        ]
        if subset.empty:
            continue
        figure = go.Figure()
        figure.add_trace(
            go.Scatter(
                x=subset["forecast_origin"],
                y=subset["actual_annualized_vol"] * 100,
                mode="lines",
                name="已实现代理",
                line=dict(color="#607d96", width=1.2),
            )
        )
        figure.add_trace(
            go.Scatter(
                x=subset["forecast_origin"],
                y=subset["annualized_vol_dynamic"] * 100,
                mode="lines",
                name="动态预测",
                line=dict(color="#20c6b7", width=1.4),
            )
        )
        figure.update_layout(
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(8,20,34,.65)",
            height=360,
            title=f"{title} · 事后诊断区间",
            yaxis_title="年化波动率 / %",
            margin=dict(l=20, r=20, t=60, b=20),
        )
        st.plotly_chart(
            figure,
            width="stretch",
            alt=f"{title}期间的已实现波动率与动态组合预测对比图",
        )
    st.caption("事件日期只用于结果切片，实时预测模型和权重从未读取这些标签。")

with method_tab:
    st.markdown(
        """
        **研究口径**

        - 目标：下一交易日 Garman–Klass 日度方差代理。
        - 验证：严格滚动样本外，模型每 22 个交易日重训。
        - 特征信息集：预测起点收盘及以前。
        - 动态权重：最近 252 个样本外损失，按当前市场状态筛选后反比例赋权。
        - 高频限制：公开 5 分钟接口仅有近期滚动历史，不能覆盖 2015/2020。
        - 统计检验：Newey–West Diebold–Mariano 与 block-bootstrap 近似 MCS。

        详细假设、参数和限制见项目 `reports/research_report.md` 与
        `reports/data_source.md`。
        """
    )
    st.download_button(
        "下载样本外预测 CSV",
        data=forecasts.to_csv(index=False).encode("utf-8-sig"),
        file_name="csi300_volatility_forecasts.csv",
        mime="text/csv",
    )
    st.caption(
        "本项目只研究波动率预测，不预测指数涨跌方向，也不构成投资建议。"
    )

