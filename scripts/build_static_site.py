"""Build a dependency-free public dashboard for GitHub Pages."""

from __future__ import annotations

import html
import json
import shutil
from pathlib import Path

import pandas as pd
import plotly
import plotly.graph_objects as go
import plotly.io as pio
from plotly.subplots import make_subplots


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DOCS_ROOT = PROJECT_ROOT / "docs"
ASSETS_ROOT = DOCS_ROOT / "assets"
REPOSITORY_URL = "https://github.com/Catherine4276/csi300-volatility-research"
COLORS = {
    "actual": "#8da9c4",
    "har": "#f0b35a",
    "xgb": "#9d8df1",
    "static": "#8b7bd8",
    "dynamic": "#20c6b7",
    "close": "#62d7ff",
}


def _read_csv(relative_path: str) -> pd.DataFrame:
    return pd.read_csv(PROJECT_ROOT / relative_path)


def _model_labels(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    if "model" in result:
        result["model"] = result["model"].map(
            {
                "pred_har": "HAR-RV",
                "pred_xgb": "XGBoost",
                "pred_static": "固定组合",
                "pred_dynamic": "动态组合",
                "HAR-RV": "HAR-RV",
                "XGBoost": "XGBoost",
                "Static Ensemble": "固定组合",
                "Dynamic Ensemble": "动态组合",
            }
        ).fillna(result["model"])
    return result


def _table(
    frame: pd.DataFrame,
    columns: dict[str, str],
    formatters: dict[str, str] | None = None,
) -> str:
    formatters = formatters or {}
    view = frame.loc[:, list(columns)].rename(columns=columns).copy()
    for column, pattern in formatters.items():
        if column in view:
            view[column] = view[column].map(
                lambda value: ""
                if pd.isna(value)
                else pattern.format(value)
            )
    return view.to_html(
        index=False,
        border=0,
        classes="data-table",
        escape=True,
    )


def _build_metrics() -> str:
    metrics = _model_labels(_read_csv("experiments/experiment_1_overall.csv"))
    return _table(
        metrics,
        {
            "model": "模型",
            "n": "样本数",
            "mae": "MAE",
            "rmse": "RMSE",
            "qlike": "QLIKE",
            "r2_ots": "R² OOS",
        },
        {
            "MAE": "{:.3e}",
            "RMSE": "{:.3e}",
            "QLIKE": "{:.5f}",
            "R² OOS": "{:.4f}",
        },
    )


def _build_regime() -> str:
    summary = _read_csv("reports/regime_summary.csv")
    return _table(
        summary,
        {
            "regime_label": "市场状态",
            "days": "交易日数",
            "start_date": "首次出现",
            "end_date": "最近出现",
            "mean_rv_w": "平均5日RV",
            "transitions": "该状态进入次数",
        },
        {"平均5日RV": "{:.6e}"},
    )


def _build_tests() -> str:
    dm = _model_labels(_read_csv("experiments/dm_vs_Static Ensemble.csv"))
    mcs = _read_csv("experiments/model_confidence_set.csv")
    return (
        "<h3>Diebold–Mariano：相对固定组合</h3>"
        + _table(
            dm,
            {
                "model": "模型",
                "benchmark": "基准",
                "mean_loss_diff": "平均损失差",
                "dm_stat": "DM 统计量",
                "dm_pvalue": "p 值",
                "n": "样本数",
            },
            {
                "平均损失差": "{:.5f}",
                "DM 统计量": "{:.3f}",
                "p 值": "{:.4g}",
            },
        )
        + "<h3>近似 Model Confidence Set</h3>"
        + _table(
            mcs,
            {
                "model": "模型",
                "mean_loss": "平均 QLIKE",
                "p_value": "p 值",
                "included": "保留",
                "elimination_rank": "淘汰顺序",
            },
            {"平均 QLIKE": "{:.5f}", "p 值": "{:.4g}"},
        )
    )


def _build_events() -> str:
    sections: list[str] = []
    for path, title in (
        ("experiments/experiment_3_2015.csv", "2015 年异常波动"),
        ("experiments/experiment_4_2020.csv", "2020 年疫情冲击"),
    ):
        event = _model_labels(_read_csv(path))
        sections.append(
            f"<h3>{title}</h3>"
            + _table(
                event,
                {
                    "model": "模型",
                    "n": "样本数",
                    "mae": "MAE",
                    "rmse": "RMSE",
                    "qlike": "QLIKE",
                    "r2_ots": "R² OOS",
                },
                {
                    "MAE": "{:.3e}",
                    "RMSE": "{:.3e}",
                    "QLIKE": "{:.5f}",
                    "R² OOS": "{:.4f}",
                },
            )
        )
    return "".join(sections)


def _build_sensitivity() -> str:
    weights = _read_csv("experiments/experiment_6_weight_sensitivity.csv")
    thresholds = _read_csv("experiments/experiment_7_threshold_sensitivity.csv")
    return (
        "<h3>固定权重敏感性</h3>"
        + _table(
            weights,
            {
                "weight_xgb": "XGBoost 权重",
                "qlike": "QLIKE",
                "rmse": "RMSE",
                "r2_ots": "R² OOS",
            },
            {"XGBoost 权重": "{:.2f}", "QLIKE": "{:.5f}", "RMSE": "{:.3e}", "R² OOS": "{:.4f}"},
        )
        + "<h3>市场状态阈值敏感性</h3>"
        + _table(
            thresholds,
            {
                "low_quantile": "低状态分位",
                "high_quantile": "高状态分位",
                "high_regime_share": "高状态占比",
                "regime_switches": "切换次数",
                "qlike": "QLIKE",
                "r2_ots": "R² OOS",
            },
            {
                "低状态分位": "{:.2f}",
                "高状态分位": "{:.2f}",
                "高状态占比": "{:.2%}",
                "QLIKE": "{:.5f}",
                "R² OOS": "{:.4f}",
            },
        )
    )


def _chart_html(figure: go.Figure, div_id: str) -> str:
    """Render a self-contained chart fragment with local Plotly assets."""

    figure.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(8,20,34,.72)",
        font={"family": "Microsoft YaHei, Noto Sans SC, sans-serif", "color": "#dbe8f5"},
        margin={"l": 56, "r": 28, "t": 62, "b": 48},
        legend={"orientation": "h", "y": 1.08, "x": 0},
        hovermode="x unified",
        hoverlabel={
            "bgcolor": "#0b1d31",
            "bordercolor": "#315a78",
            "font": {"color": "#e8f5ff", "size": 12},
        },
        dragmode="zoom",
    )
    figure.update_xaxes(
        gridcolor="rgba(75,115,145,.18)",
        showspikes=True,
        hoverformat="%Y-%m-%d",
    )
    figure.update_yaxes(gridcolor="rgba(75,115,145,.18)")
    return pio.to_html(
        figure,
        include_plotlyjs=False,
        full_html=False,
        config={
            "displaylogo": False,
            "responsive": True,
            "scrollZoom": True,
            "modeBarButtonsToRemove": ["lasso2d", "select2d"],
        },
        default_width="100%",
        default_height="560px",
        div_id=div_id,
    )


def _build_interactive_charts() -> tuple[str, str, str]:
    forecasts = pd.read_csv(
        PROJECT_ROOT / "reports" / "forecasts.csv",
        parse_dates=["forecast_origin", "target_date"],
    )
    x = forecasts["forecast_origin"].dt.strftime("%Y-%m-%d")

    forecast_figure = make_subplots(specs=[[{"secondary_y": True}]])
    series = (
        ("actual_annualized_vol", "次日已实现代理", COLORS["actual"]),
        ("annualized_vol_har", "HAR-RV 预测", COLORS["har"]),
        ("annualized_vol_xgb", "XGBoost 预测", COLORS["xgb"]),
        ("annualized_vol_dynamic", "动态组合预测", COLORS["dynamic"]),
    )
    for column, label, color in series:
        forecast_figure.add_trace(
            go.Scattergl(
                x=x,
                y=forecasts[column] * 100.0,
                mode="lines",
                name=label,
                line={"color": color, "width": 1.2},
                hovertemplate=f"{label}: %{{y:.2f}}%<extra></extra>",
            ),
            secondary_y=False,
        )
    forecast_figure.add_trace(
        go.Scattergl(
            x=x,
            y=forecasts["close"],
            mode="lines",
            name="指数收盘价",
            line={"color": COLORS["close"], "width": 1, "dash": "dot"},
            opacity=0.6,
            hovertemplate="指数收盘价: %{y:.2f}<extra></extra>",
        ),
        secondary_y=True,
    )
    forecast_figure.update_layout(title="逐日样本外预测与次日已实现波动率")
    forecast_figure.update_yaxes(title_text="年化波动率 / %", secondary_y=False)
    forecast_figure.update_yaxes(
        title_text="指数收盘价",
        secondary_y=True,
        showgrid=False,
    )
    forecast_html = _chart_html(forecast_figure, "interactive-forecast-chart")

    cumulative_figure = go.Figure()
    for column, label, color in (
        ("qlike_har", "HAR-RV", COLORS["har"]),
        ("qlike_xgb", "XGBoost", COLORS["xgb"]),
        ("qlike_static", "固定组合", COLORS["static"]),
        ("qlike_dynamic", "动态组合", COLORS["dynamic"]),
    ):
        cumulative_figure.add_trace(
            go.Scattergl(
                x=x,
                y=forecasts[column].cumsum(),
                mode="lines",
                name=label,
                line={"color": color, "width": 1.5},
                hovertemplate=f"{label} 累计 QLIKE: %{{y:.2f}}<extra></extra>",
            )
        )
    cumulative_figure.update_layout(title="累计样本外 QLIKE（越低越好）")
    cumulative_figure.update_yaxes(title_text="累计 QLIKE")
    cumulative_html = _chart_html(cumulative_figure, "interactive-qlike-chart")

    weight_figure = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.08,
        row_heights=[0.68, 0.32],
    )
    weight_figure.add_trace(
        go.Scattergl(
            x=x,
            y=forecasts["weight_xgb"],
            mode="lines",
            name="XGBoost 权重",
            line={"color": COLORS["close"], "width": 1.4},
            fill="tozeroy",
            fillcolor="rgba(98,215,255,.08)",
            hovertemplate="XGBoost 权重: %{y:.1%}<extra></extra>",
        ),
        row=1,
        col=1,
    )
    weight_figure.add_trace(
        go.Scattergl(
            x=x,
            y=forecasts["regime"],
            mode="lines",
            name="市场状态",
            line={"color": COLORS["har"], "width": 1.2},
            customdata=forecasts["regime_label"],
            hovertemplate="市场状态: %{customdata}<extra></extra>",
        ),
        row=2,
        col=1,
    )
    weight_figure.update_layout(title="动态权重与市场状态")
    weight_figure.update_yaxes(title_text="XGBoost 权重", range=[0, 1], row=1, col=1)
    weight_figure.update_yaxes(
        title_text="市场状态",
        tickmode="array",
        tickvals=[0, 1, 2],
        ticktext=["低波动", "正常波动", "高波动"],
        row=2,
        col=1,
    )
    weight_html = _chart_html(weight_figure, "interactive-regime-chart")
    return cumulative_html, forecast_html, weight_html


def _card(label: str, value: str, note: str) -> str:
    return (
        '<article class="metric-card">'
        f'<div class="metric-label">{html.escape(label)}</div>'
        f'<div class="metric-value">{html.escape(value)}</div>'
        f'<div class="metric-note">{html.escape(note)}</div>'
        "</article>"
    )


def build() -> Path:
    ASSETS_ROOT.mkdir(parents=True, exist_ok=True)
    for chart_name in (
        "forecast_vs_realized.png",
        "cumulative_qlike.png",
        "dynamic_weight.png",
    ):
        shutil.copy2(
            PROJECT_ROOT / "reports" / "charts" / chart_name,
            ASSETS_ROOT / chart_name,
        )
    shutil.copy2(
        PROJECT_ROOT / "reports" / "forecasts.csv",
        ASSETS_ROOT / "forecasts.csv",
    )
    shutil.copy2(
        Path(plotly.__file__).resolve().parent / "package_data" / "plotly.min.js",
        ASSETS_ROOT / "plotly.min.js",
    )
    cumulative_chart, forecast_chart, weight_chart = _build_interactive_charts()

    latest = json.loads(
        (PROJECT_ROOT / "reports" / "latest_forecast.json").read_text(encoding="utf-8")
    )
    metrics = _read_csv("experiments/experiment_1_overall.csv")
    dynamic_qlike = metrics.loc[
        metrics["model"].eq("pred_dynamic"), "qlike"
    ].iloc[0]
    cards = "".join(
        (
            _card(
                "下一交易日动态波动率",
                f"{latest['annualized_vol_dynamic'] * 100:.2f}%",
                "年化口径",
            ),
            _card(
                "XGBoost 动态权重",
                f"{latest['weight_xgb']:.1%}",
                "由既往样本外损失推导",
            ),
            _card("当前市场状态", latest["regime_label"], latest["forecast_origin"][:10]),
            _card("动态组合整体 QLIKE", f"{dynamic_qlike:.5f}", "方差预测损失"),
        )
    )

    risk_class = "risk-high" if latest["high_risk_alert"] else "risk-normal"
    risk_text = (
        "高波动状态或滚动阈值已触发；该信号只用于研究，不是交易建议。"
        if latest["high_risk_alert"]
        else "当前未触发高波动预警；状态和误差仍需随每日数据复核。"
    )
    page = f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>沪深300 AI 波动率研究仪表盘</title>
  <meta name="description" content="沪深300 HAR-RV、XGBoost、市场状态与动态组合的严格样本外波动率研究。">
  <link rel="stylesheet" href="assets/style.css">
  <script src="assets/plotly.min.js"></script>
</head>
<body>
  <header class="hero">
    <div class="eyebrow">QUANTITATIVE RESEARCH / CSI 300</div>
    <h1>波动率预测失效诊断与动态组合防御</h1>
    <p>严格滚动样本外 · HAR-RV / XGBoost · 市场状态识别 · 统计检验</p>
    <div class="hero-meta">
      信息截至 {html.escape(latest["forecast_origin"][:10])} ·
      静态公开归档，不依赖本机服务 · 电脑悬停 / 手机点按查看具体数据
    </div>
  </header>

  <main>
    <section class="kpi-grid" aria-label="核心指标">{cards}</section>
    <div class="{risk_class}"><strong>风险状态：{html.escape(latest["regime_label"])}</strong>
      · {html.escape(risk_text)}
    </div>

    <div class="tabs">
      <input type="radio" name="tab" id="tab-overview" checked>
      <input type="radio" name="tab" id="tab-forecast">
      <input type="radio" name="tab" id="tab-regime">
      <input type="radio" name="tab" id="tab-tests">
      <input type="radio" name="tab" id="tab-events">
      <input type="radio" name="tab" id="tab-method">
      <nav class="tab-labels" aria-label="研究页面">
        <label for="tab-overview">总览</label>
        <label for="tab-forecast">预测诊断</label>
        <label for="tab-regime">状态与权重</label>
        <label for="tab-tests">统计检验</label>
        <label for="tab-events">事件与稳健性</label>
        <label for="tab-method">方法与数据</label>
      </nav>

      <section class="tab-panel panel-overview">
        <h2>整体样本外比较</h2>
        <p class="lead">所有指标由 2010-01-04 至 2026-11-09 的逐日样本外预测计算。</p>
        {_build_metrics()}
        <figure class="interactive-figure" role="img" aria-label="四种模型的交互式累计样本外QLIKE损失曲线">
          {cumulative_chart}
          <figcaption>累计 QLIKE 越低越好。将鼠标移到图表上可查看具体日期和四种模型的累计损失。</figcaption>
        </figure>
      </section>

      <section class="tab-panel panel-forecast">
        <h2>预测与实际波动</h2>
        <figure class="interactive-figure" role="img" aria-label="沪深300收盘价、实际波动率与三种模型预测的交互图">
          {forecast_chart}
          <figcaption>悬停日期可同时查看指数收盘价、次日已实现波动率和三种模型预测；手机端点按图表查看。</figcaption>
        </figure>
        <a class="button" href="assets/forecasts.csv" download>下载完整 4,090 条样本外预测 CSV</a>
      </section>

      <section class="tab-panel panel-regime">
        <h2>状态识别与动态权重</h2>
        <figure class="interactive-figure" role="img" aria-label="动态XGBoost权重与市场状态的交互图">
          {weight_chart}
          <figcaption>悬停日期可查看动态 XGBoost 权重和对应市场状态。阈值只使用先前 252 个交易日，连续 3 日确认后才切换。</figcaption>
        </figure>
        {_build_regime()}
      </section>

      <section class="tab-panel panel-tests">
        <h2>统计检验</h2>
        {_build_tests()}
      </section>

      <section class="tab-panel panel-events">
        <h2>事件复盘与敏感性</h2>
        {_build_events()}
        {_build_sensitivity()}
      </section>

      <section class="tab-panel panel-method">
        <h2>研究口径与限制</h2>
        <ul>
          <li>预测目标：下一交易日 Garman–Klass 日度方差代理。</li>
          <li>模型每 22 个交易日重训，XGBoost 每约 252 个样本外交易日重新调参。</li>
          <li>动态权重只读取已经实现的既往样本外损失。</li>
          <li>公开 5 分钟接口只有近期滚动历史，长期实验没有将其冒充高频 RV。</li>
          <li>波动率预测不是指数涨跌预测，也不构成投资建议。</li>
        </ul>
        <p>
          <a class="button secondary" href="{REPOSITORY_URL}">查看 GitHub 开源仓库</a>
          <a class="button secondary" href="{REPOSITORY_URL}/blob/main/reports/research_report.md">阅读完整研究报告</a>
          <a class="button secondary" href="{REPOSITORY_URL}/blob/main/data/README.md">数据来源与许可边界</a>
        </p>
      </section>
    </div>
  </main>

  <footer>
    沪深300 AI 波动率研究 · 静态公开仪表盘 · 研究信号不构成投资建议
  </footer>
  <script>
    (() => {{
      const resizeCharts = () => {{
        window.setTimeout(() => {{
          document.querySelectorAll(".js-plotly-plot").forEach((chart) => {{
            if (window.Plotly) Plotly.Plots.resize(chart);
          }});
        }}, 80);
      }};
      document.querySelectorAll(".tab-labels label").forEach((label) => {{
        label.addEventListener("click", resizeCharts);
      }});
      window.addEventListener("resize", resizeCharts);
    }})();
  </script>
</body>
</html>
"""
    index_path = DOCS_ROOT / "index.html"
    index_path.write_text(page, encoding="utf-8")
    return index_path


if __name__ == "__main__":
    print(build())

