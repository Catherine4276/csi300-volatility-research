# 数据来源与处理说明

## 主数据

- 来源：东方财富公开历史行情接口
  `https://push2his.eastmoney.com/api/qt/stock/kline/get`
- 标的：沪深300指数（secid = `1.000300`）
- 下载方式：项目脚本 `scripts/prepare_data.py`
- 主要字段：日期、开盘、收盘、最高、最低、成交量、成交额等
- 复权：指数价格，`fqt=0`
- 原始文件及校验和：`data/raw/` 下的 CSV 与 manifest JSON

## 5 分钟数据

- 接口同源，`klt=5`
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
