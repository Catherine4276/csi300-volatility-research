# 分享与启动说明

## 在你自己的电脑上启动

当前项目的 `.venv` 已经安装好，直接双击根目录：

```text
start_dashboard.cmd
```

或运行：

```powershell
.\start_dashboard.cmd
```

浏览器打开 `http://localhost:8501`。关闭运行窗口即可停止服务。

## 发给别人

发送 `Stock_Volatility_Research_share.zip`，不要单独复制 `.venv`。虚拟环境与本机
操作系统、Python 安装路径绑定，换电脑后通常不能使用。

接收者步骤：

1. 安装 Python 3.11 或更高版本，并确保勾选添加到 PATH；
2. 解压整个压缩包；
3. 双击 `start_dashboard.cmd`；
4. 首次启动会自动创建 `.venv`、安装锁定依赖并打开仪表盘。

如果只想查看项目，已有 `data`、`reports`、`experiments` 和 `models` 文件已包含在
压缩包中，不需要先重新下载数据。

## 更新最新数据并重跑研究

双击：

```text
run_research.cmd
```

它会重新下载公开行情、执行全部测试、跑滚动样本外回测、生成报告并做最终验证。
该步骤需要网络并可能运行数分钟。

## 更适合同事协作的方式

压缩包适合一次性分享。如果需要多人持续修改：

1. 创建 Git 仓库；
2. 将项目提交到 GitHub、GitLab 或其他代码平台；
3. 同事克隆仓库后运行 `start_dashboard.cmd`。

`.gitignore` 已排除 `.venv`、缓存和临时文件。

## 注意

- `localhost` 只指向接收者自己的电脑，不能把本地地址当成公共网站分享。
- 若需要公网链接，应部署到 Streamlit Community Cloud、服务器或容器平台。
- 高风险预警是研究信号，不是投资建议。

