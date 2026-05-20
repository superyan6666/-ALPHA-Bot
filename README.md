# 📊 每日投研简报系统

自动化生成每日市场投研简报，包含A股、港股、美股行情分析，自动发送到钉钉群。

## ✨ 功能特性

- 📈 **市场分析**：A股、港股、美股实时行情
- 💹 **宏观研判**：美元指数、铜金比、VIX、美债收益率
- 💰 **资金流向**：北向资金、南向资金监控
- 🔥 **热点板块**：行业轮动追踪
- 📰 **资讯精选**：AI解读重要新闻
- ⚠️ **风险预警**：异常信号监控
- 🤖 **自动推送**：每日定时发送到钉钉

## 🚀 快速开始

### 环境要求

- Python 3.11+
- 钉钉机器人 Webhook

### 安装依赖

```bash
pip install -r requirements.txt
```

### 本地运行

```bash
# 设置钉钉 Webhook 环境变量
export DINGTALK_WEBHOOK="https://oapi.dingtalk.com/robot/send?access_token=your_token"

# 运行简报
python daily_briefing.py
```

## 🔧 GitHub Actions 部署

### 1. 创建 GitHub 仓库

将代码推送到 GitHub 仓库。

### 2. 设置 Secrets

在 GitHub 仓库中添加以下 Secrets：

| Secret 名称 | 说明 |
|------------|------|
| `DINGTALK_WEBHOOK` | 钉钉机器人 Webhook 地址 |

### 3. 工作流配置

工作流文件位于 `.github/workflows/daily_briefing.yml`

**执行时间**：
- 每日 09:30 (北京时间) - 开盘前简报
- 每日 21:30 (北京时间) - 收盘后简报

### 4. 手动触发

在 GitHub Actions 页面可以手动触发工作流。

## 📁 项目结构

```
.
├── daily_briefing.py      # 主简报生成脚本
├── run_briefing.py        # GitHub Actions 启动脚本
├── main.py               # 数据获取核心模块
├── factors_config.py     # 因子配置
├── requirements.txt      # 依赖列表
├── .github/
│   └── workflows/
│       └── daily_briefing.yml  # GitHub Actions 配置
├── core/                 # 核心配置
├── data/                 # 数据模块
├── pipelines/            # 策略流水线
└── optimization/         # 优化模块
```

## 📊 简报内容

每日简报包含以下板块：

1. **🎯 核心结论** - 当日市场要点总结
2. **🌐 宏观环境** - 美元指数、铜金比、VIX等
3. **🇺🇸 美股技术面** - 标普500趋势分析
4. **🇭🇰 港股大盘** - 恒生指数 + 南向资金
5. **🇨🇳 A股技术面** - 沪深300 + 市场宽度
6. **💰 资金流向** - 北向/南向资金
7. **🔥 热点板块** - 行业轮动
8. **📰 资讯精选** - AI解读新闻
9. **⚠️ 风险关注** - 异常信号

## 🛠️ 配置说明

### 环境变量

| 变量名 | 说明 | 必需 |
|--------|------|------|
| `DINGTALK_WEBHOOK` | 钉钉机器人 Webhook | 是 |

### 定时任务配置

修改 `.github/workflows/daily_briefing.yml` 中的 cron 表达式：

```yaml
schedule:
  - cron: '30 1 * * *'  # UTC时间，对应北京时间 09:30
```

## 📝 License

MIT License

