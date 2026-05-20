#!/bin/bash
#===============================================================================
# 增量部署脚本 - 用于更新 Oracle 服务器上的 ALPHA-Bot
#===============================================================================

set -e

APP_DIR="/opt/alpha-bot"
VENV_DIR="${APP_DIR}/venv"
LOG_DIR="/var/log/alpha-bot"
WEBHOOK_ENV_FILE="/etc/alpha-bot/env.conf"

# 颜色输出
GREEN='\033[0;32m'
NC='\033[0m'
log_info() { echo -e "${GREEN}[INFO]${NC} $1"; }

echo "========================================"
echo "  ALPHA-Bot 增量部署"
echo "========================================"

# 备份当前状态
log_info "备份当前状态..."
if [[ -f "${APP_DIR}/daily_briefing.py" ]]; then
    BACKUP_DIR="/tmp/alpha-bot-backup-$(date +%Y%m%d_%H%M%S)"
    mkdir -p "${BACKUP_DIR}"
    cp -r "${APP_DIR}"/* "${BACKUP_DIR}/"
    log_info "已备份到 ${BACKUP_DIR}"
fi

# 更新代码
log_info "更新代码..."
cd "${APP_DIR}"
if [[ -d ".git" ]]; then
    git pull origin main
    log_info "代码已更新"
else
    log_info "非 git 仓库，请手动更新代码"
fi

# 安装依赖
log_info "安装/更新依赖..."
source "${VENV_DIR}/bin/activate"
pip install --upgrade pandas requests akshare yfinance beautifulsoup4 lxml

# 测试运行
log_info "测试运行..."
timeout 120 python daily_briefing.py

if [[ $? -eq 0 ]]; then
    log_info "测试通过!"
else
    log_info "测试失败，请检查日志"
fi

log_info "部署完成!"
