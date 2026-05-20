#!/bin/bash
#===============================================================================
# 健康检查脚本 - 部署后验证
#===============================================================================

set -e

VENV_DIR="/opt/alpha-bot/venv"
LOG_DIR="/var/log/alpha-bot"
WEBHOOK_ENV_FILE="/etc/alpha-bot/env.conf"

GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m'

pass() { echo -e "${GREEN}✓${NC} $1"; }
fail() { echo -e "${RED}✗${NC} $1"; }
warn() { echo -e "${YELLOW}⚠${NC} $1"; }

echo "========================================"
echo "  ALPHA-Bot 健康检查"
echo "========================================"

# 1. 检查 Python 环境
echo ""
echo "[1/7] 检查 Python 环境..."
if command -v python3 &> /dev/null; then
    pass "Python3 已安装: $(python3 --version)"
else
    fail "Python3 未安装"
fi

# 2. 检查虚拟环境
echo ""
echo "[2/7] 检查虚拟环境..."
if [[ -d "${VENV_DIR}" ]]; then
    pass "虚拟环境存在: ${VENV_DIR}"
else
    fail "虚拟环境不存在"
fi

# 3. 检查依赖包
echo ""
echo "[3/7] 检查依赖包..."
source "${VENV_DIR}/bin/activate"
REQUIRED_PKGS="pandas requests akshare yfinance"
for pkg in $REQUIRED_PKGS; do
    if pip show "$pkg" &> /dev/null; then
        pass "$pkg 已安装"
    else
        fail "$pkg 未安装"
    fi
done

# 4. 检查代码文件
echo ""
echo "[4/7] 检查代码文件..."
if [[ -f "/opt/alpha-bot/daily_briefing.py" ]]; then
    pass "daily_briefing.py 存在"
else
    fail "daily_briefing.py 不存在"
fi

# 5. 检查钉钉配置
echo ""
echo "[5/7] 检查钉钉 Webhook..."
if [[ -f "${WEBHOOK_ENV_FILE}" ]]; then
    if grep -q "YOUR_TOKEN" "${WEBHOOK_ENV_FILE}" 2>/dev/null; then
        warn "钉钉 Webhook 未配置（使用默认值）"
    else
        pass "钉钉 Webhook 已配置"
    fi
else
    warn "钉钉配置文件不存在"
fi

# 6. 检查定时任务
echo ""
echo "[6/7] 检查定时任务..."
if crontab -l 2>/dev/null | grep -q "alpha-bot"; then
    pass "定时任务已配置"
    echo "    当前 crontab:"
    crontab -l | grep alpha-bot | sed 's/^/    /'
else
    warn "定时任务未配置"
fi

# 7. 检查磁盘空间
echo ""
echo "[7/7] 检查磁盘空间..."
DISK_USAGE=$(df -h / | tail -1 | awk '{print $5}' | sed 's/%//')
if [[ $DISK_USAGE -lt 80 ]]; then
    pass "磁盘空间充足 (${DISK_USAGE}%)"
else
    warn "磁盘空间使用较高 (${DISK_USAGE}%)"
fi

echo ""
echo "========================================"
echo "  检查完成"
echo "========================================"
