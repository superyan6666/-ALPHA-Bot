#!/bin/bash
#===============================================================================
# 甲骨文服务器初始化脚本 - 只需执行一次
# 用于配置 ALPHA-Bot 运行环境和 SSH 访问
#===============================================================================

set -e

APP_DIR="/opt/alpha-bot"
VENV_DIR="${APP_DIR}/venv"
LOG_DIR="/var/log/alpha-bot"

GREEN='\033[0;32m'
NC='\033[0m'
log_info() { echo -e "${GREEN}[INFO]${NC} $1"; }

echo "========================================"
echo "  ALPHA-Bot 甲骨文服务器初始化"
echo "========================================"

# 检查 root 权限
if [[ $EUID -ne 0 ]]; then
   echo "请使用 root 权限运行: sudo su -"
   exit 1
fi

# 安装系统依赖
log_info "安装系统依赖..."
if command -v apt-get &> /dev/null; then
    apt-get update
    apt-get install -y python3 python3-pip python3-venv git curl rsync
else
    yum install -y python3 python3-pip python3-venv git curl rsync
fi

# 创建目录
log_info "创建目录..."
mkdir -p "${APP_DIR}"
mkdir -p "${LOG_DIR}"

# 创建 Python 虚拟环境
log_info "创建虚拟环境..."
if [[ ! -d "${VENV_DIR}" ]]; then
    python3 -m venv "${VENV_DIR}"
fi

# 安装依赖
log_info "安装 Python 依赖..."
source "${VENV_DIR}/bin/activate"
pip install --upgrade pip
pip install pandas requests akshare yfinance beautifulsoup4 lxml numpy

# 设置日志轮转
log_info "配置日志轮转..."
cat > /etc/logrotate.d/alpha-bot << 'EOF'
/var/log/alpha-bot/*.log {
    daily
    missingok
    rotate 14
    compress
    delaycompress
    notifempty
    create 0644 root root
}
EOF

# 创建启动脚本
log_info "创建启动脚本..."
cat > "${APP_DIR}/run.sh" << 'EOF'
#!/bin/bash
export TZ="Asia/Shanghai"
source /opt/alpha-bot/venv/bin/activate
cd /opt/alpha-bot
python daily_briefing.py
EOF
chmod +x "${APP_DIR}/run.sh"

log_info "初始化完成!"
echo ""
echo "下一步:"
echo "1. 将代码同步到 /opt/alpha-bot/ (通过 GitHub Actions 自动完成)"
echo "2. 在 GitHub 仓库设置 Secrets"
echo "3. 推送代码触发 Actions"
echo ""
