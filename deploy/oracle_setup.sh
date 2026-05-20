#!/bin/bash
#===============================================================================
# Oracle Cloud 甲骨文服务器部署脚本
# 用于初始化服务器环境和部署 ALPHA-Bot 每日简报系统
#===============================================================================

set -e

# 配置变量
APP_NAME="alpha-bot"
APP_DIR="/opt/${APP_NAME}"
VENV_DIR="${APP_DIR}/venv"
LOG_DIR="/var/log/${APP_NAME}"
WEBHOOK_ENV_FILE="/etc/${APP_NAME}/env.conf"

# 颜色输出
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

log_info() { echo -e "${GREEN}[INFO]${NC} $1"; }
log_warn() { echo -e "${YELLOW}[WARN]${NC} $1"; }
log_error() { echo -e "${RED}[ERROR]${NC} $1"; }

# 检查 root 权限
check_root() {
    if [[ $EUID -ne 0 ]]; then
        log_error "请使用 root 权限运行此脚本"
        exit 1
    fi
}

# 系统环境检查
check_system() {
    log_info "检查系统环境..."

    if [[ -f /etc/oracle-release ]]; then
        log_info "检测到 Oracle Linux 系统"
        OS="oracle"
    elif [[ -f /etc/lsb-release ]]; then
        . /etc/lsb-release
        if [[ "$DISTRIB_ID" == "Ubuntu" ]]; then
            OS="ubuntu"
            log_info "检测到 Ubuntu 系统"
        fi
    fi

    log_info "系统环境检查完成"
}

# 安装 Python 环境
install_python() {
    log_info "安装 Python 环境..."

    if command -v python3 &> /dev/null; then
        PYTHON_VERSION=$(python3 --version | cut -d' ' -f2 | cut -d'.' -f1,2)
        log_info "Python ${PYTHON_VERSION} 已安装"
    else
        log_warn "正在安装 Python..."
        if [[ "$OS" == "ubuntu" ]]; then
            apt-get update
            apt-get install -y python3 python3-pip python3-venv
        else
            yum install -y python3 python3-pip python3-venv
        fi
    fi

    # 安装系统依赖
    log_info "安装系统依赖..."
    if [[ "$OS" == "ubuntu" ]]; then
        apt-get install -y python3-dev gcc git curl wget
    else
        yum install -y python3-devel gcc git curl wget
    fi
}

# 创建应用目录结构
create_dirs() {
    log_info "创建目录结构..."

    mkdir -p "${APP_DIR}"
    mkdir -p "${LOG_DIR}"
    mkdir -p "$(dirname ${WEBHOOK_ENV_FILE})"

    chown -R root:root "${APP_DIR}"
    chmod 755 "${APP_DIR}"
}

# 克隆或同步代码
deploy_code() {
    log_info "部署代码..."

    if [[ -d "${APP_DIR}/.git" ]]; then
        log_info "代码已存在，执行更新..."
        cd "${APP_DIR}"
        git pull origin main
    else
        log_info "请设置代码仓库地址，或手动复制代码到 ${APP_DIR}"
        log_warn "示例: git clone https://github.com/YOUR_USERNAME/alpha-bot.git ${APP_DIR}"
    fi
}

# 创建 Python 虚拟环境
create_venv() {
    log_info "创建 Python 虚拟环境..."

    if [[ ! -d "${VENV_DIR}" ]]; then
        python3 -m venv "${VENV_DIR}"
    fi

    source "${VENV_DIR}/bin/activate"

    log_info "升级 pip..."
    pip install --upgrade pip

    log_info "安装 Python 依赖..."
    pip install pandas>=2.1.0 requests>=2.31.0 akshare>=1.10.0 yfinance>=0.2.30 beautifulsoup4>=4.12.0 lxml>=4.9.0 numpy>=1.26.0
}

# 配置钉钉 Webhook
config_webhook() {
    log_info "配置钉钉 Webhook..."

    if [[ ! -f "${WEBHOOK_ENV_FILE}" ]]; then
        cat > "${WEBHOOK_ENV_FILE}" << 'EOF'
# ALPHA-Bot 钉钉 Webhook 配置
# 请替换为你的钉钉机器人 Webhook 地址
DINGTALK_WEBHOOK="https://oapi.dingtalk.com/robot/send?access_token=YOUR_TOKEN_HERE"
EOF
        chmod 600 "${WEBHOOK_ENV_FILE}"
        log_warn "请编辑 ${WEBHOOK_ENV_FILE} 填入你的钉钉 Webhook 地址"
    else
        log_info "Webhook 配置已存在"
    fi
}

# 设置定时任务
setup_cron() {
    log_info "配置定时任务..."

    # 读取现有 crontab
    crontab -l > /tmp/current_crontab 2>/dev/null || true

    # 添加每日简报定时任务（工作日 09:30）
    CRON_JOB="30 9 * * 1-5 source ${VENV_DIR}/bin/activate && ${APP_DIR}/run_briefing.py >> ${LOG_DIR}/daily_briefing.log 2>&1"

    # 检查是否已存在
    if grep -q "${APP_DIR}/run_briefing.py" /tmp/current_crontab 2>/dev/null; then
        log_info "定时任务已配置"
    else
        echo "${CRON_JOB}" >> /tmp/current_crontab
        crontab /tmp/current_crontab
        log_info "定时任务已添加：每个工作日 09:30 执行"
    fi

    rm /tmp/current_crontab
}

# 创建日志轮转配置
setup_logrotate() {
    log_info "配置日志轮转..."

    cat > /etc/logrotate.d/${APP_NAME} << EOF
${LOG_DIR}/*.log {
    daily
    missingok
    rotate 14
    compress
    delaycompress
    notifempty
    create 0644 root root
}
EOF
}

# 主函数
main() {
    echo "========================================"
    echo "  ALPHA-Bot Oracle 部署脚本"
    echo "========================================"

    check_root
    check_system
    install_python
    create_dirs
    deploy_code
    create_venv
    config_webhook
    setup_cron
    setup_logrotate

    echo ""
    echo "========================================"
    log_info "部署完成!"
    echo "========================================"
    echo ""
    echo "后续步骤:"
    echo "1. 编辑 ${WEBHOOK_ENV_FILE} 填入钉钉 Webhook"
    echo "2. 测试运行: source ${VENV_DIR}/bin/activate && python ${APP_DIR}/run_briefing.py"
    echo "3. 查看日志: tail -f ${LOG_DIR}/daily_briefing.log"
    echo "4. 查看定时任务: crontab -l"
    echo ""
}

main "$@"
