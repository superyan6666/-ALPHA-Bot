#!/usr/bin/env python3
"""
GitHub Actions 启动脚本

确保每日简报在 GitHub Actions 环境中正常运行
"""
import os
import sys
import logging
from datetime import datetime

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
log = logging.getLogger(__name__)

def main():
    log.info("🚀 启动每日投研简报")
    
    os.environ['TZ'] = 'Asia/Shanghai'
    
    try:
        from daily_briefing import main as briefing_main
        briefing_main()
    except ImportError:
        log.error("❌ 无法导入 daily_briefing 模块")
        sys.exit(1)
    except Exception as e:
        log.error(f"❌ 执行失败: {e}", exc_info=True)
        sys.exit(1)

if __name__ == '__main__':
    main()

