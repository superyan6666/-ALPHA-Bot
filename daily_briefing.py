import os
import sys
import time
from datetime import datetime, timedelta
import pytz

TZ_BJS = pytz.timezone('Asia/Shanghai')

os.environ['RUN_MODE'] = 'normal'
os.environ['PUSH_EMPTY_RESULT'] = 'true'
os.environ['DATA_CACHE_MODE'] = 'online'
os.environ['LOG_LEVEL'] = 'INFO'

def _today_str():
    return datetime.now(TZ_BJS).strftime('%Y-%m-%d')

def _now_str():
    return datetime.now(TZ_BJS).strftime('%Y-%m-%d %H:%M:%S')

if __name__ == '__main__':
    print(f"[{_now_str()}] 🚀 启动每日投研简报引擎...")
    
    try:
        from main import get_signals, send_dingtalk, save_pushed_state, load_pushed_state
        
        print(f"[{_now_str()}] 📊 正在获取市场数据和生成信号...")
        sigs, watch, pushed, pool_size, m_msg, total_mkt = get_signals()
        
        print(f"[{_now_str()}] 📈 信号获取完成 - 核心: {len(sigs.get('Core', []))}, 卫星: {len(sigs.get('Satellite', []))}")
        
        if m_msg:
            print(f"[{_now_str()}] 📋 市场分析摘要:")
            lines = m_msg.split('\n')[:10]
            for line in lines:
                print(f"  {line[:80]}..." if len(line) > 80 else f"  {line}")
            if len(m_msg.split('\n')) > 10:
                print("  ...(更多内容)")
        
        print(f"[{_now_str()}] 📤 发送钉钉通知...")
        send_dingtalk(sigs, watch, pool_size, total_mkt, m_msg)
        
        if any(sigs.values()):
            print(f"[{_now_str()}] 💾 保存推送状态...")
            save_pushed_state(pushed)
        
        print(f"[{_now_str()}] ✅ 每日投研简报发送完成!")
        
    except Exception as e:
        print(f"[{_now_str()}] ❌ 执行失败: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)