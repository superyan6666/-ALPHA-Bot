import sys
import re

with open('factors_config.py', 'r', encoding='utf-8') as f:
    text = f.read()

text = text.replace("Factor(lambda d: d.get('rsi', 50) > 80, -10, f_risk, \"- 🌡️ **短期过热**：RSI偏高短线超买，操作需要进一步缩减仓位\"),", "Factor(lambda d: d.get('rsi', 50) > 80, -15, f_risk, \"- 🌡️ **短期过热**：RSI偏高短线超买，随时面临获利盘砸盘 (重度扣分)\"),")
text = text.replace("Factor(lambda d: d.get('consecutive_down', 0) >= 4, -15, f_risk, \"- 🔪 **飞刀预警**：近期连续阴线急跌，左侧接飞刀风险大 (重度扣分)\"),", "Factor(lambda d: d.get('consecutive_down', 0) >= 4, -25, f_risk, \"- 🔪 **飞刀预警**：近期连续阴线急跌，左侧接飞刀风险大 (重度扣分)\"),")

vpr = "Factor(lambda d: d.get('has_obv_break', False), 10, tw * f_mom, \"- 💸 **真金白银**：模型监控到真实的资金在创纪录净流入\", FactorGroup.VOL),\n        Factor(lambda d: 3.0 <= d.get('pct_chg', 0) <= 8.0 and d.get('vol_ratio', 1.0) > 1.2 and d.get('turnover', 0) < 20.0, 15, f_mom, \"- 🌊 **量价共振**：涨幅健康且温和放量，趋势刚启动的绝佳特征\", FactorGroup.MOM),"
text = text.replace("Factor(lambda d: d.get('has_obv_break', False), 10, tw * f_mom, \"- 💸 **真金白银**：模型监控到真实的资金在创纪录净流入\", FactorGroup.VOL),", vpr)

exh = "Factor(lambda d: d.get('dist_ma20', 0) > 25, -15, f_risk, \"- 🚫 **追高预警**：目前涨得太急离均线太远，随时面临暴跌回调\"),\n        Factor(lambda d: d.get('turnover', 0) > 25.0 and d.get('upper_shadow_pct', 0) > 10.0, -25, f_risk, \"- 🌋 **天量见天价**：极度放量且带有明显上影线，主力极可能在派发筹码，坚决回避！\"),"
text = text.replace("Factor(lambda d: d.get('dist_ma20', 0) > 25, -15, f_risk, \"- 🚫 **追高预警**：目前涨得太急离均线太远，随时面临暴跌回调\"),", exh)

with open('factors_config.py', 'w', encoding='utf-8') as f:
    f.write(text)

with open('main.py', 'r', encoding='utf-8') as f:
    main_text = f.read()

main_text = main_text.replace("atr_stop = data['close_val'] - 2.0 * data['atr_val']\n    stop = max(atr_stop, row[C.S_PRICE] * 0.88)", "atr_stop = data['close_val'] - 1.5 * data['atr_val']\n    stop = atr_stop")

with open('main.py', 'w', encoding='utf-8') as f:
    f.write(main_text)

print('Patched successfully!')
