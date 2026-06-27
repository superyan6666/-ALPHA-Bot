import re

with open("main.py", "r", encoding="utf-8") as f:
    content = f.read()

# 1. Add imports at the top
import_str = """
import traceback
from ml_engine import XGBoostLTR
from feature_engine import build_ml_features
import os
"""
if "from ml_engine import XGBoostLTR" not in content:
    content = content.replace("import pandas as pd", "import pandas as pd\n" + import_str)

# 2. Replace get_signals loop
old_loop = """    try:
        for f in as_completed(futures, timeout=1200): 
            row = futures[f]
            try:
                hist = f.result()
                result = process_stock(row, hist, now, m_ok, idx_ret, hot_sectors_map)
                if result:
                    data, stop, risk = result
                    
                    score, level, reas = apply_scoring(data, now, m_regime, vol_surge, win_stats, is_fallback)
                    
                    if score >= 70: 
                        target1_price = calc_target_price(row[C.S_PRICE], stop, data)
                        
                        money_msg = format_money_risk_msg(row[C.S_PRICE], stop, target1_price)
                        tranche_msg = generate_tranche_plan(row[C.S_PRICE], score, m_ok, m_overheated)
                        plan_b_msg = generate_plan_b(row[C.S_PRICE], stop, data['ma20_val'])
                        hold_msg = generate_hold_period(data['adx'], data['price_pct'], data['has_chip_break'])
                        
                        confirmed_data.append(Signal(
                            code=row[C.S_CODE], name=row[C.S_NAME], price=row[C.S_PRICE],
                            pct_chg=f"{row[C.S_PCT]}%", score=score, level=level,
                            trigger_time=now.strftime('%H:%M'), reasons=reas,
                            stop_loss=round(stop, 2), target1=target1_price,
                            ma10=round(data['ma10_val'], 2),
                            money_risk_msg=money_msg, tranche_plan_msg=tranche_msg,
                            plan_b_msg=plan_b_msg, hold_period_msg=hold_msg
                        ))
                    elif score >= 60:  
                        watchlist_data.append((row[C.S_NAME], row[C.S_CODE], score, row[C.S_PRICE]))
                        
            except Exception as e:
                log.warning(f"⚠️ 计算个股 {row[C.S_CODE]} 时发生特征异常或被过滤: {e}")
                pass
    except FuturesTimeoutError:
        log.warning("⚠️ 后台运算达到极值，提前熔断保存已有成果。")
    finally:
        ex2.shutdown(wait=False, cancel_futures=True)"""

new_loop = """    all_hists = []
    stock_infos = {}
    try:
        for f in as_completed(futures, timeout=1200): 
            row = futures[f]
            try:
                hist = f.result()
                hist_ml = hist.copy()
                hist_ml.rename(columns={
                    C.H_DATE: 'date', C.H_OPEN: 'open', C.H_HIGH: 'high',
                    C.H_LOW: 'low', C.H_CLOSE: 'close', C.H_VOL: 'vol'
                }, inplace=True)
                hist_ml['code'] = row[C.S_CODE]
                all_hists.append(hist_ml)
                
                result = process_stock(row, hist, now, m_ok, idx_ret, hot_sectors_map)
                if result:
                    data, stop, risk = result
                    stock_infos[row[C.S_CODE]] = {
                        'row': row, 'data': data, 'stop': stop, 'risk': risk
                    }
            except Exception as e:
                log.warning(f"⚠️ 计算个股 {row[C.S_CODE]} 时发生异常: {e}")
                pass
    except FuturesTimeoutError:
        log.warning("⚠️ 后台运算达到极值，提前熔断保存已有成果。")
    finally:
        ex2.shutdown(wait=False, cancel_futures=True)

    if not all_hists:
        return [], [], pushed, len(pool), m_msg, len(df_clean)

    # ML Feature Engineering
    panel = pd.concat(all_hists, ignore_index=True)
    panel['date'] = pd.to_datetime(panel['date'])
    panel = panel.sort_values(['date', 'code'])
    
    try:
        panel = build_ml_features(panel)
        feature_cols = ['sm_corr', 'clv', 'volatility_5d', 'vol_ratio', 'alpha_reversal_5d', 'alpha_024_approx',
                        'market_ret_20d', 'market_ret_60d', 'market_vol_20d', 'cn_10y_trend']
        feature_success = True
    except Exception as e:
        log.error(f"🚨 ML Feature Computation Failed: {e}")
        log.error(traceback.format_exc())
        m_msg += "\\n\\n> ⚠️ **风控告警**：ML特征计算异常，今日信号基于中性基准 (0.5)！\\n\\n"
        feature_success = False

    # Extract today's cross section
    today_str = now.strftime('%Y-%m-%d')
    today_panel = panel[panel['date'] == pd.to_datetime(today_str)].copy()
    
    # Load Model
    model_path = '.quantbot_data/prod_xgb_model.json'
    if not os.path.exists(model_path):
        log.critical("🚨 致命错误：XGBoost模型文件缺失，系统终止！")
        raise FileNotFoundError(f"Missing model file: {model_path} (Rule B3.6 Crash)")
        
    ltr = XGBoostLTR()
    ltr.load_model(model_path)
    
    if feature_success:
        xgb_preds = ltr.predict(today_panel, feature_cols)
        if np.isnan(xgb_preds).all():
            log.critical("🚨 致命错误：XGBoost输出全部NaN，模型逻辑失效！")
            raise ValueError("XGBoost output all NaNs (Rule B3.6 Crash)")
        today_panel['xgb_score'] = xgb_preds
    else:
        today_panel['xgb_score'] = 0.5
        
    # Sort and Filter Top 5
    candidates = today_panel.sort_values('xgb_score', ascending=False)
    candidates = candidates[candidates['xgb_score'] >= 0.5]  # Dynamic threshold
    
    for _, ml_row in candidates.iterrows():
        code = ml_row['code']
        if code not in stock_infos: continue
        info = stock_infos[code]
        row, data, stop = info['row'], info['data'], info['stop']
        
        # Approximate scaling for UI score consistency
        score = float(ml_row['xgb_score']) * 100 
        level = "⚡ AI选股"
        reas = [f"XGB_Score:{ml_row['xgb_score']:.3f}"]
        
        target1_price = calc_target_price(row[C.S_PRICE], stop, data)
        money_msg = format_money_risk_msg(row[C.S_PRICE], stop, target1_price)
        tranche_msg = generate_tranche_plan(row[C.S_PRICE], score, m_ok, m_overheated)
        plan_b_msg = generate_plan_b(row[C.S_PRICE], stop, data['ma20_val'])
        hold_msg = generate_hold_period(data['adx'], data['price_pct'], data['has_chip_break'])
        
        confirmed_data.append(Signal(
            code=row[C.S_CODE], name=row[C.S_NAME], price=row[C.S_PRICE],
            pct_chg=f"{row[C.S_PCT]}%", score=score, level=level,
            trigger_time=now.strftime('%H:%M'), reasons=reas,
            stop_loss=round(stop, 2), target1=target1_price,
            ma10=round(data['ma10_val'], 2),
            money_risk_msg=money_msg, tranche_plan_msg=tranche_msg,
            plan_b_msg=plan_b_msg, hold_period_msg=hold_msg
        ))
"""

if old_loop in content:
    content = content.replace(old_loop, new_loop)
    with open("main.py", "w", encoding="utf-8") as f:
        f.write(content)
    print("Patched main.py successfully.")
else:
    print("Could not find old loop in main.py! Trying a looser match...")
    # fallback: search for 'for f in as_completed' to 'finally: ex2.shutdown'
    match = re.search(r'(    try:\n\s*for f in as_completed.*?finally:\n\s*ex2\.shutdown\(wait=False, cancel_futures=True\))', content, re.DOTALL)
    if match:
        content = content.replace(match.group(1), new_loop)
        with open("main.py", "w", encoding="utf-8") as f:
            f.write(content)
        print("Patched main.py successfully via regex.")
    else:
        print("Still could not find the block to replace!")
