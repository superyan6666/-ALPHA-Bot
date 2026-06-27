from main import get_signals
import logging

logging.basicConfig(level=logging.INFO)

sigs, watch, pushed, pool_size, m_msg, total_mkt = get_signals()
print(f"Pool size: {pool_size}")
print(f"Total Market: {total_mkt}")

total_signals = sum(len(lst) for lst in sigs.values()) if isinstance(sigs, dict) else len(sigs)
print(f"Generated {total_signals} confirmed signals")

if isinstance(sigs, dict):
    for horizon, s_list in sigs.items():
        if s_list:
            print(f"\n[{horizon} Signals]")
            for s in s_list:
                print(f"  {s.name} ({s.code}) - Score: {s.score:.2f} - Target: {s.target1} - Stop: {s.stop_loss}")
else:
    for s in sigs:
        print(f"  {s.name} ({s.code}) - Score: {s.score:.2f} - Target: {s.target1} - Stop: {s.stop_loss}")
