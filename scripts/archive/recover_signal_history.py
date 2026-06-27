"""recover_signal_history.py
从 Git 历史中恢复过去的推送信号，补录到 signal_history.jsonl。
用法: python recover_signal_history.py
"""
import subprocess
import json
import os
import re
from datetime import datetime

HISTORY_PATH = '.quantbot_data/signal_history.jsonl'


def get_pushed_state_history():
    """从 git log 中提取 pushed_state.json 的历史版本。"""
    # Get all commits that touched pushed_state.json
    result = subprocess.run(
        ['git', 'log', '--all', '--pretty=format:%H|%ai|%s', '--follow', '--', 'pushed_state.json'],
        capture_output=True, text=True, encoding='utf-8', errors='replace',
        cwd=os.path.dirname(os.path.abspath(__file__)) or '.'
    )

    if result.returncode != 0:
        print(f"git log failed: {result.stderr}")
        return []

    commits = []
    for line in result.stdout.strip().split('\n'):
        if not line.strip():
            continue
        parts = line.split('|', 2)
        if len(parts) >= 2:
            commits.append({
                'hash': parts[0].strip(),
                'date': parts[1].strip(),
                'msg': parts[2].strip() if len(parts) > 2 else ''
            })

    return commits


def get_file_at_commit(commit_hash, filepath):
    """获取某个 commit 时的文件内容。"""
    result = subprocess.run(
        ['git', 'show', f'{commit_hash}:{filepath}'],
        capture_output=True, text=True, encoding='utf-8', errors='replace',
        cwd=os.path.dirname(os.path.abspath(__file__)) or '.'
    )
    if result.returncode != 0:
        return None
    try:
        return json.loads(result.stdout)
    except json.JSONDecodeError:
        return None


def load_existing_signals():
    """加载已有的信号历史，避免重复。"""
    existing = set()
    if os.path.exists(HISTORY_PATH):
        with open(HISTORY_PATH, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        rec = json.loads(line)
                        key = f"{rec['signal_date']}_{rec['code']}"
                        existing.add(key)
                    except (json.JSONDecodeError, KeyError):
                        pass
    return existing


def recover():
    print("🔍 扫描 Git 历史中的推送记录...")

    existing = load_existing_signals()
    print(f"已有 {len(existing)} 条历史信号，将跳过重复。")

    commits = get_pushed_state_history()
    print(f"找到 {len(commits)} 个涉及 pushed_state.json 的 commit。")

    # Also get advisory_tracker.json history
    at_result = subprocess.run(
        ['git', 'log', '--all', '--pretty=format:%H|%ai|%s', '--follow', '--', 'advisory_tracker.json'],
        capture_output=True, text=True, encoding='utf-8', errors='replace',
        cwd=os.path.dirname(os.path.abspath(__file__)) or '.'
    )
    at_commits = []
    if at_result.returncode == 0:
        for line in at_result.stdout.strip().split('\n'):
            if not line.strip():
                continue
            parts = line.split('|', 2)
            if len(parts) >= 2:
                at_commits.append({'hash': parts[0].strip(), 'date': parts[1].strip()})

    print(f"找到 {len(at_commits)} 个涉及 advisory_tracker.json 的 commit。")

    new_records = []
    seen_codes_dates = set(existing)  # Avoid duplicates

    # Extract from advisory_tracker.json (richer data)
    for commit in at_commits:
        tracker_data = get_file_at_commit(commit['hash'], 'advisory_tracker.json')
        if not tracker_data or not isinstance(tracker_data, dict):
            continue

        for code, info in tracker_data.items():
            entry_date = info.get('entry_date', '')
            if not entry_date:
                continue
            key = f"{entry_date}_{code}"
            if key in seen_codes_dates:
                continue

            record = {
                'signal_date': entry_date,
                'category': 'Resonance',
                'code': code,
                'name': info.get('name', ''),
                'entry_price': 0,  # Not available from tracker
                'score': 0,
                'rank_t1': 0, 'rank_t5': 0, 'rank_t10': 0, 'rank_t20': 0,
                'core_rank': 0,
                'stop_loss': float(info.get('stop', 0)),
                'target1': float(info.get('target', 0)),
                'market_context': f"[recovered from git: {commit['hash'][:8]}]",
                'actual_ret_t1': None, 'actual_ret_t5': None,
                'actual_ret_t10': None, 'actual_ret_t20': None,
                'csi300_ret_t1': None, 'csi300_ret_t5': None,
                'csi300_ret_t10': None, 'csi300_ret_t20': None,
                'excess_ret_t20': None,
                'hit_target': None, 'hit_stop': None,
                'max_gain_pct': None, 'max_drawdown_pct': None,
                'outcome_status': 'PENDING',
                'last_updated': datetime.now().strftime('%Y-%m-%d'),
                'source': 'git_recovery'
            }
            new_records.append(record)
            seen_codes_dates.add(key)

    if not new_records:
        print("未找到新的历史信号可补录。")
        return

    # Sort by date
    new_records.sort(key=lambda r: r['signal_date'])

    # Append to history
    os.makedirs(os.path.dirname(HISTORY_PATH), exist_ok=True)
    with open(HISTORY_PATH, 'a', encoding='utf-8') as f:
        for rec in new_records:
            f.write(json.dumps(rec, ensure_ascii=False) + '\n')

    print(f"✅ 成功补录 {len(new_records)} 条历史信号到 {HISTORY_PATH}")
    for r in new_records:
        print(f"  - {r['signal_date']} | {r['code']} ({r['name']}) | target={r['target1']} stop={r['stop_loss']}")


if __name__ == '__main__':
    recover()
