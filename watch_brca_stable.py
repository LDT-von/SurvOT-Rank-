#!/usr/bin/env python3
"""
Monitor BRCA stable & norank_control experiments side-by-side.
Usage: python3 watch_brca_stable.py [refresh_seconds]
"""
import os, sys, time, csv, glob, re
from datetime import datetime

REFRESH = int(sys.argv[1]) if len(sys.argv) > 1 else 10

EXPS = {
    'stable': {
        'dir': 'results/dct_v3.3_score_first_brca_stable',
        'log': 'results/dct_v3.3_score_first_brca_stable/stable_run.log',
    },
    'norank': {
        'dir': 'results/dct_v3.3_score_first_brca_norank_control',
        'log': 'results/dct_v3.3_score_first_brca_norank_control/norank_run.log',
    },
}


def find_epoch_dir(exp_dir):
    """Find the per-run subdirectory containing epoch_curve CSVs."""
    for root, dirs, files in os.walk(exp_dir):
        for f in files:
            if f.startswith('epoch_curve_fold') and f.endswith('.csv'):
                return root
    return None


def read_fold_results(epoch_dir):
    results = {}
    if not epoch_dir:
        return results
    for fold in range(5):
        path = os.path.join(epoch_dir, f'epoch_curve_fold{fold}.csv')
        if not os.path.exists(path):
            continue
        rows = []
        with open(path) as f:
            reader = csv.DictReader(f)
            for r in reader:
                rows.append(r)
        if not rows:
            continue
        best_i = max(range(len(rows)), key=lambda i: float(rows[i].get('val_cindex', 0)))
        best_val = float(rows[best_i].get('val_cindex', 0))
        best_ep = best_i + 1
        last_val = float(rows[-1].get('val_cindex', 0))
        last5_vals = [float(r.get('val_cindex', 0)) for r in rows[-5:]]
        last5 = sum(last5_vals) / len(last5_vals)
        n_ep = len(rows)
        
        # Parse log for event/bin info
        train_events = val_events = ''
        train_bins = val_bins = ''
        train_ipcw = ''
        for r in rows:
            if r.get('train_events'):
                train_events = r['train_events']
            if r.get('val_events'):
                val_events = r['val_events']
            if r.get('train_bins'):
                train_bins = r['train_bins']
            if r.get('val_bins'):
                val_bins = r['val_bins']
            if r.get('train_ipcw_pairs'):
                train_ipcw = r['train_ipcw_pairs']
        
        results[fold] = {
            'best': best_val, 'best_ep': best_ep,
            'last': last_val, 'last5': last5,
            'n_ep': n_ep, 'train_events': train_events,
            'val_events': val_events, 'train_ipcw': train_ipcw,
        }
    return results


def read_log_tail(log_path, lines=5):
    if not os.path.exists(log_path):
        return []
    try:
        with open(log_path) as f:
            all_lines = f.readlines()
        return [l.rstrip() for l in all_lines[-lines:]]
    except:
        return []


def get_current_fold_from_log(log_path):
    lines = read_log_tail(log_path, 3)
    for l in reversed(lines):
        m = re.search(r'\[Fold (\d)\]', l)
        if m:
            return int(m.group(1))
    return None


def print_progress(n_ep, max_ep=35):
    if n_ep is None:
        return '[          ]'
    pct = n_ep / max_ep
    filled = int(pct * 10)
    return '[' + '#' * filled + ' ' * (10 - filled) + ']'


def fmt(val):
    if val is None:
        return '--'
    return f'{val:.4f}' if isinstance(val, float) else str(val)


import math


def main():
    try:
        while True:
            os.system('clear' if os.name != 'nt' else 'cls')
            now = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            print(f'{"="*80}')
            print(f'  BRCA DCT v3.3 Stable vs No-Rank Monitor  |  {now}')
            print(f'{"="*80}')

            all_bests = {}
            all_lasts = {}
            all_last5s = {}
            
            for name, exp in EXPS.items():
                epoch_dir = find_epoch_dir(exp['dir'])
                results = read_fold_results(epoch_dir)
                cur_fold = get_current_fold_from_log(exp['log'])
                
                label = f'  {name.upper()}'
                if cur_fold is not None:
                    label += f' (Fold {cur_fold} active)'
                print(f'\n{label}')
                print(f'  {"Fold":<8} {"Progress":<14} {"Best C-idx":<12} {"Best@":<8} {"Last C-idx":<12} {"Last5":<12} {"Events":<14} {"IPCW prs":<10}')
                print(f'  {"-"*90}')
                
                folds_completed = []
                for fold in range(5):
                    if fold in results:
                        r = results[fold]
                        bar = print_progress(r['n_ep'])
                        events_str = f't={r["train_events"]}/v={r["val_events"]}' if r.get('train_events') else ''
                        ipcw_str = r.get('train_ipcw', '')
                        print(f'    {fold:<6} {bar} {r["n_ep"]:>3d}/35  {r["best"]:<12.4f} {r["best_ep"]:<8} {r["last"]:<12.4f} {fmt(r.get("last5")):<12} {events_str:<14} {ipcw_str}')
                        folds_completed.append(r['best'])
                    else:
                        bar = print_progress(None)
                        print(f'    {fold:<6} {bar}  --/35  {"--":<12} {"--":<8} {"--":<12}')
                
                if folds_completed:
                    mean = sum(folds_completed) / len(folds_completed)
                    std = math.sqrt(sum((x - mean)**2 for x in folds_completed) / len(folds_completed)) if len(folds_completed) > 1 else 0
                    all_bests[name] = (mean, std)
                    
                    # Also last
                    lasts = [results[f]['last'] for f in sorted(results.keys())]
                    last5s = [results[f].get('last5', results[f]['last']) for f in sorted(results.keys()) if results[f].get('last5') is not None]
                    last_mean = sum(lasts) / len(lasts) if lasts else 0
                    all_lasts[name] = last_mean
                    all_last5s[name] = sum(last5s) / len(last5s) if last5s else 0
                    
                    print(f'  {"-"*90}')
                    print(f'  Best: {mean:.4f}±{std:.4f}  |  Last: {last_mean:.4f}  |  Gap: {mean - last_mean:.4f}  |  {len(folds_completed)}/5 folds')

                # Show recent log tail
                tail = read_log_tail(exp['log'], 4)
                if tail:
                    print(f'\n  --- Latest log ---')
                    for l in tail:
                        # Only show val/train lines, skip progress bars
                        if 'val cindex' in l or 'train_loss' in l or 'best cindex' in l or 'stopped' in l or 'Fold' in l and 'start' in l:
                            print(f'  {l[:120]}')

            # Comparison summary
            if len(all_bests) >= 2:
                b_mean = all_bests['stable'][0] - all_bests['norank'][0]
                b_winner = 'stable' if b_mean > 0 else 'norank'
                l_mean = all_lasts.get('stable', 0) - all_lasts.get('norank', 0)
                l_winner = 'stable' if l_mean > 0 else 'norank'
                print(f'\n{"="*80}')
                print(f'  Comparison: stable vs norank')
                print(f'  Best diff: {b_mean:+.4f} ({b_winner} better)')
                if 'stable' in all_lasts and 'norank' in all_lasts:
                    print(f'  Last diff: {l_mean:+.4f} ({l_winner} better)')
                    
                    gap_s = all_bests['stable'][0] - all_lasts['stable']
                    gap_n = all_bests['norank'][0] - all_lasts['norank']
                    print(f'  Best-Last gap: stable={gap_s:.4f}  norank={gap_n:.4f}')

            # Process check
            import subprocess
            result = subprocess.run(['ps', 'aux'], capture_output=True, text=True)
            procs = [l for l in result.stdout.split('\n') if 'survot_rank.cli' in l and 'python' in l]
            print(f'\n  Python processes: {len(procs)}  |  Ctrl+C to exit  |  next refresh in {REFRESH}s...')
            if procs:
                for p in procs[:3]:
                    print(f'  {p[:120]}')

            time.sleep(REFRESH)
    except KeyboardInterrupt:
        print('\n  Done.')


if __name__ == '__main__':
    main()
