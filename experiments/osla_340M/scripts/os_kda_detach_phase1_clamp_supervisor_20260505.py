#!/usr/bin/env python3
import json
import os
import re
import signal
import subprocess
import time
from datetime import datetime
from pathlib import Path

ROOT = Path('/DATA/disk1/cyzhou/OSLA')
FLAME = ROOT / 'flame'
CONFIG_DIR = ROOT / 'experiments/osla_340M/configs'
EXP_DIR = ROOT / 'experiments/osla_340M/exp'
BASE_CONFIG = CONFIG_DIR / 'os_kda_340M.json'
PYTHON = Path('/DATA/disk1/cyzhou/miniconda3/envs/osla/bin/python')
TORCHRUN = Path('/DATA/disk1/cyzhou/miniconda3/envs/osla/bin/torchrun')
TOKENIZER = 'fla-hub/delta_net-1.3B-100B'
DATASET = 'HuggingFaceFW/fineweb-edu'
DATASET_NAME = 'sample-10BT'
PORT = 29520
SCREEN_STEPS = 1600
FULL_STEPS = 20480
SCREEN_GNORM_LIMIT = 50.0
FULL_GNORM_LIMIT = 120.0
POLL_SECONDS = 30
STAMP = datetime.now().strftime('%Y%m%d-%H%M%S')
RUN_ROOT = EXP_DIR / f'os-kda-detach-phase1-clamp-supervisor-{STAMP}'
RUN_ROOT.mkdir(parents=True, exist_ok=True)
SUP_LOG = RUN_ROOT / 'supervisor.log'

STEP_PAT = re.compile(r'step:\s*(\d+).*?loss:\s*([0-9.eE+-]+).*?memory:\s*([^ ]+).*?tps:\s*([0-9,]+)', re.I)
LR_PAT = re.compile(r'lr:\s*([0-9.eE+-]+)\s+gnorm:\s*([0-9.eE+-]+)', re.I)
ANSI_PAT = re.compile(r'\x1b\[[0-9;]*m')

CANDIDATES = [
    {'eta': 0.03, 'd_max': 1e9, 'beta_aware': True},
    {'eta': 0.03, 'd_max': 2.0, 'beta_aware': True},
    {'eta': 0.03, 'd_max': 1.5, 'beta_aware': True},
    {'eta': 0.03, 'd_max': 1.2, 'beta_aware': True},
    {'eta': 0.01, 'd_max': 2.0, 'beta_aware': True},
    {'eta': 0.01, 'd_max': 1.5, 'beta_aware': True},
    {'eta': 0.01, 'd_max': 1.2, 'beta_aware': True},
]


def log(msg):
    line = f'[{datetime.now().isoformat(timespec="seconds")}] {msg}'
    print(line, flush=True)
    with SUP_LOG.open('a') as f:
        f.write(line + '\n')


def fmt_float(x):
    if x == 1e9:
        return '1e9'
    s = str(x).replace('.', 'p')
    return s.replace('-', 'm')


def tag_for(kind, cand):
    ba = 'ba1' if cand['beta_aware'] else 'ba0'
    decay = 'no-dd' if kind == 'none' else 'dd-gamma0p999'
    return f'os-kda-340M-detach-phase1-{decay}-eta{fmt_float(cand["eta"])}-dmax{fmt_float(cand["d_max"])}-{ba}'


def write_config(kind, cand):
    cfg = json.loads(BASE_CONFIG.read_text())
    cfg['use_osgm'] = True
    cfg['osgm_eta'] = cand['eta']
    cfg['osgm_use_denominator'] = False
    cfg['osgm_d_min'] = 0.0
    cfg['osgm_d_max'] = cand['d_max']
    cfg['osgm_beta_aware'] = cand['beta_aware']
    if kind == 'none':
        cfg['osgm_decay_mode'] = 'none'
        cfg['osgm_decay_gamma'] = 1.0
    elif kind == 'constant':
        cfg['osgm_decay_mode'] = 'constant'
        cfg['osgm_decay_gamma'] = 0.999
    else:
        raise ValueError(kind)
    path = CONFIG_DIR / f'{tag_for(kind, cand)}.json'
    path.write_text(json.dumps(cfg, indent=4) + '\n')
    return path


def parse_rows(log_path):
    text = log_path.read_text(errors='ignore') if log_path.exists() else ''
    text = ANSI_PAT.sub('', text)
    rows = []
    pending = None
    for line in text.splitlines():
        m = STEP_PAT.search(line)
        if m:
            pending = [int(m.group(1)), float(m.group(2)), m.group(3), int(m.group(4).replace(',', '')), None, None]
            continue
        m = LR_PAT.search(line)
        if m and pending is not None:
            pending[4] = float(m.group(1))
            pending[5] = float(m.group(2))
            rows.append(tuple(pending))
            pending = None
    lowered = text.lower()
    bad_event = any(s in lowered for s in [
        'traceback', 'runtimeerror', 'out of memory', 'nan encountered', 'inf encountered', 'childfailederror'
    ])
    completed = 'training completed' in lowered
    return rows, completed, bad_event


def launch(kind, cand, steps, online, phase):
    cfg_path = write_config(kind, cand)
    tag = tag_for(kind, cand)
    exp_name = f'{phase}-{tag}-{steps}step-{datetime.now().strftime("%Y%m%d-%H%M%S")}'
    if phase == 'full':
        exp_name = f'{tag}-full-{datetime.now().strftime("%Y%m%d-%H%M%S")}'
    exp = EXP_DIR / exp_name
    (exp / 'logs').mkdir(parents=True, exist_ok=True)
    run_log = exp / 'run.log'
    cmd = [
        str(PYTHON), str(TORCHRUN),
        '--nnodes=1', '--nproc_per_node=8', '--rdzv_backend', 'c10d',
        '--rdzv_endpoint', f'localhost:{PORT}', '--local-ranks-filter', '0',
        '--role', 'rank', '--tee', '3', '--log-dir', str(exp / 'logs'),
        '-m', 'flame.train',
        '--job.config_file', 'flame/models/fla.toml',
        '--job.dump_folder', str(exp),
        '--model.config', str(cfg_path),
        '--model.tokenizer_path', TOKENIZER,
        '--optimizer.name', 'AdamW', '--optimizer.eps', '1e-15', '--optimizer.lr', '1e-3',
        '--lr_scheduler.warmup_steps', '1024', '--lr_scheduler.lr_min', '0.1', '--lr_scheduler.decay_type', 'cosine',
        '--training.batch_size', '1', '--training.seq_len', '65536', '--training.context_len', '4096', '--training.varlen',
        '--training.gradient_accumulation_steps', '1', '--training.steps', str(steps),
        '--training.max_norm', '1.0', '--training.skip_nan_inf',
        '--training.data_parallel_replicate_degree', '8', '--training.data_parallel_shard_degree', '1',
        '--training.dataset', DATASET, '--training.dataset_name', DATASET_NAME, '--training.dataset_split', 'train',
        '--training.num_workers', '32', '--training.prefetch_factor', '2', '--training.seed', '42',
        '--activation_checkpoint.mode', 'selective', '--activation_checkpoint.selective_ac_option', '2',
        '--checkpoint.interval', '2048', '--checkpoint.load_step', '-1', '--checkpoint.keep_latest_k', '2',
        '--metrics.log_freq', '1',
    ]
    env = os.environ.copy()
    env['PYTHONPATH'] = f'{ROOT}:{FLAME}'
    env['CUDA_VISIBLE_DEVICES'] = '0,1,2,3,4,5,6,7'
    env['OSKDA_DETACH_PHASE1'] = '1'
    if online:
        env.pop('WANDB_MODE', None)
    else:
        env['WANDB_MODE'] = 'offline'
    log(f'LAUNCH phase={phase} kind={kind} online={online} steps={steps} exp={exp} cfg={cfg_path}')
    f = run_log.open('w')
    proc = subprocess.Popen(cmd, cwd=str(FLAME), env=env, stdout=f, stderr=subprocess.STDOUT, preexec_fn=os.setsid)
    (exp / 'launcher.pid').write_text(str(proc.pid) + '\n')
    return proc, f, exp, run_log


def stop_proc(proc):
    if proc.poll() is not None:
        return
    try:
        os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
    except ProcessLookupError:
        return
    for _ in range(20):
        if proc.poll() is not None:
            return
        time.sleep(1)
    try:
        os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
    except ProcessLookupError:
        pass


def monitor(proc, log_handle, exp, run_log, steps, gnorm_limit, allow_continue=False):
    last_report = 0
    while True:
        time.sleep(POLL_SECONDS)
        rows, completed, bad_event = parse_rows(run_log)
        last = rows[-1] if rows else None
        max_g = max((r[5] for r in rows), default=0.0)
        max_after_200 = max((r[5] for r in rows if r[0] >= 200), default=0.0)
        if last and (last[0] >= last_report + 100 or completed):
            last_report = last[0]
            log(f'MONITOR exp={exp.name} step={last[0]} loss={last[1]:.4f} gnorm={last[5]:.2f} max_after200={max_after_200:.2f}')
        if bad_event:
            log(f'FAIL event exp={exp.name}')
            stop_proc(proc)
            log_handle.close()
            return False, rows
        if max_after_200 > gnorm_limit:
            log(f'FAIL gnorm exp={exp.name} max_after200={max_after_200:.2f} limit={gnorm_limit}')
            stop_proc(proc)
            log_handle.close()
            return False, rows
        ret = proc.poll()
        if ret is not None:
            log_handle.close()
            rows, completed, bad_event = parse_rows(run_log)
            if ret == 0 and completed:
                log(f'PASS completed exp={exp.name} rows={len(rows)} max_after200={max((r[5] for r in rows if r[0] >= 200), default=0.0):.2f}')
                return True, rows
            log(f'FAIL exit exp={exp.name} ret={ret} completed={completed} rows={len(rows)}')
            return False, rows
        if allow_continue and last and last[0] >= steps:
            log(f'PASS reached target exp={exp.name} step={last[0]}')
            return True, rows


def run_screen_pair(cand):
    results = {}
    for kind in ['none', 'constant']:
        proc, f, exp, run_log = launch(kind, cand, SCREEN_STEPS, online=False, phase='screen')
        ok, rows = monitor(proc, f, exp, run_log, SCREEN_STEPS, SCREEN_GNORM_LIMIT)
        results[kind] = (ok, exp, rows)
        if not ok:
            return False, results
    return True, results


def run_full_pair(cand):
    full_results = {}
    for kind in ['none', 'constant']:
        proc, f, exp, run_log = launch(kind, cand, FULL_STEPS, online=True, phase='full')
        ok, rows = monitor(proc, f, exp, run_log, FULL_STEPS, FULL_GNORM_LIMIT)
        full_results[kind] = (ok, exp, rows)
        if not ok:
            return False, full_results
    return True, full_results


def main():
    log(f'START supervisor root={RUN_ROOT}')
    log(f'candidates={CANDIDATES}')
    selected = None
    for cand in CANDIDATES:
        log(f'TRY candidate={cand}')
        ok, screen_results = run_screen_pair(cand)
        summary = {k: {'ok': v[0], 'exp': str(v[1]), 'last': (v[2][-1] if v[2] else None), 'max_after200': max((r[5] for r in v[2] if r[0] >= 200), default=0.0)} for k, v in screen_results.items()}
        (RUN_ROOT / f'screen_summary_{tag_for("none", cand)}.json').write_text(json.dumps(summary, indent=2, default=str) + '\n')
        log(f'SCREEN summary={summary}')
        if not ok:
            continue
        selected = cand
        log(f'SELECT candidate={cand}; starting full pair')
        ok_full, full_results = run_full_pair(cand)
        full_summary = {k: {'ok': v[0], 'exp': str(v[1]), 'last': (v[2][-1] if v[2] else None), 'max_after200': max((r[5] for r in v[2] if r[0] >= 200), default=0.0)} for k, v in full_results.items()}
        (RUN_ROOT / f'full_summary_{tag_for("none", cand)}.json').write_text(json.dumps(full_summary, indent=2, default=str) + '\n')
        log(f'FULL summary={full_summary}')
        if ok_full:
            log(f'DONE success candidate={cand}')
            (RUN_ROOT / 'SUCCESS.json').write_text(json.dumps({'candidate': cand, 'full_summary': full_summary}, indent=2, default=str) + '\n')
            return
        log(f'FULL failed candidate={cand}; moving to next candidate')
    log('DONE no stable full pair found')
    (RUN_ROOT / 'FAILED.json').write_text(json.dumps({'selected': selected}, indent=2, default=str) + '\n')

if __name__ == '__main__':
    main()
