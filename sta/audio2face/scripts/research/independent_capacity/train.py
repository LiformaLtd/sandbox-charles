"""Isolated capacity trainer: same data/objective, independently initialized extra channels."""
from pathlib import Path
import argparse
import copy
import hashlib
import importlib.util
import json
import math
import os
import sys
import time
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from initialization import initialize_student

APP = Path(__file__).resolve().parents[3]
OUT = Path(os.environ.get('A2F_MULTIVOICE_OUTPUT', APP / '.local/compact-capacity-v1/control96')).resolve()
WORK = Path(os.environ['A2F_RESEARCH_WORK']).resolve()
# Import torch/NumPy before the optional ONNX-only compatibility path.
if os.environ.get('A2F_ONNX_SITE_PACKAGES'):
    sys.path.append(os.environ['A2F_ONNX_SITE_PACKAGES'])
import onnx
from onnx import numpy_helper

spec = importlib.util.spec_from_file_location('frozen_model', OUT / 'sources/model.py')
base = importlib.util.module_from_spec(spec)
spec.loader.exec_module(base)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def data(split, active):
    rows = [r for r in json.loads((OUT / 'manifest.json').read_text()) if r['split'] == split]
    classes = json.loads((OUT / 'phone-classes.json').read_text())
    digest = hashlib.sha256((OUT / 'manifest.json').read_bytes() + Path(__file__).read_bytes())
    for r in rows:
        if sha(r['audio']) != r['sha256_audio']:
            raise ValueError(f'Changed audio: {r["stem"]}')
        digest.update((OUT / 'labels' / (r['stem'] + '.npz')).read_bytes())
    key = digest.hexdigest()
    cache = OUT / ('features-' + split + '-' + key[:16] + '.npz')
    if cache.exists():
        with np.load(cache) as z:
            return {k: torch.from_numpy(z[k].copy()) for k in z.files}, key
    values = {k: [] for k in ['mel', 'weights', 'gaps', 'valid', 'phones', 'features', 'voice']}
    voices = sorted({r['engine'] + ':' + r['voice'] for r in rows})
    offsets = np.arange(15, 45) * 16000 // 60
    waves = []
    for row in rows:
        pcm = np.fromfile(row['audio'], '<f4')
        with np.load(OUT / 'labels' / (row['stem'] + '.npz')) as z:
            if str(z['audio_sha256']) != row['sha256_audio']:
                raise ValueError('Label/audio hash mismatch')
            weights, gaps, features = z['weights'][:, active], z['gaps'], z['speech_features']
        frames = (len(pcm) * 60 + 15999) // 16000
        if len(weights) != frames or len(features) != frames:
            raise ValueError('Incomplete teacher track')
        labels = np.full(frames, -100, np.int64)
        times = np.arange(frames) / 60
        for phone in row['phones']:
            labels[(times >= phone['start']) & (times < phone['end'])] = classes.index(phone['phone'])
        cursor = 0
        for start in range(-8000, len(pcm) - 4000, 8000):
            wave = np.zeros(16000, np.float32)
            lo, hi = max(start, 0), min(start + 16000, len(pcm))
            wave[lo - start:hi - start] = pcm[lo:hi]
            valid = (start + offsets >= 0) & (start + offsets < len(pcm))
            count = int(valid.sum())
            for name, source, shape, dtype in [
                ('weights', weights, (30, len(active)), np.float32),
                ('gaps', gaps, (30, 9), np.float32),
                ('features', features, (30, 768), np.float16),
                ('phones', labels, (30,), np.int64),
            ]:
                target = np.full(shape, -100 if name == 'phones' else 0, dtype=dtype)
                target[valid] = source[cursor:cursor + count]
                values[name].append(target)
            values['valid'].append(valid)
            values['voice'].append(voices.index(row['engine'] + ':' + row['voice']))
            waves.append(wave)
            cursor += count
        if cursor != frames:
            raise ValueError(f'Window coverage mismatch: {row["stem"]}')
    def logmel(pcm):
        windows = np.lib.stride_tricks.sliding_window_view(np.pad(pcm, ((0, 0), (200, 200))), 400, axis=1)[:, ::160]
        fft = np.fft.rfft(windows * base.HANN, n=512, axis=-1)
        power = (fft.real ** 2 + fft.imag ** 2).astype(np.float32)
        # Match the previous ablation's explicit reduction. Batched matmul in
        # this NumPy/Accelerate build reports invalid floating-point flags.
        mel = np.einsum('btf,mf->btm', power, base.MEL, optimize=False)
        return np.log1p(mel * 10).transpose(0, 2, 1).copy()
    values['mel'] = np.concatenate([logmel(np.stack(waves[i:i + 128])) for i in range(0, len(waves), 128)])
    arrays = {k: np.asarray(v) for k, v in values.items()}
    if not all(np.isfinite(v).all() for v in arrays.values()):
        raise ValueError('Nonfinite data')
    temporary = cache.with_name(cache.stem + f'.{os.getpid()}.npz')
    np.savez(temporary, **arrays)
    temporary.replace(cache)
    return {k: torch.from_numpy(v) for k, v in arrays.items()}, key


class TrainingModel(nn.Module):
    def __init__(self, student, classes):
        super().__init__()
        self.student = student
        self.phone_head = nn.Linear(student.encoder.out_channels, classes)
        self.feature_head = nn.Linear(student.encoder.out_channels, 768)

    def forward(self, mel):
        s = self.student
        encoded = s.blocks(F.gelu(s.encoder(mel), approximate='tanh'))
        lo = torch.index_select(encoded, 2, s.left)
        hi = torch.index_select(encoded, 2, s.left + 1)
        features = (lo + s.fraction * (hi - lo))[:, :, 15:45]
        pred = s.output(features).transpose(1, 2) * s.scale + s.mean
        return pred, self.phone_head(features.transpose(1, 2)), self.feature_head(features.transpose(1, 2))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--arm', choices=['control', 'phoneme', 'features', 'combined'], required=True)
    parser.add_argument('--width', type=int, choices=[96, 128], required=True)
    parser.add_argument('--epochs', type=int, default=24)
    parser.add_argument('--seed', type=int, default=37)
    parser.add_argument('--threads', type=int, default=4)
    parser.add_argument('--lr', type=float, default=.00012)
    parser.add_argument('--baseline', type=Path, help='Explicit initialization for a separate follow-up study')
    parser.add_argument('--steps-per-epoch', type=int, help='Fix update budget when a follow-up changes corpus size')
    parser.add_argument('--clean-replay', action='store_true',
                        help='Second-stage engineering arm: 48 new + 16 clean legacy windows, with an Easton validation guard')
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    rng = torch.Generator().manual_seed(args.seed + 1000)
    suffix = '-replay' if args.clean_replay else ''
    run = OUT / 'runs' / f'{args.arm}{suffix}-s{args.seed}'
    run.mkdir(parents=True, exist_ok=True)
    baseline = args.baseline or WORK / 'arkit-student-v3/james-arkit-student-v3.onnx'
    graph = onnx.load(baseline)
    values = {t.name: numpy_helper.to_array(t).copy() for t in graph.graph.initializer}
    rig = np.load(WORK / 'arkit-student/rig-metrics.npz')
    active = rig['active']
    student = base.Student(args.width, values['student.mean'], values['student.scale'], active)
    initialization = initialize_student(student, values)
    classes = json.loads((OUT / 'phone-classes.json').read_text())
    net = TrainingModel(student, len(classes))
    train, train_sha = data('train', active)
    dev, dev_sha = data('dev', active)
    replay = replay_dev = replay_fingerprint = None
    if args.clean_replay:
        os.environ['A2F_ABLATION_OUTPUT'] = str(OUT / 'replay-features')
        sys.path.insert(0, str(APP / 'scripts/research/compact_ablation'))
        import train as replay_data
        roots = [WORK / 'arkit-student-v3', WORK / 'arkit-student-v3/extra']
        replay = [torch.from_numpy(np.concatenate([replay_data.dataset(r, 'train', active, 16000)[i]
                                                   for r in roots])) for i in range(4)]
        replay_dev = [torch.from_numpy(v) for v in replay_data.dataset(roots[0], 'dev', active, 16000)]
        replay_fingerprint = {str(r): replay_data.input_fingerprint(str(r), 'train') for r in roots}
        replay_fingerprint['easton_dev'] = replay_data.input_fingerprint(str(roots[0]), 'dev')
        torch.set_num_threads(args.threads)
    basis = torch.from_numpy(rig['gap_basis'][active])
    rest = torch.from_numpy(rig['rest'] + rig['gap_offset'])
    names = json.loads((WORK / 'arkit-student/channels.json').read_text())
    lips = torch.tensor([1.7 if names[i].startswith(('mouth', 'jaw')) else 1. for i in active])
    config = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
    recipe = dict(initialization=initialization, initialization_code_sha256=sha(Path(__file__).with_name('initialization.py')), config=config, baseline_sha256=sha(baseline), train_sha256=train_sha, dev_sha256=dev_sha,
                  code_sha256=sha(__file__), model_sha256=sha(OUT / 'sources/model.py'),
                  clean_replay_fingerprints=replay_fingerprint,
                  phone_weight=.1, feature_weight=.1,
                  selection='macro voice gap RMSE + .25 * (1 - contact F1)' + ('; include clean Easton dev as fifth voice' if args.clean_replay else ''),
                  evaluation_policy='test split never used for selection')
    recipe_sha = hashlib.sha256(json.dumps(recipe, sort_keys=True).encode()).hexdigest()
    log_path = run / 'training.json'
    if log_path.exists() and json.loads(log_path.read_text()).get('recipe_sha256') != recipe_sha:
        raise ValueError('Run recipe changed; use a new output directory')
    if (run / 'model.onnx').exists():
        raise ValueError('Completed run exists; evaluate it or use a fresh study directory')
    def loss(pred, target, gaps, valid):
        w = valid[..., None]
        count = valid.sum().clamp(min=1)
        shape = (F.smooth_l1_loss((pred - target) / student.scale, torch.zeros_like(pred), reduction='none', beta=.5) * w * lips).sum() / (count * len(active))
        geom = pred.clamp(0, 1) @ basis + rest
        gap = (F.smooth_l1_loss(geom, gaps, reduction='none', beta=.14) * w).sum() / (count * 9)
        adjacent = valid[:, 1:] & valid[:, :-1]
        velocity = (((pred.diff(dim=1) - target.diff(dim=1)).abs() / student.scale) * adjacent[..., None]).sum() / (count * len(active))
        actual, estimate = gaps.median(-1).values, geom.median(-1).values
        contact = actual <= .1
        closure = (F.binary_cross_entropy_with_logits((.1 - estimate) / .16, contact.float(), reduction='none') * (1 + contact) * valid).sum() / count
        near = ((actual[:, 1:] < .35) | (actual[:, :-1] < .35)) & adjacent
        timing = (F.smooth_l1_loss(estimate.diff(dim=1), actual.diff(dim=1), reduction='none', beta=.1) * near).sum() / near.sum().clamp(min=1)
        return shape + 1.2 * gap + .12 * velocity + .22 * closure + .45 * timing

    @torch.no_grad()
    def evaluate():
        net.eval()
        predictions = torch.cat([net(v)[0].clamp(0, 1) for v in dev['mel'].split(64)])
        scores = []
        for voice in dev['voice'].unique():
            ix = dev['voice'] == voice
            valid = dev['valid'][ix]
            truth = dev['gaps'][ix][valid]
            pred = (predictions[ix] @ basis + rest)[valid]
            a, b = truth.median(-1).values <= .1, pred.median(-1).values <= .1
            f1 = float(2 * (a & b).sum() / (a.sum() + b.sum()).clamp(min=1))
            rmse = float((pred - truth).square().mean().sqrt())
            scores.append(dict(voice_index=int(voice), lip_gap_rmse=rmse, contact_f1=f1))
        score = float(np.mean([s['lip_gap_rmse'] + .25 * (1 - s['contact_f1']) for s in scores]))
        if replay_dev is not None:
            x, _, truth, valid = replay_dev
            pred = torch.cat([net(v)[0].clamp(0, 1) for v in x.split(64)])
            pg, tg = (pred @ basis + rest)[valid], truth[valid]
            a, b = tg.median(-1).values <= .1, pg.median(-1).values <= .1
            f1 = float(2 * (a & b).sum() / (a.sum() + b.sum()).clamp(min=1))
            rmse = float((pg - tg).square().mean().sqrt())
            score = (score * 4 + rmse + .25 * (1 - f1)) / 5
            scores.append(dict(voice_index='easton_guard', lip_gap_rmse=rmse, contact_f1=f1))
        return score, scores

    groups = [torch.nonzero(train['voice'] == v, as_tuple=True)[0] for v in train['voice'].unique()]
    if len(groups) != 8:
        raise ValueError('Expected eight balanced training voices')
    steps = args.steps_per_epoch or math.ceil(len(train['mel']) / 64)
    if steps < 1:
        raise ValueError('Expected a positive update budget')
    optimizer = torch.optim.AdamW(net.parameters(), lr=args.lr, weight_decay=.015)
    schedule = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, steps * args.epochs, eta_min=args.lr * .1)
    score, initial = evaluate()
    best, best_epoch, best_state = score, 0, copy.deepcopy(student.state_dict())
    log = dict(recipe=recipe, recipe_sha256=recipe_sha, initial_dev=initial, history=[], steps_per_epoch=steps,
               train_windows=len(train['mel']), phone_classes=classes)
    start_epoch = 0
    checkpoint = run / 'checkpoint.pt'
    if checkpoint.exists():
        state = torch.load(checkpoint, weights_only=False)
        if state['recipe_sha256'] != recipe_sha:
            raise ValueError('Checkpoint recipe mismatch')
        net.load_state_dict(state['model'])
        optimizer.load_state_dict(state['optimizer'])
        schedule.load_state_dict(state['scheduler'])
        rng.set_state(state['rng'])
        best, best_epoch, best_state = state['best'], state['best_epoch'], state['best_state']
        start_epoch, log = state['epoch'], state['log']
    begun = time.monotonic()
    for epoch in range(start_epoch, args.epochs):
        net.train()
        losses = []
        for _ in range(steps):
            # Pilot: eight windows per voice. Replay follow-up: six per voice
            # plus 16 clean Easton windows; no old mixed-fit v1/v2 labels.
            per_voice = 6 if args.clean_replay else 8
            index = torch.cat([g[torch.randint(len(g), (per_voice,), generator=rng)] for g in groups])
            x, y, g, m, p, f = [train[k][index] for k in ['mel', 'weights', 'gaps', 'valid', 'phones', 'features']]
            if replay is not None:
                old = torch.randint(len(replay[0]), (16,), generator=rng)
                x, y, g, m = [torch.cat((new, previous[old])) for new, previous in zip((x, y, g, m), replay)]
                p = torch.cat((p, torch.full((16, 30), -100, dtype=torch.int64)))
                f = torch.cat((f, torch.zeros(16, 30, 768, dtype=torch.float16)))
            optimizer.zero_grad(set_to_none=True)
            pred, phone, features = net(x)
            objective = loss(pred, y, g, m)
            if args.arm in ['phoneme', 'combined']:
                target = p.flatten()
                if (target != -100).any():
                    objective = objective + .1 * F.cross_entropy(phone.reshape(-1, len(classes)), target, ignore_index=-100)
            if args.arm in ['features', 'combined']:
                a = F.layer_norm(features, (768,))
                b = F.layer_norm(f.float(), (768,))
                feature_mask = m.clone()
                if replay is not None:
                    feature_mask[-16:] = False
                objective = objective + .1 * F.smooth_l1_loss(a[feature_mask], b[feature_mask], beta=.5)
            if not torch.isfinite(objective):
                raise FloatingPointError('Nonfinite training objective')
            objective.backward()
            nn.utils.clip_grad_norm_(net.parameters(), 1., error_if_nonfinite=True)
            optimizer.step()
            schedule.step()
            losses.append(float(objective.detach()))
        score, scores = evaluate()
        if score < best:
            best, best_epoch, best_state = score, epoch + 1, copy.deepcopy(student.state_dict())
        log['history'].append(dict(epoch=epoch + 1, dev=scores, selection=score,
                                   training_loss=float(np.mean(losses)), elapsed=time.monotonic() - begun))
        log['best_epoch'] = best_epoch
        log_path.write_text(json.dumps(log, indent=2))
        state = dict(model=net.state_dict(), optimizer=optimizer.state_dict(), scheduler=schedule.state_dict(),
                     rng=rng.get_state(), best=best, best_epoch=best_epoch, best_state=best_state,
                     epoch=epoch + 1, log=log, recipe_sha256=recipe_sha)
        torch.save(state, run / 'checkpoint.part.pt')
        (run / 'checkpoint.part.pt').replace(checkpoint)
        print(args.arm, epoch + 1, 'selection', round(score, 5), 'best', best_epoch,
              'elapsed', round(time.monotonic() - begun), flush=True)
    student.load_state_dict(best_state)
    student.eval()
    torch.save(student.state_dict(), run / 'student.pt')
    model = base.WaveformModel(student).eval()
    torch.onnx.export(model, torch.zeros(1, 16000), str(run / 'model.onnx'), input_names=['pcm'],
                      output_names=['blendshapes'], opset_version=17, dynamo=False)
    exported = onnx.load(run / 'model.onnx')
    onnx.checker.check_model(exported)
    if any('head' in v.name for v in exported.graph.initializer):
        raise ValueError('Auxiliary head leaked into deployment export')
    log.update(bytes=(run / 'model.onnx').stat().st_size, model_sha256=sha(run / 'model.onnx'),
               parameters=sum(p.numel() for p in student.parameters()), best_selection=best,
               seconds=time.monotonic() - begun)
    log_path.write_text(json.dumps(log, indent=2))


if __name__ == '__main__':
    main()
