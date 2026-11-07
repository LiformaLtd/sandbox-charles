"""Geometry-timed closure scores; phonemes identify events, never move them."""
from collections import Counter
import numpy as np

FPS = 60


def runs(mask):
    edges = np.flatnonzero(np.diff(np.r_[False, mask, False]))
    return [(int(a), int(b)) for a, b in zip(edges[::2], edges[1::2])]


def associate(phones, events, frames, lead=.100, trail=.035):
    """Keep unique teacher/phone associations; expose ambiguity instead of guessing.

    An episode may start before the window if it continues into it. The 100 ms
    padding is an association tolerance, not an onset constraint or time shift.
    """
    labials = [p for p in phones if p['phone'].lower() in ['m', 'b', 'p']]
    candidates = [[j for j, (a, b) in enumerate(events)
                   if b - a >= 2 and b / FPS >= p['start'] - lead and a / FPS <= p['end'] + trail]
                  for p in labials]
    claims = Counter(j for js in candidates for j in js)
    output = []
    for phone, js in zip(labials, candidates):
        if not js:
            status = 'no_teacher_contact'
        elif len(js) > 1:
            status = 'multiple_teacher_events'
        elif claims[js[0]] > 1:
            status = 'shared_with_neighbor'
        elif events[js[0]][0] == 0 or events[js[0]][1] == frames:
            status = 'clip_truncated'
        else:
            status = 'unique'
        output.append(dict(phone=phone, status=status, candidates=js,
                           event=events[js[0]] if status == 'unique' else None))
    return output


def score(records, lead=.100):
    """Each record is (manifest row, raw teacher gaps, predicted gaps)."""
    result = {p: dict(phonemes=0, association_counts=Counter(), teacher_events=0,
                     overlap_hits=0, well_timed_hits=0, teacher_onset_offsets_ms=[],
                     onset_errors_ms=[], release_errors_ms=[], iou=[]) for p in ['m', 'b', 'p']}
    details = []
    for row, truth, pred in records:
        actual = runs(np.median(truth, axis=-1) <= .1)
        predicted = runs(np.median(pred, axis=-1) <= .1)
        associated = associate(row.get('phones', []), actual, len(truth), lead)
        eligible = []
        for item in associated:
            phone = item['phone']['phone'].lower()
            r = result[phone]
            r['phonemes'] += 1
            r['association_counts'][item['status']] += 1
            if item['status'] == 'unique':
                a, b = item['event']
                r['teacher_events'] += 1
                offset = (a / FPS - item['phone']['start']) * 1000
                r['teacher_onset_offsets_ms'].append(offset)
                eligible.append((phone, a, b, item))
        pairs = []
        for i, (phone, a, b, _) in enumerate(eligible):
            for j, (c, d) in enumerate(predicted):
                overlap = min(b, d) - max(a, c)
                if overlap > 0:
                    pairs.append((-overlap, abs(a - c) + abs(b - d), i, j, phone, a, b, c, d))
        matched, used = set(), set()
        for _, _, i, j, phone, a, b, c, d in sorted(pairs):
            if i in matched or j in used:
                continue
            matched.add(i)
            used.add(j)
            r = result[phone]
            iou = (min(b, d) - max(a, c)) / (max(b, d) - min(a, c))
            r['overlap_hits'] += 1
            r['well_timed_hits'] += int(iou >= .5 and abs(c - a) <= 2)
            r['onset_errors_ms'].append((c - a) * 1000 / FPS)
            r['release_errors_ms'].append((d - b) * 1000 / FPS)
            r['iou'].append(iou)
        details.append(dict(stem=row['stem'], associations=associated,
                            predicted_events=predicted, uniquely_scored=len(eligible), hits=len(matched)))
    def dist(values):
        if not values:
            return dict(n=0)
        return dict(n=len(values), mean=float(np.mean(values)), median=float(np.median(values)),
                    p05=float(np.quantile(values, .05)), p95=float(np.quantile(values, .95)))
    for value in result.values():
        value['association_counts'] = dict(value['association_counts'])
        value['teacher_onset_offsets_ms'] = dist(value['teacher_onset_offsets_ms'])
        for key in ['onset_errors_ms', 'release_errors_ms']:
            errors = value[key]
            value[key] = dict(**dist(errors), mae=float(np.mean(np.abs(errors))) if errors else None)
        value['iou'] = dist(value['iou'])
    return dict(method=dict(association_lead_ms=lead * 1000, association_trail_ms=35,
                            threshold_cm=.1, min_teacher_frames=2, fps=FPS,
                            overlap_hit='any temporal overlap with teacher contact; one-to-one prediction matching',
                            well_timed_hit='overlap IoU >= .5 and onset error <= 2 frames (33.3 ms)',
                            timing='signed errors relative to actual teacher closure, not acoustic phoneme onset',
                            exclusions='ambiguous/shared or clip-truncated teacher events reported, not silently assigned'),
                summary=result, details=details)
