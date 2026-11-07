"""Record the confirmation protocol before generating or inspecting predictions."""
from pathlib import Path
import hashlib
import json
import os

APP = Path(__file__).resolve().parents[3]
PILOT = Path(os.environ.get('A2F_MULTIVOICE_PILOT', APP / '.local/compact-multivoice')).resolve()
CONFIRM = Path(os.environ.get('A2F_CONFIRMATION_ROOT', APP / '.local/compact-v4-confirmation')).resolve()


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    shared = ['sources/label.py', 'sources/model.py', 'sources/pcm.js', 'sources/supertonic.js',
              'assets/aligner/model.onnx', 'assets/aligner/lexicon.json']
    for relative in shared:
        if sha(CONFIRM / relative) != sha(PILOT / relative):
            raise ValueError(f'Confirmation source/asset differs: {relative}')
    protocol = dict(
        seeds=[37, 73, 101],
        training='unchanged control --clean-replay, 24 epochs, lr 0.00012, width96, initialized from released v3',
        training_code_sha256=sha(APP / 'scripts/research/multivoice/train.py'),
        planned_sha256=sha(CONFIRM / 'planned.json'),
        definition_sha256=json.loads((CONFIRM / 'assets.json').read_text())['definition_sha256'],
        primary='Mean across seeds of macro-voice gap RMS versus fixed v3; report seed variability and paired crossed seed/voice/text bootstrap',
        representative='Minimum best_selection on five-voice validation cohort; ties resolved by seed number; select before fresh test evaluation',
        candidate_screen=dict(fresh_macro_gap_reduction_at_least=.05, fresh_each_engine_gap_reduction_at_least=0,
                              fresh_macro_contact_f1_delta_at_least=0, fresh_each_voice_rms_relative_regression_at_most=.02,
                              historical_easton_mean_gap_relative_regression_at_most=.02,
                              historical_easton_mean_contact_f1_delta_at_least=-.01),
        interpretation='Engineering screen, not universal voice certification. No fresh-test tuning. Historical sets are regression checks. Physical iPhone remains manual.',
        visual_cases=['test-phonon-trevor-000', 'test-phonon-pippa-005', 'test-supertonic-m3-009',
                      'test-supertonic-f3-014', 'test-phonon-trevor-018', 'test-supertonic-f3-022'])
    target = CONFIRM / 'protocol.json'
    if target.exists():
        if json.loads(target.read_text()) != protocol:
            raise ValueError('Frozen protocol changed; create a new study')
        print('Existing frozen protocol matches; preserving its exact bytes.')
        return
    if (CONFIRM / 'evaluation').exists():
        raise ValueError('Cannot freeze a new confirmation protocol after evaluation')
    target.write_text(json.dumps(protocol, indent=2))
    print('Frozen', target)


if __name__ == '__main__':
    main()
