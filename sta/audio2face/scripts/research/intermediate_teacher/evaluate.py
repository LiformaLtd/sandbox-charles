"""Lock selections before scoring existing held-out regression cohorts."""
from common import *
import argparse
import numpy as np,onnxruntime as ort,onnx
import importlib.util
_spec=importlib.util.spec_from_file_location('frozen_scale_evaluator',APP/'scripts/research/data_scale/evaluate.py')
_module=importlib.util.module_from_spec(_spec);_spec.loader.exec_module(_module)
evaluate_cohort=_module.evaluate_cohort

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--mentor-only',action='store_true');args=parser.parse_args()
    protocol=lock();models={'baseline':BASELINE,'mentor':OUT/'mentor-s37/model.onnx'}
    if not args.mentor_only:
        for seed in protocol['students']['seeds']:
            for arm in ['control','distill']:models[f'{arm}-s{seed}']=OUT/f'{arm}-s{seed}/model.onnx'
    selections={}
    for name,path in models.items():
        if name=='baseline':selections[name]={'sha256':sha(path)};continue
        record=json.loads((path.parent/'complete.json').read_text())
        if sha(path)!=record['export']['sha256']:raise ValueError('Selected model changed')
        selections[name]=record
    if not args.mentor_only:
        for seed in protocol['students']['seeds']:
            a,b=[selections[f'{arm}-s{seed}'] for arm in ['control','distill']]
            for field in ['sampler_sha256','initial_parameters_sha256','steps','protocol_sha256']:
                if a[field]!=b[field]:raise ValueError('Unmatched control: '+field)
    write(OUT/('mentor-selection-lock.json' if args.mentor_only else 'selection-lock.json'),selections)
    cohorts={'feature_confirmation':DATA/'.local/compact-feature-v1/confirmation/variants',
             'scale_confirmation':SCALE/'datasets/confirmation','historical':DATA/'.local/compact-gain-v2'}
    rig=np.load(WORK/'arkit-student/rig-metrics.npz');reports={}
    for name,path in models.items():
        result_path=OUT/f'evaluation-{name}.json'
        if result_path.exists():
            report=json.loads(result_path.read_text())
            if report['sha256']!=sha(path):raise ValueError('Invalid cached report')
        else:
            options=ort.SessionOptions();options.intra_op_num_threads=2;options.inter_op_num_threads=1
            session=ort.InferenceSession(str(path),sess_options=options,providers=['CPUExecutionProvider'])
            cache=OUT/'predictions'/sha(path);cache.mkdir(parents=True,exist_ok=True)
            report={'sha256':sha(path)}
            for cohort,root in cohorts.items():
                if sha(root/'manifest.json')!=protocol['heldout_cohorts'][str(root/'manifest.json')]:raise ValueError('Held-out manifest changed')
                rows=[r for r in json.loads((root/'manifest.json').read_text()) if r.get('split')=='test' and r['condition']=='original']
                if not rows:raise ValueError('Empty held-out cohort')
                report[cohort]=evaluate_cohort(session,rows,root/'labels',cache,rig['gap_basis'],rig['rest']+rig['gap_offset'],rig['active'])
                print(name,cohort,report[cohort]['conditions']['original']['macro_gap_rmse_mm'],flush=True)
            write(result_path,report)
        reports[name]=report
    def stats(name,cohort):return reports[name][cohort]['conditions']['original']
    summary={'cohort_status':'Previously inspected held-out recordings; not new confirmation','mentor_development_gate':selections['mentor']['mentor_gate_pass']}
    if not args.mentor_only:
        effects={}
        for cohort in cohorts:
            changes=[];f1=[];engine=[];opening=[];closure=[]
            for seed in protocol['students']['seeds']:
                a,b=[stats(f'{arm}-s{seed}',cohort) for arm in ['control','distill']]
                changes.append(1-b['macro_gap_rmse_mm']/a['macro_gap_rmse_mm']);f1.append(b['macro_contact_f1']-a['macro_contact_f1'])
                engine.append({e:1-b['engines'][e]['macro_gap_rmse_mm']/a['engines'][e]['macro_gap_rmse_mm'] for e in ['phonon','supertonic']})
                opening.append(np.mean([v['opening_rmse_mm'] for v in b['voices'].values()])-np.mean([v['opening_rmse_mm'] for v in a['voices'].values()]))
                closure.append(sum(b['closure_audit'][p]['well_timed_hits']-a['closure_audit'][p]['well_timed_hits'] for p in ['m','b','p']))
            effects[cohort]=dict(relative_gap_reduction_by_seed=changes,mean_relative_gap_reduction=float(np.mean(changes)),contact_f1_delta_by_seed=f1,mean_contact_f1_delta=float(np.mean(f1)),engine_effects=engine,opening_rms_delta_mm_by_seed=opening,well_timed_closure_delta_by_seed=closure)
        e=effects['feature_confirmation'];checks=dict(gap=e['mean_relative_gap_reduction']>=.03,contact=e['mean_contact_f1_delta']>=0,opening=np.mean(e['opening_rms_delta_mm_by_seed'])<=0,closures=np.mean(e['well_timed_closure_delta_by_seed'])>=0,engines=all(np.mean([x[k] for x in e['engine_effects']])>=0 for k in ['phonon','supertonic']))
        for cohort in ['scale_confirmation','historical']:
            baseline=stats('baseline',cohort)['macro_gap_rmse_mm']
            checks[cohort]=np.mean([stats(f'distill-s{s}',cohort)['macro_gap_rmse_mm'] for s in protocol['students']['seeds']])<=1.02*baseline
        summary.update(effects=effects,checks={k:bool(v) for k,v in checks.items()},promising=bool(all(checks.values())))
    write(OUT/('mentor-heldout-summary.json' if args.mentor_only else 'summary.json'),summary)
    print(json.dumps(summary,indent=2),flush=True)
if __name__=='__main__':main()
