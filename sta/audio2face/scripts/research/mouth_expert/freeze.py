from common import *
import subprocess

def main():
 sources=[HERE/n for n in ['common.py','prepare.py','expert.py','score.py','render.html','render.mjs','freeze.py']]+[APP/'docs/plans/mouth-expert-compatibility.md',APP/'src/rig.js',APP/'src/blendshape-rig.js',APP/'scripts/research/multivoice/closure_metrics.py']
 weights={n:sha(VENDOR/n) for n in ['lipreader.ckpt','syncnet_v2.model']};files={str(p.relative_to(OUT)):sha(p) for name in ['tracks','audio'] for p in (OUT/name).iterdir()}
 for p in sources:freeze(OUT/'sources/frozen'/p.relative_to(APP),p.read_bytes())
 versions={n:subprocess.check_output(['git','rev-parse','HEAD'],cwd=VENDOR/n,text=True).strip() for n in ['av-guidance','auto_avsr','syncnet']}
 freeze(OUT/'protocol.json',dict(sources={str(p.relative_to(APP)):sha(p) for p in sources},weights=weights,git_commits=versions,case_files_sha256=files,cases_sha256=sha(OUT/'cases.json'),rig_sha256={n:sha(APP/'.local'/n) for n in ['rig.bin','rig.json','arkit-student/james-blendshapes.bin']},render_setup='ROI inspected on a non-scored calibration frame; no expert James score used for selection.'))
 print('Frozen cases, geometric corruptions, expert checkpoints, sources and protocol.',flush=True)
if __name__=='__main__':main()
