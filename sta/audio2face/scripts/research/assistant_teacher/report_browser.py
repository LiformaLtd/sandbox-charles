"""Summarize independent browser trials without treating timing noise as a win."""
from common import *
import numpy as np

def main():
    result={}
    for arm in ['v4','control','distilled']:
        timing=[];parity=[];memory=[];allocation=[]
        for trial in range(1,4):
            rows=json.loads((OUT/f'browser/parity/run-{trial}.json').read_text());take=[r for r in rows if r['model']==arm and r['variant']=='fixed'];timing.append(next(r['median_ms'] for r in take if r['name']=='phonon-packed'));parity.extend(r['max'] for r in take)
            record=json.loads((OUT/f'memory/run-{trial}'/(arm+'.json')).read_text());memory.append(record['peak']/2**20);allocation.append(record['allocation']);assert record['drawEnd']>record['drawStart'];assert not record['errors']
        result[arm]=dict(long_block_median_ms_trials=timing,median_ms=float(np.median(timing)),nominal_500ms_realtime=float(500/np.median(timing)),max_coefficient_delta=max(parity),peak_physical_mib_trials=memory,allocation=allocation)
    assert all(a==result['v4']['allocation'][0] for r in result.values() for a in r['allocation'])
    freeze(OUT/'browser-summary.json',dict(models=result,scope='Desktop WebKit, M5 Max64GiB, normal animated James with accelerated prerecorded STA; TTS, initial rig loading, network and browser parent excluded. Identical graph; small timing/footprint differences do not prove speed/memory gains.'))
    print(json.dumps(result,indent=2))

if __name__=='__main__':main()
