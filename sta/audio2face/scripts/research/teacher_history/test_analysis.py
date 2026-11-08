"""Unit conventions, pooled RMS, time bins and frozen-artifact integrity."""
import unittest
import numpy as np
from common import *
from summarize import metrics, aggregate

class AnalysisChecks(unittest.TestCase):
 def test_units_and_euclidean_surface(self):
  result=metrics(np.ones((2,3,3)),np.ones((2,9)))
  self.assertAlmostEqual(result['gap_rms_mm'],10)
  self.assertAlmostEqual(result['surface_rms_mm'],10*np.sqrt(3))
 def test_pool_squared_errors(self):
  rows=[dict(surface_rms_mm=v,gap_rms_mm=v) for v in (0,2)]
  self.assertAlmostEqual(aggregate(rows)['gap_rms_mm'],np.sqrt(2))
 def test_bins(self):
  p=json.loads((OUT/'protocol.json').read_text());t=np.arange(225)/60
  counts=[int(((t>=lo)&(t<hi)).sum()) for lo,hi in p['bins']+[p['primary']]]
  self.assertEqual(counts,[15,30,30,60,60,180])
 def test_frozen_integrity(self):
  p=json.loads((OUT/'protocol.json').read_text());done=json.loads((OUT/'completion.json').read_text())
  self.assertEqual(sha(BASELINE),EXPECTED)
  self.assertEqual(sha(OUT/'cases.json'),p['cases_sha256'])
  for name,digest in p['source_sha256'].items():self.assertEqual(sha(APP/name),digest,name)
  for name,digest in done['files_sha256'].items():self.assertEqual(sha(OUT/name),digest,name)
  self.assertEqual(len(list((OUT/'tracks').glob('*/*.npz'))),304)
  self.assertEqual(len(list((OUT/'tracks').glob('*/student.npy'))),16)
  self.assertEqual(len(done['checks']),16)
  for row in done['checks']:
   for key in ('windows_identical','repeat_identical','reset_identical','student_identical'):self.assertTrue(row[key])
  for case in json.loads((OUT/'cases.json').read_text()):
   for item in case['audio'].values():self.assertEqual(sha(item['path']),item['sha256'])
  self.assertEqual(len(set(done['noise_sha256'].values())),4)

if __name__=='__main__':unittest.main()
