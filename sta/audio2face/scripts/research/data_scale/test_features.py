import json,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
import features


class MappedDatasetTests(unittest.TestCase):
    def test_cross_source_order_and_repeated_indices_preserve_targets(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp)
            for name,size,offset in [('a',2,0),('b',3,100)]:
                folder=root/'features'/name;folder.mkdir(parents=True)
                (folder/'complete.json').write_text(json.dumps(dict(voices=['first','second'])))
                for key,shape in [('mel',(64,101)),('weights',(30,39)),('gaps',(30,9)),('valid',(30,))]:
                    values=np.stack([np.full(shape,i+offset,np.float32) for i in range(size)])
                    np.save(folder/(key+'.npy'),values)
                np.save(folder/'voice.npy',np.arange(size)%2)
            with patch.object(features,'OUT',root):
                dataset=features.Dataset(['a','b']);batch=dataset.batch([4,0,2,4,1])
                np.testing.assert_array_equal(batch['weights'][:,0,0],[102,0,100,102,1])
                np.testing.assert_array_equal(batch['mel'][:,0,0],[102,0,100,102,1])
                batch['weights'][:]=999
                np.testing.assert_array_equal(dataset.batch([4,0])['weights'][:,0,0],[102,0])
                self.assertEqual(dataset.voices.tolist(),['first','second','first','second','first'])


if __name__=='__main__':unittest.main()
