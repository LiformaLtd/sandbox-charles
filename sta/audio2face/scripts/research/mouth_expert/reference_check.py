"""Official real-video SyncNet sanity example; kept distinct from James tests."""
from common import *
import argparse,torch
sys.path.insert(0,str(OUT/'vendor-deps'));sys.path.insert(0,str(VENDOR/'syncnet'))
from SyncNetInstance import SyncNetInstance

def main():
 torch.set_num_threads(8);net=SyncNetInstance(device='cpu');net.loadParameters(str(VENDOR/'syncnet_v2.model'));options=argparse.Namespace(tmp_dir=str(OUT/'sync-reference'),reference='example',batch_size=20,vshift=15);offset,confidence,distances=net.evaluate(options,str(VENDOR/'sync-example.avi'))
 result=dict(offset=int(offset),confidence=float(confidence),min_distance=float(distances.mean(axis=0).min()),source_expected_approximately=dict(offset=3,confidence=10.021,min_distance=5.353),video_sha256=sha(VENDOR/'sync-example.avi'));assert offset==3 and confidence>9
 freeze(OUT/'sync-real-reference.json',result);print(result)
if __name__=='__main__':main()
