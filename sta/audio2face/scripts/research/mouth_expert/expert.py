"""Frozen official AVSR expert; no Lightning/training wrapper or CUDA required."""
from common import *
import argparse,re,yaml,numpy as np,torch,sentencepiece
sys.path.insert(0,str(VENDOR/'auto_avsr'))
from espnet.nets.pytorch_backend.e2e_asr_conformer_av import E2E

class Expert:
 def __init__(self):
  torch.set_num_threads(8);torch.manual_seed(37)
  args=argparse.Namespace(**yaml.safe_load((VENDOR/'auto_avsr/configs/model/audiovisual_backbone/resnet_conformer.yaml').read_text()));weights=torch.load(VENDOR/'lipreader.ckpt',map_location='cpu',weights_only=True)
  self.net=E2E(weights['ctc.ctc_lo.weight'].shape[0],args).eval();self.net.load_state_dict(weights,strict=True);self.net.requires_grad_(False)
  self.sp=sentencepiece.SentencePieceProcessor(model_file=str(VENDOR/'auto_avsr/spm/unigram/unigram5000.model'))
  units=[line.split() for line in (VENDOR/'auto_avsr/spm/unigram/unigram5000_units.txt').read_text().splitlines()];self.tokens={token:int(number) for token,number in units}
 def target(self,text):
  # Identical piece-to-ID mapping to Auto-AVSR TextTransform.
  return torch.tensor([[self.tokens.get(s,self.tokens['<unk>']) for s in self.sp.encode(re.sub('[^A-Z]', ' ', text.upper()).strip(),out_type=str)]],dtype=torch.long)
 def video(self,frames,std=.4):
  import cv2
  x=np.stack([cv2.resize(f[48:176,48:176],(88,88),interpolation=cv2.INTER_LINEAR) for f in frames]).astype(np.float32)/255
  gray=.2989*x[:,:,:,0]+.587*x[:,:,:,1]+.114*x[:,:,:,2]
  return torch.from_numpy(((gray-.421)/std)[:,None])[None]
 def audio(self,pcm,count):
  y=np.zeros(count*640,np.float32);y[:min(len(y),len(pcm))]=pcm[:len(y)];t=torch.from_numpy(y[:,None]);return torch.nn.functional.layer_norm(t,t.shape,eps=1e-8)[None]
 @torch.inference_mode()
 def score(self,frames,pcm,text,std=.4):
  v=self.video(frames,std);a=self.audio(pcm,len(frames));target=self.target(text)
  output=self.net(v,a,torch.tensor([v.shape[1]]),torch.tensor([a.shape[1]]),target)
  return dict(loss=float(output[0]),ctc=float(output[1]),attention=float(output[2]),token_accuracy=float(output[3]))
