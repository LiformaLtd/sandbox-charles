"""Offline differentiable James geometry; distinct from the browser renderer."""
from common import *
import numpy as np,struct,torch
from torch.nn import functional as F
from pytorch3d.structures import Meshes
from pytorch3d.renderer.blending import softmax_rgb_blend
class Unlit(torch.nn.Module):
 def forward(self,fragments,meshes,**kwargs):return softmax_rgb_blend(meshes.sample_textures(fragments),fragments,BlendParams(sigma=1e-3,gamma=1e-2,background_color=(0,0,0)),znear=.1,zfar=300)
from pytorch3d.renderer import MeshRenderer,MeshRasterizer,SoftPhongShader,PerspectiveCameras,look_at_view_transform,RasterizationSettings,PointLights,TexturesVertex,BlendParams,Materials

def smooth(x):return np.clip(x,0,1)**2*(3-2*np.clip(x,0,1))
def packed():
 data=(APP/'.local/arkit-student/james-blendshapes.bin').read_bytes();offset=4;groups=[]
 for n in [24002,5602,5]:
  b=np.zeros((53,n,3),np.float32);count=struct.unpack_from('<I',data,offset)[0];offset+=4
  for _ in range(count):
   channel,npoints,scale=struct.unpack_from('<IIf',data,offset);offset+=12;p=np.frombuffer(data,dtype='<i2',count=npoints*4,offset=offset).reshape(-1,4);offset+=npoints*8;b[channel,p[:,0]]=p[:,1:]*scale
  groups.append(b)
 assert offset==len(data);return groups

def cluster(v,b,faces,color,skin=False):
 # Preserve every original mouth vertex; cluster static outside regions only.
 protected=(abs(v[:,0])<3.5)&(v[:,1]>165.5)&(v[:,1]<169.5)&(v[:,2]>3) if skin else np.zeros(len(v),bool)
 grid=np.floor(v/(.2 if skin else .1)).astype(np.int64);keys=np.column_stack((grid,np.where(protected,np.arange(len(v))+1,0)));_,index=np.unique(keys,axis=0,return_inverse=True);n=index.max()+1;counts=np.bincount(index);out=np.zeros((n,3),np.float32);np.add.at(out,index,v);out/=counts[:,None];basis=np.zeros((53,n,3),np.float32)
 for c in range(53):np.add.at(basis[c],index,b[c]);basis[c]/=counts[:,None]
 colors=np.zeros((n,3),np.float32);np.add.at(colors,index,color);colors/=counts[:,None]
 f=index[faces];f=f[(f[:,0]!=f[:,1])&(f[:,1]!=f[:,2])&(f[:,0]!=f[:,2])];_,unique=np.unique(np.sort(f,axis=1),axis=0,return_index=True);f=f[np.sort(unique)]
 # Keep neutral-facing front/side surfaces, with a wide crop margin.
 p=out[f];normal=np.cross(p[:,1]-p[:,0],p[:,2]-p[:,0]);f=f[normal[:,2]>=-.15*np.linalg.norm(normal,axis=-1)]
 return out,basis,f,colors

class Renderer:
 def __init__(self,size=80):
  self.size=size;meta=json.loads((APP/'.local/rig.json').read_text());data=(APP/'.local/rig.bin').read_bytes();skin,tongue,jaw=packed();parts=[];offset=0
  for name,item in meta['meshes'].items():
   v=np.frombuffer(data,'<f4',count=item['vertexCount']*3,offset=item['positionOffset']).reshape(-1,3).copy();faces=np.frombuffer(data,'<u4',count=item['indexCount'],offset=item['indexOffset']).reshape(-1,3).astype(np.int64);b=skin if name=='skin' else tongue if name=='tongue' else np.zeros((53,len(v),3),np.float32)
   if name=='skin':
    outer=np.maximum(smooth((abs(v[:,0])-4)/4),smooth((-v[:,2]-8)/5));b=b.copy()*smooth((v[:,1]-157-4*outer)/5)[None,:,None]
   rgb=np.array([.725,.518,.416] if name=='skin' else [.525,.235,.38] if name=='tongue' else [.85,.79,.70] if name.endswith('Teeth') else [.82,.80,.77],np.float32);color=np.tile(rgb,(len(v),1))
   if name.endswith('Eye'):
    pivot=np.array(item['pivot']);rad=np.linalg.norm(v[:,:2]-pivot[:2],axis=-1);color[(v[:,2]>=pivot[2])&(rad<=.58)]=[.41,.39,.30];color[(v[:,2]>=pivot[2])&(rad<=.24)]=[.09,.08,.09]
   v,b,faces,color=cluster(v,b,faces,color,name=='skin');parts.append((name,torch.from_numpy(v),torch.from_numpy(b),torch.from_numpy(faces),torch.from_numpy(color)));offset+=len(v)
  self.parts=parts;self.jaw=torch.from_numpy(jaw);self.neutral_jaw=torch.tensor(meta['jaw']);self.neutral_basis=self.frame(self.neutral_jaw[None])[0];self.neutral_center=self.neutral_jaw.mean(0)
  R,T=look_at_view_transform(eye=((0,1.4,69),),at=((0,0,-2),),up=((0,1,0),));scale=1.;focal=(112/np.tan(np.deg2rad(17)))*(224/160)
  self.camera=PerspectiveCameras(focal_length=((focal,focal),),principal_point=((40.,22.4),),R=R,T=T,image_size=((size,size),),in_ndc=False)
  settings=RasterizationSettings(image_size=size,blur_radius=float(np.log(1/1e-3-1)*1e-3),faces_per_pixel=16,bin_size=0,cull_backfaces=False,perspective_correct=True)
  lights=PointLights(location=((0,15,50),),ambient_color=((.60,.60,.60),),diffuse_color=((.50,.50,.50),),specular_color=((.04,.04,.04),));materials=Materials(shininess=32)
  self.photo=torch.from_numpy(np.load(OUT/'neutral-bgr.npy')).permute(2,0,1).float()[None]
  kept=[]
  for name,v,b,f,c in self.parts:
   if name.endswith('Eye'):continue
   screen=self.camera.transform_points_screen((v-torch.tensor([0,169,0]))[None])[0,:,:2];full=screen+torch.tensor([72,112]);grid=(full/223*2-1).reshape(1,-1,1,2)
   if name=='skin':c=F.grid_sample(self.photo[:,[2,1,0]]/255,grid,align_corners=True,padding_mode='border')[0,:,:,0].T
   tri=screen[f];lo=tri.amin(1);hi=tri.amax(1);take=(lo[:,0]<size+32)&(hi[:,0]>-32)&(lo[:,1]<size+32)&(hi[:,1]>-32);kept.append((name,v,b,f[take],c))
  clean=[]
  for name,v,b,f,c in kept:
   used=torch.unique(f.flatten());used=used[v[used,2]>0];mapping=torch.full((len(v),),-1,dtype=torch.long);mapping[used]=torch.arange(len(used));f=mapping[f];f=f[(f>=0).all(1)];v,b,c=v[used],b[:,used],c[used]
   if name=='skin':
    region=(v[:,0].abs()<3.5)&(v[:,1]>165.5)&(v[:,1]<169.5)&(v[:,2]>3);tri=f[region[f].all(1)];v=torch.cat((v,v[tri].mean(1)));b=torch.cat((b,b[:,tri].mean(2)),1);c=torch.cat((c,c[tri].mean(1)))
   clean.append((name,v,b,f,c))
  self.parts=clean
  self.renderer=MeshRenderer(MeshRasterizer(cameras=self.camera,raster_settings=settings),Unlit())
  self.photo=torch.from_numpy(np.load(OUT/'neutral-bgr.npy')).permute(2,0,1).float()[None]
  axis=torch.minimum(torch.arange(size),torch.arange(size).flip(0)).float();ramp=torch.clamp(axis/6,0,1);self.feather=ramp[None,None,:,None]*ramp[None,None,None,:]

 def frame(self,p):
  horizontal=F.normalize(p[:,1]-p[:,0],dim=-1);rear=(p[:,0]+p[:,1])/2;front=F.normalize(p[:,4]-rear,dim=-1);up=F.normalize(torch.cross(front,horizontal,dim=-1),dim=-1);front=torch.cross(horizontal,up,dim=-1);return torch.stack((horizontal,up,front),-1)
 def raw(self,weights):
  weights=weights.double()
  n=len(weights);ones=torch.ones((n,1),dtype=weights.dtype);w=torch.cat((weights,ones),1);jaw=self.neutral_jaw+torch.einsum('nc,cvd->nvd',w,self.jaw.double());rotation=self.frame(jaw)@self.neutral_basis.double().T;position=jaw.mean(1)-torch.einsum('d,nbd->nb',self.neutral_center.double(),rotation);verts=[];faces=[];colors=[];radii=[];offset=0
  for name,v,b,f,c in self.parts:
   p=v[None]+torch.einsum('nc,cvd->nvd',w,b.double())
   if name=='lowerTeeth':p=torch.einsum('nvd,nbd->nvb',p,rotation)+position[:,None]
   region=(v[:,0].abs()<3.5)&(v[:,1]>165.5)&(v[:,1]<169.5)&(v[:,2]>3)
   radii.append(torch.where(region,1.1,3.) if name=='skin' else torch.full((len(v),),1.5))
   p=p-torch.tensor([0,169,0]);verts.append(p);faces.append(f+offset);colors.append(c);offset+=len(v)
  v=torch.cat(verts,1);c=torch.cat(colors);radius_values=torch.cat(radii);view=v@self.camera.R[0].double()+self.camera.T[0].double();depth=view[...,2];screen=torch.stack((40.-self.camera.focal_length[0,0].double()*view[...,0]/depth,22.4-self.camera.focal_length[0,1].double()*view[...,1]/depth),-1)
  images=[]
  for xy,z in zip(screen,depth):
   # Compact C1 kernels: bucket changes occur strictly outside zero support.
   radius=3.;points=xy.detach().numpy();base=np.floor(points).astype(np.int64);delta=np.array([(x,y) for y in range(-3,5) for x in range(-3,5)])
   pixels=base[:,None,:]+delta[None];ids=np.broadcast_to(np.arange(len(points))[:,None],pixels.shape[:2]);ok=(pixels[...,0]>=0)&(pixels[...,0]<self.size)&(pixels[...,1]>=0)&(pixels[...,1]<self.size)
   ids=torch.from_numpy(ids[ok].copy());pixel=pixels[ok];flat=torch.from_numpy((pixel[:,1]*self.size+pixel[:,0]).copy());grid=torch.from_numpy(pixel.copy()).double()+.5
   dist=((xy[ids]-grid)**2).sum(-1);kernel=torch.clamp(1-dist/radius_values[ids].square(),min=0)**2;valid=kernel>0
   # Ragged per-pixel reductions avoid allocating the largest bucket for every pixel.
   logits=torch.where(valid,torch.log(kernel.clamp_min(1e-30))-z[ids]/.05,-1e9);maximum=torch.full((self.size*self.size,),-1e9,dtype=xy.dtype);maximum.scatter_reduce_(0,flat,logits.detach(),reduce='amax',include_self=True)
   value=torch.exp(logits-maximum[flat])*valid;denominator=torch.zeros_like(maximum).index_add(0,flat,value);numerator=torch.zeros((self.size*self.size,3),dtype=xy.dtype).index_add(0,flat,value[:,None]*c[ids]);coverage=1-torch.exp(-3*torch.zeros_like(maximum).index_add(0,flat,kernel));rgb=numerator/denominator.clamp_min(1e-30)[:,None]*coverage[:,None]
   images.append(rgb.reshape(self.size,self.size,3).permute(2,0,1))
  return torch.stack(images)[:,[2,1,0]]*255
 def __call__(self,weights):
  patch=(self.raw(weights)-self.photo[:,:,112:192,72:152])*self.feather
  return torch.clamp(self.photo+F.pad(patch,(72,72,112,32)),0,255).to(weights.dtype)
