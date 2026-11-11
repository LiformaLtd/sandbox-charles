import type { AvatarMesh } from './types';

export const INNER_UP = [78,191,80,81,82,13,312,311,310,415,308];
export const INNER_DOWN = [78,95,88,178,87,14,317,402,318,324,308];
const OUTER_UP = [61,185,40,39,37,0,267,269,270,409,291];
const OUTER_DOWN = [61,146,91,181,84,17,314,405,321,375,291];
const POINTS=478;

/** Hard lip contact with harmonic skin follow and pinned-boundary area repair. */
export class LipContactConstraint {
  private handles=Array.from(new Set([...INNER_UP,...INNER_DOWN,...OUTER_UP,...OUTER_DOWN]));
  private handleIndex=new Int16Array(POINTS).fill(-1);
  private free=new Uint8Array(POINTS);
  private influence:Float64Array;
  private movement:Float64Array;
  private baseline=new Float32Array(POINTS*2);
  private work=new Float64Array(POINTS*2);
  private curve=new Float64Array(22);
  private curveScratch=new Float64Array(22);
  private targetAreas:Float64Array;
  private referenceAreas:Float64Array;
  private triangles:number[]=[];
  unresolvedTriangles=0;
  constructor(mesh:AvatarMesh,reference:Float32Array){
    const count=this.handles.length;this.influence=new Float64Array(POINTS*count);this.movement=new Float64Array(count*2);
    this.handles.forEach((v,i)=>{this.handleIndex[v]=i;this.influence[v*count+i]=1;});
    const graph=Array.from({length:POINTS},()=>new Set<number>());
    for(let t=0;t<mesh.triangles.length;t+=3){const a=mesh.triangles[t],b=mesh.triangles[t+1],c=mesh.triangles[t+2];
      if(Math.max(a,b,c)>=POINTS)continue;
      for(const [u,v] of [[a,b],[b,c],[c,a]]){graph[u].add(v);graph[v].add(u);}
    }
    const distance=new Int16Array(POINTS).fill(1000),queue=this.handles.slice();for(const v of queue)distance[v]=0;
    for(let q=0;q<queue.length;q++){const v=queue[q];for(const n of graph[v])if(distance[n]>distance[v]+1){distance[n]=distance[v]+1;queue.push(n);}}
    const neighbors=graph.map((set,v)=>Array.from(set,n=>({n,w:1/Math.max(.005,Math.hypot(reference[v*2]-reference[n*2],reference[v*2+1]-reference[n*2+1]))**2})));
    for(let v=0;v<POINTS;v++)if(distance[v]>0&&distance[v]<5&&v<468)this.free[v]=1;
    for(let iteration=0;iteration<180;iteration++){
      let change=0;
      for(let v=0;v<POINTS;v++)if(this.free[v]){
        const list=neighbors[v];let sum=0;for(const e of list)sum+=e.w;
        for(let h=0;h<count;h++){let value=0;for(const e of list)value+=this.influence[e.n*count+h]*e.w;value/=sum;
          change=Math.max(change,Math.abs(value-this.influence[v*count+h]));this.influence[v*count+h]=value;}
      }
      if(change<1e-9)break;
    }
    for(let t=0;t<mesh.triangles.length;t+=3){const a=mesh.triangles[t],b=mesh.triangles[t+1],c=mesh.triangles[t+2];
      if(Math.max(a,b,c)<POINTS&&[a,b,c].some(v=>this.free[v]||this.handleIndex[v]>=0))this.triangles.push(a,b,c);
    }
    // Nearly collinear fitted triangles have no reliable geometric winding.
    // Infer it from adjacent nondegenerate triangles sharing an edge.
    const edgeFaces=new Map<string,{direction:number;area:number}[]>();
    for(let t=0;t<mesh.triangles.length;t+=3){const a=mesh.triangles[t],b=mesh.triangles[t+1],c=mesh.triangles[t+2];
      if(Math.max(a,b,c)>=POINTS)continue;
      const area=(reference[b*2]-reference[a*2])*(reference[c*2+1]-reference[a*2+1])-(reference[b*2+1]-reference[a*2+1])*(reference[c*2]-reference[a*2]);
      if(Math.abs(area)<=1e-8)continue;
      for(const [u,v] of [[a,b],[b,c],[c,a]]){const key=`${Math.min(u,v)}:${Math.max(u,v)}`;const list=edgeFaces.get(key)??[];list.push({direction:u<v?1:-1,area});edgeFaces.set(key,list);}
    }
    this.targetAreas=new Float64Array(this.triangles.length/3);
    this.referenceAreas=new Float64Array(this.triangles.length/3);
    for(let t=0;t<this.triangles.length;t+=3){const a=this.triangles[t]*2,b=this.triangles[t+1]*2,c=this.triangles[t+2]*2;
      this.referenceAreas[t/3]=(reference[b]-reference[a])*(reference[c+1]-reference[a+1])-(reference[b+1]-reference[a+1])*(reference[c]-reference[a]);
      if(Math.abs(this.referenceAreas[t/3])<=1e-8){
        let vote=0;for(const [u,v] of [[a/2,b/2],[b/2,c/2],[c/2,a/2]]){
          for(const neighbor of edgeFaces.get(`${Math.min(u,v)}:${Math.max(u,v)}`)??[])vote-=neighbor.area*neighbor.direction*(u<v?1:-1);
        }
        if(vote)this.referenceAreas[t/3]=Math.sign(vote)*1e-6;
      }
    }
  }
  apply(output:Float32Array,amount:number){
    this.unresolvedTriangles=0;if(amount<=1e-6)return;
    const count=this.handles.length,base=this.baseline,points=this.work; base.set(output);points.set(output);this.movement.fill(0);
    const curve=this.curve;
    for(let p=0;p<11;p++)for(let axis=0;axis<2;axis++)curve[p*2+axis]=(base[INNER_UP[p]*2+axis]+base[INNER_DOWN[p]*2+axis])/2;
    // Reject isolated fit noise while retaining a shared, curved seal.
    for(let iteration=0;iteration<2;iteration++){
      this.curveScratch.set(curve);
      for(let p=1;p<10;p++)for(let axis=0;axis<2;axis++)curve[p*2+axis]=this.curveScratch[(p-1)*2+axis]*.2+this.curveScratch[p*2+axis]*.6+this.curveScratch[(p+1)*2+axis]*.2;
    }
    for(let p=1;p<10;p++){
      const tx=curve[(p+1)*2]-curve[(p-1)*2],ty=curve[(p+1)*2+1]-curve[(p-1)*2+1],length=Math.hypot(tx,ty)||1;
      const nx=tx/length,ny=ty/length;
      for(let side=0;side<2;side++){
        const inner=side?INNER_DOWN:INNER_UP,outer=side?OUTER_DOWN:OUTER_UP,i=inner[p],o=outer[p];
        const ox=base[inner[p+1]*2]-base[inner[p-1]*2],oy=base[inner[p+1]*2+1]-base[inner[p-1]*2+1],oldLength=Math.hypot(ox,oy)||1;
        const ux=ox/oldLength,uy=oy/oldLength,dx=base[o*2]-base[i*2],dy=base[o*2+1]-base[i*2+1];
        const tangent=dx*ux+dy*uy,normal=-dx*uy+dy*ux;
        const rimX=tangent*nx-normal*ny,rimY=tangent*ny+normal*nx;
        this.movement[this.handleIndex[i]*2]=(curve[p*2]-base[i*2])*amount;
        this.movement[this.handleIndex[i]*2+1]=(curve[p*2+1]-base[i*2+1])*amount;
        this.movement[this.handleIndex[o]*2]=(curve[p*2]+rimX-base[o*2])*amount;
        this.movement[this.handleIndex[o]*2+1]=(curve[p*2+1]+rimY-base[o*2+1])*amount;
      }
    }
    for(let v=0;v<POINTS;v++)if(this.free[v]||this.handleIndex[v]>=0){
      for(let axis=0;axis<2;axis++){let shift=0;for(let h=0;h<count;h++)shift+=this.influence[v*count+h]*this.movement[h*2+axis];points[v*2+axis]+=shift;}
    }
    // Set full-contact handles from a single rounded midpoint: exact equality.
    if(amount===1)for(let p=1;p<10;p++)for(let axis=0;axis<2;axis++){
      const up=INNER_UP[p],down=INNER_DOWN[p];const mid=Math.fround(curve[p*2+axis]);
      points[up*2+axis]=mid;points[down*2+axis]=mid;
    }
    for(let t=0;t<this.triangles.length;t+=3){const a=this.triangles[t]*2,b=this.triangles[t+1]*2,c=this.triangles[t+2]*2;
      const raw=(base[b]-base[a])*(base[c+1]-base[a+1])-(base[b+1]-base[a+1])*(base[c]-base[a]);
      const reference=this.referenceAreas[t/3];
      this.targetAreas[t/3]=Math.sign(reference)*Math.max(Math.abs(raw),Math.abs(reference)*.01);
    }
    // Contact/rim handles remain pinned. Only surrounding skin can relax.
    // Double precision avoids stalled repairs on nearly collinear fitted cells.
    for(let iteration=0;iteration<1024;iteration++){
      let violations=0;
      for(let t=0;t<this.triangles.length;t+=3){
        const ai=this.triangles[t],bi=this.triangles[t+1],ci=this.triangles[t+2],a=ai*2,b=bi*2,c=ci*2;
        const rest=this.targetAreas[t/3];if(Math.abs(rest)<1e-9)continue;
        const sign=Math.sign(rest),area=((points[b]-points[a])*(points[c+1]-points[a+1])-(points[b+1]-points[a+1])*(points[c]-points[a]))*sign;
        const wanted=1e-7;if(area>=wanted-1e-9)continue;violations++;
        const ax=(points[b+1]-points[c+1])*sign,ay=(points[c]-points[b])*sign;
        const bx=(points[c+1]-points[a+1])*sign,by=(points[a]-points[c])*sign;
        const cx=(points[a+1]-points[b+1])*sign,cy=(points[b]-points[a])*sign;
        const denom=this.free[ai]*(ax*ax+ay*ay)+this.free[bi]*(bx*bx+by*by)+this.free[ci]*(cx*cx+cy*cy);
        if(denom<1e-14)continue;const scale=(wanted-area)/denom;
        if(this.free[ai]){points[a]+=ax*scale;points[a+1]+=ay*scale;}
        if(this.free[bi]){points[b]+=bx*scale;points[b+1]+=by*scale;}
        if(this.free[ci]){points[c]+=cx*scale;points[c+1]+=cy*scale;}
      }
      if(!violations)break;
    }
    output.set(points);points.set(output);
    for(let t=0;t<this.triangles.length;t+=3){const a=this.triangles[t]*2,b=this.triangles[t+1]*2,c=this.triangles[t+2]*2;
      const rest=this.targetAreas[t/3];const area=(points[b]-points[a])*(points[c+1]-points[a+1])-(points[b+1]-points[a+1])*(points[c]-points[a]);
      if(Math.abs(rest)>1e-9&&area*Math.sign(rest)<-1e-8)this.unresolvedTriangles++;
    }
  }
}
