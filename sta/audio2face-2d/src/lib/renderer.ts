import earcut from 'earcut';
import { INNER_UP,INNER_DOWN,POINTS } from './shape';
import type { AvatarMesh,Crop } from './types';

const VERTEX=`attribute vec2 a_position; attribute vec2 a_uv; uniform vec4 u_crop; varying vec2 v_uv;
void main(){ vec2 p=(a_position-u_crop.xy)/u_crop.zw;gl_Position=vec4(p.x*2.-1.,1.-p.y*2.,0.,1.);v_uv=a_uv;gl_PointSize=3.;}`;
const FRAGMENT=`precision mediump float;varying vec2 v_uv;uniform sampler2D u_texture;uniform int u_solid;uniform vec4 u_color;
void main(){gl_FragColor=u_solid==1?u_color:texture2D(u_texture,v_uv);}`;

export class FaceRenderer {
  private gl: WebGLRenderingContext;
  private program: WebGLProgram;
  private texture: WebGLTexture;
  private buffers: WebGLBuffer[]=[];
  private source: WebGLBuffer;
  private destination: WebGLBuffer;
  private indices: WebGLBuffer;
  private wire: WebGLBuffer;
  private background: WebGLBuffer;
  private quad: WebGLBuffer;
  private mouth: WebGLBuffer;
  private mouthIndices: WebGLBuffer;
  private mouthUV: WebGLBuffer;
  private positionLocation: number;
  private uvLocation: number;
  private cropLocation: WebGLUniformLocation | null;
  private solidLocation: WebGLUniformLocation | null;
  private colorLocation: WebGLUniformLocation | null;
  private edges: Uint16Array<ArrayBuffer>;
  private mouthPoints=new Float32Array(20*2);
  private mouthList=[...INNER_UP,...INNER_DOWN.slice(1,-1).reverse()];
  private mouthCoordinates:number[]=[];
  private imageWidth=1;private imageHeight=1;
  private destroyed=false;private image:HTMLImageElement|undefined;

  constructor(readonly canvas:HTMLCanvasElement,readonly mesh:AvatarMesh) {
    const gl=canvas.getContext('webgl',{ alpha:false,antialias:true,premultipliedAlpha:false });
    if(!gl)throw new Error('This browser could not create a WebGL canvas.');
    this.gl=gl;
    const compile=(type:number,code:string)=>{
      const shader=gl.createShader(type);if(!shader)throw new Error('Could not allocate a mesh shader.');
      gl.shaderSource(shader,code);gl.compileShader(shader);
      if(!gl.getShaderParameter(shader,gl.COMPILE_STATUS)){const message=gl.getShaderInfoLog(shader);gl.deleteShader(shader);throw new Error(`Mesh shader failed: ${message}`);}
      return shader;
    };
    const program=gl.createProgram();if(!program)throw new Error('Could not create the mesh program.');
    const vs=compile(gl.VERTEX_SHADER,VERTEX);const fs=compile(gl.FRAGMENT_SHADER,FRAGMENT);
    gl.attachShader(program,vs);gl.attachShader(program,fs);gl.linkProgram(program);gl.deleteShader(vs);gl.deleteShader(fs);
    if(!gl.getProgramParameter(program,gl.LINK_STATUS))throw new Error('Could not link the mesh shaders.');
    this.program=program;gl.useProgram(program);
    this.positionLocation=gl.getAttribLocation(program,'a_position');this.uvLocation=gl.getAttribLocation(program,'a_uv');
    this.cropLocation=gl.getUniformLocation(program,'u_crop');this.solidLocation=gl.getUniformLocation(program,'u_solid');this.colorLocation=gl.getUniformLocation(program,'u_color');
    const buffer=(target:number,data:BufferSource,usage:number=gl.STATIC_DRAW)=>{
      const value=gl.createBuffer();if(!value)throw new Error('Could not allocate a mesh buffer.');
      gl.bindBuffer(target,value);gl.bufferData(target,data,usage);this.buffers.push(value);return value;
    };
    this.source=buffer(gl.ARRAY_BUFFER,new Float32Array([...mesh.sourceUV,...mesh.ring]));
    this.destination=buffer(gl.ARRAY_BUFFER,new Float32Array((POINTS+36)*2),gl.DYNAMIC_DRAW);
    this.indices=buffer(gl.ELEMENT_ARRAY_BUFFER,new Uint16Array(mesh.triangles));
    const seen=new Set<string>();const edges:number[]=[];
    for(let i=0;i<mesh.triangles.length;i+=3){
      const t=mesh.triangles.slice(i,i+3);
      for(const [a,b] of [[t[0],t[1]],[t[1],t[2]],[t[2],t[0]]]){
        const key=`${Math.min(a,b)}:${Math.max(a,b)}`;if(!seen.has(key)){seen.add(key);edges.push(a,b);}
      }
    }
    this.edges=new Uint16Array(edges);this.wire=buffer(gl.ELEMENT_ARRAY_BUFFER,this.edges);
    this.background=buffer(gl.ARRAY_BUFFER,new Float32Array([0,0,1,0,1,1,0,1]));
    this.quad=buffer(gl.ELEMENT_ARRAY_BUFFER,new Uint16Array([0,1,2,0,2,3]));
    this.mouth=buffer(gl.ARRAY_BUFFER,this.mouthPoints,gl.DYNAMIC_DRAW);
    this.mouthUV=buffer(gl.ARRAY_BUFFER,this.mouthPoints);
    this.mouthIndices=buffer(gl.ELEMENT_ARRAY_BUFFER,new Uint16Array(54),gl.DYNAMIC_DRAW);
    const texture=gl.createTexture();if(!texture)throw new Error('Could not allocate the neutral texture.');this.texture=texture;
    gl.bindTexture(gl.TEXTURE_2D,texture);gl.texParameteri(gl.TEXTURE_2D,gl.TEXTURE_MIN_FILTER,gl.LINEAR);
    gl.texParameteri(gl.TEXTURE_2D,gl.TEXTURE_MAG_FILTER,gl.LINEAR);
    gl.texParameteri(gl.TEXTURE_2D,gl.TEXTURE_WRAP_S,gl.CLAMP_TO_EDGE);gl.texParameteri(gl.TEXTURE_2D,gl.TEXTURE_WRAP_T,gl.CLAMP_TO_EDGE);
    gl.enable(gl.BLEND);gl.blendFunc(gl.SRC_ALPHA,gl.ONE_MINUS_SRC_ALPHA);
  }
  async load(signal:AbortSignal){
    const image=new Image();this.image=image;
    await new Promise<void>((resolve,reject)=>{
      const abort=()=>{image.src='';reject(new DOMException('Avatar load cancelled.','AbortError'));};
      signal.addEventListener('abort',abort,{once:true});
      const clean=()=>{signal.removeEventListener('abort',abort);image.onload=null;image.onerror=null;};
      image.onload=()=>{clean();resolve();};image.onerror=()=>{clean();reject(new Error('The neutral texture could not be loaded.'));};
      image.src=this.mesh.image;
    });
    signal.throwIfAborted();if(this.destroyed)return;
    this.imageWidth=image.naturalWidth;this.imageHeight=image.naturalHeight;
    const gl=this.gl;gl.bindTexture(gl.TEXTURE_2D,this.texture);gl.pixelStorei(gl.UNPACK_FLIP_Y_WEBGL,0);
    gl.texImage2D(gl.TEXTURE_2D,0,gl.RGBA,gl.RGBA,gl.UNSIGNED_BYTE,image);
  }
  draw(points:Float32Array,{wireframe,closeup,cavity}:{wireframe:boolean;closeup:boolean;cavity:boolean}){
    if(this.destroyed||!this.image)return;
    const gl=this.gl;const css=this.canvas.getBoundingClientRect();const ratio=Math.min(devicePixelRatio||1,2);
    const w=Math.max(1,Math.round(css.width*ratio));const h=Math.max(1,Math.round(css.height*ratio));
    if(this.canvas.width!==w||this.canvas.height!==h){this.canvas.width=w;this.canvas.height=h;}
    gl.viewport(0,0,w,h);gl.clearColor(.045,.059,.086,1);gl.clear(gl.COLOR_BUFFER_BIT);gl.useProgram(this.program);gl.bindTexture(gl.TEXTURE_2D,this.texture);
    let crop:Crop={...this.mesh.crop};
    if(closeup){
      const uv=this.mesh.sourceUV;const cx=(uv[61*2]+uv[291*2])/2;const cy=(uv[13*2+1]+uv[14*2+1])/2;
      const mouthWidth=Math.hypot(uv[291*2]-uv[61*2],uv[291*2+1]-uv[61*2+1]);
      crop={x:cx-mouthWidth*1.3,y:cy-mouthWidth*.9,w:mouthWidth*2.6,h:mouthWidth*1.8};
    }
    // Fit, never stretch, the cropped image region to the viewport.
    const physical=crop.w*this.imageWidth/(crop.h*this.imageHeight);const aspect=w/h;
    if(physical>aspect){const newHeight=crop.w*this.imageWidth/aspect/this.imageHeight;crop.y-=(newHeight-crop.h)/2;crop.h=newHeight;}
    else {const newWidth=crop.h*this.imageHeight*aspect/this.imageWidth;crop.x-=(newWidth-crop.w)/2;crop.w=newWidth;}
    gl.uniform4f(this.cropLocation,crop.x,crop.y,crop.w,crop.h);
    gl.uniform1i(this.solidLocation,0);
    this.attributes(this.background,this.background);gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER,this.quad);gl.drawElements(gl.TRIANGLES,6,gl.UNSIGNED_SHORT,0);
    if(cavity){
      this.mouthCoordinates.length=0;
      for(let i=0;i<this.mouthList.length;i++){
        const p=this.mouthList[i];this.mouthPoints[2*i]=points[2*p];this.mouthPoints[2*i+1]=points[2*p+1];
        this.mouthCoordinates.push(points[2*p],points[2*p+1]);
      }
      const indices=new Uint16Array(earcut(this.mouthCoordinates));
      gl.bindBuffer(gl.ARRAY_BUFFER,this.mouth);gl.bufferSubData(gl.ARRAY_BUFFER,0,this.mouthPoints);
      this.attributes(this.mouth,this.mouthUV);gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER,this.mouthIndices);gl.bufferData(gl.ELEMENT_ARRAY_BUFFER,indices,gl.DYNAMIC_DRAW);
      gl.uniform1i(this.solidLocation,1);gl.uniform4f(this.colorLocation,.075,.032,.04,1);gl.drawElements(gl.TRIANGLES,indices.length,gl.UNSIGNED_SHORT,0);
    }
    gl.bindBuffer(gl.ARRAY_BUFFER,this.destination);gl.bufferSubData(gl.ARRAY_BUFFER,0,points);
    this.attributes(this.destination,this.source);gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER,this.indices);
    gl.uniform1i(this.solidLocation,0);gl.drawElements(gl.TRIANGLES,this.mesh.triangles.length,gl.UNSIGNED_SHORT,0);
    if(wireframe){
      gl.uniform1i(this.solidLocation,1);gl.uniform4f(this.colorLocation,.33,.92,.77,.55);
      gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER,this.wire);gl.drawElements(gl.LINES,this.edges.length,gl.UNSIGNED_SHORT,0);
      gl.uniform4f(this.colorLocation,.71,1,.89,.9);gl.drawArrays(gl.POINTS,0,POINTS);
    }
  }
  private attributes(position:WebGLBuffer,uv:WebGLBuffer){
    const gl=this.gl;
    gl.bindBuffer(gl.ARRAY_BUFFER,position);gl.enableVertexAttribArray(this.positionLocation);gl.vertexAttribPointer(this.positionLocation,2,gl.FLOAT,false,0,0);
    gl.bindBuffer(gl.ARRAY_BUFFER,uv);gl.enableVertexAttribArray(this.uvLocation);gl.vertexAttribPointer(this.uvLocation,2,gl.FLOAT,false,0,0);
  }
  dispose(){
    this.destroyed=true;this.image=undefined;
    for(const buffer of this.buffers)this.gl.deleteBuffer(buffer);
    this.gl.deleteTexture(this.texture);this.gl.deleteProgram(this.program);
  }
}
