import earcut from 'earcut';
import {INNER_UP,INNER_DOWN} from './shape';
import {OVAL} from './pose';
import type {FacePlacement} from './pose';
import type {NeutralGeometry} from './neutral-neck';
const VERTEX=`attribute vec3 a_position;attribute vec2 a_uv;attribute float a_alpha;uniform vec2 u_size;varying vec2 v_uv;varying float v_alpha;
void main(){gl_Position=vec4(a_position.x/u_size.x*2.-1.,1.-a_position.y/u_size.y*2.,clamp(a_position.z/u_size.x,-.9,.9),1.);v_uv=a_uv;v_alpha=a_alpha;}`;
const FRAGMENT=`precision mediump float;varying vec2 v_uv;varying float v_alpha;uniform sampler2D u_texture;uniform int u_mode;
void main(){vec4 color=u_mode==1?vec4(.075,.025,.035,1.):u_mode==2?vec4(.45,1.,.75,.65):u_mode==3?vec4(1.,.72,.3,.55):texture2D(u_texture,v_uv);
float alpha=color.a*(u_mode>=2?1.:v_alpha);gl_FragColor=vec4(color.rgb*alpha,alpha);}`;
type Surface={position:WebGLBuffer;uv:WebGLBuffer;alpha:WebGLBuffer;index:WebGLBuffer;wire:WebGLBuffer;triangles:Uint16Array<ArrayBuffer>;edges:Uint16Array<ArrayBuffer>};
const EYES=[[33,7,163,144,145,153,154,155,133,173,157,158,159,160,161,246],[263,249,390,373,374,380,381,382,362,398,384,385,386,387,388,466]];
export class FaceRenderer{
 readonly canvas=document.createElement('canvas');
 private gl:WebGLRenderingContext;private program:WebGLProgram;private texture:WebGLTexture;
 private face:Surface;private neutral:Surface|undefined;
 private posLocation:number;private uvLocation:number;private alphaLocation:number;
 private mouth=[...INNER_UP,...INNER_DOWN.slice(1,-1).reverse()];
 constructor(image:ImageBitmap,placement:FacePlacement,triangles:number[]){
  const gl=this.canvas.getContext('webgl',{alpha:true,antialias:true,premultipliedAlpha:true});
  if(!gl)throw new Error('WebGL is unavailable. Open this app in Chrome.');this.gl=gl;
  const shader=(type:number,source:string)=>{
   const s=gl.createShader(type)!;gl.shaderSource(s,source);gl.compileShader(s);
   if(!gl.getShaderParameter(s,gl.COMPILE_STATUS))throw new Error(gl.getShaderInfoLog(s)||'Shader compilation failed');return s;
  };
  const program=gl.createProgram()!,vs=shader(gl.VERTEX_SHADER,VERTEX),fs=shader(gl.FRAGMENT_SHADER,FRAGMENT);
  gl.attachShader(program,vs);gl.attachShader(program,fs);gl.linkProgram(program);gl.deleteShader(vs);gl.deleteShader(fs);
  if(!gl.getProgramParameter(program,gl.LINK_STATUS))throw new Error('Face shader linking failed');this.program=program;gl.useProgram(program);
  this.posLocation=gl.getAttribLocation(program,'a_position');this.uvLocation=gl.getAttribLocation(program,'a_uv');this.alphaLocation=gl.getAttribLocation(program,'a_alpha');
  const uv=Array.from(placement.uv);let cx=0,cy=0;
  for(const i of OVAL){cx+=uv[i*2]/36;cy+=uv[i*2+1]/36}
  for(const i of OVAL)uv.push(cx+(uv[i*2]-cx)*1.18,cy+(uv[i*2+1]-cy)*1.18);
  const alpha=new Float32Array(514);alpha.fill(1,0,478);
  this.face=this.surface(new Float32Array(uv),alpha,triangles);
  this.texture=gl.createTexture()!;gl.bindTexture(gl.TEXTURE_2D,this.texture);
  gl.texParameteri(gl.TEXTURE_2D,gl.TEXTURE_MIN_FILTER,gl.LINEAR);gl.texParameteri(gl.TEXTURE_2D,gl.TEXTURE_MAG_FILTER,gl.LINEAR);
  gl.texParameteri(gl.TEXTURE_2D,gl.TEXTURE_WRAP_S,gl.CLAMP_TO_EDGE);gl.texParameteri(gl.TEXTURE_2D,gl.TEXTURE_WRAP_T,gl.CLAMP_TO_EDGE);
  gl.pixelStorei(gl.UNPACK_FLIP_Y_WEBGL,0);gl.texImage2D(gl.TEXTURE_2D,0,gl.RGBA,gl.RGBA,gl.UNSIGNED_BYTE,image);
 }
 private surface(uv:Float32Array,alpha:Float32Array,triangles:number[]):Surface{
  const gl=this.gl;
  const buffer=(target:number,data:BufferSource,usage:number=gl.STATIC_DRAW)=>{const b=gl.createBuffer()!;gl.bindBuffer(target,b);gl.bufferData(target,data,usage);return b};
  const list=new Uint16Array(triangles),edges:number[]=[];
  for(let t=0;t<triangles.length;t+=3)edges.push(triangles[t],triangles[t+1],triangles[t+1],triangles[t+2],triangles[t+2],triangles[t]);
  const lines=new Uint16Array(edges);
  return {position:buffer(gl.ARRAY_BUFFER,new Float32Array(alpha.length*3),gl.DYNAMIC_DRAW),uv:buffer(gl.ARRAY_BUFFER,new Float32Array(uv)),alpha:buffer(gl.ARRAY_BUFFER,new Float32Array(alpha)),
   index:buffer(gl.ELEMENT_ARRAY_BUFFER,list),wire:buffer(gl.ELEMENT_ARRAY_BUFFER,lines),triangles:list,edges:lines};
 }
 setNeutralGeometry(geometry:NeutralGeometry){
  if(this.neutral)this.release(this.neutral);
  const triangles=geometry.triangles.slice();
  // The neutral pass fills texture holes; the old moving mouth/eyes cannot show through.
  for(const loop of [this.mouth,...EYES]){
   const polygon=loop.flatMap(i=>[geometry.uv[i*2],geometry.uv[i*2+1]]);
   triangles.push(...earcut(polygon).map(i=>loop[i]));
  }
  this.neutral=this.surface(geometry.uv,geometry.alpha,triangles);
 }
 private bind(surface:Surface,points:Float32Array){
  const gl=this.gl,attribute=(buffer:WebGLBuffer,location:number,size:number)=>{
   gl.bindBuffer(gl.ARRAY_BUFFER,buffer);gl.enableVertexAttribArray(location);gl.vertexAttribPointer(location,size,gl.FLOAT,false,0,0);
  };
  attribute(surface.position,this.posLocation,3);gl.bufferSubData(gl.ARRAY_BUFFER,0,points);
  attribute(surface.uv,this.uvLocation,2);attribute(surface.alpha,this.alphaLocation,1);
 }
 private paint(surface:Surface,points:Float32Array){
  const gl=this.gl;this.bind(surface,points);gl.enable(gl.DEPTH_TEST);gl.depthFunc(gl.LEQUAL);
  gl.uniform1i(gl.getUniformLocation(this.program,'u_mode'),0);gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER,surface.index);
  gl.bufferData(gl.ELEMENT_ARRAY_BUFFER,surface.triangles,gl.DYNAMIC_DRAW);gl.drawElements(gl.TRIANGLES,surface.triangles.length,gl.UNSIGNED_SHORT,0);
 }
 private wireframe(surface:Surface,points:Float32Array,mode:number){
  const gl=this.gl;this.bind(surface,points);gl.disable(gl.DEPTH_TEST);gl.uniform1i(gl.getUniformLocation(this.program,'u_mode'),mode);
  gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER,surface.wire);gl.drawElements(gl.LINES,surface.edges.length,gl.UNSIGNED_SHORT,0);
 }
 draw(points:Float32Array,width:number,height:number,wireframe:boolean,cavity:boolean,neutralPoints?:Float32Array|null,neutralOnly=false){
  const gl=this.gl;if(this.canvas.width!==width||this.canvas.height!==height){this.canvas.width=width;this.canvas.height=height}
  gl.viewport(0,0,width,height);gl.clearColor(0,0,0,0);gl.clear(gl.COLOR_BUFFER_BIT|gl.DEPTH_BUFFER_BIT);gl.useProgram(this.program);
  gl.uniform2f(gl.getUniformLocation(this.program,'u_size'),width,height);gl.bindTexture(gl.TEXTURE_2D,this.texture);
  gl.enable(gl.BLEND);gl.blendFunc(gl.ONE,gl.ONE_MINUS_SRC_ALPHA);
  if(neutralPoints&&this.neutral)this.paint(this.neutral,neutralPoints);
  if(!neutralOnly||!neutralPoints){
   // The speaking surface owns its depth pass, above the neutral underlay.
   gl.clear(gl.DEPTH_BUFFER_BIT);this.bind(this.face,points);
   if(cavity){
    const polygon=this.mouth.flatMap(i=>[points[i*3],points[i*3+1]]),list=new Uint16Array(earcut(polygon).map(i=>this.mouth[i]));
    gl.disable(gl.DEPTH_TEST);gl.uniform1i(gl.getUniformLocation(this.program,'u_mode'),1);gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER,this.face.index);
    gl.bufferData(gl.ELEMENT_ARRAY_BUFFER,list,gl.DYNAMIC_DRAW);gl.drawElements(gl.TRIANGLES,list.length,gl.UNSIGNED_SHORT,0);
   }
   this.paint(this.face,points);
  }
  if(wireframe){
   if(neutralPoints&&this.neutral)this.wireframe(this.neutral,neutralPoints,3);
   if(!neutralOnly||!neutralPoints)this.wireframe(this.face,points,2);
  }
  return this.canvas;
 }
 private release(surface:Surface){for(const buffer of [surface.position,surface.uv,surface.alpha,surface.index,surface.wire])this.gl.deleteBuffer(buffer)}
 dispose(){this.release(this.face);if(this.neutral)this.release(this.neutral);this.gl.deleteTexture(this.texture);this.gl.deleteProgram(this.program);this.gl.getExtension('WEBGL_lose_context')?.loseContext()}
}
