/// <reference lib="webworker" />
import {FaceLandmarker,FilesetResolver} from '@mediapipe/tasks-vision';
// MediaPipe's Emscripten loader calls self.import in module workers.
// Prepared local WASM glue exports ModuleFactory onto globalThis.
(self as unknown as {import:(url:string)=>Promise<unknown>}).import=url=>import(/* @vite-ignore */ url);
(self as unknown as {custom_dbg:(message:string)=>void}).custom_dbg=message=>console.warn(message);
let engine:Promise<FaceLandmarker>|undefined;
self.onmessage=async({data})=>{
 const {id,image}=data;
 try{
  engine??=FilesetResolver.forVisionTasks('/assets/vision-wasm').then(files=>FaceLandmarker.createFromOptions(files,{
   baseOptions:{modelAssetPath:'/assets/face_landmarker.task',delegate:'CPU'},runningMode:'IMAGE',numFaces:1,
   outputFacialTransformationMatrixes:true,outputFaceBlendshapes:false,minFaceDetectionConfidence:.4,minFacePresenceConfidence:.4
  })).catch(error=>{engine=undefined;throw error});
  const face=(await engine).detect(image),landmarks=face.faceLandmarks[0],matrix=face.facialTransformationMatrixes[0]?.data;
  const fit=landmarks?.length===478&&matrix?{points:Float32Array.from(landmarks.flatMap(p=>[p.x,p.y,p.z])),matrix:Float32Array.from(matrix)}:null;
  postMessage({id,fit});
 }catch(error){postMessage({id,error:error instanceof Error?error.message:String(error)})}
 finally{image.close()}
};
