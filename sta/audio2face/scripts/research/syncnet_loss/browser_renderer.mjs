// Local research-only renderer RPC; no models or speech leave this computer.
import http from 'node:http';
import {writeFile} from 'node:fs/promises';
const {webkit}=await import(process.env.PLAYWRIGHT_MODULE);
const browser=await webkit.launch({executablePath:process.env.WEBKIT_EXECUTABLE});
const page=await browser.newPage({viewport:{width:224,height:224},deviceScaleFactor:1});
await page.goto(process.env.DIAGNOSTIC_URL+'/scripts/research/syncnet_loss/browser-render.html');await page.waitForFunction(()=>window.ready);
let serial=Promise.resolve();
const server=http.createServer((request,response)=>{
 let body='';request.on('data',chunk=>body+=chunk);request.on('end',()=>{serial=serial.then(async()=>{
  try{const weights=JSON.parse(body);const images=await page.evaluate(weights=>window.renderClip(weights),weights);response.writeHead(200,{'content-type':'application/json'});response.end(JSON.stringify(images));}
  catch(error){response.writeHead(500);response.end(String(error));}
 });});
});server.listen(0,'127.0.0.1',async()=>{const endpoint='http://127.0.0.1:'+server.address().port;await writeFile(process.env.SYNCNET_RENDERER_ENDPOINT,JSON.stringify({endpoint,pid:process.pid}));console.log('Research renderer ready at',endpoint);});
async function stop(){server.close();await browser.close();process.exit(0);}process.on('SIGTERM',stop);process.on('SIGINT',stop);
