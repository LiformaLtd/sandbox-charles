import {createServer as createViteServer} from 'vite';
import {createServer as createHttpServer} from 'node:http';
import {spawn} from 'node:child_process';
// Exercise the real SvelteKit proxy without spending GPU time. This upstream
// deliberately rejects a browser Origin, just like the real model service.
let posts=0;
const upstream=createHttpServer(async(request,response)=>{
 response.setHeader('content-type','application/json');
 if(request.url==='/stats')return response.end(JSON.stringify({posts}));
 if(request.url==='/health')return response.end(JSON.stringify({model_ready:true,detail:'Test service'}));
 if(request.method==='GET')return response.end(JSON.stringify({sets:[]}));
 if(request.headers.origin){response.statusCode=403;return response.end(JSON.stringify({detail:'Unexpected browser Origin at model server'}));}
 let bytes=0;for await(const chunk of request)bytes+=chunk.length;
 posts++;response.statusCode=202;response.end(JSON.stringify({received_origin:request.headers.origin??null,bytes,content_type:request.headers['content-type']}));
});
await new Promise(resolve=>upstream.listen(0,'127.0.0.1',resolve));
const previous=process.env.LTX25_API_URL;
process.env.LTX25_API_URL=`http://127.0.0.1:${upstream.address().port}`;
process.env.LOOPLAB_TEST_RUN='1';
const server=await createViteServer({server:{host:'127.0.0.1',port:5173,strictPort:false}});
try{
 await server.listen();
 const address=server.httpServer.address();
 const child=spawn(process.execPath,['node_modules/@playwright/test/cli.js','test'],{stdio:'inherit',env:{...process.env,LOOPLAB_TEST_URL:`http://127.0.0.1:${address.port}`,LOOPLAB_STUB_URL:process.env.LTX25_API_URL}});
 process.exitCode=await new Promise(resolve=>child.on('exit',code=>resolve(code??1)));
}finally{
 await server.close();await new Promise(resolve=>upstream.close(resolve));
 if(previous===undefined)delete process.env.LTX25_API_URL;else process.env.LTX25_API_URL=previous;
}
