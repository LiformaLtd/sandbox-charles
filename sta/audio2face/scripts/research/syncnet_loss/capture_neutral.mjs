// Capture the unmodified browser renderer for the fixed offline background.
import {writeFile} from 'node:fs/promises';
const {webkit}=await import(process.env.PLAYWRIGHT_MODULE);
const browser=await webkit.launch({executablePath:process.env.WEBKIT_EXECUTABLE});
try{
 const page=await browser.newPage({viewport:{width:448,height:224},deviceScaleFactor:1});
 await page.goto(process.env.DIAGNOSTIC_URL+'/scripts/research/mouth_expert/render.html');
 await page.waitForFunction(()=>window.ready);
 const images=await page.evaluate(()=>window.renderWeights(Array(52).fill(0)));
 await writeFile(process.env.SYNCNET_NEUTRAL_PNG,Buffer.from(images[1],'base64'));
}finally{await browser.close();}
