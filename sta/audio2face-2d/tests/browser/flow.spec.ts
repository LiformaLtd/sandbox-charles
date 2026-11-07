import { test,expect } from '@playwright/test';

test('real speech inference, seek, avatar and mapping switch, upload and local text',async({page})=>{
 page.on('response',r=>{if(r.status()>=400)console.log('HTTP',r.status(),r.url());});const errors:string[]=[];page.on('pageerror',e=>errors.push(e.message));page.on('console',m=>{if(m.type()==='error')console.log('BROWSER ERROR',m.text());});
 await page.goto('/?runtime=cpu');await expect(page.getByRole('button',{name:'Animate speech'})).toBeEnabled();
 await expect(page.locator('#avatar-select option')).toHaveCount(18);
 await page.screenshot({path:'../../../../outputs/audio2face-2d-initial.png',fullPage:true});
 await page.getByRole('button',{name:'Animate speech'}).click();
 await expect(page.locator('.status-line')).toContainText('Ready.',{timeout:120000});
 await expect(page.locator('.transport-title')).toContainText('shape frames');
 const before=await page.locator('canvas').screenshot();
 await page.locator('audio').evaluate((a:HTMLAudioElement)=>{a.pause();a.currentTime=0.75;});
 await page.waitForTimeout(100);const after=await page.locator('canvas').screenshot();expect(before.equals(after)).toBe(false);
 await page.getByText('Show mesh',{exact:true}).click();
 await page.getByRole('button',{name:'Mouth close-up',exact:true}).click();
 await page.screenshot({path:'../../../../outputs/audio2face-2d-mesh.png',fullPage:true});
 await page.locator('#avatar-select').selectOption('01d55f04');await expect(page.locator('.viewer-top strong')).toHaveText('Andrew Watson');
 await page.locator('#mapping-select').selectOption('local');await expect(page.locator('.status-line')).toContainText('Ready.');
 await page.getByRole('button',{name:'Upload',exact:true}).click();await page.locator('#audio-file').setInputFiles('static/assets/speech/test-supertonic-m3-000.wav');
 await page.getByRole('button',{name:'Animate speech'}).click();await expect(page.locator('.status-line')).toContainText('Ready.',{timeout:120000});
 await page.getByRole('button',{name:'Type text',exact:true}).click();await page.locator('#speech-text').fill('Mum packed a blue bag.');
 await page.getByRole('button',{name:'Generate & play'}).click();await expect(page.locator('.transcript')).toHaveText('Mum packed a blue bag.',{timeout:150000});
 await expect(page.locator('.transcript')).toHaveText('Mum packed a blue bag.');
 await page.getByRole('button',{name:'Stop and reset'}).click();await expect(page.locator('.status-line')).toContainText('Stopped.');
 expect(errors).toEqual([]);
});


test('all actual avatar meshes and both maps load without browser errors',async({page})=>{
 const errors:string[]=[];page.on('pageerror',e=>errors.push(e.message));
 await page.goto('/?runtime=cpu');await expect(page.getByRole('button',{name:'Animate speech'})).toBeEnabled();
 const ids=await page.locator('#avatar-select option').evaluateAll(options=>options.map(o=>(o as HTMLOptionElement).value));
 for(const id of ids){
  await page.locator('#avatar-select').selectOption(id);await expect(page.getByRole('button',{name:'Animate speech'})).toBeEnabled();
  await expect(page.getByRole('alert')).toHaveCount(0);
  await page.locator('#mapping-select').selectOption('local');await expect(page.getByRole('button',{name:'Animate speech'})).toBeEnabled();
  await page.locator('#mapping-select').selectOption('linear');await expect(page.getByRole('button',{name:'Animate speech'})).toBeEnabled();
 }
 await page.setViewportSize({width:390,height:844});
 expect(await page.evaluate(()=>document.documentElement.scrollWidth<=innerWidth)).toBe(true);
 await page.screenshot({path:'../../../../outputs/audio2face-2d-mobile.png',fullPage:true});expect(errors).toEqual([]);
});
