// Names, gender labels and accents from the pinned Gradium Phonon voice catalogue.
export const VOICE_CATALOGUE_REVISION='0ee17205f254b5abc43b266c8a0f215565c352f7';
export const DEFAULT_VOICE='easton';
export const VOICES=[
 {id:'zoey',name:'Zoey',gender:'female',accent:'US',source:'voices/NbpkqMVS3CJeq2j8.safetensors'},
 {id:'juniper',name:'Juniper',gender:'female',accent:'US',source:'voices/NIWIOv5vTi5eRMj8.safetensors'},
 {id:'freya',name:'Freya',gender:'female',accent:'UK',source:'voices/GgfEkEJtxZR7gnpy.safetensors'},
 {id:'maeve',name:'Maeve',gender:'female',accent:'UK',source:'voices/6PWnV0Nq4wu7RVBT.safetensors'},
 {id:'easton',name:'Easton',gender:'male',accent:'US',source:'voices/LMDIYE0Oa377Qh6Z.safetensors'},
 {id:'trevor',name:'Trevor',gender:'male',accent:'US',source:'voices/0zLApsZfVLLNaEGl.safetensors'},
 {id:'alfie',name:'Alfie',gender:'male',accent:'UK',source:'voices/X6NceqF285Gr04gb.safetensors'},
 {id:'hugh',name:'Hugh',gender:'male',accent:'UK',source:'voices/YyKcxPG7iIgS3jsK.safetensors'}
] as const;
export function getVoice(id:string){const voice=VOICES.find(v=>v.id===id);if(!voice)throw new Error('Choose an available Phonon voice.');return voice}
export function voiceURL(id:string){getVoice(id);return `/assets/lf-tts/${id}.safetensors`}
