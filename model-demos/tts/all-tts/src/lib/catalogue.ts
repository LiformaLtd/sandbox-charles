import phononVoices from './phonon/english-voices.json';
import pocketModels from './models.json';
export type Family = 'phonon' | 'supertonic' | 'pocket';
export type Mode = 'browser' | 'server' | 'native' | 'mlx';
export const modeLabels: Record<Mode, string> = { browser: 'Browser', server: 'Local server', native: 'Local native · CPU/Q8', mlx: 'Local MLX · Apple GPU' };
export interface Model { id: string; label: string; modes: Mode[]; licence: string; checkpoint?: string }
export const families: { id: Family; label: string; models: Model[] }[] = [
 { id: 'phonon', label: 'Phonon', models: [
  { id: 'phonon-31m-en', label: '31M English', modes: ['browser', 'native', 'mlx'], licence: 'Apache 2.0', checkpoint: 'hf://gradium/phonon-31m-en@92741c5b07d085ff840e26fea8ce43110b87852f' }
 ] },
 { id: 'supertonic', label: 'Supertonic', models: [
  { id: 'supertonic-3', label: 'Supertonic 3 · 99M · standard', modes: ['server', 'browser'], licence: 'OpenRAIL' }
 ] },
 { id: 'pocket', label: 'Pocket TTS', models: pocketModels.map(m => ({ id: m.id, label: `${m.size} · ${m.label}`, modes: m.id === 'english_18m_2026-10' ? ['server', 'browser'] : ['server'], licence: 'CC BY 4.0' })) }
];
export function familyById(id: Family) { return families.find(f => f.id === id)!; }
export function modelById(family: Family, id: string) { return familyById(family).models.find(m => m.id === id)!; }
export function voiceOptions(family: Family, mode: Mode) {
 if (family === 'phonon') return Object.entries(phononVoices).map(([group, voices]) => ({ group, voices: voices.map(v => ({ value: v.name, label: v.name })) }));
 if (family === 'supertonic') return [{ group: 'Presets', voices: ['F1','F2','F3','F4','F5','M1','M2','M3','M4','M5'].map(v => ({ value: v, label: `${v[0] === 'F' ? 'Female' : 'Male'} · ${v}` })) }];
 const voices = mode === 'browser' ? ['alba','marius','azelma','cosette','eponine','fantine','javert','jean'] : ['alba','anna','marius'];
 return [{ group: 'Presets', voices: voices.map(v => ({ value: v, label: v[0].toUpperCase() + v.slice(1) })) }, { group: 'Your voice', voices: [{ value: '__clone', label: 'Clone from WAV or M4A' }] }];
}
export const phononVoiceFiles = new Map(Object.values(phononVoices).flat().map(v => [v.name, `https://huggingface.co/gradium/phonon-voices/resolve/0ee17205f254b5abc43b266c8a0f215565c352f7/${v.file}`]));
