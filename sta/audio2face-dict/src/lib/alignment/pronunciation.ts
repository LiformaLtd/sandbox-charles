import { toARPABET } from 'phonemize';

// Phonemize can emit extended ARPABET symbols. TinyAligner uses the CMU inventory.
const equivalents: Readonly<Record<string, string>> = {
  AX: 'AH', AXR: 'ER', EL: 'L', SAW: 'AO',
  EN: 'AH N', IN: 'IY N', UN: 'UW N', TS: 'T S',
};

export function predictPhones(word: string): string[] {
  const pronunciation = toARPABET(word.toLowerCase(), { stripStress: true }).trim();
  if (!pronunciation) return [];
  return pronunciation.split(/\s+/).flatMap((phone) => {
    const unstressed = phone.replace(/[012]$/, '');
    return (equivalents[unstressed] ?? unstressed).split(' ');
  });
}
