export const VISEME_PAIRS = Object.freeze({
  'P/B': ['P','B'], 'F/V': ['F','V'], 'S/Z': ['S','Z'],
  'TH/DH': ['TH','DH'], 'SH/ZH': ['SH','ZH'], 'T/D': ['T','D'], 'K/G': ['K','G'],
});
export const DEFAULT_VISEME_PAIRS = Object.freeze(['P/B']);
export const RECOMMENDED_VISEME_PAIRS = Object.freeze(['P/B','S/Z','SH/ZH','K/G']);
export const ALL_VISEME_PAIRS = Object.freeze(Object.keys(VISEME_PAIRS));

export function substitutionMap(pairs = DEFAULT_VISEME_PAIRS) {
  if(!Array.isArray(pairs))throw new TypeError('Viseme pairs must be an array');
  const map = new Map();
  for(const pair of new Set(pairs)) {
    if(!Object.hasOwn(VISEME_PAIRS,pair))throw new RangeError('Unsupported viseme pair: '+pair);
    const [a,b]=VISEME_PAIRS[pair];
    map.set(a,{to:b,pair});map.set(b,{to:a,pair});
  }
  return map;
}
