import averages from './average-voices.json';

export const averagedVoices = averages.groups;
export const averageVoiceUrl = (id: string) => `/api/average-voices/${id}?revision=${averages.revision}`;
export interface BlendRecipe { sourceA: string; sourceB: string; percentB: number }
export interface SavedBlend extends BlendRecipe { id: string; name: string; createdAt: string }
export const blendVoiceKey = (id: string) => `blend-${id}`;
export const blendVoiceUrl = (id: string) => `/api/blends/${id}`;
export const blendDescription = (recipe: BlendRecipe) =>
	`${recipe.sourceA} ${100 - recipe.percentB}% + ${recipe.sourceB} ${recipe.percentB}%`;
