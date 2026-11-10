export interface Job {
  id: string; status: 'queued' | 'running' | 'completed' | 'failed' | 'cancelled';
  request: { seed: number; frames?:number; seconds?:number; kind?:'idle'|'talking' }; error?: string; depends_on?: string | null;
}
export interface LoopSet {
  id: string; status: 'generating' | 'blocked' | 'completed' | 'cancelled'; ready_count: number; total: number;
  replaced_jobs?: number;
  config: { prompt: string; negative_prompt?: string; nag?: boolean; width: number; height: number; frames?: number; idle_prompt?:string; idle_negative_prompt?:string };
  clips: { slot: number; job: Job }[];
}
export interface Draft { image: File | null; prompt: string; negativePrompt: string; idlePrompt:string; idleNegativePrompt:string; size: string }
export interface Clip { id: string; slot: number; seed: number; url: string; seconds?:number }
export interface Playback {
  armed: boolean; playing: boolean; current: string | null; slot: number | null;
  elapsed: number; transitions: number; buffering: boolean;
}

export function isLoopSet(value: unknown): value is LoopSet {
  if (!value || typeof value !== 'object') return false;
  const set = value as Partial<LoopSet>;
  return typeof set.id === 'string' && /^[a-f0-9]{64}$/.test(set.id) &&
    !!set.config && typeof set.config.prompt === 'string' &&
    (set.config.negative_prompt === undefined || typeof set.config.negative_prompt === 'string') &&
    Number.isInteger(set.config.width) && Number.isInteger(set.config.height) &&
    ['generating','blocked','completed','cancelled'].includes(set.status ?? '') &&
    Number.isInteger(set.ready_count) && (set.total === 5 || set.total === 11) &&
    Array.isArray(set.clips) && set.clips.length === set.total &&
    set.clips.every(clip => !!clip && typeof clip === 'object' && Number.isInteger(clip.slot) && clip.slot >= 0 && clip.slot < set.total! &&
      typeof clip.job?.id === 'string' &&
      ['queued','running','completed','failed','cancelled'].includes(clip.job.status) &&
      Number.isInteger(clip.job.request?.seed));
}

export function parseLoopSet(value: unknown): LoopSet {
  if (!isLoopSet(value)) throw new Error('The model server returned invalid loop-set metadata. Your existing set was kept.');
  return value;
}

export const DEFAULT_NEGATIVE_PROMPT = 'speaking, talking, lip movements, open mouth, singing';
