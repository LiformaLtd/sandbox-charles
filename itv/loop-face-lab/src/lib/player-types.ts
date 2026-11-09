export interface Clip{id:string;slot:number;seed:number;url:string}
export interface Playback{armed:boolean;playing:boolean;current:string|null;slot:number|null;elapsed:number;transitions:number;buffering:boolean}
