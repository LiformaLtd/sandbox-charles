export interface TimedToken {
  word: string;
  start: number;
  end: number;
}

export interface AlignedWord extends TimedToken {
  phones: TimedToken[];
}
