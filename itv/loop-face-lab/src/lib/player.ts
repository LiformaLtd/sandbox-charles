import type { Clip, Playback } from './player-types';

interface Deck {
  video: HTMLVideoElement; id: string | null; ready: boolean;
  loading: Promise<boolean> | null; cancel?: () => void; gain?: GainNode;
}

/** Two predecoded decks paint one persistent canvas; the canvas is never cleared at a join. */
export class LoopPlayer {
  private decks: Deck[];
  private pool: Clip[] = [];
  private active = 0;
  private wanted = false;
  private playing = false;
  private booting = false;
  private switching = false;
  private retiring: Deck | null = null;
  private destroyed = false;
  private version = 0;
  private raf = 0;
  private transitions = 0;
  private lastEmit = 0;
  private anchor: HTMLImageElement | null = null;
  private overlay: HTMLCanvasElement;
  private blendStart = 0;
  private audio: AudioContext | null = null;
  private master: GainNode | null = null;
  private sound = false;
  private volume = .7;
  private context: CanvasRenderingContext2D;

  constructor(private canvas: HTMLCanvasElement, private notify: (value: Playback) => void,
              private fail: (message: string) => void,
              private decorate: (ctx: CanvasRenderingContext2D, id: string, time: number) => void) {
    const context = canvas.getContext('2d', { alpha: false });
    if (!context) throw new Error('Canvas playback is unavailable in this browser.');
    this.context = context;
    this.overlay = document.createElement('canvas');
    this.decks = [0, 1].map(() => {
      const video = document.createElement('video');
      video.preload = 'auto'; video.playsInline = true; video.muted = true; video.loop = true;
      return { video, id: null, ready: false, loading: null };
    });
    this.paint = this.paint.bind(this);
    this.emit();
  }

  setAnchor(url: string, width: number, height: number) {
    this.canvas.width = width; this.canvas.height = height;
    const image = new Image(); this.anchor = image;
    image.onload = () => { if (!this.playing && this.anchor === image) this.draw(image); };
    image.onerror = () => this.fail('Could not load the shared endpoint image.');
    image.src = url;
  }

  setClips(clips: Clip[]) {
    this.pool = clips;
    if (this.wanted && !this.playing) void this.begin();
    else if (this.playing) void this.prepareNext();
    this.emit();
  }

  private emit() {
    const current = this.pool.find(clip => clip.id === this.decks[this.active]?.id);
    this.notify({ armed: this.wanted, playing: this.playing, current: current?.id ?? null,
      slot: current?.slot ?? null, elapsed: this.decks[this.active]?.video.currentTime ?? 0,
      transitions: this.transitions, buffering: this.wanted && !this.playing });
    this.lastEmit = performance.now();
  }

  private load(deck: Deck, clip: Clip): Promise<boolean> {
    if (deck.id === clip.id && deck.loading) return deck.loading;
    if (deck.id === clip.id && deck.ready) return Promise.resolve(true);
    deck.cancel?.(); deck.video.pause(); deck.id = clip.id; deck.ready = false;
    deck.loading = new Promise<boolean>((resolve, reject) => {
      const clean = () => {
        clearTimeout(timeout); deck.video.removeEventListener('loadeddata', loaded);
        deck.video.removeEventListener('error', failed); deck.cancel = undefined;
      };
      const loaded = () => { clean(); deck.ready = true; resolve(true); };
      const failed = () => { clean(); reject(new Error(`Clip ${clip.slot + 1} could not be decoded.`)); };
      const timeout = setTimeout(() => { clean(); reject(new Error(`Clip ${clip.slot + 1} did not buffer in time.`)); }, 20000);
      deck.cancel = () => { clean(); resolve(false); };
      deck.video.addEventListener('loadeddata', loaded, { once: true });
      deck.video.addEventListener('error', failed, { once: true });
      deck.video.src = clip.url; deck.video.load();
    });
    return deck.loading;
  }

  private async prepareNext() {
    if (!this.pool.length || !this.playing) return;
    const deck = this.decks[1 - this.active];
    if (deck === this.retiring) return;
    const activeId = this.decks[this.active].id;
    if (deck.id && this.pool.some(c => c.id === deck.id) &&
        (this.pool.length === 1 || deck.id !== activeId)) return;
    const candidates = this.pool.filter(clip => clip.id !== activeId);
    const chosen = candidates.length ? candidates[Math.floor(Math.random() * candidates.length)] : this.pool[0];
    try { await this.load(deck, chosen); }
    catch (error) { if (!this.destroyed) { this.stop(); this.fail(String(error)); } }
  }

  start() { this.wanted = true; this.emit(); void this.begin(); }

  private async begin() {
    if (this.booting || this.playing || !this.pool.length || !this.wanted) return;
    this.booting = true; const version = this.version;
    try {
      const deck = this.decks[this.active];
      const selected = this.pool.find(c => c.id === deck.id) ?? this.pool[0];
      if (!await this.load(deck, selected) || !this.wanted || version !== this.version) return;
      await deck.video.play();
      if (!this.wanted || version !== this.version) { deck.video.pause(); return; }
      this.playing = true; this.setDeckGains(); this.emit();
      this.raf = requestAnimationFrame(this.paint); void this.prepareNext();
    } catch (error) {
      if (!this.destroyed) { this.wanted = false; this.fail(`Playback could not start. Press Start to try again. ${String(error)}`); this.emit(); }
    } finally {
      this.booting = false;
      if (version !== this.version && this.wanted && this.pool.length) void this.begin();
    }
  }

  stop() {
    this.wanted = false; this.playing = false; cancelAnimationFrame(this.raf);
    for (const deck of this.decks) deck.video.pause();
    this.redraw();
    this.emit();
  }

  reset() {
    this.stop(); this.version++; this.transitions = 0; this.pool = []; this.active = 0; this.retiring = null;
    for (const deck of this.decks) {
      deck.cancel?.(); deck.id = null; deck.ready = false; deck.loading = null;
      deck.video.removeAttribute('src'); deck.video.load();
    }
    this.emit();
  }

  private draw(source: CanvasImageSource, alpha = 1) {
    this.context.globalAlpha = alpha;
    if (source === this.anchor && this.anchor) {
      const scale = Math.max(this.canvas.width / this.anchor.naturalWidth, this.canvas.height / this.anchor.naturalHeight);
      const width = this.canvas.width / scale, height = this.canvas.height / scale;
      this.context.drawImage(source, (this.anchor.naturalWidth - width) / 2, (this.anchor.naturalHeight - height) / 2,
        width, height, 0, 0, this.canvas.width, this.canvas.height);
    } else this.context.drawImage(source, 0, 0, this.canvas.width, this.canvas.height);
    this.context.globalAlpha = 1;
  }

  redraw() {
    const deck = this.decks[this.active];
    if (deck.video.readyState >= 2) {
      this.draw(deck.video);
      this.decorate(this.context, deck.id ?? '', deck.video.currentTime);
    }
  }

  private paint(now: number) {
    if (!this.playing) return;
    const video = this.decks[this.active].video;
    if (video.readyState >= 2) {
      this.draw(video);
      // Briefly converge onto the shared image at each endpoint. This softens
      // VAE reconstruction differences; it cannot repair arbitrary model motion.
      if (this.anchor?.complete && this.anchor.naturalWidth && video.currentTime > 9.88) {
        this.draw(this.anchor, Math.min(1, (video.currentTime - 9.88) / .12));
      }
      if (this.blendStart && now - this.blendStart < 100) this.draw(this.overlay, 1 - (now - this.blendStart) / 100);
    }
    this.decorate(this.context, this.decks[this.active].id ?? '', video.currentTime);
    if (video.currentTime >= Math.min(10, video.duration - .015) && !this.switching) void this.advance();
    if (now - this.lastEmit > 200) this.emit();
    this.raf = requestAnimationFrame(this.paint);
  }

  private async advance() {
    this.switching = true; const version = this.version;
    try {
      await this.prepareNext();
      const previous = this.decks[this.active], next = this.decks[1 - this.active];
      if (next.loading && !await next.loading) return;
      if (!this.wanted || version !== this.version || !next.ready) return;
      this.overlay.width = this.canvas.width; this.overlay.height = this.canvas.height;
      this.overlay.getContext('2d')!.drawImage(this.canvas, 0, 0);
      next.video.currentTime = 0;
      await next.video.play();
      if (!this.wanted || version !== this.version) { next.video.pause(); return; }
      this.active = 1 - this.active; this.transitions++; this.blendStart = performance.now();
      this.retiring = previous;
      if (this.audio && this.sound) {
        const now = this.audio.currentTime;
        next.gain!.gain.setValueAtTime(0, now); next.gain!.gain.linearRampToValueAtTime(1, now + .05);
        previous.gain!.gain.setValueAtTime(1, now); previous.gain!.gain.linearRampToValueAtTime(0, now + .05);
      }
      setTimeout(() => {
        if (version === this.version && this.decks[this.active] !== previous) {
          previous.video.pause(); previous.id = null; previous.ready = false; previous.loading = null;
          this.retiring = null;
          void this.prepareNext();
        }
      }, 60);
      this.emit();
    } catch (error) { this.stop(); this.fail(`Could not join the next clip: ${String(error)}`); }
    finally { this.switching = false; }
  }

  private setDeckGains() {
    if (!this.audio) return;
    this.decks.forEach((deck, index) => { deck.gain!.gain.value = index === this.active ? 1 : 0; });
  }

  async setAudio(enabled: boolean, volume = this.volume) {
    this.sound = enabled; this.volume = volume;
    for (const deck of this.decks) deck.video.muted = !enabled;
    if (enabled && !this.audio) {
      this.audio = new AudioContext(); this.master = this.audio.createGain(); this.master.connect(this.audio.destination);
      for (const deck of this.decks) {
        deck.gain = this.audio.createGain();
        this.audio.createMediaElementSource(deck.video).connect(deck.gain); deck.gain.connect(this.master);
      }
      this.setDeckGains();
    }
    if (this.master) this.master.gain.value = enabled ? volume : 0;
    if (enabled) await this.audio!.resume();
  }

  destroy() {
    this.reset(); this.destroyed = true; this.anchor = null;
    if (this.audio) void this.audio.close();
  }
}
