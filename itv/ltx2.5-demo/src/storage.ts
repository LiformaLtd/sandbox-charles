export interface Draft {
  first: File | null; last: File | null; prompt: string;
  size: string; frames: number; seed: number;
}
export interface Job {
  id: string; status: 'submitting' | 'queued' | 'running' | 'completed' | 'failed';
  error?: string | null; created?: number; updated?: number;
  request?: { seed: number; requested_seed?: number; frames: number };
}
export interface Output { id: string; blob: Blob }
export const ACTIVE = 'ltx25.activeJob.v1';
export const OUTPUT = 'ltx25.output.v1';

let database: Promise<IDBDatabase> | undefined;
function open() {
  database ??= new Promise((resolve, reject) => {
    const req = indexedDB.open('ltx25-demo', 1);
    req.onupgradeneeded = () => req.result.createObjectStore('records');
    req.onsuccess = () => resolve(req.result);
    req.onerror = () => reject(req.error);
    req.onblocked = () => reject(new Error('Close other demo tabs to update browser storage.'));
  });
  return database;
}
export async function read<T>(key: string): Promise<T | undefined> {
  const db = await open();
  return new Promise((resolve, reject) => {
    const tx = db.transaction('records', 'readonly');
    const req = tx.objectStore('records').get(key);
    tx.oncomplete = () => resolve(req.result);
    tx.onerror = () => reject(tx.error);
    tx.onabort = () => reject(tx.error);
  });
}
export async function write(key: string, value: unknown): Promise<void> {
  const db = await open();
  return new Promise((resolve, reject) => {
    const tx = db.transaction('records', 'readwrite');
    tx.objectStore('records').put(value, key);
    tx.oncomplete = () => resolve();
    tx.onerror = () => reject(tx.error);
    tx.onabort = () => reject(tx.error);
  });
}
export async function forget(key: string): Promise<void> {
  const db = await open();
  return new Promise((resolve, reject) => {
    const tx = db.transaction('records', 'readwrite');
    tx.objectStore('records').delete(key);
    tx.oncomplete = () => resolve();
    tx.onabort = () => reject(tx.error);
    tx.onerror = () => reject(tx.error);
  });
}
