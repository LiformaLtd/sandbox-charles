let database: Promise<IDBDatabase> | undefined;
function open() {
  database ??= new Promise((resolve, reject) => {
    const request = indexedDB.open('loop-face-lab', 1);
    request.onupgradeneeded = () => request.result.createObjectStore('records');
    request.onsuccess = () => resolve(request.result);
    request.onerror = () => reject(request.error);
    request.onblocked = () => reject(new Error('Close other Loop Lab tabs to update storage.'));
  });
  return database;
}
export async function read<T>(key: string): Promise<T | undefined> {
  const db = await open();
  return new Promise((resolve, reject) => {
    const tx = db.transaction('records'); const request = tx.objectStore('records').get(key);
    tx.oncomplete = () => resolve(request.result);
    tx.onerror = tx.onabort = () => reject(tx.error);
  });
}
export async function write(key: string, value: unknown) {
  const db = await open();
  return new Promise<void>((resolve, reject) => {
    const tx = db.transaction('records', 'readwrite'); tx.objectStore('records').put(value, key);
    tx.oncomplete = () => resolve(); tx.onerror = tx.onabort = () => reject(tx.error);
  });
}
