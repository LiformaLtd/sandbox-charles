export function openSessionStore() {
  return new Promise((resolve, reject) => {
    const request = indexedDB.open('lito-demo', 1);
    request.onupgradeneeded = () => request.result.createObjectStore('sessions');
    request.onerror = () => reject(request.error);
    request.onsuccess = () => resolve(request.result);
  });
}

export function readSession(db) {
  return new Promise((resolve, reject) => {
    const request = db.transaction('sessions').objectStore('sessions').get('current');
    request.onsuccess = () => resolve(request.result ?? {});
    request.onerror = () => reject(request.error);
  });
}

export function saveSession(db, session) {
  return new Promise((resolve, reject) => {
    const transaction = db.transaction('sessions', 'readwrite');
    transaction.oncomplete = resolve;
    transaction.onabort = () => reject(transaction.error);
    transaction.objectStore('sessions').put(session, 'current');
  });
}
