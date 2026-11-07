const recoveryKey = 'james-preload-recovery-at';
const retryWindow = 60_000;

// A tab may outlive the deployment containing its lazy-loaded JS chunks.
// Only reload for a confirmed missing same-origin build asset, preserving input.
export function installPreloadRecovery({ target = window, saveDraft, fetchAsset = fetch, now = Date.now }) {
  let checking = false;
  const recover = async (event) => {
    if (checking) return;
    const match = event.payload?.message?.match(/https?:\/\/[^\s"'<>]+|\/assets\/[^\s"'<>]+/);
    // WebKit can omit the failed URL. The old entry script also disappears
    // after a deployment, so check that URL when the error has no asset URL.
    const asset = match?.[0] ?? target.document?.querySelector('script[type="module"][src]')?.src;
    if (!asset) return;
    let url;
    try { url = new URL(asset, target.location.href); } catch { return; }
    if (url.origin !== target.location.origin || !/^\/assets\/.+\.(?:js|css)$/.test(url.pathname)) return;
    checking = true;
    try {
      const storage = target.sessionStorage;
      const previous = Number(storage.getItem(recoveryKey));
      if (previous && now() - previous < retryWindow) return;
      const response = await fetchAsset(url.href, { method: 'HEAD', cache: 'no-store' });
      if (response.status !== 404) return;
      saveDraft();
      storage.setItem(recoveryKey, String(now()));
      target.location.reload();
    } catch {
      // Keep the original error visible if offline or input cannot be saved.
    } finally {
      checking = false;
    }
  };
  target.addEventListener('vite:preloadError', recover);
  return () => target.removeEventListener('vite:preloadError', recover);
}
