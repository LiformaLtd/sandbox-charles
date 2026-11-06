import { redirects } from '../vercel.json';

// Start hosted downloads at the upstream URL. WebKit can reject a second
// cross-origin redirect when the request started at a same-origin asset path.
const hostedAssets = import.meta.env.MODE === 'hosted'
  ? new Map(redirects.map(({ source, destination }) => [source, destination]))
  : null;

export function modelAssetUrl(path) {
  return hostedAssets?.get(path) ?? path;
}
