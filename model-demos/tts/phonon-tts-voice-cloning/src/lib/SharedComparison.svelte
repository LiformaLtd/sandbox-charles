<script lang="ts">
  import { api, type Voice, type Health, type ComparisonBatch } from './api';
  import { formatError, versionName } from './voice-workflow';

  let { source, versions, selectedId, health, disabled, onactivity, onrefresh, onerror }: {
    source: Voice; versions: Voice[]; selectedId: string; health: Health | undefined; disabled: boolean;
    onactivity: (message: string) => void; onrefresh: () => Promise<void>; onerror: (error: unknown) => void;
  } = $props();
  let sourceId = $state('');
  let transcript = $state('');
  let confirmed = $state(false);
  let working = $state(false);
  let cancelling = $state(false);
  let message = $state('');
  const latest = $derived(versions.flatMap(v => v.sharedComparison ? [v.sharedComparison] : []).sort((a, b) => b.createdAt.localeCompare(a.createdAt))[0]);
  $effect(() => {
    if (source.id !== sourceId) {
      sourceId = source.id;
      transcript = latest?.transcript || versions.find(v => v.transcript)?.transcript || '';
      confirmed = false; message = '';
    }
  });

  async function transcribe() {
    if (disabled || working) return;
    working = true; onactivity('Transcribing the saved recording for comparison…');
    try {
      const result = await api<{ text: string }>(`voices/${source.id}/transcribe`, { method: 'POST' });
      transcript = result.text; confirmed = false;
      message = 'Check these words against your saved recording before measuring.';
    } catch (error) { onerror(error); }
    finally { working = false; onactivity(''); await onrefresh(); }
  }

  async function measure() {
    if (disabled || working || !confirmed || transcript.trim().length < 3) return;
    working = true; message = ''; onactivity('Measuring shared comparison errors…');
    try {
      const result = await api<ComparisonBatch | { cancelled: true }>(`voices/${source.id}/compare`, {
        method: 'POST', headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ transcript: transcript.trim(), confirmed, voiceIds: versions.map(v => v.id) })
      });
      message = 'cancelled' in result ? 'Comparison cancelled. Earlier measurements are kept.' : `Measured ${result.results.length} versions in ${result.elapsedSeconds.toFixed(2)} seconds. No fitting was run.`;
    } catch (error) { onerror(error); }
    finally { working = false; cancelling = false; onactivity(''); await onrefresh(); }
  }

  async function cancel() {
    if (!health?.evaluation || cancelling) return;
    cancelling = true;
    try { await api(`evaluations/${health.evaluation.id}/cancel`, { method: 'POST' }); }
    catch (error) { cancelling = false; onerror(error); }
  }
</script>

<div class="shared-comparison" aria-labelledby="shared-heading">
  <h3 id="shared-heading">Compare versions on one shared error</h3>
  <p>All versions say the words in “{source.name}” and are scored with the same settings. This measures existing voices; it does not run fitting or change them.</p>
  {#if health?.supportsSharedComparison && health.refinementReady}
    <button class="secondary full" onclick={transcribe} disabled={disabled || working}>Get recording transcript for comparison</button>
    <label for="shared-transcript">Words spoken in the saved recording</label>
    <textarea id="shared-transcript" rows="3" maxlength="1000" bind:value={transcript} oninput={() => { confirmed = false; }} disabled={disabled || working}></textarea>
    <label class="consent"><input type="checkbox" bind:checked={confirmed} disabled={disabled || working} /><span>I checked these comparison words against the saved recording.</span></label>
    <button class="secondary full" onclick={measure} disabled={disabled || working || !confirmed || transcript.trim().length < 3 || versions.length > 100}>Measure shared errors for all versions</button>
    {#if versions.length > 100}<p>Compare up to 100 versions at once.</p>{/if}
    {#if health.evaluation}
      <div class="fit-progress" role="status"><strong>{health.evaluation.stage}</strong><progress value={health.evaluation.step} max={health.evaluation.total}></progress><span>{health.evaluation.step} / {health.evaluation.total} versions</span><button class="text-button" onclick={cancel} disabled={cancelling}>{cancelling ? 'Cancelling…' : 'Cancel comparison'}</button></div>
    {/if}
  {:else}<p>Restart the lab with the comparison models installed to enable shared scoring.</p>{/if}
  {#if message}<p role="status">{message}</p>{/if}
  {#if latest}
    <p class="fine-print">Lower shared error is better for this recording. Unlike the fitting errors above, these values share one scale. This is not a percentage or an accent guarantee.</p>
    <div class="shared-table"><table aria-label="Shared comparison errors"><thead><tr><th>Voice version</th><th>Shared error</th></tr></thead><tbody>
      {#each versions as version (version.id)}<tr class:selected-row={version.id === selectedId}><td>{versionName(version, versions)}</td><td>{version.sharedComparison?.comparisonKey === latest.comparisonKey ? formatError(version.sharedComparison.error) : version.sharedComparison ? 'Measure again' : 'Not measured'}</td></tr>{/each}
    </tbody></table></div>
    {#if latest.transcript !== transcript.trim()}<p class="comparison-warning">The shown results use the last measured transcript. Measure again to use your edited words.</p>{/if}
    <details><summary>Shared scoring settings and measured words</summary>
      <p>Measured {new Date(latest.createdAt).toLocaleString()} · temperature {latest.protocol.sampling.temperature} · seed {latest.protocol.sampling.seed}.</p>
      <p>“{latest.transcript}”</p><p>Speaker weight {latest.protocol.weights.speaker}, acoustic weight {latest.protocol.weights.mel}, spectrum weight {latest.protocol.weights.spectrum}, duration weight {latest.protocol.weights.duration}. No parameter penalty. Scores with other settings are marked “Measure again”.</p>
      <p>This checks reproduction of the reference recording. It is not an independent test on new speech; listen to other sentences too.</p>
    </details>
  {/if}
</div>

<style>
  .shared-comparison{border:1px solid #dfd6ec;border-radius:11px;background:#f8f5fd;padding:16px;margin:18px 0 22px}
  h3{margin:0;font-size:14px}p{font-size:11px;line-height:1.7;color:#897893}label{margin-top:16px}textarea{min-height:100px;font-size:12px}
  .fine-print{text-align:left;line-height:1.7}.shared-table{overflow-x:auto;margin-top:15px}table{width:100%;border-collapse:collapse;font-size:11px;color:#6e5b80}th,td{padding:10px 7px;border-bottom:1px solid #e4dced;text-align:left}th:last-child,td:last-child{text-align:right;font-variant-numeric:tabular-nums;white-space:nowrap}.selected-row{background:#eee6f8}details{margin-top:15px;font-size:10px;color:#897893}summary{cursor:pointer}
</style>
