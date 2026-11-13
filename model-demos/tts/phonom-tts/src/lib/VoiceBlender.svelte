<script lang="ts">
	import { onMount } from 'svelte';
	import catalogue from './voices-catalogue.json';
	import { blendDescription, MIN_BLEND_PERCENT, MAX_BLEND_PERCENT, type BlendRecipe, type SavedBlend } from './voice-blends';
	let { disabled, canPreview, onpreview, onvoiceschange, onworking }: {
		disabled: boolean; canPreview: boolean;
		onpreview: (voice: { id: string; name: string; source: Blob }) => Promise<void>;
		onvoiceschange: (voices: SavedBlend[], selectedId?: string) => void;
		onworking: (working: boolean) => void;
	} = $props();
	let sourceA = $state('Alfie'), sourceB = $state('Hugh'), percentB = $state(50);
	let name = $state(''), preparing = $state(false), failure = $state(''), message = $state('');
	let saved = $state<SavedBlend[]>([]);
	let confirming = $state('');
	const recipe = $derived({ sourceA, sourceB, percentB } satisfies BlendRecipe);
	const description = $derived(blendDescription(recipe));
	let disposed = false;
	let listVersion = 0;
	onMount(() => {
		void load();
		return () => { disposed = true; };
	});
	async function responseData(response: Response) {
		const body = await response.json();
		if (!response.ok) throw new Error(body.detail || 'The voice operation failed.');
		return body;
	}
	function fail(error: unknown) {
		console.error('Voice blending operation failed', error);
		failure = error instanceof Error ? error.message : 'Could not blend these voices.';
	}
	async function load() {
		const version = ++listVersion;
		try {
			const voices = await responseData(await fetch('/api/blends', { cache: 'no-store' }));
			if (!disposed && version === listVersion) { saved = voices; onvoiceschange(saved); }
		} catch (error) { if (!disposed && version === listVersion) fail(error); }
	}
	async function create(save: boolean) {
		if (disabled || preparing) return;
		preparing = true; onworking(true); failure = ''; message = '';
		const chosen = { ...recipe };
		const label = name.trim() || blendDescription(chosen);
		try {
			const response = await fetch('/api/blends', { method: 'POST', headers: { 'Content-Type': 'application/json' },
				body: JSON.stringify({ ...chosen, action: save ? 'save' : 'preview', name: label }) });
			if (save) {
				const voice: SavedBlend = await responseData(response);
				if (disposed) return;
				++listVersion;
				saved = [voice, ...saved]; onvoiceschange(saved, voice.id);
				message = `Saved “${voice.name}” and selected it in the Voice menu.`;
			} else {
				if (!response.ok) await responseData(response);
				const bytes = await response.blob();
				if (disposed) return;
				onworking(false);
				await onpreview({ id: `preview-${crypto.randomUUID()}`, name: `Blend preview · ${blendDescription(chosen)}`, source: bytes });
			}
		} catch (error) { if (!disposed) fail(error); }
		finally { preparing = false; onworking(false); }
	}
	async function remove(id: string) {
		if (disabled || preparing) return;
		preparing = true; onworking(true); failure = ''; message = '';
		try {
			await responseData(await fetch(`/api/blends/${id}`, { method: 'DELETE' }));
			++listVersion;
			saved = saved.filter(v => v.id !== id); onvoiceschange(saved); confirming = '';
			message = 'Saved blend deleted.';
		} catch (error) { fail(error); }
		finally { preparing = false; onworking(false); }
	}
</script>

<section aria-labelledby="blend-heading">
	<h2 id="blend-heading">Blend voices <span>Experimental</span></h2>
	<p>Combine two catalogue voices to make a new voice. Preview uses the text above and Browser Q8 with your selected model and device.</p>
	<div class="sources">
		{#each ['A', 'B'] as letter}
			<label>Voice {letter}
				<select aria-label={`Blend voice ${letter}`} value={letter === 'A' ? sourceA : sourceB} disabled={disabled || preparing} onchange={e => { if (letter === 'A') sourceA = e.currentTarget.value; else sourceB = e.currentTarget.value; }}>
					{#each Object.entries(catalogue) as [group, voices]}
						<optgroup label={group}>{#each voices as voice}<option value={voice.name}>{voice.name}</option>{/each}</optgroup>
					{/each}
				</select>
			</label>
		{/each}
	</div>
	<label class="slider" for="blend-weight">Blend balance</label>
	<div class="balance"><span>{sourceA} {100 - percentB}%</span><span>{sourceB} {percentB}%</span></div>
	<input id="blend-weight" type="range" min={MIN_BLEND_PERCENT} max={MAX_BLEND_PERCENT} step="1" bind:value={percentB} disabled={disabled || preparing} />
	<p class="hint">0% B is voice A alone; 100% B is voice B alone. Beyond either endpoint, one voice is subtracted from the other. Larger extrapolations may reduce clarity.</p>
	<label class="name">New voice name <input type="text" bind:value={name} placeholder={description} maxlength="80" disabled={disabled || preparing} /></label>
	<div class="actions">
		<button type="button" onclick={() => create(false)} disabled={disabled || preparing || !canPreview}>Preview blend · Browser Q8</button>
		<button type="button" class="secondary" onclick={() => create(true)} disabled={disabled || preparing}>Save as new voice</button>
	</div>
	{#if preparing}<p role="status">Preparing your blend…</p>{/if}
	{#if message}<p role="status">{message}</p>{/if}
	{#if failure}<p class="error" role="alert">{failure} <button type="button" class="secondary" onclick={load} disabled={disabled || preparing}>Reload saved blends</button></p>{/if}
	<p class="hint">Preview plays in Streaming result below. Blends are experimental; tone, accent and clarity may change unpredictably. Saved blends work in Browser Q8 and stay on this computer across reloads and port changes.</p>
	{#if saved.length}
		<h3>Saved blends</h3>
		<ul>{#each saved as item (item.id)}
			<li><div><strong>{item.name}</strong><small>{blendDescription(item)}</small></div><div class="actions">
				<button type="button" class="secondary" disabled={disabled || preparing} onclick={() => onvoiceschange(saved, item.id)}>Use voice</button>
				<a href={`/api/blends/${item.id}`} download={`${item.name}.safetensors`}>Download voice</a>
				{#if confirming === item.id}<button type="button" class="secondary" disabled={disabled || preparing} onclick={() => remove(item.id)}>Confirm delete</button><button type="button" class="secondary" onclick={() => { confirming = ''; }}>Keep</button>
				{:else}<button type="button" class="secondary" disabled={disabled || preparing} onclick={() => { confirming = item.id; }}>Delete</button>{/if}
			</div></li>
		{/each}</ul>
	{/if}
</section>

<style>
	section { border: 1px solid #3c4061; background: #1b2035; border-radius: 16px; padding: clamp(14px, 4vw, 24px); margin-top: 16px; }
	h2 { margin: 0; font-size: 1.1rem; } h2 span { color: #bfb0ff; font-size: .75rem; font-weight: 400; margin-left: 8px; }
	p, small { color: #bcbfda; font-size: .85rem; } .sources { display: grid; grid-template-columns: 1fr 1fr; gap: 16px; }
	label { display: block; } select, input[type=text] { display: block; width: 100%; margin-top: 6px; min-width: 0; }
	select, input[type=text], button { font: inherit; border-radius: 8px; padding: 8px 12px; }
	select, input[type=text] { border: 1px solid #5a6084; background: #121524; color: inherit; }
	.slider, .name { margin-top: 16px; } .balance, .actions { display: flex; flex-wrap: wrap; gap: 10px; align-items: center; } .balance { justify-content: space-between; font-size: .85rem; margin-top: 8px; }
	input[type=range] { width: 100%; accent-color: #c5b6ff; margin: 12px 0; } .actions { margin-top: 12px; }
	button { border: 1px solid #c5b6ff; background: #c5b6ff; color: #1b1437; font-weight: 650; cursor: pointer; }
	button.secondary { border-color: #67608e; background: transparent; color: #ded3ff; } button:disabled { opacity: .5; cursor: default; }
	a { color: #ded3ff; font-size: .85rem; } .error { color: #ffb4ad; overflow-wrap: anywhere; } .hint { margin-bottom: 0; }
	h3 { font-size: 1rem; margin-bottom: 0; } ul { list-style: none; padding: 0; } li { border-top: 1px solid #3c4061; padding: 12px 0; } strong, small { display: block; overflow-wrap: anywhere; }
	@media(max-width: 540px) { .sources { grid-template-columns: 1fr; } }
</style>
