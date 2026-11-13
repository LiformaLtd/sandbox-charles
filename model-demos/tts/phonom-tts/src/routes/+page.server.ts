import { modelAvailability } from '../lib/server/model-assets';
import type { PageServerLoad } from './$types';

export const load: PageServerLoad = async () => ({ modelAvailability: await modelAvailability() });
