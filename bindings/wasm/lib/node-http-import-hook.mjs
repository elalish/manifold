import {register} from 'node:module';

export async function load(url, context, nextLoad) {
  if (!url.startsWith('https://') && !url.startsWith('http://')) {
    return nextLoad(url, context);
  }

  const res = await fetch(url);
  return {
    format: 'module',
    shortCircuit: true,
    source: await res.text(),
  } 
}

register('./node-http-import-hook.mjs', import.meta.url);