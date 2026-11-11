import { parse } from 'acorn';
import { transform } from 'esbuild';
import { createHash } from 'node:crypto';
import { readFile, mkdir, writeFile } from 'node:fs/promises';
import { createRequire } from 'node:module';

// Build only the pinned upstream Emscripten engine for this browser worker.
// Its Node entry points and bundled example models are not browser dependencies.
const require = createRequire(import.meta.url);
const source = await readFile(require.resolve('@sctg/sentencepiece-js'), 'utf8');
const expected = 'de24f2d49b0cb6f607e4afb51ca932ccc4b38d6fb412667bc0860e2a512d6392';
if (createHash('sha256').update(source).digest('hex') !== expected) {
  throw new Error('SentencePiece upstream source changed; review the browser build before upgrading.');
}
const ast = parse(source, { ecmaVersion: 'latest', sourceType: 'module' });
const declarations = ast.body.filter((node) => node.type === 'VariableDeclaration').flatMap((node) => node.declarations);
const engine = declarations.find((node) => node.id.name === 'createSentencePieceModule');
if (!engine) throw new Error('Pinned SentencePiece engine was not found.');
const platformFlags = [];
function visit(node) {
  if (!node || typeof node !== 'object') return;
  if (node.type === 'VariableDeclarator' && node.id.name === 'ENVIRONMENT_IS_NODE') platformFlags.push(node.init);
  for (const value of Object.values(node)) {
    if (Array.isArray(value)) value.forEach(visit);
    else if (value && typeof value === 'object') visit(value);
  }
}
visit(engine);
if (platformFlags.length !== 1) throw new Error('Unexpected SentencePiece platform selection.');
const flag = platformFlags[0];
const replacements = [{ start: flag.start, end: flag.end }];
function browserReferences(node, parent) {
  if (!node || typeof node !== 'object') return;
  if (node.type === 'Identifier' && node.name === 'ENVIRONMENT_IS_NODE' && parent?.type !== 'VariableDeclarator') replacements.push(node);
  for (const value of Object.values(node)) {
    if (Array.isArray(value)) value.forEach((child) => browserReferences(child, node));
    else if (value && typeof value === 'object') browserReferences(value, node);
  }
}
browserReferences(engine);
let browserEngine = source.slice(engine.init.start, engine.init.end);
for (const node of replacements.sort((a, b) => b.start - a.start)) {
  const start = node.start - engine.init.start, end = node.end - engine.init.start;
  browserEngine = browserEngine.slice(0, start) + 'false' + browserEngine.slice(end);
}
const result = await transform(`export default ${browserEngine};`, {
  format: 'esm', target: 'es2022', minifySyntax: true, treeShaking: true
});
// A browser build must contain no Node imports, rather than substituting empty shims.
const built = parse(result.code, { ecmaVersion: 'latest', sourceType: 'module' });
const imports = [];
function check(node) {
  if (!node || typeof node !== 'object') return;
  if (node.type === 'ImportDeclaration' || node.type === 'ImportExpression') imports.push(node);
  if (node.type === 'CallExpression' && node.callee.name === 'require') imports.push(node);
  for (const value of Object.values(node)) {
    if (Array.isArray(value)) value.forEach(check);
    else if (value && typeof value === 'object') check(value);
  }
}
check(built);
if (imports.length) throw new Error('Unexpected imports remain in the standalone browser engine.');
const directory = new URL('../src/lib/pocket/onnx/generated/', import.meta.url);
await mkdir(directory, { recursive: true });
await writeFile(new URL('sentencepiece.js', directory), '// Browser build of @sctg/sentencepiece-js 1.3.3, Apache-2.0.\n' + result.code);
console.log('Built pinned SentencePiece browser engine without Node entry points.');
