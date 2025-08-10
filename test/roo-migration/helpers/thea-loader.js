const path = require('path')
const fs = require('fs')
const esbuild = require('esbuild')

// In-repo loader: bundles Thea TS entrypoints to CJS and externalizes host-only deps
const repoRoot = path.resolve(__dirname, '../../..')
const theaRoot = repoRoot
const cacheDir = path.join(repoRoot, 'test', 'roo-migration', '.cache')

function ensureDir(p) {
  fs.mkdirSync(p, { recursive: true })
}

function resolveTheaPath(relOrAbs) {
  return path.isAbsolute(relOrAbs) ? relOrAbs : path.join(theaRoot, relOrAbs)
}

function loadTheaModule(relTsPath) {
  const entry = resolveTheaPath(relTsPath)
  if (!fs.existsSync(entry)) {
    throw new Error(`Thea entry not found: ${entry}`)
  }
  ensureDir(cacheDir)
  const outFile = path.join(cacheDir, relTsPath.replace(/[\\/]/g, '__').replace(/\.(ts|tsx)$/, '.cjs'))
  ensureDir(path.dirname(outFile))

  esbuild.buildSync({
    entryPoints: [entry],
    bundle: true,
    platform: 'node',
    format: 'cjs',
    target: 'node18',
    outfile: outFile,
    sourcemap: false,
    absWorkingDir: theaRoot,
    tsconfig: path.join(theaRoot, 'tsconfig.json'),
    loader: { '.ts': 'ts', '.tsx': 'ts', '.json': 'json' },
    // Externalize modules that require VS Code host or network/ports during tests
    external: ['vscode', 'tcp-port-used'],
  })

  delete require.cache[outFile]
  return require(outFile)
}

module.exports = { loadTheaModule, resolveTheaPath, theaRoot }
