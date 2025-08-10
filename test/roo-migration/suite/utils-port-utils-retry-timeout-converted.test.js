const assert = require('assert')
const { loadTheaModule } = require('../helpers/thea-loader')

// The Thea module imports 'tcp-port-used'. Our loader externalizes it, and we provide a stub under test/roo-migration/node_modules.

describe('Roo-migration: utils/port-utils.retry-timeout', function () {
  const mod = loadTheaModule('src/utils/port-utils.ts')
  const { isPortAvailable, findAvailablePort, waitForPortAvailable, waitForPortInUse } = mod
  const tcp = require('tcp-port-used')

  let savedEnv = {}

  beforeEach(function () {
    savedEnv = { JEST_WORKER_ID: process.env.JEST_WORKER_ID, NODE_ENV: process.env.NODE_ENV }
    delete process.env.JEST_WORKER_ID
    delete process.env.NODE_ENV
  })

  afterEach(function () {
    if (savedEnv.JEST_WORKER_ID !== undefined) process.env.JEST_WORKER_ID = savedEnv.JEST_WORKER_ID
    if (savedEnv.NODE_ENV !== undefined) process.env.NODE_ENV = savedEnv.NODE_ENV
  })

  it('isPortAvailable returns false on errors and true/false accordingly', async function () {
    tcp.check = async () => { throw new Error('Network error') }
    const r0 = await isPortAvailable(3000)
    assert.strictEqual(r0, false)
    tcp.check = async () => false
    const r1 = await isPortAvailable(3001)
    assert.strictEqual(r1, true)
    tcp.check = async () => true
    const r2 = await isPortAvailable(3002)
    assert.strictEqual(r2, false)
  })

  it('findAvailablePort retries then last resort', async function () {
    let calls = 0
    tcp.check = async () => { calls++; return true }
    await assert.rejects(() => findAvailablePort(4000, 'localhost', undefined, 5), /No available ports/)
    assert.ok(calls >= 5)
  })

  it('waitForPortAvailable retries with backoff and then succeeds', async function () {
    let attempts = 0
    tcp.waitUntilFree = async () => { attempts++; if (attempts < 3) throw new Error('in use') }
    const t0 = Date.now()
    await waitForPortAvailable(3000, 'localhost', 100, 30000, 'test', 5)
    const dt = Date.now() - t0
    assert.ok(attempts === 3)
    assert.ok(dt >= 200)
  })

  it('waitForPortInUse retries with jitter and honors NODE_ENV test skip', async function () {
    let attempts = 0
    process.env.NODE_ENV = 'test'
    tcp.waitUntilUsed = async () => { attempts++; throw new Error('not ready') }
    await waitForPortInUse(3000)
    assert.strictEqual(attempts, 0)
  })
})
