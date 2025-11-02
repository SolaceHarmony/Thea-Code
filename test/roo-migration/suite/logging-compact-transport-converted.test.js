import { strictEqual, ok } from 'assert'
import { existsSync, readdirSync, lstatSync, unlinkSync, rmdirSync, mkdirSync, readFileSync } from 'fs'
import { join } from 'path'
import { tmpdir } from 'os'
import { loadTheaModule } from '../helpers/thea-loader'

describe('Roo-migration: utils/logging/CompactTransport', function () {
  const { CompactTransport } = loadTheaModule('src/utils/logging/CompactTransport.ts')

  const root = join(tmpdir(), 'roo-compact-transport-tests')
  const testDir = join(root, `logs-${Date.now()}`)
  const testLogPath = join(testDir, 'test.log')
  let originalWrite

  function cleanupDir(dirPath) {
    if (!existsSync(dirPath)) return
    for (const name of readdirSync(dirPath)) {
      const p = join(dirPath, name)
      const st = lstatSync(p)
      if (st.isDirectory()) cleanupDir(p)
      else unlinkSync(p)
    }
    try { rmdirSync(dirPath) } catch {}
  }

  beforeEach(function () {
    originalWrite = process.stdout.write
    process.stdout.write = () => true
    cleanupDir(testDir)
    mkdirSync(testDir, { recursive: true })
  })

  afterEach(function () {
    process.stdout.write = originalWrite
    cleanupDir(testDir)
  })

  it('creates new log file on first write and appends entries', function () {
    const transport = new CompactTransport({
      level: 'fatal',
      fileOutput: { enabled: true, path: testLogPath },
    })

    transport.write({ t: Date.now(), l: 'info', m: 'test message' })

    let content = readFileSync(testLogPath, 'utf8').trim().split('\n')
    strictEqual(content.length, 2)

    const first = JSON.parse(content[0])
    const second = JSON.parse(content[1])
    strictEqual(first.l, 'info')
    strictEqual(first.m, 'Log session started')
    strictEqual(second.l, 'info')
    strictEqual(second.m, 'test message')

    transport.write({ t: Date.now(), l: 'info', m: 'first' })
    transport.write({ t: Date.now(), l: 'info', m: 'second' })

    content = readFileSync(testLogPath, 'utf8').trim().split('\n')
    strictEqual(content.length, 4)
    const third = JSON.parse(content[2])
    const fourth = JSON.parse(content[3])
    strictEqual(third.m, 'first')
    strictEqual(fourth.m, 'second')

    transport.close()
  })

  it('writes session end marker on close', function () {
    const transport = new CompactTransport({
      level: 'info',
      fileOutput: { enabled: true, path: testLogPath },
    })

    transport.write({ t: Date.now(), l: 'info', m: 'test' })
    transport.close()

    const lines = readFileSync(testLogPath, 'utf8').trim().split('\n')
    const last = JSON.parse(lines[lines.length - 1])
    strictEqual(last.l, 'info')
    strictEqual(last.m, 'Log session ended')
  })

  it('handles deep nested directories for file output', function () {
    const deepDir = join(testDir, 'deep/nested/path')
    const deepPath = join(deepDir, 'test.log')
    const transport = new CompactTransport({
      fileOutput: { enabled: true, path: deepPath },
    })

    try {
      transport.write({ t: Date.now(), l: 'info', m: 'test' })
      ok(existsSync(deepPath))
    } finally {
      transport.close()
      cleanupDir(join(testDir, 'deep'))
    }
  })

  it('handles rapid concurrent writes', async function () {
    const transport = new CompactTransport({
      level: 'fatal',
      fileOutput: { enabled: true, path: testLogPath },
    })

    const entries = Array.from({ length: 50 }, (_, i) => ({ t: Date.now(), l: 'info', m: `test ${i}` }))
    await Promise.all(entries.map((e) => Promise.resolve(transport.write(e))))

    const lines = readFileSync(testLogPath, 'utf8').trim().split('\n')
    strictEqual(lines.length, entries.length + 1)
    transport.close()
  })

  it('converts absolute timestamps to deltas (file output)', function () {
    const transport = new CompactTransport({ level: 'info', fileOutput: { enabled: true, path: testLogPath } })
    const t0 = Date.now()

    transport.write({ t: t0, l: 'info', m: 'first' })
    transport.write({ t: t0 + 100, l: 'info', m: 'second' })
    transport.close()

    const lines = readFileSync(testLogPath, 'utf8').trim().split('\n')
    const e1 = JSON.parse(lines[1])
    const e2 = JSON.parse(lines[2])
    ok(e1.t >= 0 && e1.t < 50)
    strictEqual(e2.t, 100)
  })
})
