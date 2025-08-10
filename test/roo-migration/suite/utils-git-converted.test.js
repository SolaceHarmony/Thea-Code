const assert = require('assert')
const Module = require('module')
const { loadTheaModule } = require('../helpers/thea-loader')

// Inline mocks for child_process.exec and util.promisify
function withMockedExec(fn) {
  const originalLoad = Module._load
  const mocks = {
    'child_process': {
      exec: (cmd, opts, cb) => {
        const map = withMockedExec._responses || new Map()
        if (map.has(cmd)) return cb(null, map.get(cmd))
        cb(new Error('Unexpected command: ' + cmd))
      }
    },
    'util': {
      promisify: (execFn) => async (cmd, opts) => new Promise((resolve, reject) => {
        execFn(cmd, opts || {}, (err, res) => err ? reject(err) : resolve(res))
      })
    },
    '../../integrations/misc/extract-text': {
      truncateOutput: (t) => t
    }
  }
  Module._load = function (request, parent, isMain) {
    if (request in mocks) return mocks[request]
    return originalLoad.apply(this, arguments)
  }
  try { return fn() } finally { Module._load = originalLoad }
}

function setResponses(map) { withMockedExec._responses = map }

describe('Roo-migration: utils/git', function () {
  it('searchCommits returns results for grep', async function () {
    const res = await withMockedExec(async () => {
      setResponses(new Map([
        ['git --version', { stdout: 'git version 2.39.2', stderr: '' }],
        ['git rev-parse --git-dir', { stdout: '.git', stderr: '' }],
        ['git log -n 10 --format="%H%n%h%n%s%n%an%n%ad" --date=short --grep="test" --regexp-ignore-case', {
          stdout: [
            'abc123def456','abc123','fix: test commit','John Doe','2024-01-06',
            'def456abc789','def456','feat: new feature','Jane Smith','2024-01-05'
          ].join('\n'), stderr: ''
        }]
      ]))
      const { searchCommits } = loadTheaModule('src/utils/git.ts')
      return await searchCommits('test', '/tmp')
    })
    assert.strictEqual(res.length, 2)
    assert.strictEqual(res[0].shortHash, 'abc123')
  })

  it('searchCommits falls back to hash search', async function () {
    const res = await withMockedExec(async () => {
      setResponses(new Map([
        ['git --version', { stdout: 'git version 2.39.2', stderr: '' }],
        ['git rev-parse --git-dir', { stdout: '.git', stderr: '' }],
        ['git log -n 10 --format="%H%n%h%n%s%n%an%n%ad" --date=short --grep="abc123" --regexp-ignore-case', { stdout: '', stderr: '' }],
        ['git log -n 10 --format="%H%n%h%n%s%n%an%n%ad" --date=short --author-date-order abc123', { stdout: [
          'abc123def456','abc123','fix: test commit','John Doe','2024-01-06',
          'def456abc789','def456','feat: new feature','Jane Smith','2024-01-05'
        ].join('\n'), stderr: '' }]
      ]))
      const { searchCommits } = loadTheaModule('src/utils/git.ts')
      return await searchCommits('abc123', '/tmp')
    })
    assert.strictEqual(res.length, 2)
  })

  it('getCommitInfo formats output', async function () {
    const out = await withMockedExec(async () => {
      setResponses(new Map([
        ['git --version', { stdout: 'git version 2.39.2', stderr: '' }],
        ['git rev-parse --git-dir', { stdout: '.git', stderr: '' }],
        ['git show --format="%H%n%h%n%s%n%an%n%ad%n%b" --no-patch abc123', { stdout: [
          'abc123def456','abc123','fix: test commit','John Doe','2024-01-06','Body'].join('\n'), stderr: '' }],
        ['git show --stat --format="" abc123', { stdout: '1 file changed', stderr: '' }],
        ['git show --format="" abc123', { stdout: '@@ -1 +1 @@\n+new', stderr: '' }]
      ]))
      const { getCommitInfo } = loadTheaModule('src/utils/git.ts')
      return await getCommitInfo('abc123', '/tmp')
    })
    assert.ok(out.includes('Commit: abc123'))
    assert.ok(out.includes('Files Changed:'))
  })

  it('getWorkingState returns clean message when no changes', async function () {
    const out = await withMockedExec(async () => {
      setResponses(new Map([
        ['git --version', { stdout: 'git version 2.39.2', stderr: '' }],
        ['git rev-parse --git-dir', { stdout: '.git', stderr: '' }],
        ['git status --short', { stdout: '', stderr: '' }]
      ]))
      const { getWorkingState } = loadTheaModule('src/utils/git.ts')
      return await getWorkingState('/tmp')
    })
    assert.strictEqual(out, 'No changes in working directory')
  })
})
