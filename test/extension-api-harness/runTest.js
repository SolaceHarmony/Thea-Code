const { run } = require('../roo-migration/suite/index')

async function main() {
  try {
    console.log('[info] Using extension-api-harness (alias to legacy roo-migration)')
    await run()
  } catch (err) {
    console.error('Failed to run Extension API harness tests:', err)
    process.exit(1)
  }
}

main().then(r => )
