const path = require('path')
const { run } = require('./suite/index')

async function main() {
  try {
    await run()
  } catch (err) {
    console.error('Failed to run Roo migration tests:', err)
    process.exit(1)
  }
}

main()
