const path = require('path')
const Mocha = require('mocha')
const { globSync } = require('glob')

module.exports = {
  run: function () {
    const mocha = new Mocha({ ui: 'bdd', color: true, timeout: 20000 })

    const testsRoot = path.resolve(__dirname)

    return new Promise((resolve, reject) => {
      try {
        const files = globSync('**/*.test.js', { cwd: testsRoot })
        files.forEach(f => mocha.addFile(path.resolve(testsRoot, f)))
        mocha.run(failures => {
          if (failures > 0) reject(new Error(`${failures} tests failed`))
          else resolve()
        })
      } catch (e) {
        reject(e)
      }
    })
  }
}
