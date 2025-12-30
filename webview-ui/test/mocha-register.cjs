const path = require("node:path")
const fs = require("node:fs")

const configFile = path.resolve(__dirname, "../babel.config.cjs")
const hasConfig = fs.existsSync(configFile)

require("@babel/register")({
  extensions: [".ts", ".tsx", ".js", ".jsx"],
  ignore: [/node_modules\/(@babel|core-js|regenerator-runtime)/],
  babelrc: false,
  configFile: hasConfig ? configFile : undefined,
})

require("./mocha-setup")
