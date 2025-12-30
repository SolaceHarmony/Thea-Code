const specGlobs = ["src/**/*.test.ts", "src/**/*.test.tsx"]

const envSpec = process.env.MOCHA_SPEC
const spec = envSpec ? envSpec.split(",").map((entry) => entry.trim()).filter(Boolean) : specGlobs

module.exports = {
  extension: ["ts", "tsx"],
  spec,
  require: ["./test/mocha-register.cjs"],
  timeout: 20000,
  reporter: "spec",
}
