import Mocha from "mocha"

export async function run(): Promise<void> {
  const mocha = new Mocha({ ui: "tdd", timeout: 30000, reporter: "spec" })
  // Initialize TDD globals for inline definitions
  // @ts-ignore
  mocha.suite.emit("pre-require", global, "global", mocha)

  suite("SMOKE", () => {
    test("passes", () => {})
  })

  return await new Promise<void>((resolve, reject) =>
    mocha.run((failures) => (failures === 0 ? resolve() : reject(new Error(`${failures} tests failed.`))))
  )
}
