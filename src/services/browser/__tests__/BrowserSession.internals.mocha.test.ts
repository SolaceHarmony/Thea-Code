import assert from "node:assert/strict"
import sinon from "sinon"

// Import class under test
import { BrowserSession } from "../../browser/BrowserSession"

// Minimal mock for vscode.ExtensionContext with only globalState we need
const createMockContext = () => {
  const store = new Map<string, unknown>()
  const globalState = {
    get: <T = unknown>(key: string): T | undefined => store.get(key) as T | undefined,
    update: (key: string, value: unknown): Promise<void> => {
      if (value === undefined) {
        store.delete(key)
      } else {
        store.set(key, value)
      }
      return Promise.resolve()
    },
  }
  return { globalState } as unknown as import("vscode").ExtensionContext
}

describe("BrowserSession internals", () => {
  let ctx: import("vscode").ExtensionContext
  let session: BrowserSession

  afterEach(() => {
    sinon.restore()
  })

  beforeEach(() => {
    ctx = createMockContext()
    session = new BrowserSession(ctx)
  })

  describe("getRootDomain (private)", () => {
    it("handles localhost with ports and www prefix", () => {
      const getRootDomain = (session as unknown as { [k: string]: (u: string) => string })["getRootDomain"].bind(
        session
      ) as (u: string) => string

      assert.equal(getRootDomain("http://localhost:3000/path"), "localhost:3000")
      assert.equal(getRootDomain("https://www.example.com/foo"), "example.com")
    })

    it("returns input for invalid URLs", () => {
      const getRootDomain = (session as unknown as { [k: string]: (u: string) => string })["getRootDomain"].bind(
        session
      ) as (u: string) => string

      const input = "not a url"
      assert.equal(getRootDomain(input), input)
    })
  })

  describe("navigatePageToUrl (private)", () => {
    it("calls page.goto with expected options and waits for HTML stability", async () => {
      // Arrange custom timeouts via globalState
      await ctx.globalState.update("pageGotoTimeoutMs", 1234)
      await ctx.globalState.update("htmlStableTimeoutMs", 2345)

      // Fake Page implementing the minimal surface we use
      const gotoStub = sinon.stub().resolves()
      const fake = { goto: gotoStub }
      const fakePage = fake as unknown as import("puppeteer-core").Page

      // Stub waitTillHTMLStable to avoid loop and assert it's called with configured timeout
      const waitStub = sinon
              .stub(
                session as unknown as {
                  waitTillHTMLStable: (
                    p: import("puppeteer-core").Page,
                    t?: number
                  ) => Promise<void>
                },
                "waitTillHTMLStable"
              )
              .resolves()

      const navigate = (session as unknown as { [k: string]: (p: any, u: string) => Promise<void> })[
        "navigatePageToUrl"
      ].bind(session) as (p: any, u: string) => Promise<void>

      // Act
      await navigate(fakePage, "https://example.com/path")

      // Assert goto called with both events and custom timeout
      sinon.assert.calledOnce(gotoStub)
      const args = gotoStub.getCall(0).args
      assert.equal(args[0], "https://example.com/path")
      const opts = args[1] as { timeout: number; waitUntil: string[] }
      assert.equal(opts.timeout, 1234)
      assert.ok(Array.isArray(opts.waitUntil), "waitUntil should be an array")
      assert.ok(opts.waitUntil.includes("domcontentloaded"))
      assert.ok(opts.waitUntil.includes("networkidle2"))

      // Assert stable wait invoked with custom timeout
      sinon.assert.calledOnce(waitStub)
      const wsArgs = waitStub.getCall(0).args
      assert.equal(wsArgs[0], fakePage)
      assert.equal(wsArgs[1], 2345)
    })
  })
})
