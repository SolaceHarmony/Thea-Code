import { JSDOM } from "jsdom"
import Module from "module"
import path from "node:path"
import expectLib from "expect"

const dom = new JSDOM("<!doctype html><html><body></body></html>", { url: "https://localhost" })

const { window } = dom

for (const property of Object.getOwnPropertyNames(window)) {
  if (!(property in globalThis)) {
    // @ts-ignore
    globalThis[property] = (window as any)[property]
  }
}

// Ensure common globals exist
if (!globalThis.window) {
  // @ts-ignore
  globalThis.window = window
}
if (!globalThis.document) {
  // @ts-ignore
  globalThis.document = window.document
}
if (!globalThis.navigator) {
  // @ts-ignore
  globalThis.navigator = window.navigator
}

// React Testing Library expects this flag for act() warnings
// @ts-ignore
globalThis.IS_REACT_ACT_ENVIRONMENT = true

if (!globalThis.requestAnimationFrame) {
  globalThis.requestAnimationFrame = (cb: FrameRequestCallback) => setTimeout(() => cb(Date.now()), 0)
}

if (!globalThis.cancelAnimationFrame) {
  globalThis.cancelAnimationFrame = (handle: ReturnType<typeof setTimeout>) => clearTimeout(handle)
}

// Ignore CSS imports by returning an empty module
require.extensions[".css"] = () => {}

const mochaModule = require("mocha")
const { cleanup } = require("@testing-library/react") as typeof import("@testing-library/react")

// Minimal crypto.getRandomValues implementation
if (!globalThis.crypto) {
  // @ts-ignore
  globalThis.crypto = {}
}
if (typeof globalThis.crypto.getRandomValues !== "function") {
  // @ts-ignore
  globalThis.crypto.getRandomValues = (buffer: Uint8Array) => {
    for (let i = 0; i < buffer.length; i += 1) {
      buffer[i] = Math.floor(Math.random() * 256)
    }
    return buffer
  }
}

// Provide a stubbed matchMedia implementation similar to the Jest setup
if (typeof window.matchMedia !== "function") {
  // @ts-ignore
  window.matchMedia = (query: string) => ({
    matches: false,
    media: query,
    onchange: null,
    addListener: () => {},
    removeListener: () => {},
    addEventListener: () => {},
    removeEventListener: () => {},
    dispatchEvent: () => false,
  })
}

// Register toolkit custom elements for tests before any components render
// eslint-disable-next-line @typescript-eslint/no-var-requires
const { allComponents, provideVSCodeDesignSystem } = require("@vscode/webview-ui-toolkit")
provideVSCodeDesignSystem().register(allComponents)

const projectRoot = path.resolve(__dirname, "..")
const aliasMap: Record<string, string> = {
  "@/": path.join(projectRoot, "src") + path.sep,
  zod: path.join(projectRoot, "node_modules", "zod"),
  "pretty-bytes": path.join(projectRoot, "src", "__mocks__", "pretty-bytes.js"),
  "src/i18n/setup": path.join(projectRoot, "src", "__mocks__", "i18n", "setup.ts"),
  "src/i18n/TranslationContext": path.join(
    projectRoot,
    "src",
    "__mocks__",
    "i18n",
    "TranslationContext.tsx",
  ),
  axios: path.join(projectRoot, "src", "__mocks__", "axios.ts"),
}

// Use Jest's expect library so existing assertions keep working
// @ts-ignore
globalThis.expect = expectLib

// Extend expect with testing-library matchers
require("@testing-library/jest-dom")

// Provide mocha aliases expected by the existing test suite
const addEachHelper = (target: any) => {
  if (!target || target.each) {
    return
  }

  const toRows = (cases: unknown): unknown[][] => {
    if (Array.isArray(cases)) {
      if (cases.length === 0) {
        return []
      }
      return Array.isArray(cases[0]) ? (cases as unknown[][]) : (cases as unknown[]).map((value) => [value])
    }
    if (cases && typeof cases === "object" && "raw" in (cases as any)) {
      const raw = String.raw(cases as TemplateStringsArray)
      return raw
        .trim()
        .split("\n")
        .map((line) => line.split("|").map((cell) => cell.trim()))
    }
    return []
  }

  target.each = (cases: unknown) =>
    (title: string, fn: (...args: unknown[]) => unknown) => {
      toRows(cases).forEach((row) => {
        target(title, () => fn(...row))
      })
    }
}

addEachHelper(mochaModule.it)

if (!Object.prototype.hasOwnProperty.call(globalThis, "it")) {
  Object.defineProperty(globalThis, "it", {
    configurable: true,
    enumerable: true,
    set(value: any) {
      addEachHelper(value)
      Object.defineProperty(globalThis, "it", {
        value,
        writable: true,
        configurable: true,
        enumerable: true,
      })
    },
  })
}

const ensureLazyAlias = (property: string, target: string) => {
  if (!Object.getOwnPropertyDescriptor(globalThis, property)) {
    Object.defineProperty(globalThis, property, {
      configurable: true,
      enumerable: true,
      get() {
        // @ts-ignore
        return globalThis[target]
      },
      set(value: unknown) {
        Object.defineProperty(globalThis, property, {
          configurable: true,
          enumerable: true,
          writable: true,
          value,
        })
      },
    })
  }
}

ensureLazyAlias("beforeAll", "before")
ensureLazyAlias("afterAll", "after")
ensureLazyAlias("test", "it")

const ensureMochaAliases = () => {
  if (!globalThis.test && typeof globalThis.it === "function") {
    // @ts-ignore
    globalThis.test = globalThis.it
  }
  if (!globalThis.beforeAll && typeof globalThis.before === "function") {
    // @ts-ignore
    globalThis.beforeAll = globalThis.before
  }
  if (!globalThis.afterAll && typeof globalThis.after === "function") {
    // @ts-ignore
    globalThis.afterAll = globalThis.after
  }
  if (typeof globalThis.it === "function") {
    addEachHelper(globalThis.it)
  }
  if (!globalThis.test || !globalThis.beforeAll || !globalThis.afterAll) {
    setImmediate(ensureMochaAliases)
  }
}

ensureMochaAliases()

const NodeModule = Module as unknown as {
  _resolveFilename: typeof Module._resolveFilename
}

const originalResolveFilename = NodeModule._resolveFilename
function resolveWithAliases(request: string) {
  if (request.startsWith("@/")) {
    return path.join(aliasMap["@/"], request.slice(2))
  }
  if (request === "zod") {
    return aliasMap.zod
  }
  if (request === "pretty-bytes") {
    return aliasMap["pretty-bytes"]
  }
  if (request === "axios") {
    return aliasMap.axios
  }
  if (request === "src/i18n/setup" || request === "../setup" || request === "./setup") {
    return aliasMap["src/i18n/setup"]
  }
  if (
    request === "src/i18n/TranslationContext" ||
    request === "../TranslationContext" ||
    request === "./TranslationContext"
  ) {
    return aliasMap["src/i18n/TranslationContext"]
  }
  return undefined
}
NodeModule._resolveFilename = function patchedResolveFilename(
  request: string,
  parent: NodeModule | null,
  isMain: boolean,
  options?: NodeModule.RequireResolveOptions,
) {
  const mapped = resolveWithAliases(request)
  return originalResolveFilename.call(this, mapped ?? request, parent, isMain, options)
}

// Clean up React Testing Library DOM between tests
const waitForAfterEach = () => {
  if (typeof globalThis.afterEach === "function") {
    globalThis.afterEach(() => {
      cleanup()
    })
  } else {
    setImmediate(waitForAfterEach)
  }
}

waitForAfterEach()
