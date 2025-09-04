import { EmbeddedMcpProvider } from "../services/mcp/providers/EmbeddedMcpProvider"

declare const global: typeof globalThis & { __MCP_PROVIDER__?: EmbeddedMcpProvider }

// Ensure predictable, quiet MCP behavior in unit tests
process.env.NODE_ENV = process.env.NODE_ENV || "test"
process.env.THEA_E2E = process.env.THEA_E2E || "1"
process.env.THEA_DISABLE_MCP_SDK = process.env.THEA_DISABLE_MCP_SDK || "1"
process.env.THEA_SKIP_MCP_PORT_WAIT = process.env.THEA_SKIP_MCP_PORT_WAIT || "1"
process.env.THEA_SILENT_MCP_LOGS = process.env.THEA_SILENT_MCP_LOGS || "1"

export const mochaHooks = {
  async beforeAll() {
    // Start a single embedded MCP server for the whole test run
    if (!global.__MCP_PROVIDER__) {
      const provider = await EmbeddedMcpProvider.create({ port: 0, hostname: "127.0.0.1" })
      await provider.start()
      global.__MCP_PROVIDER__ = provider
    }
  },
  async afterAll() {
    if (global.__MCP_PROVIDER__) {
      try { await global.__MCP_PROVIDER__.stop() } catch {}
      global.__MCP_PROVIDER__ = undefined
    }
  },
}
