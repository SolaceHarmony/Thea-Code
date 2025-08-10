import genericProviderSetup from "./generic-provider-mock/setup"
import mcpSetup from "./mcp-mock-server/setup"
import openaiSetup from "./openai-mock/setup"

module.exports = async () => {
	// Start the generic provider mock that handles Ollama, OpenAI, Anthropic, etc.
	await genericProviderSetup()
	// Expose port to tests if available (for backward compatibility)
	if ((globalThis as any).__GENERIC_MOCK_PORT__) {
		console.log(`GlobalSetup: Generic Provider mock port ${(globalThis as any).__GENERIC_MOCK_PORT__}`)
	}
	await mcpSetup()
	await openaiSetup()
	// Note: McpToolExecutor initialization removed to avoid conflicts with mock server
	// Individual tests that need it will initialize it themselves with mocks
}
