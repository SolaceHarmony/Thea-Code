import { providerMocks } from "./setup"

export default async () => {
	console.log("\n🛑 Stopping All Provider Mock Servers...")
	
	const stopPromises = Object.entries(providerMocks).map(async ([providerName, mock]) => {
		try {
			await mock.stop()
			console.log(`✅ ${providerName.charAt(0).toUpperCase() + providerName.slice(1)} Mock Server stopped`)
		} catch (error) {
			console.error(`❌ Failed to stop ${providerName} mock server:`, error)
		}
	})
	
	await Promise.all(stopPromises)
	
	// Clear global variables
	const providers = ['openai', 'anthropic', 'bedrock', 'gemini', 'vertex', 'mistral', 'deepseek', 'ollama']
	providers.forEach(provider => {
		delete (globalThis as any)[`__${provider.toUpperCase()}_MOCK_PORT__`]
	})
	delete (globalThis as any).__PROVIDER_MOCKS__
	delete (globalThis as any).__PROVIDER_PORTS__
	delete (globalThis as any).__GENERIC_MOCK_PORT__
	delete (globalThis as any).__OLLAMA_PORT__ // Backward compatibility
	
	// Clear environment variables
	delete process.env.OPENAI_BASE_URL
	delete process.env.ANTHROPIC_BASE_URL
	delete process.env.AWS_BEDROCK_ENDPOINT
	delete process.env.GEMINI_BASE_URL
	delete process.env.GOOGLE_VERTEX_ENDPOINT
	delete process.env.MISTRAL_BASE_URL
	delete process.env.DEEPSEEK_BASE_URL
	delete process.env.OLLAMA_BASE_URL
	
	console.log("🎉 All Provider Mock Servers stopped.")
}