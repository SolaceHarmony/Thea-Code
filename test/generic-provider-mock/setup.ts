import GenericProviderMock, { 
	PROVIDER_CONFIGS, 
	mockOpenAI, 
	mockAnthropic, 
	mockBedrock, 
	mockGemini, 
	mockVertex, 
	mockMistral, 
	mockDeepSeek, 
	mockOllama 
} from "./server"

// Collection of all provider mocks
const providerMocks = {
	openai: mockOpenAI,
	anthropic: mockAnthropic,
	bedrock: mockBedrock,
	gemini: mockGemini,
	vertex: mockVertex,
	mistral: mockMistral,
	deepseek: mockDeepSeek,
	ollama: mockOllama,
}

export default async () => {
	console.log("\n🚀 Starting All Provider Mock Servers...")
	
	const ports: Record<string, number> = {}
	
	// Start all provider mock servers
	for (const [providerName, mock] of Object.entries(providerMocks)) {
		try {
			const port = await mock.start()
			ports[providerName] = port
			
			// Store ports globally for tests
			;(globalThis as any)[`__${providerName.toUpperCase()}_MOCK_PORT__`] = port
			
			console.log(`✅ ${providerName.charAt(0).toUpperCase() + providerName.slice(1)} Mock Server: http://127.0.0.1:${port}`)
		} catch (error) {
			console.error(`❌ Failed to start ${providerName} mock server:`, error)
		}
	}
	
	// Set environment variables for dynamic model testing
	if (ports.openai) process.env.OPENAI_BASE_URL = `http://127.0.0.1:${ports.openai}`
	if (ports.anthropic) process.env.ANTHROPIC_BASE_URL = `http://127.0.0.1:${ports.anthropic}`
	if (ports.bedrock) process.env.AWS_BEDROCK_ENDPOINT = `http://127.0.0.1:${ports.bedrock}`
	if (ports.gemini) process.env.GEMINI_BASE_URL = `http://127.0.0.1:${ports.gemini}`
	if (ports.vertex) process.env.GOOGLE_VERTEX_ENDPOINT = `http://127.0.0.1:${ports.vertex}`
	if (ports.mistral) process.env.MISTRAL_BASE_URL = `http://127.0.0.1:${ports.mistral}`
	if (ports.deepseek) process.env.DEEPSEEK_BASE_URL = `http://127.0.0.1:${ports.deepseek}`
	if (ports.ollama) {
		process.env.OLLAMA_BASE_URL = `http://127.0.0.1:${ports.ollama}`
		// Backward compatibility
		;(globalThis as any).__OLLAMA_PORT__ = ports.ollama
	}
	
	// Store the collection globally for easy access
	;(globalThis as any).__PROVIDER_MOCKS__ = providerMocks
	;(globalThis as any).__PROVIDER_PORTS__ = ports
	
	console.log(`\n🎉 All ${Object.keys(ports).length} Provider Mock Servers Started!`)
	console.log("📋 Available endpoints:")
	Object.entries(ports).forEach(([provider, port]) => {
		console.log(`   ${provider}: http://127.0.0.1:${port}`)
	})
}

// Export everything for individual testing
export { providerMocks, PROVIDER_CONFIGS }