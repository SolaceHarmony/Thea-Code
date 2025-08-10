import GenericProviderMock, { 
	MockProviderConfig, 
	PROVIDER_CONFIGS,
	mockOpenAI,
	mockAnthropic,
	mockOllama,
	mockGeneric
} from "./server"

/**
 * Test scenario for verifying provider behavior
 */
export interface TestScenario {
	name: string
	description: string
	request: {
		endpoint: string
		method: "GET" | "POST" | "DELETE"
		body?: any
		headers?: Record<string, string>
	}
	expectedResponse?: {
		status?: number
		bodyContains?: string[]
		bodyMatches?: RegExp[]
		headers?: Record<string, string>
	}
	responseOverride?: any
}

/**
 * Common test scenarios for different provider behaviors
 */
export const TEST_SCENARIOS: Record<string, TestScenario[]> = {
	basic_chat: [
		{
			name: "simple_message",
			description: "Basic chat completion without streaming",
			request: {
				endpoint: "/v1/chat/completions",
				method: "POST",
				body: {
					model: "gpt-4",
					messages: [{ role: "user", content: "Hello" }],
					stream: false,
				},
			},
			expectedResponse: {
				status: 200,
				bodyContains: ["choices", "message", "content"],
			},
		},
		{
			name: "streaming_message",
			description: "Chat completion with streaming",
			request: {
				endpoint: "/v1/chat/completions",
				method: "POST",
				body: {
					model: "gpt-4",
					messages: [{ role: "user", content: "Hello" }],
					stream: true,
				},
			},
			expectedResponse: {
				status: 200,
				headers: { "content-type": "text/event-stream" },
			},
		},
	],
	tool_use: [
		{
			name: "openai_tools",
			description: "OpenAI function calling",
			request: {
				endpoint: "/v1/chat/completions",
				method: "POST",
				body: {
					model: "gpt-4",
					messages: [{ role: "user", content: "Use the weather tool" }],
					tools: [{
						type: "function",
						function: {
							name: "get_weather",
							description: "Get weather for a location",
							parameters: {
								type: "object",
								properties: {
									location: { type: "string" },
								},
								required: ["location"],
							},
						},
					}],
					stream: false,
				},
			},
			expectedResponse: {
				status: 200,
				bodyContains: ["tool_calls", "function", "get_weather"],
			},
		},
		{
			name: "anthropic_tools",
			description: "Anthropic tool use format",
			request: {
				endpoint: "/v1/messages",
				method: "POST",
				body: {
					model: "claude-3-sonnet-20240229",
					messages: [{ role: "user", content: "Use a tool" }],
					tools: [{
						name: "calculator",
						description: "Perform calculations",
						input_schema: {
							type: "object",
							properties: {
								expression: { type: "string" },
							},
							required: ["expression"],
						},
					}],
					stream: false,
				},
			},
			expectedResponse: {
				status: 200,
				bodyContains: ["tool_use", "calculator"],
			},
		},
	],
	thinking: [
		{
			name: "claude_thinking",
			description: "Claude with thinking tags",
			request: {
				endpoint: "/v1/messages",
				method: "POST",
				body: {
					model: "claude-3-5-sonnet-thinking",
					messages: [{ role: "user", content: "Think about this" }],
					stream: false,
				},
			},
			expectedResponse: {
				status: 200,
				bodyMatches: [/<think>.*<\/think>/],
			},
		},
	],
	error_handling: [
		{
			name: "invalid_model",
			description: "Request with invalid model",
			request: {
				endpoint: "/v1/chat/completions",
				method: "POST",
				body: {
					model: "invalid-model",
					messages: [{ role: "user", content: "Test" }],
				},
			},
			responseOverride: {
				error: {
					message: "Model not found",
					type: "invalid_request_error",
					code: "model_not_found",
				},
			},
			expectedResponse: {
				status: 404,
				bodyContains: ["error", "model_not_found"],
			},
		},
		{
			name: "rate_limit",
			description: "Rate limit error response",
			request: {
				endpoint: "/v1/chat/completions",
				method: "POST",
				body: {
					model: "gpt-4",
					messages: [{ role: "user", content: "Test" }],
				},
			},
			responseOverride: {
				error: {
					message: "Rate limit exceeded",
					type: "rate_limit_error",
					code: "rate_limit_exceeded",
				},
			},
			expectedResponse: {
				status: 429,
				bodyContains: ["error", "rate_limit"],
			},
		},
	],
}

/**
 * Helper class for testing with the generic provider mock
 */
export class ProviderMockTestHelper {
	private mock: GenericProviderMock
	private baseUrl: string = ""

	constructor(mock: GenericProviderMock) {
		this.mock = mock
	}

	/**
	 * Start the mock server and return the URL
	 */
	async start(port?: number): Promise<string> {
		const actualPort = await this.mock.start(port)
		this.baseUrl = `http://127.0.0.1:${actualPort}`
		return this.baseUrl
	}

	/**
	 * Stop the mock server
	 */
	async stop(): Promise<void> {
		await this.mock.stop()
	}

	/**
	 * Get the base URL of the running server
	 */
	getBaseUrl(): string {
		if (!this.baseUrl) {
			throw new Error("Mock server not started. Call start() first.")
		}
		return this.baseUrl
	}

	/**
	 * Run a test scenario
	 */
	async runScenario(scenario: TestScenario): Promise<{
		success: boolean
		errors: string[]
		response?: any
	}> {
		const errors: string[] = []
		
		// Set up response override if specified
		if (scenario.responseOverride) {
			const key = `${scenario.request.endpoint}_${scenario.request.method}`
			this.mock.setResponseOverride(key, scenario.responseOverride)
		}

		try {
			const url = `${this.baseUrl}${scenario.request.endpoint}`
			const response = await fetch(url, {
				method: scenario.request.method,
				headers: {
					"Content-Type": "application/json",
					...scenario.request.headers,
				},
				body: scenario.request.body ? JSON.stringify(scenario.request.body) : undefined,
			})

			// Check status code
			if (scenario.expectedResponse?.status && response.status !== scenario.expectedResponse.status) {
				errors.push(`Expected status ${scenario.expectedResponse.status}, got ${response.status}`)
			}

			// Check headers
			if (scenario.expectedResponse?.headers) {
				for (const [key, value] of Object.entries(scenario.expectedResponse.headers)) {
					const actual = response.headers.get(key)
					if (actual !== value) {
						errors.push(`Expected header ${key}="${value}", got "${actual}"`)
					}
				}
			}

			// Check body content
			const contentType = response.headers.get("content-type")
			let body: any

			if (contentType?.includes("text/event-stream")) {
				body = await response.text()
			} else if (contentType?.includes("application/json")) {
				body = await response.json()
			} else {
				body = await response.text()
			}

			// Check body contains
			if (scenario.expectedResponse?.bodyContains) {
				const bodyStr = typeof body === "string" ? body : JSON.stringify(body)
				for (const expected of scenario.expectedResponse.bodyContains) {
					if (!bodyStr.includes(expected)) {
						errors.push(`Body does not contain "${expected}"`)
					}
				}
			}

			// Check body matches regex
			if (scenario.expectedResponse?.bodyMatches) {
				const bodyStr = typeof body === "string" ? body : JSON.stringify(body)
				for (const regex of scenario.expectedResponse.bodyMatches) {
					if (!regex.test(bodyStr)) {
						errors.push(`Body does not match pattern ${regex}`)
					}
				}
			}

			return {
				success: errors.length === 0,
				errors,
				response: body,
			}
		} catch (error) {
			errors.push(`Request failed: ${error}`)
			return {
				success: false,
				errors,
			}
		} finally {
			// Clear override
			if (scenario.responseOverride) {
				this.mock.clearResponseOverrides()
			}
		}
	}

	/**
	 * Run multiple scenarios and return results
	 */
	async runScenarios(scenarios: TestScenario[]): Promise<Map<string, {
		success: boolean
		errors: string[]
		response?: any
	}>> {
		const results = new Map()
		
		for (const scenario of scenarios) {
			const result = await this.runScenario(scenario)
			results.set(scenario.name, result)
		}
		
		return results
	}

	/**
	 * Get request log from the mock server
	 */
	getRequestLog(): any[] {
		return this.mock.getRequestLog()
	}

	/**
	 * Clear request log
	 */
	clearRequestLog(): void {
		this.mock.clearRequestLog()
	}

	/**
	 * Set a custom response for a specific endpoint
	 */
	setCustomResponse(key: string, response: any): void {
		this.mock.setResponseOverride(key, response)
	}

	/**
	 * Clear all custom responses
	 */
	clearCustomResponses(): void {
		this.mock.clearResponseOverrides()
	}
}

/**
 * Create a test helper for a specific provider type
 */
export function createTestHelper(providerType: keyof typeof PROVIDER_CONFIGS): ProviderMockTestHelper {
	const config = PROVIDER_CONFIGS[providerType]
	const mock = new GenericProviderMock(config)
	return new ProviderMockTestHelper(mock)
}

/**
 * Create a test helper with custom configuration
 */
export function createCustomTestHelper(config: MockProviderConfig): ProviderMockTestHelper {
	const mock = new GenericProviderMock(config)
	return new ProviderMockTestHelper(mock)
}

/**
 * Quick test runner for common scenarios
 */
export async function testProviderBehavior(
	providerType: keyof typeof PROVIDER_CONFIGS,
	scenarioTypes: (keyof typeof TEST_SCENARIOS)[] = ["basic_chat"]
): Promise<{
	passed: number
	failed: number
	results: Map<string, any>
}> {
	const helper = createTestHelper(providerType)
	const allResults = new Map()
	let passed = 0
	let failed = 0

	try {
		await helper.start()
		
		for (const scenarioType of scenarioTypes) {
			const scenarios = TEST_SCENARIOS[scenarioType] || []
			const results = await helper.runScenarios(scenarios)
			
			for (const [name, result] of results) {
				allResults.set(`${scenarioType}/${name}`, result)
				if (result.success) {
					passed++
				} else {
					failed++
				}
			}
		}
	} finally {
		await helper.stop()
	}

	return { passed, failed, results: allResults }
}

// Export pre-configured helpers for common providers
export const openAITestHelper = new ProviderMockTestHelper(mockOpenAI)
export const anthropicTestHelper = new ProviderMockTestHelper(mockAnthropic)
export const ollamaTestHelper = new ProviderMockTestHelper(mockOllama)
export const genericTestHelper = new ProviderMockTestHelper(mockGeneric)