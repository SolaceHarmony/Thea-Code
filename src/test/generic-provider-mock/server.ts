import express from "express"
import http from "http"
import type { Request, Response } from "express"
import { findAvailablePort, waitForPortInUse } from "../../utils/port-utils"

const HOST = "127.0.0.1"

// Configuration for different provider behaviors
export interface MockProviderConfig {
	name: string
	models: Array<{ id: string; object?: string; capabilities?: string[] }>
	defaultModel: string
	supportedFormats: ("openai" | "anthropic" | "gemini" | "ollama")[]
	supportsTools: boolean
	supportsStreaming: boolean
	supportsThinking?: boolean
	responsePatterns?: {
		simple?: string
		withTools?: string
		withThinking?: string
		error?: string
	}
	customHeaders?: Record<string, string>
}

// Default configurations for different provider types
export const PROVIDER_CONFIGS: Record<string, MockProviderConfig> = {
	openai: {
		name: "OpenAI",
		models: [
			{ id: "gpt-4", capabilities: ["chat", "tools"] },
			{ id: "gpt-3.5-turbo", capabilities: ["chat", "tools"] },
			{ id: "o1-preview", capabilities: ["chat", "reasoning"] },
		],
		defaultModel: "gpt-4",
		supportedFormats: ["openai"],
		supportsTools: true,
		supportsStreaming: true,
		responsePatterns: {
			simple: "This is a mock OpenAI response.",
			withTools: '{"type":"tool_use","id":"call_123","name":"test_tool","arguments":"{\\"arg1\\":\\"value1\\"}"}',
		},
	},
	anthropic: {
		name: "Anthropic",
		models: [
			{ id: "claude-3-opus-20240229", capabilities: ["chat", "tools", "vision"] },
			{ id: "claude-3-sonnet-20240229", capabilities: ["chat", "tools", "vision"] },
			{ id: "claude-3-5-sonnet-20241022", capabilities: ["chat", "tools", "vision", "thinking"] },
		],
		defaultModel: "claude-3-sonnet-20240229",
		supportedFormats: ["anthropic"],
		supportsTools: true,
		supportsStreaming: true,
		supportsThinking: true,
		responsePatterns: {
			simple: "This is a mock Anthropic response.",
			withTools: '<tool_name>test_tool</tool_name>\\n<parameters>{"arg1": "value1"}</parameters>',
			withThinking: "<think>Let me think about this...</think>This is my response.",
		},
	},
	bedrock: {
		name: "AWS Bedrock",
		models: [
			{ id: "anthropic.claude-3-opus-20240229-v1:0", capabilities: ["chat", "tools", "vision"] },
			{ id: "anthropic.claude-3-sonnet-20240229-v1:0", capabilities: ["chat", "tools", "vision"] },
			{ id: "anthropic.claude-3-haiku-20240307-v1:0", capabilities: ["chat", "tools", "vision"] },
			{ id: "amazon.titan-text-premier-v1:0", capabilities: ["chat"] },
		],
		defaultModel: "anthropic.claude-3-sonnet-20240229-v1:0",
		supportedFormats: ["openai"],
		supportsTools: true,
		supportsStreaming: true,
		responsePatterns: {
			simple: "This is a mock Bedrock response.",
			withTools: '{"type":"tool_use","id":"bedrock_123","name":"test_tool","arguments":"{\\"param\\":\\"value\\"}"}',
		},
	},
	gemini: {
		name: "Google Gemini",
		models: [
			{ id: "gemini-pro", capabilities: ["chat", "tools"] },
			{ id: "gemini-pro-vision", capabilities: ["chat", "vision"] },
			{ id: "gemini-1.5-pro", capabilities: ["chat", "tools", "vision"] },
			{ id: "gemini-1.5-flash", capabilities: ["chat", "tools"] },
		],
		defaultModel: "gemini-pro",
		supportedFormats: ["openai"],
		supportsTools: true,
		supportsStreaming: true,
		responsePatterns: {
			simple: "This is a mock Gemini response.",
			withTools: '{"type":"function_call","id":"gemini_123","name":"test_tool","arguments":"{\\"input\\":\\"value\\"}"}',
		},
	},
	vertex: {
		name: "Google Vertex AI",
		models: [
			{ id: "claude-3-opus@20240229", capabilities: ["chat", "tools", "vision"] },
			{ id: "claude-3-sonnet@20240229", capabilities: ["chat", "tools", "vision"] },
			{ id: "claude-3-haiku@20240307", capabilities: ["chat", "tools", "vision"] },
			{ id: "gemini-1.5-pro-001", capabilities: ["chat", "tools", "vision"] },
		],
		defaultModel: "claude-3-sonnet@20240229",
		supportedFormats: ["openai", "anthropic"],
		supportsTools: true,
		supportsStreaming: true,
		responsePatterns: {
			simple: "This is a mock Vertex AI response.",
			withTools: '{"type":"tool_use","id":"vertex_123","name":"test_tool","input":{"param":"value"}}',
		},
	},
	mistral: {
		name: "Mistral AI",
		models: [
			{ id: "mistral-large-latest", capabilities: ["chat", "tools"] },
			{ id: "mistral-medium-latest", capabilities: ["chat", "tools"] },
			{ id: "mistral-small-latest", capabilities: ["chat", "tools"] },
			{ id: "codestral-latest", capabilities: ["chat", "code"] },
		],
		defaultModel: "mistral-large-latest",
		supportedFormats: ["openai"],
		supportsTools: true,
		supportsStreaming: true,
		responsePatterns: {
			simple: "This is a mock Mistral response.",
			withTools: '{"type":"function_call","id":"mistral_123","function":{"name":"test_tool","arguments":"{\\"param\\":\\"value\\"}"}',
		},
	},
	deepseek: {
		name: "DeepSeek",
		models: [
			{ id: "deepseek-chat", capabilities: ["chat", "tools"] },
			{ id: "deepseek-coder", capabilities: ["chat", "code"] },
			{ id: "deepseek-math", capabilities: ["chat", "math"] },
		],
		defaultModel: "deepseek-chat",
		supportedFormats: ["openai"],
		supportsTools: true,
		supportsStreaming: true,
		responsePatterns: {
			simple: "This is a mock DeepSeek response.",
			withTools: '{"type":"tool_call","id":"deepseek_123","function":{"name":"test_tool","arguments":"{\\"input\\":\\"value\\"}"}',
		},
	},
	ollama: {
		name: "Ollama",
		models: [
			{ id: "llama2", capabilities: ["chat"] },
			{ id: "mistral", capabilities: ["chat", "tools"] },
			{ id: "gemma", capabilities: ["chat"] },
		],
		defaultModel: "llama2",
		supportedFormats: ["openai", "ollama"],
		supportsTools: true,
		supportsStreaming: true,
		responsePatterns: {
			simple: "This is a mock Ollama response.",
		},
	},
	generic: {
		name: "Generic Provider",
		models: [
			{ id: "model-1", capabilities: ["chat", "tools"] },
			{ id: "model-2", capabilities: ["chat"] },
			{ id: "model-3", capabilities: ["chat", "tools", "vision"] },
		],
		defaultModel: "model-1",
		supportedFormats: ["openai", "anthropic"],
		supportsTools: true,
		supportsStreaming: true,
		responsePatterns: {
			simple: "This is a generic mock response.",
			withTools: '{"type":"tool_use","id":"tool_123","name":"generic_tool","input":{"param":"value"}}',
		},
	},
}

class GenericProviderMock {
	private app: express.Application
	private server: http.Server | null = null
	private port: number | null = null
	private config: MockProviderConfig
	private requestLog: any[] = []
	private responseOverrides: Map<string, any> = new Map()

	constructor(config: MockProviderConfig = PROVIDER_CONFIGS.generic) {
		this.config = config
		this.app = express()
		this.setupMiddleware()
		this.setupRoutes()
	}

	private setupMiddleware() {
		this.app.use(express.json({ limit: '10mb' }))
		
		// Log all requests for debugging
		this.app.use((req, res, next) => {
			this.requestLog.push({
				timestamp: new Date(),
				method: req.method,
				path: req.path,
				body: req.body,
				headers: req.headers,
			})
			
			// Add custom headers if configured
			if (this.config.customHeaders) {
				Object.entries(this.config.customHeaders).forEach(([key, value]) => {
					res.setHeader(key, value)
				})
			}
			
			next()
		})
	}

	private setupRoutes() {
		// OpenAI-compatible endpoints
		this.app.get("/v1/models", this.handleListModels.bind(this))
		this.app.post("/v1/chat/completions", this.handleChatCompletion.bind(this))
		this.app.post("/v1/completions", this.handleCompletion.bind(this))
		
		// Anthropic-compatible endpoints
		this.app.post("/v1/messages", this.handleAnthropicMessages.bind(this))
		this.app.post("/v1/messages/count_tokens", this.handleTokenCount.bind(this))
		
		// AWS Bedrock endpoints
		this.app.post("/", this.handleBedrockRequest.bind(this))
		
		// Google Vertex AI endpoints
		this.app.get("/v1/projects/:project/locations/:location/publishers/:publisher/models", this.handleVertexPublisherModels.bind(this))
		this.app.get("/v1/projects/:project/locations/:location/models", this.handleVertexFoundationModels.bind(this))
		this.app.post("/v1/projects/:project/locations/:location/publishers/:publisher/models/:model:streamGenerateContent", this.handleVertexGenerate.bind(this))
		
		// Ollama-specific endpoints
		this.app.post("/api/generate", this.handleOllamaGenerate.bind(this))
		this.app.post("/api/chat", this.handleOllamaChat.bind(this))
		this.app.get("/api/tags", this.handleOllamaTags.bind(this))
		
		// Test/debug endpoints
		this.app.get("/debug/requests", (req, res) => res.json(this.requestLog))
		this.app.post("/debug/override", this.handleOverride.bind(this))
		this.app.delete("/debug/override/:id", this.handleRemoveOverride.bind(this))
	}

	private handleListModels(req: Request, res: Response) {
		console.log(`Mock ${this.config.name}: GET /v1/models`)
		
		const models = this.config.models.map(m => ({
			id: m.id,
			object: m.object || "model",
			created: Date.now(),
			owned_by: this.config.name.toLowerCase().replace(/\s/g, "-"),
			capabilities: m.capabilities,
		}))
		
		res.json({
			data: models,
			object: "list",
		})
	}

	private async handleChatCompletion(req: Request, res: Response) {
		console.log(`Mock ${this.config.name}: POST /v1/chat/completions`)
		const { messages, stream, model, tools, tool_choice } = req.body
		
		// Check for override
		const overrideKey = `chat_${model || this.config.defaultModel}`
		if (this.responseOverrides.has(overrideKey)) {
			return res.json(this.responseOverrides.get(overrideKey))
		}
		
		// Handle tool calls if requested
		if (tools && this.config.supportsTools) {
			return this.handleToolResponse(req, res, "openai")
		}
		
		if (stream && this.config.supportsStreaming) {
			return this.handleStreamingResponse(res, "openai")
		}
		
		// Non-streaming response
		res.json({
			id: `chatcmpl-${Date.now()}`,
			object: "chat.completion",
			created: Date.now(),
			model: model || this.config.defaultModel,
			choices: [{
				index: 0,
				message: {
					role: "assistant",
					content: this.config.responsePatterns?.simple || "Mock response",
				},
				finish_reason: "stop",
			}],
			usage: {
				prompt_tokens: 10,
				completion_tokens: 15,
				total_tokens: 25,
			},
		})
	}

	private handleAnthropicMessages(req: Request, res: Response) {
		console.log(`Mock ${this.config.name}: POST /v1/messages`)
		const { messages, stream, model } = req.body
		
		if (stream && this.config.supportsStreaming) {
			return this.handleStreamingResponse(res, "anthropic")
		}
		
		// Check if we should include thinking
		const useThinking = this.config.supportsThinking && 
			model?.includes("thinking") || model?.includes("o1")
		
		const content = useThinking && this.config.responsePatterns?.withThinking
			? this.config.responsePatterns.withThinking
			: this.config.responsePatterns?.simple || "Mock response"
		
		res.json({
			id: `msg_${Date.now()}`,
			type: "message",
			role: "assistant",
			content: [{ type: "text", text: content }],
			model: model || this.config.defaultModel,
			usage: { input_tokens: 10, output_tokens: 15 },
		})
	}

	private handleStreamingResponse(res: Response, format: "openai" | "anthropic") {
		res.setHeader("Content-Type", "text/event-stream")
		res.setHeader("Cache-Control", "no-cache")
		res.setHeader("Connection", "keep-alive")
		
		const response = this.config.responsePatterns?.simple || "Mock streamed response."
		const chunks = response.split(" ")
		
		if (format === "openai") {
			// Send OpenAI-style streaming chunks
			chunks.forEach((chunk, i) => {
				const data = {
					id: `chatcmpl-${Date.now()}`,
					object: "chat.completion.chunk",
					created: Date.now(),
					model: this.config.defaultModel,
					choices: [{
						index: 0,
						delta: { content: chunk + (i < chunks.length - 1 ? " " : "") },
						finish_reason: i === chunks.length - 1 ? "stop" : null,
					}],
				}
				res.write(`data: ${JSON.stringify(data)}\n\n`)
			})
		} else {
			// Send Anthropic-style streaming
			res.write(`data: ${JSON.stringify({
				type: "message_start",
				message: {
					id: `msg_${Date.now()}`,
					type: "message",
					role: "assistant",
					usage: { input_tokens: 10, output_tokens: 0 },
				},
			})}\n\n`)
			
			res.write(`data: ${JSON.stringify({
				type: "content_block_start",
				index: 0,
				content_block: { type: "text", text: "" },
			})}\n\n`)
			
			chunks.forEach((chunk, i) => {
				res.write(`data: ${JSON.stringify({
					type: "content_block_delta",
					index: 0,
					delta: { type: "text_delta", text: chunk + (i < chunks.length - 1 ? " " : "") },
				})}\n\n`)
			})
			
			res.write(`data: ${JSON.stringify({
				type: "content_block_stop",
				index: 0,
			})}\n\n`)
		}
		
		res.write("data: [DONE]\n\n")
		res.end()
	}

	private handleToolResponse(req: Request, res: Response, format: "openai" | "anthropic") {
		const { stream, model, tools } = req.body
		
		if (!this.config.supportsTools) {
			return res.status(400).json({
				error: {
					message: "Tools not supported by this provider",
					type: "invalid_request_error",
				},
			})
		}
		
		// Get the appropriate tool response pattern
		const toolResponse = this.config.responsePatterns?.withTools || 
			'{"type":"tool_use","id":"call_123","name":"test_tool","arguments":"{}"}';
		
		if (stream) {
			res.setHeader("Content-Type", "text/event-stream")
			res.setHeader("Cache-Control", "no-cache")
			res.setHeader("Connection", "keep-alive")
			
			if (format === "openai") {
				// Stream tool call in OpenAI format
				res.write(`data: ${JSON.stringify({
					id: `chatcmpl-${Date.now()}`,
					object: "chat.completion.chunk",
					created: Date.now(),
					model: model || this.config.defaultModel,
					choices: [{
						index: 0,
						delta: {
							tool_calls: [{
								index: 0,
								id: "call_123",
								type: "function",
								function: {
									name: tools?.[0]?.function?.name || "test_tool",
									arguments: '{"result": "success"}',
								},
							}],
						},
						finish_reason: "tool_calls",
					}],
				})}\n\n`)
			}
			
			res.write("data: [DONE]\n\n")
			res.end()
		} else {
			// Non-streaming tool response
			if (format === "openai") {
				res.json({
					id: `chatcmpl-${Date.now()}`,
					object: "chat.completion",
					created: Date.now(),
					model: model || this.config.defaultModel,
					choices: [{
						index: 0,
						message: {
							role: "assistant",
							content: null,
							tool_calls: [{
								id: "call_123",
								type: "function",
								function: {
									name: tools?.[0]?.function?.name || "test_tool",
									arguments: '{"result": "success"}',
								},
							}],
						},
						finish_reason: "tool_calls",
					}],
					usage: {
						prompt_tokens: 20,
						completion_tokens: 10,
						total_tokens: 30,
					},
				})
			} else {
				res.json({
					id: `msg_${Date.now()}`,
					type: "message",
					role: "assistant",
					content: [{
						type: "tool_use",
						id: "tool_123",
						name: "test_tool",
						input: { result: "success" },
					}],
					model: model || this.config.defaultModel,
					usage: { input_tokens: 20, output_tokens: 10 },
				})
			}
		}
	}

	private handleCompletion(req: Request, res: Response) {
		console.log(`Mock ${this.config.name}: POST /v1/completions`)
		const { prompt, model, stream } = req.body
		
		if (stream) {
			// Handle streaming completions
			res.setHeader("Content-Type", "text/event-stream")
			res.setHeader("Cache-Control", "no-cache")
			res.setHeader("Connection", "keep-alive")
			
			const response = this.config.responsePatterns?.simple || "Mock completion"
			res.write(`data: ${JSON.stringify({
				id: `cmpl-${Date.now()}`,
				object: "text_completion",
				created: Date.now(),
				model: model || this.config.defaultModel,
				choices: [{
					text: response,
					index: 0,
					finish_reason: "stop",
				}],
			})}\n\n`)
			
			res.write("data: [DONE]\n\n")
			res.end()
		} else {
			res.json({
				id: `cmpl-${Date.now()}`,
				object: "text_completion",
				created: Date.now(),
				model: model || this.config.defaultModel,
				choices: [{
					text: this.config.responsePatterns?.simple || "Mock completion",
					index: 0,
					finish_reason: "stop",
				}],
				usage: {
					prompt_tokens: 5,
					completion_tokens: 10,
					total_tokens: 15,
				},
			})
		}
	}

	private handleOllamaGenerate(req: Request, res: Response) {
		console.log(`Mock ${this.config.name}: POST /api/generate`)
		const { prompt, model, stream } = req.body
		
		if (stream !== false) {
			// Ollama defaults to streaming
			res.setHeader("Content-Type", "application/x-ndjson")
			
			const response = this.config.responsePatterns?.simple || "Mock Ollama response"
			const chunks = response.split(" ")
			
			chunks.forEach((chunk, i) => {
				res.write(JSON.stringify({
					model: model || this.config.defaultModel,
					created_at: new Date().toISOString(),
					response: chunk + (i < chunks.length - 1 ? " " : ""),
					done: i === chunks.length - 1,
				}) + "\n")
			})
			
			res.end()
		} else {
			res.json({
				model: model || this.config.defaultModel,
				created_at: new Date().toISOString(),
				response: this.config.responsePatterns?.simple || "Mock Ollama response",
				done: true,
			})
		}
	}

	private handleOllamaChat(req: Request, res: Response) {
		console.log(`Mock ${this.config.name}: POST /api/chat`)
		const { messages, model, stream } = req.body
		
		if (stream !== false) {
			res.setHeader("Content-Type", "application/x-ndjson")
			
			const response = this.config.responsePatterns?.simple || "Mock Ollama chat response"
			res.write(JSON.stringify({
				model: model || this.config.defaultModel,
				created_at: new Date().toISOString(),
				message: {
					role: "assistant",
					content: response,
				},
				done: true,
			}) + "\n")
			
			res.end()
		} else {
			res.json({
				model: model || this.config.defaultModel,
				created_at: new Date().toISOString(),
				message: {
					role: "assistant",
					content: this.config.responsePatterns?.simple || "Mock Ollama chat response",
				},
				done: true,
			})
		}
	}

	private handleTokenCount(req: Request, res: Response) {
		console.log(`Mock ${this.config.name}: POST /v1/messages/count_tokens`)
		const { messages } = req.body
		
		// Simple token estimation
		const text = JSON.stringify(messages || "")
		const tokens = Math.max(1, Math.ceil(text.length / 4))
		
		res.json({ input_tokens: tokens })
	}

	private handleOverride(req: Request, res: Response) {
		const { key, response } = req.body
		this.responseOverrides.set(key, response)
		res.json({ success: true, message: `Override set for key: ${key}` })
	}

	private handleRemoveOverride(req: Request, res: Response) {
		const { id } = req.params
		const deleted = this.responseOverrides.delete(id)
		res.json({ success: deleted, message: deleted ? `Override removed: ${id}` : `No override found: ${id}` })
	}

	// AWS Bedrock handlers
	private handleBedrockRequest(req: Request, res: Response) {
		const target = req.headers['x-amz-target'] as string
		console.log(`Mock Bedrock: ${target}`)
		
		if (target === 'AmazonBedrockControlPlaneService.ListFoundationModels') {
			const models = this.config.models
				.filter(m => m.id.includes('claude') || m.id.includes('titan'))
				.map(m => ({
					modelId: m.id,
					modelName: m.id.split('.')[1] || m.id,
					providerName: m.id.includes('anthropic') ? 'Anthropic' : 'Amazon',
					inputModalities: ['TEXT'],
					outputModalities: ['TEXT'],
					responseStreamingSupported: this.config.supportsStreaming,
					customizationsSupported: [],
					inferenceTypesSupported: ['ON_DEMAND'],
					modelLifecycle: { status: 'ACTIVE' }
				}))
			
			res.json({ modelSummaries: models })
		} else {
			res.status(400).json({ error: 'Unknown Bedrock operation' })
		}
	}

	// Google Vertex AI handlers
	private handleVertexPublisherModels(req: Request, res: Response) {
		const { project, location, publisher } = req.params
		console.log(`Mock Vertex: GET publisher models for ${publisher}`)
		
		const publisherModels = this.config.models
			.filter(m => m.id.includes(publisher) || publisher === 'anthropic')
			.map(m => ({
				name: `projects/${project}/locations/${location}/publishers/${publisher}/models/${m.id}`,
				versions: [{ name: m.id, version: '001' }],
				displayName: m.id,
				description: `Mock ${publisher} model`,
				inputTokenLimit: 200000,
				outputTokenLimit: 8192,
				supportedGenerationMethods: ['generateContent', 'streamGenerateContent']
			}))
		
		res.json({ models: publisherModels })
	}

	private handleVertexFoundationModels(req: Request, res: Response) {
		const { project, location } = req.params
		console.log(`Mock Vertex: GET foundation models`)
		
		const foundationModels = this.config.models
			.filter(m => m.id.includes('gemini'))
			.map(m => ({
				name: `projects/${project}/locations/${location}/models/${m.id}`,
				baseModelId: m.id,
				displayName: m.id,
				description: `Mock foundation model`,
				inputTokenLimit: 128000,
				outputTokenLimit: 8192,
				supportedGenerationMethods: ['generateContent', 'streamGenerateContent']
			}))
		
		res.json({ models: foundationModels })
	}

	private handleVertexGenerate(req: Request, res: Response) {
		const { project, location, publisher, model } = req.params
		console.log(`Mock Vertex: Generate content for ${model}`)
		
		res.json({
			candidates: [{
				content: {
					parts: [{
						text: this.config.responsePatterns?.simple || 'Mock Vertex AI response'
					}],
					role: 'model'
				},
				finishReason: 'STOP',
				index: 0
			}],
			usageMetadata: {
				promptTokenCount: 10,
				candidatesTokenCount: 15,
				totalTokenCount: 25
			}
		})
	}

	// Enhanced Ollama handlers
	private handleOllamaTags(req: Request, res: Response) {
		console.log(`Mock Ollama: GET /api/tags`)
		
		const models = this.config.models.map(m => ({
			name: m.id,
			model: m.id,
			modified_at: new Date().toISOString(),
			size: 4000000000, // 4GB
			digest: `sha256:${Date.now()}`,
			details: {
				parent_model: '',
				format: 'gguf',
				family: 'llama',
				families: ['llama'],
				parameter_size: '7B',
				quantization_level: 'Q4_0'
			}
		}))
		
		res.json({ models })
	}

	// Public methods for server management
	async start(preferredPort?: number): Promise<number> {
		if (this.server) {
			console.log(`Mock ${this.config.name} already running on port ${this.port}`)
			return this.port!
		}

		const preferredRanges: Array<[number, number]> = [
			[10000, 10100],
			[20000, 20100],
			[30000, 30100],
		]

		this.port = await findAvailablePort(
			preferredPort || 10000,
			HOST,
			preferredRanges,
			150
		)

		return new Promise((resolve, reject) => {
			this.server = this.app.listen(this.port, HOST, () => {
				console.log(`Mock ${this.config.name} Server listening on http://${HOST}:${this.port}`)
				resolve(this.port!)
			}).on("error", reject)
		})
	}

	async stop(): Promise<void> {
		return new Promise((resolve, reject) => {
			if (this.server) {
				this.server.close((err) => {
					if (err) {
						console.error(`Failed to stop Mock ${this.config.name} Server:`, err)
						reject(err)
					} else {
						console.log(`Mock ${this.config.name} Server stopped`)
						this.server = null
						this.port = null
						resolve()
					}
				})
			} else {
				resolve()
			}
		})
	}

	getPort(): number | null {
		return this.port
	}

	getRequestLog(): any[] {
		return this.requestLog
	}

	clearRequestLog(): void {
		this.requestLog = []
	}

	setResponseOverride(key: string, response: any): void {
		this.responseOverrides.set(key, response)
	}

	clearResponseOverrides(): void {
		this.responseOverrides.clear()
	}

	updateConfig(config: Partial<MockProviderConfig>): void {
		this.config = { ...this.config, ...config }
	}
}

// Export singleton instances for all providers
export const mockOpenAI = new GenericProviderMock(PROVIDER_CONFIGS.openai)
export const mockAnthropic = new GenericProviderMock(PROVIDER_CONFIGS.anthropic)
export const mockBedrock = new GenericProviderMock(PROVIDER_CONFIGS.bedrock)
export const mockGemini = new GenericProviderMock(PROVIDER_CONFIGS.gemini)
export const mockVertex = new GenericProviderMock(PROVIDER_CONFIGS.vertex)
export const mockMistral = new GenericProviderMock(PROVIDER_CONFIGS.mistral)
export const mockDeepSeek = new GenericProviderMock(PROVIDER_CONFIGS.deepseek)
export const mockOllama = new GenericProviderMock(PROVIDER_CONFIGS.ollama)
export const mockGeneric = new GenericProviderMock(PROVIDER_CONFIGS.generic)

// Export the class for custom configurations
export default GenericProviderMock