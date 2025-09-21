import * as assert from 'assert'
import * as sinon from 'sinon'
import * as proxyquire from 'proxyquire'
import OpenAI from 'openai'

/**
 * Ollama System Role Handling Test Suite
 * 
 * This test suite validates how OllamaHandler properly handles system prompts and system role
 * messages when communicating with Ollama's OpenAI-compatible API. System role handling is
 * crucial for proper model behavior and tool instructions.
 * 
 * Key System Role Behaviors Tested:
 * 1. System prompts are converted to proper "system" role messages in OpenAI format
 * 2. Existing system messages from neutral history are preserved
 * 3. Multiple system messages are handled correctly when present in conversation history
 * 4. Tool information in system prompts remains in system role (not converted to user messages)
 * 5. Consistent behavior across different Ollama model types (llama2, mistral, gemma)
 * 
 * Why System Role Matters for Ollama:
 * - Ollama uses OpenAI-compatible API format where system role sets model behavior
 * - Tool definitions and instructions should be in system role for proper function calling
 * - System role content is not counted as conversation turns in many models
 * - Improper role assignment can affect model instruction-following capability
 */
suite("Ollama System Role Handling", () => {
	let OllamaHandler: any
	let mockCreate: sinon.SinonStub
	let mockConvertToOllamaHistory: sinon.SinonStub

	// Test across multiple common Ollama models to ensure consistent behavior
	const availableModels: string[] = ["llama2", "mistral", "gemma"]

	setup(() => {
		// Create fresh stubs for each test
		mockCreate = sinon.stub()
		mockConvertToOllamaHistory = sinon.stub()

		// Mock the OllamaHandler with proxyquire to capture API calls
		OllamaHandler = proxyquire('../ollama', {
			'openai': {
				__esModule: true,
				default: sinon.stub().callsFake(() => ({
					chat: {
						completions: {
							create: mockCreate,
						},
					},
				})),
			},
			'../transform/neutral-ollama-format': {
				convertToOllamaHistory: mockConvertToOllamaHistory,
			},
			'./openai': {
				OpenAiHandler: sinon.stub().callsFake(() => ({
					extractToolCalls: sinon.stub().returns([]),
					hasToolCalls: sinon.stub().returns(false),
					processToolUse: sinon.stub().resolves("tool result"),
				})),
			},
			'../../utils/json-xml-bridge': {
				HybridMatcher: sinon.stub().callsFake(() => ({
					update: sinon.stub().callsFake((text: string) => [{ matched: false, data: text }]),
					final: sinon.stub().returns([]),
					getDetectedFormat: sinon.stub().returns("json"),
				})),
			},
		}).OllamaHandler

		// Default mock behavior for successful streaming
		mockCreate.callsFake(({ messages }: { messages: OpenAI.Chat.ChatCompletionMessageParam[] }) => {
			// Store messages for inspection
			(mockCreate as any).lastMessages = messages
// Mock removed - needs manual implementation],
// 					}
					yield {
						choices: [{
							delta: {},
							index: 0,
						}],
						usage: {
							prompt_tokens: 10,
							completion_tokens: 5,
							total_tokens: 15,
						},
					}
				},
			}
		})
	})

	teardown(() => {
		sinon.restore()
	})

	/**
	 * System Prompt to System Role Tests
	 * 
	 * Verify that system prompts provided to createMessage are properly converted
	 * to "system" role messages in the OpenAI-compatible format sent to Ollama.
	 */
	suite("System Prompt to System Role Conversion", () => {
		availableModels.forEach((modelId) => {
			test(`should use system role for system prompts with ${modelId} model`, async () => {
				// Mock the conversion to return proper OpenAI format messages
				mockConvertToOllamaHistory.returns([
					{ role: "system", content: "You are a helpful assistant with access to the following tools: tool1, tool2, tool3" },
					{ role: "user", content: "Hello" }
				])

				const handler = new OllamaHandler({
					ollamaBaseUrl: "http://localhost:10000",
					ollamaModelId: modelId,
				})

				const neutralHistory = [
					{ role: "user", content: [{ type: "text", text: "Hello" }] },
				]

				const systemPrompt = "You are a helpful assistant with access to the following tools: tool1, tool2, tool3"

				// Call createMessage
				const stream = handler.createMessage(systemPrompt, neutralHistory)

				// Consume the stream to ensure the API is called
				for await (const _chunk of stream) {
					// Consume the response
				}

				// Verify the conversion was called with the neutral history
				assert.ok(mockConvertToOllamaHistory.calledWith(neutralHistory))

				// Get the messages that were sent to the OpenAI client
				const sentMessages = (mockCreate as any).lastMessages as OpenAI.Chat.ChatCompletionMessageParam[]

				// Verify that the system prompt was sent with the system role
				assert.notStrictEqual(sentMessages, undefined)
				assert.ok(sentMessages.length >= 2)

				// The first message should be the system prompt with role 'system'
				assert.deepStrictEqual(sentMessages[0], {
					role: "system",
					content: systemPrompt,
				})

				// The second message should be the user message
				assert.deepStrictEqual(sentMessages[1], {
					role: "user",
					content: "Hello",
				})
			})
		})

		test("should handle empty system prompts gracefully", async () => {
			mockConvertToOllamaHistory.returns([
				{ role: "user", content: "Hello without system prompt" }
			])

			const handler = new OllamaHandler({
				ollamaBaseUrl: "http://localhost:10000",
				ollamaModelId: "llama2",
			})

			const neutralHistory = [
				{ role: "user", content: [{ type: "text", text: "Hello without system prompt" }] },
			]

			// Empty system prompt
			const stream = handler.createMessage("", neutralHistory)

			for await (const _chunk of stream) {
				// Consume the response
			}

			const sentMessages = (mockCreate as any).lastMessages as OpenAI.Chat.ChatCompletionMessageParam[]

			// Should not have any system messages when system prompt is empty
			const systemMessages = sentMessages.filter(msg => msg.role === "system")
			assert.strictEqual(systemMessages.length, 0)

			// Should still have the user message
			assert.ok(sentMessages.some(msg => msg.role === "user"))
		})
	})

	/**
	 * Existing System Messages Preservation Tests
	 * 
	 * Verify that system messages already present in the neutral conversation history
	 * are properly preserved and handled, with correct priority over additional system prompts.
	 */
	suite("Existing System Messages Preservation", () => {
		availableModels.forEach((modelId) => {
			test(`should preserve existing system messages from neutral history with ${modelId} model`, async () => {
				// Mock conversion to include existing system message
				mockConvertToOllamaHistory.returns([
					{ role: "system", content: "Existing system message" },
					{ role: "user", content: "Hello" }
				])

				const handler = new OllamaHandler({
					ollamaBaseUrl: "http://localhost:10000",
					ollamaModelId: modelId,
				})

				const neutralHistory = [
					{ role: "system", content: [{ type: "text", text: "Existing system message" }] },
					{ role: "user", content: [{ type: "text", text: "Hello" }] },
				]

				// Additional system prompt should not override existing system message
				const additionalSystemPrompt = "Additional system prompt"

				const stream = handler.createMessage(additionalSystemPrompt, neutralHistory)

				for await (const _chunk of stream) {
					// Consume the response
				}

				const sentMessages = (mockCreate as any).lastMessages as OpenAI.Chat.ChatCompletionMessageParam[]

				// Should preserve the existing system message from history
				assert.notStrictEqual(sentMessages, undefined)
				assert.ok(sentMessages.length >= 2)

				// The first message should be the existing system message
				assert.deepStrictEqual(sentMessages[0], {
					role: "system",
					content: "Existing system message",
				})

				// The second message should be the user message
				assert.deepStrictEqual(sentMessages[1], {
					role: "user",
					content: "Hello",
				})

				// The additional system prompt should not be added as a separate system message
				// when there's already a system message in the history
				const systemMessages = sentMessages.filter(msg => msg.role === "system")
				assert.strictEqual(systemMessages.length, 1, "Should only have one system message")
			})
		})

		availableModels.forEach((modelId) => {
			test(`should handle multiple system messages from neutral history with ${modelId} model`, async () => {
				// Mock conversion to include multiple system messages
				mockConvertToOllamaHistory.returns([
					{ role: "system", content: "System message 1" },
					{ role: "system", content: "System message 2" },
					{ role: "user", content: "Hello" }
				])

				const handler = new OllamaHandler({
					ollamaBaseUrl: "http://localhost:10000",
					ollamaModelId: modelId,
				})

				const neutralHistory = [
					{ role: "system", content: [{ type: "text", text: "System message 1" }] },
					{ role: "system", content: [{ type: "text", text: "System message 2" }] },
					{ role: "user", content: [{ type: "text", text: "Hello" }] },
				]

				// Call createMessage with empty system prompt to preserve history
				const stream = handler.createMessage("", neutralHistory)

				for await (const _chunk of stream) {
					// Consume the response
				}

				const sentMessages = (mockCreate as any).lastMessages as OpenAI.Chat.ChatCompletionMessageParam[]

				// Verify that both system messages were preserved
				assert.notStrictEqual(sentMessages, undefined)
				assert.ok(sentMessages.length >= 3)

				// The first two messages should be the system messages in order
				assert.deepStrictEqual(sentMessages[0], {
					role: "system",
					content: "System message 1",
				})

				assert.deepStrictEqual(sentMessages[1], {
					role: "system",
					content: "System message 2",
				})

				// The third message should be the user message
				assert.deepStrictEqual(sentMessages[2], {
					role: "user",
					content: "Hello",
				})
			})
		})
	})

	/**
	 * Tool Information Handling Tests
	 * 
	 * Verify that tool-related information and complex conversation histories
	 * with tool use maintain proper role assignments, particularly ensuring
	 * tool definitions remain in system role rather than being converted to user messages.
	 */
	suite("Tool Information Role Handling", () => {
		availableModels.forEach((modelId) => {
			test(`should not convert tool information to user messages with ${modelId} model`, async () => {
				// Mock conversion to properly handle tool-related messages
				mockConvertToOllamaHistory.returns([
					{ role: "system", content: "You have access to the following tools: calculator" },
					{ role: "user", content: "Use a tool" },
					{ role: "assistant", content: "I will use a tool\n\nFunction: calculator(expression='2+2')" },
					{ role: "user", content: "Tool result: 4" }
				])

				const handler = new OllamaHandler({
					ollamaBaseUrl: "http://localhost:10000",
					ollamaModelId: modelId,
				})

				const neutralHistory = [
					{
						role: "user",
						content: [{ type: "text", text: "Use a tool" }],
					},
					{
						role: "assistant",
						content: [
							{ type: "text", text: "I will use a tool" },
							{
								type: "tool_use",
								id: "tool1",
								name: "calculator",
								input: { expression: "2+2" },
							},
						],
					},
					{
						role: "tool",
						content: [
							{
								type: "tool_result",
								tool_use_id: "tool1",
								content: [{ type: "text", text: "4" }],
							},
						],
					},
				]

				const systemPrompt = "You have access to the following tools: calculator"

				const stream = handler.createMessage(systemPrompt, neutralHistory)

				for await (const _chunk of stream) {
					// Consume the response
				}

				const sentMessages = (mockCreate as any).lastMessages as OpenAI.Chat.ChatCompletionMessageParam[]

				// Verify that the system prompt was sent with the system role
				assert.notStrictEqual(sentMessages, undefined)
				assert.deepStrictEqual(sentMessages[0], {
					role: "system",
					content: systemPrompt,
				})

				// Verify that tool definitions are not leaked into user messages
				const userMessages = sentMessages.filter(msg => 
					msg.role === "user" && 
					typeof msg.content === "string" && 
					msg.content.includes("calculator")
				)
				assert.strictEqual(userMessages.length, 0, "Tool definitions should not appear in user messages")

				// Verify system message contains tool information properly
				const systemMessages = sentMessages.filter(msg => msg.role === "system")
				assert.ok(systemMessages.length >= 1, "Should have at least one system message")
				assert.ok(systemMessages[0].content.includes("calculator"), "System message should contain tool information")
			})
		})

		test("should maintain role integrity in complex conversations", async () => {
			// Mock a complex conversation with multiple role types
			mockConvertToOllamaHistory.returns([
				{ role: "system", content: "You are an assistant with tools" },
				{ role: "user", content: "Calculate something" },
				{ role: "assistant", content: "I'll help with calculation" },
				{ role: "user", content: "Thank you" }
			])

			const handler = new OllamaHandler({
				ollamaBaseUrl: "http://localhost:10000",
				ollamaModelId: "llama2",
			})

			const complexHistory = [
				{ role: "user", content: [{ type: "text", text: "Calculate something" }] },
				{ role: "assistant", content: [{ type: "text", text: "I'll help with calculation" }] },
				{ role: "user", content: [{ type: "text", text: "Thank you" }] },
			]

			const systemPrompt = "You are an assistant with tools"

			const stream = handler.createMessage(systemPrompt, complexHistory)

			for await (const _chunk of stream) {
				// Consume the response
			}

			const sentMessages = (mockCreate as any).lastMessages as OpenAI.Chat.ChatCompletionMessageParam[]

			// Verify role distribution is correct
			const roleCount = sentMessages.reduce((acc, msg) => {
				acc[msg.role] = (acc[msg.role] || 0) + 1
				return acc
			}, {} as Record<string, number>)

			assert.ok(roleCount.system >= 1, "Should have system messages")
			assert.ok(roleCount.user >= 1, "Should have user messages") 
			assert.ok(roleCount.assistant >= 0, "May have assistant messages")
			
			// System message should be first
			assert.strictEqual(sentMessages[0].role, "system", "First message should be system role")
		})
	})
// Mock cleanup
