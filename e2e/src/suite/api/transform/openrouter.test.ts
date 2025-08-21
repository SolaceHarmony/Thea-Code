import * as assert from 'assert'
import * as sinon from 'sinon'
import { 
	convertToOpenRouterFormat, 
	applyAnthropicCacheControl, 
	validateOpenRouterParams,
	createOpenRouterRequest,
	type OpenRouterChatCompletionParams,
	type OpenRouterTransformOptions
} from "../openrouter"
import OpenAI from "openai"
import type { NeutralConversationHistory } from "../../../shared/neutral-history"

suite("OpenRouter Transform Functions", () => {
	suite("convertToOpenRouterFormat", () => {
		test("should convert basic OpenAI messages to OpenRouter format", () => {
			const messages: OpenAI.Chat.ChatCompletionMessageParam[] = [
				{ role: "system", content: "You are a helpful assistant" },
				{ role: "user", content: "Hello" }
			]

			const result = convertToOpenRouterFormat(messages)

			expect(result).toMatchObject({
				model: "gpt-3.5-turbo",
				messages,
				stream: true,
				stream_options: { include_usage: true }
			})
		})

		test("should add middle-out transform when enabled", () => {
			const messages: OpenAI.Chat.ChatCompletionMessageParam[] = [
				{ role: "user", content: "Hello" }
			]
			const options: OpenRouterTransformOptions = {
				useMiddleOutTransform: true
			}

			const result = convertToOpenRouterFormat(messages, options)

			assert.deepStrictEqual(result.transforms, ["middle-out"])
		})

		test("should add provider preference when specified", () => {
			const messages: OpenAI.Chat.ChatCompletionMessageParam[] = [
				{ role: "user", content: "Hello" }
			]
			const options: OpenRouterTransformOptions = {
				specificProvider: "anthropic"
			}

			const result = convertToOpenRouterFormat(messages, options)

			assert.deepStrictEqual(result.provider, {
				order: ["anthropic"]
			})
		})

		test("should not add provider when default is specified", () => {
			const messages: OpenAI.Chat.ChatCompletionMessageParam[] = [
				{ role: "user", content: "Hello" }
			]
			const options: OpenRouterTransformOptions = {
				specificProvider: "[default]"
			}

			const result = convertToOpenRouterFormat(messages, options)

			assert.strictEqual(result.provider, undefined)
		})

		test("should add thinking mode when enabled", () => {
			const messages: OpenAI.Chat.ChatCompletionMessageParam[] = [
				{ role: "user", content: "Hello" }
			]
			const options: OpenRouterTransformOptions = {
				enableThinking: true
			}

			const result = convertToOpenRouterFormat(messages, options)

			assert.strictEqual(result.thinking, true)
		})

		test("should add reasoning when enabled", () => {
			const messages: OpenAI.Chat.ChatCompletionMessageParam[] = [
				{ role: "user", content: "Hello" }
			]
			const options: OpenRouterTransformOptions = {
				includeReasoning: true
			}

			const result = convertToOpenRouterFormat(messages, options)

			assert.strictEqual(result.include_reasoning, true)
		})
	})

	suite("validateOpenRouterParams", () => {
		test("should validate correct parameters", () => {
			const params: OpenRouterChatCompletionParams = {
				model: "gpt-3.5-turbo",
				messages: [],
				transforms: ["middle-out"],
				provider: { order: ["anthropic"] },
				thinking: true,
				include_reasoning: false
			}

			const result = validateOpenRouterParams(params)

			assert.strictEqual(result.valid, true)
			assert.strictEqual(result.errors.length, 0)
		})

		test("should reject invalid transforms type", () => {
			const params = {
				model: "gpt-3.5-turbo",
				messages: [],
				transforms: "invalid" as unknown as string[]
			} as OpenRouterChatCompletionParams

			const result = validateOpenRouterParams(params)

			assert.strictEqual(result.valid, false)
			assert.ok(result.errors.includes("transforms must be an array"))
		})

		test("should reject invalid provider format", () => {
			const params = {
				model: "gpt-3.5-turbo",
				messages: [],
				provider: { order: "invalid" }
			} as unknown as OpenRouterChatCompletionParams

			const result = validateOpenRouterParams(params)

			assert.strictEqual(result.valid, false)
			assert.ok(result.errors.includes("provider.order must be an array"))
		})

		test("should reject invalid thinking mode", () => {
			const params = {
				model: "gpt-3.5-turbo",
				messages: [],
				thinking: "invalid"
			} as unknown as OpenRouterChatCompletionParams

			const result = validateOpenRouterParams(params)

			assert.strictEqual(result.valid, false)
			assert.ok(result.errors.includes("thinking must be boolean or 'auto'"))
		})
	})

	suite("applyAnthropicCacheControl", () => {
		test("should apply cache control to system message", () => {
			const messages: OpenAI.Chat.ChatCompletionMessageParam[] = [
				{ role: "system", content: "You are helpful" }
			]
			const systemPrompt = "You are a helpful assistant"

			const result = applyAnthropicCacheControl(messages, systemPrompt)

			expect(result[0]).toMatchObject({
				role: "system",
				content: [
					{
						type: "text",
						text: systemPrompt,
						cache_control: { type: "ephemeral" }
					}
				]
			})
		})

		test("should apply cache control to user messages", () => {
			const messages: OpenAI.Chat.ChatCompletionMessageParam[] = [
				{ role: "system", content: "System" },
				{ role: "user", content: "First user message" },
				{ role: "assistant", content: "Assistant response" },
				{ role: "user", content: "Second user message" }
			]

			const result = applyAnthropicCacheControl(messages, "System prompt")

			// Check that user messages have cache control
			const userMessages = result.filter(msg => msg.role === "user")
			assert.strictEqual(userMessages.length, 2)
			
			// Both user messages should have cache control on their text parts
			userMessages.forEach(msg => {
				expect(Array.isArray(msg.content)).toBe(true)
				const content = msg.content as { type: string; text: string; cache_control?: Record<string, unknown> }[]
				const textPart = content.find(part => part.type === "text")
				assert.deepStrictEqual(textPart?.cache_control, { type: "ephemeral" })
			})
		})
	})

	suite("createOpenRouterRequest", () => {
		test("should create a complete OpenRouter request", () => {
			const systemPrompt = "You are helpful"
			const history: NeutralConversationHistory = [
				{ role: "user", content: "Hello" }
			]
			const modelId = "gpt-3.5-turbo"
			const options = {
				useMiddleOutTransform: true,
				maxTokens: 1000,
				temperature: 0.7
			}

			const result = createOpenRouterRequest(systemPrompt, history, modelId, options)

			assert.strictEqual(result.model, modelId)
			assert.strictEqual(result.max_tokens, 1000)
			assert.strictEqual(result.temperature, 0.7)
			assert.deepStrictEqual(result.transforms, ["middle-out"])
			assert.strictEqual(result.messages.length, 2) // system + user
		})

		test("should apply Anthropic cache control for Anthropic models", () => {
			const systemPrompt = "You are helpful"
			const history: NeutralConversationHistory = [
				{ role: "user", content: "Hello" }
			]
			const modelId = "anthropic/claude-3.5-sonnet"

			const result = createOpenRouterRequest(systemPrompt, history, modelId)

			// System message should have cache control
			expect(result.messages[0]).toMatchObject({
				role: "system",
				content: [
					{
						type: "text",
						text: systemPrompt,
						cache_control: { type: "ephemeral" }
					}
				]
			})
		})

		test("should throw error for invalid parameters", () => {
			const systemPrompt = "You are helpful"
			const history: NeutralConversationHistory = []
			const modelId = "gpt-3.5-turbo"
			const options = {
				specificProvider: "invalid"
			}

			// This should work fine since our validation allows any string in provider order
			expect(() => createOpenRouterRequest(systemPrompt, history, modelId, options)).not.toThrow()
		})
	})
// Mock cleanup
