import * as assert from 'assert'
import * as sinon from 'sinon'
import * as proxyquire from 'proxyquire'

import type { ApiHandlerOptions } from "../../../shared/api"
import type {
	NeutralContentBlock,
	NeutralConversationHistory,
	NeutralMessageContent,
	NeutralTextContentBlock,
} from "../../../shared/neutral-history"
import type { ApiStreamChunk } from "../../transform/stream"

const GEMINI_MODEL_ID = "gemini-2.0-flash-thinking-exp-1219"

type GeminiHistoryPayload = Array<{ role: string; parts: Array<Record<string, unknown>> }>

type GeminiHandlerConstructor = typeof import("../gemini").GeminiHandler

type GeminiHandlerInstance = InstanceType<GeminiHandlerConstructor>

interface MockGeminiUsageMetadata {
	promptTokenCount?: number
	candidatesTokenCount?: number
	totalTokenCount?: number
}

interface MockGeminiStreamChunk {
	text(): string | undefined
	functionCalls(): unknown[] | undefined
	usageMetadata?: MockGeminiUsageMetadata
}

interface MockGeminiStreamResult {
	stream: AsyncGenerator<MockGeminiStreamChunk, void, unknown>
	response: Promise<{ usageMetadata?: MockGeminiUsageMetadata }>
}

interface MockGeminiContentResult {
	response: {
		text(): string
		usageMetadata?: MockGeminiUsageMetadata
	}
}

interface GenerateContentStreamParams {
	contents: unknown
	generationConfig?: Record<string, unknown>
}

interface GenerateContentParams {
	contents: unknown
	generationConfig?: Record<string, unknown>
}

interface MockGenerativeModel {
	generateContentStream: sinon.SinonStub<[GenerateContentStreamParams], Promise<MockGeminiStreamResult>>
	generateContent: sinon.SinonStub<[GenerateContentParams], Promise<MockGeminiContentResult>>
}

interface MockGeminiClient {
	getGenerativeModel: sinon.SinonStub<
		[{ model: string; systemInstruction?: string }, { baseUrl?: string }?],
		MockGenerativeModel
	>
}

suite("GeminiHandler", () => {
	let GeminiHandlerClass: GeminiHandlerConstructor
	let handler: GeminiHandlerInstance
	let mockOptions: ApiHandlerOptions
	let mockGenerateContentStream: MockGenerativeModel["generateContentStream"]
	let mockGenerateContent: MockGenerativeModel["generateContent"]
	let mockClient: MockGeminiClient
	let convertToGeminiHistoryStub: sinon.SinonStub<[NeutralConversationHistory], GeminiHistoryPayload>

	setup(() => {
		mockGenerateContentStream = sinon.stub<
			[GenerateContentStreamParams],
			Promise<MockGeminiStreamResult>
		>()
		mockGenerateContent = sinon.stub<
			[GenerateContentParams],
			Promise<MockGeminiContentResult>
		>()

		const mockModel: MockGenerativeModel = {
			generateContentStream: mockGenerateContentStream,
			generateContent: mockGenerateContent,
		}

		const getGenerativeModelStub = sinon.stub<
			[{ model: string; systemInstruction?: string }, { baseUrl?: string }?],
			MockGenerativeModel
		>().returns(mockModel)

		mockClient = {
			getGenerativeModel: getGenerativeModelStub,
		}

		const GoogleGenerativeAIStub = sinon.stub().returns(mockClient)

		convertToGeminiHistoryStub = sinon
			.stub<[NeutralConversationHistory], GeminiHistoryPayload>()
			.callsFake((history: NeutralConversationHistory) =>
				history.map((message) => {
					const normalizedContent: NeutralMessageContent =
						typeof message.content === "string"
							? [{ type: "text", text: message.content } as NeutralTextContentBlock]
							: message.content
					return {
						role: message.role,
						parts: normalizedContent.map((block: NeutralContentBlock) => {
							if (block.type === "text") {
								return { text: (block as NeutralTextContentBlock).text }
							}
							return { json: JSON.stringify(block) }
						}),
					}
				}),
			)

		const geminiModule = proxyquire("../gemini", {
			"@google/generative-ai": {
				__esModule: true,
				GoogleGenerativeAI: GoogleGenerativeAIStub,
			},
			"../transform/neutral-gemini-format": {
				__esModule: true,
				convertToGeminiHistory: convertToGeminiHistoryStub,
			},
		}) as { GeminiHandler: GeminiHandlerConstructor }

		GeminiHandlerClass = geminiModule.GeminiHandler

		mockOptions = {
			apiModelId: GEMINI_MODEL_ID,
			geminiApiKey: "test-key",
			apiKey: "legacy-test-key",
		}

		handler = new GeminiHandlerClass(mockOptions)
	})

	teardown(() => {
		sinon.restore()
	})

	suite("constructor", () => {
		test("initializes with provided config", () => {
			assert.ok(handler instanceof GeminiHandlerClass)
			assert.strictEqual(handler.getModel().id, GEMINI_MODEL_ID)
		})

		test("uses default model when none provided", () => {
			const handlerWithoutModel = new GeminiHandlerClass({
				geminiApiKey: "test-key",
			})
			const model = handlerWithoutModel.getModel()
			assert.ok(model.id)
		})

		test("passes base URL when provided", () => {
			const handlerWithBaseUrl = new GeminiHandlerClass({
				...mockOptions,
				googleGeminiBaseUrl: "https://custom.googleapis.com",
			})

			void handlerWithBaseUrl.createMessage("prompt", [])

			assert.ok(
				mockClient.getGenerativeModel.calledWithMatch(
					sinon.match.object,
					sinon.match({ baseUrl: "https://custom.googleapis.com" }),
				),
			)
		})
	})

	suite("createMessage", () => {
		const systemPrompt = "You are a helpful assistant."

		setup(() => {
			mockGenerateContentStream.resetBehavior()
			mockGenerateContentStream.resetHistory()

			const streamChunks: MockGeminiStreamChunk[] = [
				{
					text: () => "Test response part 1",
					functionCalls: () => undefined,
					usageMetadata: {
						promptTokenCount: 10,
						candidatesTokenCount: 5,
						totalTokenCount: 15,
					},
				},
				{
					text: () => " part 2",
					functionCalls: () => undefined,
					usageMetadata: undefined,
				},
			]

		mockGenerateContentStream.callsFake(() =>
			Promise.resolve({
				stream: (async function* () {
					await Promise.resolve()
					for (const chunk of streamChunks) {
						yield chunk
					}
				})(),
				response: Promise.resolve({
					usageMetadata: {
						promptTokenCount: 10,
						candidatesTokenCount: 5,
						totalTokenCount: 15,
					},
				}),
			}),
		)
		})

		test("produces a stream of chunks", async () => {
			const neutralMessages: NeutralConversationHistory = [
				{
					role: "user",
					content: [{ type: "text", text: "Test message" }],
				},
			]

			const stream = handler.createMessage(systemPrompt, neutralMessages)
			const chunks: ApiStreamChunk[] = []
			for await (const chunk of stream) {
				chunks.push(chunk)
			}

			assert.strictEqual(chunks.length, 3)
			assert.strictEqual(chunks[0].type, "text")
			assert.strictEqual(chunks[1].type, "text")
			assert.deepStrictEqual(chunks[2], {
				type: "usage",
				inputTokens: 10,
				outputTokens: 5,
			})
			assert.ok(mockGenerateContentStream.calledOnce)
		})

		test("passes converted history to SDK", async () => {
			const history: NeutralConversationHistory = [
				{
					role: "user",
					content: [{ type: "text", text: "Hello" }],
				},
			]

			const stream = handler.createMessage(systemPrompt, history)
			for await (const _chunk of stream) {
				// drain stream
			}

			assert.ok(convertToGeminiHistoryStub.calledWith(history))
		})
	})

	suite("completePrompt", () => {
		test("returns generated text", async () => {
			mockGenerateContent.resolves({
				response: {
					text: () => "Test response",
				},
			})

			const result = await handler.completePrompt("Test prompt")
			assert.strictEqual(result, "Test response")
			assert.ok(mockGenerateContent.calledOnce)
		})

		test("returns empty string when no text", async () => {
			mockGenerateContent.resolves({
				response: {
					text: () => "",
				},
			})

			const result = await handler.completePrompt("Test prompt")
			assert.strictEqual(result, "")
		})
	})

	suite("countTokens", () => {
		test("counts tokens for text content", async () => {
			const content: NeutralMessageContent = [{ type: "text", text: "Hello world" }]
			const result = await handler.countTokens(content)
			assert.ok(result > 0)
		})

		test("handles mixed content", async () => {
			const content: NeutralMessageContent = [
				{ type: "text", text: "Describe the image" },
				{ type: "image_url", image_url: { url: "https://example.com/image.png" } },
			]
			const result = await handler.countTokens(content)
			assert.ok(result >= 258)
		})
	})
})
