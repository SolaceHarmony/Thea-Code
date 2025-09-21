import type * as vscode from "vscode"
import { expect } from 'chai'
import { ApiHandlerOptions } from "../../../shared/api"
import { VsCodeLmHandler } from "../vscode-lm"
import * as assert from 'assert'
import { NeutralConversationHistory } from "../../../shared/neutral-history"
import * as sinon from 'sinon'

// Mock vscode namespace
// Mock needs manual implementation
		type = "text"
		constructor(public value: string) {}
	}

	class MockLanguageModelToolCallPart {
		type = "tool_call"
		constructor(
			public callId: string,
			public name: string,
			public input: Record<string, unknown>,
		) {}
	}

	type MockLanguageModelPart = MockLanguageModelTextPart | MockLanguageModelToolCallPart
// Mock removed - needs manual implementation)) as (callback: (e: vscode.ConfigurationChangeEvent) => void) => { dispose: sinon.SinonStub }),
// 		},
// 		CancellationTokenSource: sinon.stub(() => ({
// 			token: {
// 				isCancellationRequested: false,
// 				onCancellationRequested: sinon.stub(),
// 			},
// 			cancel: sinon.stub(),
// 			dispose: sinon.stub(),
// 		})),
// 		CancellationError: class CancellationError extends Error {
// 			constructor() {
// 				super("Operation cancelled")
// 				this.name = "CancellationError"
// 			}
		},
		LanguageModelChatMessage: {
			Assistant: sinon.stub((content: string | MockLanguageModelPart[]) => ({
				role: "assistant",
				content: Array.isArray(content) ? content : [new MockLanguageModelTextPart(content as string)], // eslint-disable-line @typescript-eslint/no-unnecessary-type-assertion
			})),
			User: sinon.stub((content: string | MockLanguageModelPart[]) => ({
				role: "user",
				content: Array.isArray(content) ? content : [new MockLanguageModelTextPart(content as string)], // eslint-disable-line @typescript-eslint/no-unnecessary-type-assertion
			})),
		},
		LanguageModelTextPart: MockLanguageModelTextPart,
		LanguageModelToolCallPart: MockLanguageModelToolCallPart,
		lm: {
			selectChatModels: sinon.stub(),
		},
	}
// Mock cleanup
const mockLanguageModelChat = {
	id: "test-model",
	name: "Test Model",
	vendor: "test-vendor",
	family: "test-family",
	version: "1.0",
	maxInputTokens: 4096,
	sendRequest: sinon.stub(),
	countTokens: sinon.stub(),
}

suite("VsCodeLmHandler", () => {
	let handler: VsCodeLmHandler
	const defaultOptions: ApiHandlerOptions = {
		vsCodeLmModelSelector: {
			vendor: "test-vendor",
			family: "test-family",
		},
	}

	setup(() => {
		sinon.restore()
		handler = new VsCodeLmHandler(defaultOptions)
	})

	teardown(() => {
		handler.dispose()
	})

	suite("constructor", () => {
		test("should initialize with provided options", () => {
			assert.notStrictEqual(handler, undefined)
			assert.ok(vscode.workspace.onDidChangeConfiguration.called)
		})

		test("should handle configuration changes", () => {
			const mockOnDidChangeConfiguration = vscode.workspace.onDidChangeConfiguration as sinon.SinonStub
			// Verify the mock was called and get the callback
			assert.ok(mockOnDidChangeConfiguration.called)
			const callArgs = mockOnDidChangeConfiguration.mock.calls[0] as [(e: vscode.ConfigurationChangeEvent) => void]
			const callback = callArgs[0]
			callback({ affectsConfiguration: () => true })
			// Should reset client when config changes
			assert.strictEqual(handler["client"], null)
		})
	})

	suite("createClient", () => {
		test("should create client with selector", async () => {
			const mockModel = { ...mockLanguageModelChat }
			;(vscode.lm.selectChatModels as sinon.SinonStub).mockResolvedValueOnce([mockModel])

			const client = await handler["createClient"]({
				vendor: "test-vendor",
				family: "test-family",
			})

			assert.notStrictEqual(client, undefined)
			assert.strictEqual(client.id, "test-model")
			assert.ok(vscode.lm.selectChatModels.calledWith({
				vendor: "test-vendor",
				family: "test-family",
			}))
		})

		test("should return default client when no models available", async () => {
			;(vscode.lm.selectChatModels as sinon.SinonStub).mockResolvedValueOnce([])

			const client = await handler["createClient"]({})

			assert.notStrictEqual(client, undefined)
			assert.strictEqual(client.id, "default-lm")
			assert.strictEqual(client.vendor, "vscode")
		})
	})

	suite("createMessage", () => {
		setup(() => {
			const mockModel = { ...mockLanguageModelChat }
			;(vscode.lm.selectChatModels as sinon.SinonStub).mockResolvedValueOnce([mockModel])
			mockLanguageModelChat.countTokens.resolves(10)
		})

		test("should stream text responses", async () => {
			const systemPrompt = "You are a helpful assistant"
			const messages: NeutralConversationHistory = [
				{
					role: "user",
					content: [
						{
							type: "text",
							text: "Hello",
						},
					],
				},
			]

			const responseText = "Hello! How can I help you?"
			mockLanguageModelChat.sendRequest.mockResolvedValueOnce({
				stream: (function* () {
					yield new vscode.LanguageModelTextPart(responseText)
				})(),
			})

			const stream = handler.createMessage(systemPrompt, messages)
			const chunks: Array<{ type: string; text?: string; inputTokens?: number; outputTokens?: number }> = []
			for await (const chunk of stream) {
				chunks.push(chunk)
			}

			assert.strictEqual(chunks.length, 2) // Text chunk + usage chunk
			assert.deepStrictEqual(chunks[0], {
				type: "text",
				text: responseText,
			})
			expect(chunks[1]).toMatchObject({
				type: "usage",
				inputTokens: sinon.match.instanceOf(Number), // eslint-disable-line @typescript-eslint/no-unsafe-assignment
				outputTokens: sinon.match.instanceOf(Number), // eslint-disable-line @typescript-eslint/no-unsafe-assignment
			})
		})

		test("should handle tool calls", async () => {
			const systemPrompt = "You are a helpful assistant"
			const messages: NeutralConversationHistory = [
				{
					role: "user",
					content: [
						{
							type: "text",
							text: "Calculate 2+2",
						},
					],
				},
			]

			const toolCallData = {
				name: "calculator",
				arguments: { operation: "add", numbers: [2, 2] },
				callId: "call-1",
			}

			mockLanguageModelChat.sendRequest.mockResolvedValueOnce({
				stream: (function* () {
					yield new vscode.LanguageModelToolCallPart(
						toolCallData.callId,
						toolCallData.name,
						toolCallData.arguments,
					)
				})(),
			})

			const stream = handler.createMessage(systemPrompt, messages)
			const chunks: Array<{ type: string; text?: string; inputTokens?: number; outputTokens?: number }> = []
			for await (const chunk of stream) {
				chunks.push(chunk)
			}

			assert.strictEqual(chunks.length, 2) // Tool call chunk + usage chunk
			assert.deepStrictEqual(chunks[0], {
				type: "tool_use",
				id: toolCallData.callId,
				name: toolCallData.name,
				input: toolCallData.arguments,
			})
		})

		test("should handle errors", async () => {
			const systemPrompt = "You are a helpful assistant"
			const messages: NeutralConversationHistory = [
				{
					role: "user",
					content: [
						{
							type: "text",
							text: "Hello",
						},
					],
				},
			]

			mockLanguageModelChat.sendRequest.mockRejectedValueOnce(new Error("API Error"))

			await expect(async () => {
				const stream = handler.createMessage(systemPrompt, messages)
				for await (const chunk of stream) {
					// consume stream
					void chunk
				}
			}).rejects.toThrow("API Error")
		})
	})

	suite("getModel", () => {
		test("should return model info when client exists", async () => {
			const mockModel = { ...mockLanguageModelChat }
			;(vscode.lm.selectChatModels as sinon.SinonStub).mockResolvedValueOnce([mockModel])

			// Initialize client
			await handler["getClient"]()

			const model = handler.getModel()
			assert.strictEqual(model.id, "test-model")
			assert.notStrictEqual(model.info, undefined)
			assert.strictEqual(model.info.contextWindow, 4096)
		})

		test("should return fallback model info when no client exists", () => {
			const model = handler.getModel()
			assert.strictEqual(model.id, "test-vendor/test-family")
			assert.notStrictEqual(model.info, undefined)
		})
	})

	suite("completePrompt", () => {
		test("should complete single prompt", async () => {
			const mockModel = { ...mockLanguageModelChat }
			;(vscode.lm.selectChatModels as sinon.SinonStub).mockResolvedValueOnce([mockModel])

			const responseText = "Completed text"
			mockLanguageModelChat.sendRequest.mockResolvedValueOnce({
				stream: (function* () {
					yield new vscode.LanguageModelTextPart(responseText)
				})(),
			})

			const result = await handler.completePrompt("Test prompt")
			assert.strictEqual(result, responseText)
			assert.ok(mockLanguageModelChat.sendRequest.called)
		})

		test("should handle errors during completion", async () => {
			const mockModel = { ...mockLanguageModelChat }
			;(vscode.lm.selectChatModels as sinon.SinonStub).mockResolvedValueOnce([mockModel])

			mockLanguageModelChat.sendRequest.mockRejectedValueOnce(new Error("Completion failed"))

			await expect(handler.completePrompt("Test prompt")).rejects.toThrow(
				"VSCode LM completion error: Completion failed",
			)
		})
	})
// Mock cleanup
