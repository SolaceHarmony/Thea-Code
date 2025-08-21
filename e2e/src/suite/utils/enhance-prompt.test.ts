import * as assert from 'assert'
import * as sinon from 'sinon'
import * as proxyquire from 'proxyquire'

suite("Single Completion Handler", () => {
	let sandbox: sinon.SinonSandbox
	let singleCompletionModule: any
	let buildApiHandlerStub: sinon.SinonStub
	let mockHandler: any

	setup(() => {
		sandbox = sinon.createSandbox()
		
		// Create mock handler with completePrompt method
		mockHandler = {
			completePrompt: sandbox.stub(),
			createMessage: sandbox.stub(),
			getModel: sandbox.stub().returns({
				id: "test-model",
				info: {
					maxTokens: 4096,
					contextWindow: 8192,
					supportsPromptCache: false
				}
			})
		}
		
		// Create buildApiHandler stub
		buildApiHandlerStub = sandbox.stub().returns(mockHandler)
		
		// Load the module with mocked dependencies
		singleCompletionModule = proxyquire('../../../src/utils/single-completion-handler', {
			'../api': {
				buildApiHandler: buildApiHandlerStub
			}
		})
	})

	teardown(() => {
		sandbox.restore()
	})

	suite("singleCompletionHandler", () => {
		const mockApiConfig = {
			apiProvider: "openai",
			openAiApiKey: "test-key",
			openAiBaseUrl: "https://api.openai.com/v1"
		}

		test("completes prompt using the API handler", async () => {
			mockHandler.completePrompt.resolves("Enhanced prompt result")
			
			const result = await singleCompletionModule.singleCompletionHandler(
				mockApiConfig,
				"Test prompt"
			)

			assert.strictEqual(result, "Enhanced prompt result")
			assert.ok(buildApiHandlerStub.calledOnce)
			assert.ok(buildApiHandlerStub.calledWith(mockApiConfig))
			assert.ok(mockHandler.completePrompt.calledOnce)
			assert.ok(mockHandler.completePrompt.calledWith("Test prompt"))
		})

		test("throws error when no prompt text provided", async () => {
			try {
				await singleCompletionModule.singleCompletionHandler(mockApiConfig, "")
				assert.fail("Should have thrown an error")
} catch (error) {
				assert.ok(error instanceof Error)
				assert.strictEqual(error.message, "No prompt text provided")
			}
		})

		test("throws error when no API configuration provided", async () => {
			try {
				await singleCompletionModule.singleCompletionHandler(null, "Test prompt")
				assert.fail("Should have thrown an error")
} catch (error) {
				assert.ok(error instanceof Error)
				assert.strictEqual(error.message, "No valid API configuration provided")
			}
		})

		test("throws error when API configuration has no provider", async () => {
			try {
				await singleCompletionModule.singleCompletionHandler(
					{ apiProvider: null },
					"Test prompt"
				)
				assert.fail("Should have thrown an error")
} catch (error) {
				assert.ok(error instanceof Error)
				assert.strictEqual(error.message, "No valid API configuration provided")
			}
		})

		test("throws error when handler doesn't support single completions", async () => {
			// Create a handler without completePrompt method
			const incompatibleHandler = {
				createMessage: sandbox.stub(),
				getModel: sandbox.stub()
			}
			buildApiHandlerStub.returns(incompatibleHandler)

			try {
				await singleCompletionModule.singleCompletionHandler(
					mockApiConfig,
					"Test prompt"
				)
				assert.fail("Should have thrown an error")
} catch (error) {
				assert.ok(error instanceof Error)
				assert.strictEqual(error.message, "The selected API provider does not support prompt enhancement")
			}
		})

		test("passes through API errors", async () => {
			const apiError = new Error("API request failed")
			mockHandler.completePrompt.rejects(apiError)

			try {
				await singleCompletionModule.singleCompletionHandler(
					mockApiConfig,
					"Test prompt"
				)
				assert.fail("Should have thrown an error")
} catch (error) {
				assert.strictEqual(error, apiError)
			}
		})

		test("handles different API providers", async () => {
			const anthropicConfig = {
				apiProvider: "anthropic",
				anthropicApiKey: "test-key"
			}
			
			mockHandler.completePrompt.resolves("Anthropic response")
			
			const result = await singleCompletionModule.singleCompletionHandler(
				anthropicConfig,
				"Test prompt"
			)

			assert.strictEqual(result, "Anthropic response")
			assert.ok(buildApiHandlerStub.calledWith(anthropicConfig))
		})

		test("handles long prompts", async () => {
			const longPrompt = "Test ".repeat(1000)
			mockHandler.completePrompt.resolves("Response to long prompt")
			
			const result = await singleCompletionModule.singleCompletionHandler(
				mockApiConfig,
				longPrompt
			)

			assert.strictEqual(result, "Response to long prompt")
			assert.ok(mockHandler.completePrompt.calledWith(longPrompt))
		})

		test("handles special characters in prompt", async () => {
			const specialPrompt = "Test with 特殊文字 and émojis 🎉"
			mockHandler.completePrompt.resolves("Response with special chars")
			
			const result = await singleCompletionModule.singleCompletionHandler(
				mockApiConfig,
				specialPrompt
			)

			assert.strictEqual(result, "Response with special chars")
			assert.ok(mockHandler.completePrompt.calledWith(specialPrompt))
		})
	})
// Mock cleanup
