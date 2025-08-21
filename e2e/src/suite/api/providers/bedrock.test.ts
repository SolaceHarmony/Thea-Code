import * as assert from 'assert'
import * as sinon from 'sinon'
import * as proxyquire from 'proxyquire'

suite("AwsBedrockHandler", () => {
	let AwsBedrockHandler: any
	let handler: any
	let mockOptions: any
	let mockSend: sinon.SinonStub
	let mockConverseStreamCommand: sinon.SinonStub
	let mockConverseCommand: sinon.SinonStub
	let mockConvertToBedrockConverseMessages: sinon.SinonStub

	setup(() => {
		// Create fresh stubs for each test
		mockSend = sinon.stub()
		mockConverseStreamCommand = sinon.stub()
		mockConverseCommand = sinon.stub()
		mockConvertToBedrockConverseMessages = sinon.stub()

		// Use proxyquire to mock AWS SDK and format converter
		AwsBedrockHandler = proxyquire('../../../../../src/api/providers/bedrock', {
			'@aws-sdk/client-bedrock-runtime': {
				BedrockRuntimeClient: sinon.stub().callsFake(() => ({
					send: mockSend,
					config: {
						region: "us-east-1",
					},
				})),
				ConverseStreamCommand: mockConverseStreamCommand,
				ConverseCommand: mockConverseCommand,
			},
			'@aws-sdk/credential-providers': {
				fromIni: sinon.stub().returns({}),
			},
			'../transform/neutral-bedrock-format': {
				convertToBedrockConverseMessages: mockConvertToBedrockConverseMessages,
			}
		}).AwsBedrockHandler

		mockOptions = {
			awsAccessKey: "test-access-key",
			awsSecretKey: "test-secret-key",
			apiModelId: "anthropic.claude-v2",
		}
		handler = new AwsBedrockHandler(mockOptions)
	})

	teardown(() => {
		sinon.restore()
	})

	suite("constructor", () => {
		test("should initialize with AWS credentials", () => {
			assert.ok(handler instanceof AwsBedrockHandler)
			assert.strictEqual(handler.getModel().id, "anthropic.claude-v2")
		})

		test("should use profile-based credentials when enabled", () => {
			const profileHandler = new AwsBedrockHandler({
				...mockOptions,
				awsUseProfile: true,
				awsProfile: "test-profile",
			})
			assert.ok(profileHandler instanceof AwsBedrockHandler)
		})

		test("should handle custom ARN with region validation", () => {
			const arnHandler = new AwsBedrockHandler({
				...mockOptions,
				awsCustomArn: "arn:aws:bedrock:us-west-2:123456789:foundation-model/anthropic.claude-v2",
				awsRegion: "us-west-2",
			})
			assert.ok(arnHandler instanceof AwsBedrockHandler)
		})
	})

	suite("createMessage", () => {
		const systemPrompt = "You are a helpful assistant."

		setup(() => {
			// Mock the format converter to return valid Bedrock messages
			mockConvertToBedrockConverseMessages.returns([
				{ role: "user", content: [{ text: "Test message" }] }
			])

			// Mock streaming response from Bedrock
			mockSend.resolves({
				stream: {
					[Symbol.asyncIterator]: async function* () {
						// Yield usage metadata first
						yield {
							metadata: {
								usage: {
									inputTokens: 10,
									outputTokens: 5,
								},
							},
						}
						// Yield content start
						yield {
							contentBlockStart: {
								start: {
									text: "Test response",
								},
							},
						}
						// Yield content delta
						yield {
							contentBlockDelta: {
								delta: {
									text: " part 2",
								},
							},
						}
						// Yield message stop
						yield {
							messageStop: {
								stopReason: "end_turn",
							},
						}
					},
				},
			})
		})

		test("should convert neutral messages and stream Bedrock response", async () => {
			const neutralMessages = [
				{
					role: "user",
					content: [{ type: "text", text: "Test message" }],
				},
			]

			const stream = handler.createMessage(systemPrompt, neutralMessages)
			const chunks = []
			for await (const chunk of stream) {
				chunks.push(chunk)
			}

			// Verify conversion was called
			assert.ok(mockConvertToBedrockConverseMessages.calledWith(neutralMessages))
			
			// Verify Bedrock client was called
			assert.ok(mockSend.calledOnce)
			
			// Verify we got streaming chunks
			assert.ok(chunks.length > 0)
			
			// Verify usage information
			const usageChunk = chunks.find(chunk => chunk.type === "usage")
			assert.notStrictEqual(usageChunk, undefined)
			assert.strictEqual(usageChunk?.inputTokens, 10)
			assert.strictEqual(usageChunk?.outputTokens, 5)
		})

		test("should handle system prompt inclusion", async () => {
			const neutralMessages = [
				{
					role: "user",
					content: [{ type: "text", text: "Hello" }],
				},
			]

			const stream = handler.createMessage(systemPrompt, neutralMessages)
			const chunks = []
			for await (const chunk of stream) {
				chunks.push(chunk)
			}

			// Verify the system prompt and messages were passed to converter
			assert.ok(mockConvertToBedrockConverseMessages.calledOnce)
			const callArgs = mockConvertToBedrockConverseMessages.firstCall.args
			assert.strictEqual(callArgs[0], neutralMessages)
		})
	})

	suite("completePrompt", () => {
		test("should complete prompt using non-streaming Bedrock API", async () => {
			mockSend.resolves({
				output: {
					message: {
						content: [
							{ text: "Test completion response" }
						],
					},
				},
			})

			const result = await handler.completePrompt("Test prompt")
			assert.strictEqual(result, "Test completion response")
			assert.ok(mockSend.calledOnce)
		})

		test("should handle empty completion response", async () => {
			mockSend.resolves({
				output: {
					message: {
						content: [],
					},
				},
			})

			const result = await handler.completePrompt("Test prompt")
			assert.strictEqual(result, "")
		})
	})

	suite("getModel", () => {
		test("should return model information for standard Bedrock models", () => {
			const model = handler.getModel()
			assert.strictEqual(model.id, "anthropic.claude-v2")
			assert.notStrictEqual(model.info, undefined)
			assert.ok(model.info.maxTokens > 0)
			assert.ok(model.info.contextWindow > 0)
		})

		test("should handle custom ARN model IDs", () => {
			const arnHandler = new AwsBedrockHandler({
				...mockOptions,
				awsCustomArn: "arn:aws:bedrock:us-west-2:123456789:foundation-model/anthropic.claude-v2",
				apiModelId: "arn:aws:bedrock:us-west-2:123456789:foundation-model/anthropic.claude-v2",
			})
			const model = arnHandler.getModel()
			assert.strictEqual(model.id, "arn:aws:bedrock:us-west-2:123456789:foundation-model/anthropic.claude-v2")
		})

		test("should return model properties for Bedrock models", () => {
			const model = handler.getModel()
			assert.strictEqual(model.info.supportsImages, true)
			assert.strictEqual(model.info.supportsPromptCache, false)
		})
	})

	suite("countTokens", () => {
		test("should count tokens for text content", async () => {
			const neutralContent = [{ type: "text" as const, text: "Test message with some tokens" }]
			const result = await handler.countTokens(neutralContent)
			
			// Should return a reasonable token count (falls back to base provider implementation)
			assert.ok(typeof result === "number")
			assert.ok(result > 0)
		})

		test("should handle empty content", async () => {
			const neutralContent: any[] = []
			const result = await handler.countTokens(neutralContent)
			
			assert.strictEqual(result, 0)
		})
	})

	suite("ARN validation", () => {
		test("should handle invalid ARN format", async () => {
			const invalidArnHandler = new AwsBedrockHandler({
				...mockOptions,
				awsCustomArn: "invalid-arn-format",
				apiModelId: "invalid-arn-format",
			})

			const systemPrompt = "Test"
			const neutralMessages = [
				{
					role: "user",
					content: [{ type: "text", text: "Test message" }],
				},
			]

			try {
				const stream = invalidArnHandler.createMessage(systemPrompt, neutralMessages)
				const chunks = []
				for await (const chunk of stream) {
					chunks.push(chunk)
				}
				// Should have yielded error message
				const errorChunk = chunks.find(chunk => chunk.type === "text" && chunk.text?.includes("Error:"))
				assert.notStrictEqual(errorChunk, undefined)
			} catch (error) {
				// ARN validation should catch this
				assert.ok(error instanceof Error)
				assert.ok(error.message.includes("Invalid ARN format"))
			}
		})
	})
// Mock cleanup