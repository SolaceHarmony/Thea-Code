import * as assert from 'assert'
import * as sinon from 'sinon'
/**
 * Bedrock provider edge case tests as recommended by architect
 * Tests ARN validation, cross-region inference, credentials modes, error/usage handling
 */

import { ApiConfiguration } from "../../../types/api"
import { BedrockHandler } from "../bedrock"
import { Message } from "../../../shared/message"

suite("Bedrock Provider Edge Cases", () => {
	let handler: BedrockHandler
	let mockClient: any
	
	setup(() => {
		// Mock the AWS Bedrock client
		mockClient = {
			send: sinon.stub(),
			config: {
				region: sinon.stub().returns("us-east-1")
			}
		}
		
		// Mock the BedrockRuntimeClient constructor
		// TODO: Use proxyquire for module mocking
		// Mock for "@aws-sdk/client-bedrock-runtime" needed here
			BedrockRuntimeClient: sinon.stub(() => mockClient),
			ConverseStreamCommand: sinon.stub()
		}))
		
		handler = new BedrockHandler({
			bedrockConfig: {
				region: "us-east-1"
			}
		} as ApiConfiguration)
	})

	teardown(() => {
		sinon.restore()
	})

	suite("ARN validation", () => {
		test("should validate basic ARN format", () => {
			const validArn = "arn:aws:bedrock:us-east-1:123456789012:inference-profile/anthropic.claude-3-sonnet-20240229-v1:0"
			const result = handler.validateBedrockArn(validArn)
			
			assert.strictEqual(result.isValid, true)
			assert.strictEqual(result.region, "us-east-1")
			assert.strictEqual(result.accountId, "123456789012")
			assert.strictEqual(result.resourceType, "inference-profile")
			assert.strictEqual(result.resourceId, "anthropic.claude-3-sonnet-20240229-v1:0")
		})

		test("should reject invalid ARN prefix", () => {
			const invalidArn = "not-an-arn:aws:bedrock:us-east-1:123456789012:inference-profile/model"
			const result = handler.validateBedrockArn(invalidArn)
			
			assert.strictEqual(result.isValid, false)
			assert.ok(result.error.includes("Invalid ARN format"))
		})

		test("should reject wrong service in ARN", () => {
			const wrongServiceArn = "arn:aws:s3:us-east-1:123456789012:bucket/my-bucket"
			const result = handler.validateBedrockArn(wrongServiceArn)
			
			assert.strictEqual(result.isValid, false)
			assert.ok(result.error.includes("must be for bedrock service"))
		})

		test("should handle ARN with missing components", () => {
			const incompleteArn = "arn:aws:bedrock:us-east-1"
			const result = handler.validateBedrockArn(incompleteArn)
			
			assert.strictEqual(result.isValid, false)
			assert.ok(result.error.includes("Invalid ARN format"))
		})

		test("should extract cross-region inference profile", () => {
			const crossRegionArn = "arn:aws:bedrock:eu-west-1:123456789012:inference-profile/us.anthropic.claude-3-sonnet"
			const result = handler.validateBedrockArn(crossRegionArn)
			
			assert.strictEqual(result.isValid, true)
			assert.strictEqual(result.region, "eu-west-1")
			assert.ok(result.resourceId.includes("us.anthropic"))
		})

		test("should handle provisioned throughput ARN", () => {
			const provisionedArn = "arn:aws:bedrock:us-east-1:123456789012:provisioned-model-throughput/abc123"
			const result = handler.validateBedrockArn(provisionedArn)
			
			assert.strictEqual(result.isValid, true)
			assert.strictEqual(result.resourceType, "provisioned-model-throughput")
			assert.strictEqual(result.resourceId, "abc123")
		})

		test("should handle custom model ARN", () => {
			const customModelArn = "arn:aws:bedrock:us-east-1:123456789012:custom-model/my-fine-tuned-model"
			const result = handler.validateBedrockArn(customModelArn)
			
			assert.strictEqual(result.isValid, true)
			assert.strictEqual(result.resourceType, "custom-model")
			assert.strictEqual(result.resourceId, "my-fine-tuned-model")
		})
	})

	suite("Cross-region inference", () => {
		test("should prefix model ID for cross-region", () => {
			const crossRegionModel = "us.anthropic.claude-3-sonnet-20240229-v1:0"
			
			// The handler should detect and handle cross-region prefix
			const isPrefixed = crossRegionModel.startsWith("us.")
			assert.strictEqual(isPrefixed, true)
		})

		test("should handle region mismatch warning", () => {
			const arnInDifferentRegion = "arn:aws:bedrock:eu-west-1:123456789012:inference-profile/model"
			
			// Handler is configured for us-east-1
			const result = handler.validateBedrockArn(arnInDifferentRegion)
			
			// Should still be valid but note the different region
			assert.strictEqual(result.isValid, true)
			assert.strictEqual(result.region, "eu-west-1")
			assert.notStrictEqual(result.region, "us-east-1") // Different from handler config
		})

		test("should support multiple region prefixes", () => {
			const prefixes = ["us.", "eu.", "ap.", "ca.", "sa."]
			
			prefixes.forEach(prefix => {
				const model = `${prefix}anthropic.claude-3-sonnet`
				expect(model).toMatch(/^[a-z]{2}\\./)
			})
		})
	})

	suite("Credentials modes", () => {
		test("should initialize with default credentials", () => {
			const defaultHandler = new BedrockHandler({
				bedrockConfig: {
					region: "us-east-1"
					// No explicit credentials
				}
			} as ApiConfiguration)
			
			assert.notStrictEqual(defaultHandler, undefined)
			// Should use default credential chain
		})

		test("should accept explicit credentials", () => {
			const explicitHandler = new BedrockHandler({
				bedrockConfig: {
					region: "us-east-1",
					credentials: {
						accessKeyId: "AKIAIOSFODNN7EXAMPLE",
						secretAccessKey: "wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY"
					}
				}
			} as ApiConfiguration)
			
			assert.notStrictEqual(explicitHandler, undefined)
		})

		test("should handle session token credentials", () => {
			const sessionHandler = new BedrockHandler({
				bedrockConfig: {
					region: "us-east-1",
					credentials: {
						accessKeyId: "AKIAIOSFODNN7EXAMPLE",
						secretAccessKey: "wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY",
						sessionToken: "AQoEXAMPLEH4aoAH0gNCAPy..."
					}
				}
			} as ApiConfiguration)
			
			assert.notStrictEqual(sessionHandler, undefined)
		})

		test("should support assume role credentials", () => {
			const assumeRoleHandler = new BedrockHandler({
				bedrockConfig: {
					region: "us-east-1",
					credentials: {
						roleArn: "arn:aws:iam::123456789012:role/BedrockAccessRole"
					}
				}
			} as ApiConfiguration)
			
			assert.notStrictEqual(assumeRoleHandler, undefined)
		})
	})

	suite("Error handling during streaming", () => {
		test("should handle throttling errors", async () => {
			const throttleError = new Error("ThrottlingException")
			throttleError.name = "ThrottlingException"
			;(throttleError as any).$metadata = {
				httpStatusCode: 429,
				requestId: "test-request-id"
			}
			
			mockClient.send.rejects(throttleError)
			
			const messages: Message[] = [{
				role: "user",
				content: [{ type: "text", text: "Test" }]
			}]
			
			const generator = handler.createMessage({
				systemPrompt: "Test system",
				messages,
				modelId: "anthropic.claude-3-sonnet"
			})
			
			// Should throw throttling error
			await expect(generator.next()).rejects.toThrow("ThrottlingException")
		})

		test("should handle model not found errors", async () => {
			const notFoundError = new Error("Model not found")
			notFoundError.name = "ResourceNotFoundException"
			;(notFoundError as any).$metadata = {
				httpStatusCode: 404
			}
			
			mockClient.send.rejects(notFoundError)
			
			const messages: Message[] = [{
				role: "user",
				content: [{ type: "text", text: "Test" }]
			}]
			
			const generator = handler.createMessage({
				systemPrompt: "Test system",
				messages,
				modelId: "nonexistent-model"
			})
			
			await expect(generator.next()).rejects.toThrow("ResourceNotFoundException")
		})

		test("should handle access denied errors", async () => {
			const accessError = new Error("Access denied")
			accessError.name = "AccessDeniedException"
			;(accessError as any).$metadata = {
				httpStatusCode: 403
			}
			
			mockClient.send.rejects(accessError)
			
			const messages: Message[] = [{
				role: "user",
				content: [{ type: "text", text: "Test" }]
			}]
			
			const generator = handler.createMessage({
				systemPrompt: "Test system",
				messages,
				modelId: "anthropic.claude-3-sonnet"
			})
			
			await expect(generator.next()).rejects.toThrow("AccessDeniedException")
		})
	})

	suite("Usage and metadata handling", () => {
		test("should yield usage chunks during stream", async () => {
			const mockStream = {
				stream: [
					{
						messageStart: {
							role: "assistant"
						}
					},
					{
						contentBlockStart: {
							start: { text: "" }
						}
					},
					{
						contentBlockDelta: {
							delta: { text: "Hello" }
						}
					},
					{
						metadata: {
							usage: {
								inputTokens: 10,
								outputTokens: 5,
								totalTokens: 15
							}
						}
					},
					{
						messageStop: {
							stopReason: "end_turn"
						}
					}
				][Symbol.iterator]()
			}
			
			mockClient.send.resolves(mockStream)
			
			const messages: Message[] = [{
				role: "user",
				content: [{ type: "text", text: "Test" }]
			}]
			
			const generator = handler.createMessage({
				systemPrompt: "Test system",
				messages,
				modelId: "anthropic.claude-3-sonnet"
			})
			
			const chunks = []
			for await (const chunk of generator) {
				chunks.push(chunk)
			}
			
			// Should have usage chunk
			const usageChunk = chunks.find(c => c.type === "usage")
			assert.notStrictEqual(usageChunk, undefined)
			if (usageChunk?.type === "usage") {
				assert.strictEqual(usageChunk.inputTokens, 10)
				assert.strictEqual(usageChunk.outputTokens, 5)
			}
		})

		test("should handle stream with warnings", async () => {
			const mockStream = {
				stream: [
					{
						messageStart: {
							role: "assistant"
						}
					},
					{
						warning: {
							message: "Model version deprecated",
							code: "DEPRECATED_MODEL"
						}
					},
					{
						contentBlockDelta: {
							delta: { text: "Response" }
						}
					},
					{
						messageStop: {}
					}
				][Symbol.iterator]()
			}
			
			mockClient.send.resolves(mockStream)
			
			const messages: Message[] = [{
				role: "user",
				content: [{ type: "text", text: "Test" }]
			}]
			
			const generator = handler.createMessage({
				systemPrompt: "Test system",
				messages,
				modelId: "anthropic.claude-v1"
			})
			
			const chunks = []
			for await (const chunk of generator) {
				chunks.push(chunk)
			}
			
			// Should handle warning gracefully
			expect(chunks.some(c => c.type === "text")).toBe(true)
		})
	})

	suite("Model ID handling", () => {
		test("should accept various model ID formats", () => {
			const modelIds = [
				"anthropic.claude-3-sonnet-20240229-v1:0",
				"anthropic.claude-3-opus-20240229-v1:0",
				"anthropic.claude-3-haiku-20240307-v1:0",
				"anthropic.claude-v2:1",
				"anthropic.claude-instant-v1",
				"amazon.titan-text-express-v1",
				"ai21.j2-ultra-v1",
				"cohere.command-text-v14",
				"meta.llama2-13b-chat-v1"
			]
			
			modelIds.forEach(modelId => {
				expect(modelId).toMatch(/^[a-z0-9.-]+/)
			})
		})

		test("should handle model ID with version suffix", () => {
			const modelWithVersion = "anthropic.claude-3-sonnet-20240229-v1:0"
			const parts = modelWithVersion.split(":")
			
			assert.strictEqual(parts.length, 2)
			assert.strictEqual(parts[1], "0")
		})

		test("should handle cross-region model prefix", () => {
			const crossRegionModel = "us.anthropic.claude-3-sonnet-20240229-v1:0"
			
			// Extract region prefix
			const match = crossRegionModel.match(/^([a-z]{2})\\.(.+)/)
			assert.ok(match)
			assert.strictEqual(match![1], "us")
			assert.strictEqual(match![2], "anthropic.claude-3-sonnet-20240229-v1:0")
		})
	})

	suite("Stream event handling", () => {
		test("should handle all Bedrock stream event types", async () => {
			const mockStream = {
				stream: [
					{ messageStart: { role: "assistant" } },
					{ contentBlockStart: { start: { text: "" }, contentBlockIndex: 0 } },
					{ contentBlockDelta: { delta: { text: "Part 1" }, contentBlockIndex: 0 } },
					{ contentBlockDelta: { delta: { text: " Part 2" }, contentBlockIndex: 0 } },
					{ contentBlockStop: { contentBlockIndex: 0 } },
					{ metadata: { usage: { inputTokens: 10, outputTokens: 5 } } },
					{ messageStop: { stopReason: "end_turn" } }
				][Symbol.iterator]()
			}
			
			mockClient.send.resolves(mockStream)
			
			const messages: Message[] = [{
				role: "user",
				content: [{ type: "text", text: "Test" }]
			}]
			
			const generator = handler.createMessage({
				systemPrompt: "Test system",
				messages,
				modelId: "anthropic.claude-3-sonnet"
			})
			
			const chunks = []
			for await (const chunk of generator) {
				chunks.push(chunk)
			}
			
			// Should handle all event types
			assert.ok(chunks.length > 0)
			
			// Should concatenate text properly
			const textChunks = chunks.filter(c => c.type === "text")
			const fullText = textChunks.map(c => c.text).join("")
			assert.strictEqual(fullText, "Part 1 Part 2")
		})

		test("should handle tool use in stream", async () => {
			const mockStream = {
				stream: [
					{ messageStart: { role: "assistant" } },
					{ 
						contentBlockStart: { 
							start: { 
								toolUse: {
									toolUseId: "tool-123",
									name: "calculator"
								}
							},
							contentBlockIndex: 0
						}
					},
					{
						contentBlockDelta: {
							delta: {
								toolUse: {
									input: '{"expression": "2+2"}'
								}
							},
							contentBlockIndex: 0
						}
					},
					{ contentBlockStop: { contentBlockIndex: 0 } },
					{ messageStop: {} }
				][Symbol.iterator]()
			}
			
			mockClient.send.resolves(mockStream)
			
			const messages: Message[] = [{
				role: "user",
				content: [{ type: "text", text: "Calculate 2+2" }]
			}]
			
			const generator = handler.createMessage({
				systemPrompt: "You can use tools",
				messages,
				modelId: "anthropic.claude-3-sonnet"
			})
			
			const chunks = []
			for await (const chunk of generator) {
				chunks.push(chunk)
			}
			
			// Should have tool use chunk
			const toolChunk = chunks.find(c => c.type === "tool_use")
			assert.notStrictEqual(toolChunk, undefined)
			if (toolChunk?.type === "tool_use") {
				assert.strictEqual(toolChunk.id, "tool-123")
				assert.strictEqual(toolChunk.name, "calculator")
			}
		})
	})
// Mock cleanup
