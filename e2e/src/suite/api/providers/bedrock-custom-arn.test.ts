import * as assert from 'assert'
import * as sinon from 'sinon'
import * as proxyquire from 'proxyquire'
import { ApiHandlerOptions } from '../../../shared/api'

/**
 * AWS Bedrock Handler Custom ARN Test Suite
 * 
 * This test suite validates the custom ARN functionality in AwsBedrockHandler, which allows
 * users to specify their own Bedrock model ARNs instead of using the predefined model IDs.
 * 
 * Key Custom ARN Behaviors Tested:
 * 1. ARN validation using the validateBedrockArn function
 * 2. Region extraction from ARN and client configuration  
 * 3. Model ID override with the custom ARN
 * 4. Error handling for invalid ARN formats
 * 5. Successful API calls using custom ARNs
 * 
 * Custom ARN Format: arn:aws:bedrock:region:account-id:resource-type/resource-name
 * Supported resource types: foundation-model, provisioned-model, default-prompt-router, 
 * prompt-router, application-inference-profile
 */
suite("AwsBedrockHandler Custom ARN Functionality", () => {
	let AwsBedrockHandler: any
	let mockSend: sinon.SinonStub
	let mockConverseCommand: sinon.SinonStub
	let mockConverseStreamCommand: sinon.SinonStub
	let mockConvertToBedrockConverseMessages: sinon.SinonStub

	setup(() => {
		// Create fresh stubs for each test to ensure isolation
		mockSend = sinon.stub()
		mockConverseCommand = sinon.stub()
		mockConverseStreamCommand = sinon.stub()
		mockConvertToBedrockConverseMessages = sinon.stub()

		// Mock the AWS SDK and dependencies using proxyquire
		AwsBedrockHandler = proxyquire('../../../../../src/api/providers/bedrock', {
			'@aws-sdk/client-bedrock-runtime': {
				BedrockRuntimeClient: sinon.stub().callsFake((config: any) => ({
					send: mockSend,
					config: config,
				})),
				ConverseCommand: mockConverseCommand,
				ConverseStreamCommand: mockConverseStreamCommand,
			},
			'../transform/neutral-bedrock-format': {
				convertToBedrockConverseMessages: mockConvertToBedrockConverseMessages,
			},
		}).AwsBedrockHandler

		// Set up default mock behaviors
		mockConvertToBedrockConverseMessages.returns([
			{ role: 'user', content: [{ text: 'Test message' }] }
		])
	})

	teardown(() => {
		sinon.restore()
	})

	/**
	 * Model Information Tests
	 * 
	 * Verify that when a custom ARN is provided, the handler correctly uses it as the model ID
	 * and extracts the appropriate region information for client configuration.
	 */
	suite("Model Information and Configuration", () => {
		test("should use the custom ARN as the model ID", () => {
			const options: ApiHandlerOptions = {
				apiModelId: "custom-arn",
				awsCustomArn: "arn:aws:bedrock:us-east-1:123456789012:foundation-model/anthropic.claude-3-sonnet-20240229-v1:0",
				awsRegion: "us-east-1",
			}

			const handler = new AwsBedrockHandler(options)
			const model = handler.getModel()

			// The custom ARN should be used as the model ID
			assert.strictEqual(model.id, options.awsCustomArn)
			
			// Should still have model info properties
			assert.notStrictEqual(model.info, undefined)
			assert.ok(typeof model.info.maxTokens === "number")
			assert.ok(typeof model.info.contextWindow === "number")
			assert.ok(typeof model.info.supportsPromptCache === "boolean")
		})

		test("should extract region from ARN and use it for client configuration", () => {
			// Test with matching region
			const matchingOptions: ApiHandlerOptions = {
				apiModelId: "custom-arn",
				awsCustomArn: "arn:aws:bedrock:us-east-1:123456789012:foundation-model/anthropic.claude-3-sonnet-20240229-v1:0",
				awsRegion: "us-east-1",
			}

			const handler1 = new AwsBedrockHandler(matchingOptions)
			assert.strictEqual((handler1 as any).client.config.region, "us-east-1")

			// Test with mismatched region - should use ARN region
			const mismatchOptions: ApiHandlerOptions = {
				apiModelId: "custom-arn",
				awsCustomArn: "arn:aws:bedrock:eu-west-1:123456789012:foundation-model/anthropic.claude-3-sonnet-20240229-v1:0",
				awsRegion: "us-west-2",
			}

			const handler2 = new AwsBedrockHandler(mismatchOptions)
			// Should use the ARN region (eu-west-1), not the provided region (us-west-2)
			assert.strictEqual((handler2 as any).client.config.region, "eu-west-1")
		})

		test("should handle different ARN resource types", () => {
			const resourceTypes = [
				"foundation-model",
				"provisioned-model", 
				"default-prompt-router",
				"prompt-router",
				"application-inference-profile"
			]

			resourceTypes.forEach(resourceType => {
				const options: ApiHandlerOptions = {
					apiModelId: "custom-arn",
					awsCustomArn: `arn:aws:bedrock:us-east-1:123456789012:${resourceType}/test-model`,
					awsRegion: "us-east-1",
				}

				const handler = new AwsBedrockHandler(options)
				const model = handler.getModel()
				
				assert.strictEqual(model.id, options.awsCustomArn)
			})
		})
	})

	/**
	 * ARN Validation Tests
	 * 
	 * Test the ARN validation functionality that ensures custom ARNs follow the correct
	 * format and provide meaningful error messages for invalid formats.
	 */
	suite("ARN Validation", () => {
		test("should handle invalid ARN format in createMessage", async () => {
			const invalidOptions: ApiHandlerOptions = {
				apiModelId: "custom-arn",
				awsCustomArn: "invalid-arn-format",
				awsRegion: "us-east-1",
			}

			const handler = new AwsBedrockHandler(invalidOptions)
			const neutralMessages = [
				{
					role: "user" as const,
					content: [{ type: "text" as const, text: "Test message" }],
				},
			]

			const stream = handler.createMessage("System prompt", neutralMessages)
			const chunks = []
			
			for await (const chunk of stream) {
				chunks.push(chunk)
			}

			// Should receive error chunks for invalid ARN
			const errorChunks = chunks.filter(chunk => 
				chunk.type === "text" && chunk.text.includes("Invalid ARN format")
			)
			assert.ok(errorChunks.length > 0)
		})

		test("should handle invalid ARN format in completePrompt", async () => {
			const invalidOptions: ApiHandlerOptions = {
				apiModelId: "custom-arn",
				awsCustomArn: "invalid-arn-format",
				awsRegion: "us-east-1",
			}

			const handler = new AwsBedrockHandler(invalidOptions)

			try {
				await handler.completePrompt("test")
			assert.fail("Should have thrown an error for invalid ARN")
		} catch (error) {
			assert.ok(error instanceof Error)
			assert.ok(error.message.includes("Invalid ARN format"))
			} catch (error) {
			assert.fail("Unexpected error: " + error.message)
		}
		})

		test("should accept valid ARN formats", () => {
			const validArns = [
				"arn:aws:bedrock:us-east-1:123456789012:foundation-model/anthropic.claude-3-sonnet-20240229-v1:0",
				"arn:aws:bedrock:eu-west-1:987654321098:provisioned-model/my-custom-model",
				"arn:aws:bedrock:ap-southeast-1:555666777888:application-inference-profile/my-profile",
			]

			validArns.forEach(arn => {
				const options: ApiHandlerOptions = {
					apiModelId: "custom-arn",
					awsCustomArn: arn,
					awsRegion: "us-east-1",
				}

				// Should not throw during construction
				const handler = new AwsBedrockHandler(options)
				const model = handler.getModel()
				assert.strictEqual(model.id, arn)
			})
		})
	})

	/**
	 * API Integration Tests
	 * 
	 * Test that custom ARNs work correctly when making actual API calls to Bedrock,
	 * ensuring the ARN is passed through correctly and responses are handled properly.
	 */
	suite("API Integration with Custom ARNs", () => {
		test("should successfully complete prompt with valid custom ARN", async () => {
			const options: ApiHandlerOptions = {
				apiModelId: "custom-arn",
				awsCustomArn: "arn:aws:bedrock:us-east-1:123456789012:foundation-model/anthropic.claude-3-sonnet-20240229-v1:0",
				awsRegion: "us-east-1",
			}

			// Mock successful API response
			mockSend.resolves({
				output: {
					message: {
						content: [{ text: "Custom ARN response" }]
					}
				}
			})

			const handler = new AwsBedrockHandler(options)
			const response = await handler.completePrompt("test prompt")

			assert.strictEqual(response, "Custom ARN response")
			assert.ok(mockSend.calledOnce)
			
			// Verify the command was created with the custom ARN
			const commandCall = mockConverseCommand.firstCall
			assert.ok(commandCall)
			assert.strictEqual(commandCall.args[0].modelId, options.awsCustomArn)
		})

		test("should successfully stream with valid custom ARN", async () => {
			const options: ApiHandlerOptions = {
				apiModelId: "custom-arn",
				awsCustomArn: "arn:aws:bedrock:us-east-1:123456789012:foundation-model/anthropic.claude-3-sonnet-20240229-v1:0",
				awsRegion: "us-east-1",
			}

			// Mock streaming response
			mockSend.resolves({
				stream: {
					[Symbol.asyncIterator]: async function* () {
						yield {
							contentBlockDelta: {
								delta: {
									text: "Streaming "
								}
							}
						}
						yield {
							contentBlockDelta: {
								delta: {
									text: "from custom ARN"
								}
							}
						}
						yield {
							messageStop: {}
						}
					}
				}
			})

			const handler = new AwsBedrockHandler(options)
			const neutralMessages = [
				{
					role: "user" as const,
					content: [{ type: "text" as const, text: "Stream test" }],
				},
			]

			const stream = handler.createMessage("System prompt", neutralMessages)
			const chunks = []
			
			for await (const chunk of stream) {
				chunks.push(chunk)
			}

			const textChunks = chunks.filter(chunk => chunk.type === "text")
			assert.ok(textChunks.length > 0)
			
			// Verify the stream command was created with the custom ARN
			const streamCommandCall = mockConverseStreamCommand.firstCall
			assert.ok(streamCommandCall)
			assert.strictEqual(streamCommandCall.args[0].modelId, options.awsCustomArn)
		})

		test("should handle provisioned model ARNs correctly", async () => {
			const options: ApiHandlerOptions = {
				apiModelId: "custom-arn",
				awsCustomArn: "arn:aws:bedrock:us-east-1:123456789012:provisioned-model/my-provisioned-claude",
				awsRegion: "us-east-1",
			}

			mockSend.resolves({
				output: {
					message: {
						content: [{ text: "Provisioned model response" }]
					}
				}
			})

			const handler = new AwsBedrockHandler(options)
			const response = await handler.completePrompt("test")

			assert.strictEqual(response, "Provisioned model response")
			
			// Verify the provisioned model ARN was used
			const commandCall = mockConverseCommand.firstCall
			assert.strictEqual(commandCall.args[0].modelId, options.awsCustomArn)
		})
	})

	/**
	 * Error Handling Tests
	 * 
	 * Verify robust error handling for various failure scenarios specific to custom ARN usage,
	 * including permission errors, non-existent models, and region-related issues.
	 */
	suite("Custom ARN Error Handling", () => {
		test("should handle access denied errors with custom ARN", async () => {
			const options: ApiHandlerOptions = {
				apiModelId: "custom-arn",
				awsCustomArn: "arn:aws:bedrock:us-east-1:123456789012:foundation-model/anthropic.claude-3-sonnet-20240229-v1:0",
				awsRegion: "us-east-1",
			}

			// Mock access denied error
			const accessError = new Error("Access denied to model")
			accessError.name = "AccessDeniedException"
			mockSend.rejects(accessError)

			const handler = new AwsBedrockHandler(options)
			const neutralMessages = [
				{
					role: "user" as const,
					content: [{ type: "text" as const, text: "Test" }],
				},
			]

			const stream = handler.createMessage("System", neutralMessages)
			const chunks = []
			
			for await (const chunk of stream) {
				chunks.push(chunk)
			}

			// Should handle the error gracefully with error chunks
			const errorChunks = chunks.filter(chunk => 
				chunk.type === "text" && chunk.text.includes("Access denied")
			)
			assert.ok(errorChunks.length > 0)
		})

		test("should handle model not found errors with custom ARN", async () => {
			const options: ApiHandlerOptions = {
				apiModelId: "custom-arn",
				awsCustomArn: "arn:aws:bedrock:us-east-1:123456789012:foundation-model/non-existent-model",
				awsRegion: "us-east-1",
			}

			// Mock model not found error
			const notFoundError = new Error("Model not found")
			notFoundError.name = "ResourceNotFoundException"
			mockSend.rejects(notFoundError)

			const handler = new AwsBedrockHandler(options)
			
			try {
				await handler.completePrompt("test")
			assert.fail("Should have thrown an error for non-existent model")
		} catch (error) {
			assert.ok(error instanceof Error)
			assert.ok(error.message.includes("Model not found"))
			} catch (error) {
			assert.fail("Unexpected error: " + error.message)
		}
		})
	})
// Mock cleanup
