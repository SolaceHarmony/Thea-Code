import * as assert from 'assert'
import * as sinon from 'sinon'
import * as proxyquire from 'proxyquire'
import { ApiHandlerOptions } from '../../../shared/api'

/**
 * AWS Bedrock Handler InvokedModelId Test Suite
 * 
 * This test suite validates the dynamic model cost tracking functionality in AwsBedrockHandler.
 * When AWS Bedrock's prompt router invokes a different model than originally requested 
 * (e.g., cross-region inference, inference profiles), the invokedModelId in the trace
 * allows us to update the cost model configuration for accurate billing.
 * 
 * Key InvokedModelId Behaviors Tested:
 * 1. Dynamic cost model configuration updates based on actual invoked model
 * 2. Model name extraction from invokedModelId ARNs
 * 3. Region prefix handling (e.g., "us." prefixes in model names)
 * 4. Error handling for invalid invokedModelId formats
 * 5. Usage token reporting with updated model information
 * 
 * This is particularly important for:
 * - Cross-region inference where costs may differ
 * - Inference profiles that route to different models
 * - Application inference profiles with dynamic routing
 */
suite("AwsBedrockHandler InvokedModelId Cost Tracking", () => {
	let AwsBedrockHandler: any
	let mockSend: sinon.SinonStub
	let mockConverseStreamCommand: sinon.SinonStub
	let mockConvertToBedrockConverseMessages: sinon.SinonStub

	// Helper to create realistic stream events
	function createMockStream(events: any[]) {
// Mock removed - needs manual implementation
				// Always yield final metadata for usage tracking
				yield {
					metadata: {
						usage: {
							inputTokens: 100,
							outputTokens: 200,
						},
					},
				}
			},
		}
	}

	setup(() => {
		// Create fresh stubs for each test
		mockSend = sinon.stub()
		mockConverseStreamCommand = sinon.stub()
		mockConvertToBedrockConverseMessages = sinon.stub()

		// Mock the AWS SDK and dependencies using proxyquire
		AwsBedrockHandler = proxyquire('../../../../../src/api/providers/bedrock', {
			'@aws-sdk/client-bedrock-runtime': {
				BedrockRuntimeClient: sinon.stub().callsFake((config: any) => ({
					send: mockSend,
					config: config,
				})),
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
	 * Dynamic Cost Model Update Tests
	 * 
	 * Verify that when invokedModelId is present in the stream trace, the handler
	 * correctly extracts the model name and updates the cost model configuration
	 * for accurate billing calculations.
	 */
	suite("Dynamic Cost Model Updates", () => {
		test("should update cost model when invokedModelId is present in trace", async () => {
			const options: ApiHandlerOptions = {
				apiModelId: "custom-arn",
				awsCustomArn: "arn:aws:bedrock:us-west-2:699475926481:default-prompt-router/anthropic.claude:1",
				awsRegion: "us-west-2",
			}

			// Mock streaming response with invokedModelId trace
			mockSend.resolves({
				stream: createMockStream([
					// Trace event with invokedModelId and usage
					{
						trace: {
							promptRouter: {
								invokedModelId: "arn:aws:bedrock:us-west-2:699475926481:inference-profile/us.anthropic.claude-3-5-sonnet-20240620-v1:0",
								usage: {
									inputTokens: 150,
									outputTokens: 250,
								},
							},
						},
					},
					// Content events
					{
						contentBlockStart: {
							start: { text: "Response from " },
							contentBlockIndex: 0,
						},
					},
					{
						contentBlockDelta: {
							delta: { text: "invoked model" },
							contentBlockIndex: 0,
						},
					},
				])
			})

			const handler = new AwsBedrockHandler(options)
			const getModelByNameSpy = sinon.spy(handler, 'getModelByName')

			const neutralMessages = [
				{
					role: "user" as const,
					content: [{ type: "text" as const, text: "Test with router" }],
				},
			]

			const stream = handler.createMessage("System prompt", neutralMessages)
			const events = []
			
			for await (const event of stream) {
				events.push(event)
			}

			// Verify getModelByName was called with the extracted model name
			// The "us." prefix should be stripped from "us.anthropic.claude-3-5-sonnet-20240620-v1:0"
			assert.ok(getModelByNameSpy.calledWith("anthropic.claude-3-5-sonnet-20240620-v1:0"))

			// Verify usage events were generated with correct token counts
			const usageEvents = events.filter(event => event.type === "usage")
			assert.ok(usageEvents.length >= 1)

			// Should have both the trace usage and final metadata usage
			const traceUsageEvent = usageEvents.find(event => 
				event.inputTokens === 150 && event.outputTokens === 250
			)
			assert.ok(traceUsageEvent, "Should emit usage from trace event")

			const metadataUsageEvent = usageEvents.find(event => 
				event.inputTokens === 100 && event.outputTokens === 200
			)
			assert.ok(metadataUsageEvent, "Should emit usage from final metadata")
		})

		test("should handle region prefixes in model names correctly", async () => {
			const testCases = [
				{
					invokedModelId: "arn:aws:bedrock:us-east-1:123456789012:inference-profile/us.anthropic.claude-3-sonnet-20240229-v1:0",
					expectedModelName: "anthropic.claude-3-sonnet-20240229-v1:0", // "us." stripped
				},
				{
					invokedModelId: "arn:aws:bedrock:eu-west-1:123456789012:inference-profile/eu.anthropic.claude-3-haiku-20240307-v1:0", 
					expectedModelName: "anthropic.claude-3-haiku-20240307-v1:0", // "eu." stripped
				},
				{
					invokedModelId: "arn:aws:bedrock:ap-southeast-1:123456789012:foundation-model/anthropic.claude-3-sonnet-20240229-v1:0",
					expectedModelName: "anthropic.claude-3-sonnet-20240229-v1:0", // no prefix to strip
				},
			]

			for (const testCase of testCases) {
				const options: ApiHandlerOptions = {
					apiModelId: "anthropic.claude-3-5-sonnet-20241022-v2:0",
					awsRegion: "us-east-1",
				}

				mockSend.resolves({
					stream: createMockStream([
						{
							trace: {
								promptRouter: {
									invokedModelId: testCase.invokedModelId,
								},
							},
						},
					])
				})

				const handler = new AwsBedrockHandler(options)
				const getModelByNameSpy = sinon.spy(handler, 'getModelByName')

				const neutralMessages = [
					{
						role: "user" as const,
						content: [{ type: "text" as const, text: "Test region prefix" }],
					},
				]

				const stream = handler.createMessage("System", neutralMessages)
				
				// Consume the stream
				for await (const event of stream) {
					// Just consume
				}

				// Verify correct model name extraction
				assert.ok(getModelByNameSpy.calledWith(testCase.expectedModelName),
					`Should extract "${testCase.expectedModelName}" from "${testCase.invokedModelId}"`)

				// Reset spies for next iteration
				getModelByNameSpy.restore()
			}
		})

		test("should not update cost model when invokedModelId is not present", async () => {
			const options: ApiHandlerOptions = {
				apiModelId: "anthropic.claude-3-5-sonnet-20241022-v2:0",
				awsRegion: "us-east-1",
			}

			// Mock streaming response WITHOUT invokedModelId trace
			mockSend.resolves({
				stream: createMockStream([
					// Only content events, no trace
					{
						contentBlockStart: {
							start: { text: "Regular response" },
							contentBlockIndex: 0,
						},
					},
					{
						contentBlockDelta: {
							delta: { text: " without routing" },
							contentBlockIndex: 0,
						},
					},
				])
			})

			const handler = new AwsBedrockHandler(options)
			const getModelByNameSpy = sinon.spy(handler, 'getModelByName')

			const neutralMessages = [
				{
					role: "user" as const,
					content: [{ type: "text" as const, text: "Regular request" }],
				},
			]

			const stream = handler.createMessage("System", neutralMessages)
			const events = []
			
			for await (const event of stream) {
				events.push(event)
			}

			// Verify getModelByName was NOT called (no invokedModelId to process)
			assert.ok(!getModelByNameSpy.called, "Should not call getModelByName without invokedModelId")

			// Should still have final metadata usage event
			const usageEvents = events.filter(event => event.type === "usage")
			assert.strictEqual(usageEvents.length, 1)
			assert.strictEqual(usageEvents[0].inputTokens, 100)
			assert.strictEqual(usageEvents[0].outputTokens, 200)
		})
	})

	/**
	 * Error Handling Tests
	 * 
	 * Verify robust error handling for various failure scenarios during 
	 * invokedModelId processing, ensuring the system continues to function
	 * even when cost model updates fail.
	 */
	suite("InvokedModelId Error Handling", () => {
		test("should handle invalid invokedModelId format gracefully", async () => {
			const options: ApiHandlerOptions = {
				apiModelId: "anthropic.claude-3-5-sonnet-20241022-v2:0",
				awsRegion: "us-east-1",
			}

			// Mock streaming response with invalid invokedModelId format
			mockSend.resolves({
				stream: createMockStream([
					{
						trace: {
							promptRouter: {
								invokedModelId: "invalid-format-not-an-arn", // Invalid format
							},
						},
					},
					{
						contentBlockStart: {
							start: { text: "Response despite" },
							contentBlockIndex: 0,
						},
					},
					{
						contentBlockDelta: {
							delta: { text: " invalid invokedModelId" },
							contentBlockIndex: 0,
						},
					},
				])
			})

			const handler = new AwsBedrockHandler(options)
			const getModelByNameSpy = sinon.spy(handler, 'getModelByName')

			const neutralMessages = [
				{
					role: "user" as const,
					content: [{ type: "text" as const, text: "Test invalid format" }],
				},
			]

			const stream = handler.createMessage("System", neutralMessages)
			const events = []
			
			// Should not throw despite invalid invokedModelId
			for await (const event of stream) {
				events.push(event)
			}

			// Should not call getModelByName due to invalid format
			assert.ok(!getModelByNameSpy.called, "Should not call getModelByName with invalid format")

			// Should still process content and usage normally
			const textEvents = events.filter(event => event.type === "text")
			assert.ok(textEvents.length > 0, "Should still emit text events")

			const usageEvents = events.filter(event => event.type === "usage")
			assert.strictEqual(usageEvents.length, 1, "Should still emit final usage")
		})

		test("should handle errors during model lookup gracefully", async () => {
			const options: ApiHandlerOptions = {
				apiModelId: "anthropic.claude-3-5-sonnet-20241022-v2:0",
				awsRegion: "us-east-1",
			}

			// Mock streaming response with valid invokedModelId
			mockSend.resolves({
				stream: createMockStream([
					{
						trace: {
							promptRouter: {
								invokedModelId: "arn:aws:bedrock:us-east-1:123456789012:foundation-model/anthropic.claude-3-sonnet-20240229-v1:0",
							},
						},
					},
					{
						contentBlockStart: {
							start: { text: "Response despite" },
							contentBlockIndex: 0,
						},
					},
					{
						contentBlockDelta: {
							delta: { text: " model lookup error" },
							contentBlockIndex: 0,
						},
					},
				])
			})

			const handler = new AwsBedrockHandler(options)
			
			// Mock getModelByName to throw an error
			sinon.stub(handler, 'getModelByName').callsFake((modelName: string) => {
				throw new Error(`Test error looking up model: ${modelName}`)
			})

			const neutralMessages = [
				{
					role: "user" as const,
					content: [{ type: "text" as const, text: "Test error handling" }],
				},
			]

			const stream = handler.createMessage("System", neutralMessages)
			const events = []
			
			// Should not throw despite getModelByName error
			for await (const event of stream) {
				events.push(event)
			}

			// Should still process content and usage normally
			const textEvents = events.filter(event => event.type === "text")
			assert.ok(textEvents.length > 0, "Should still emit text events despite error")

			const usageEvents = events.filter(event => event.type === "usage")
			assert.strictEqual(usageEvents.length, 1, "Should still emit final usage despite error")
		})

		test("should handle missing trace.promptRouter gracefully", async () => {
			const options: ApiHandlerOptions = {
				apiModelId: "anthropic.claude-3-5-sonnet-20241022-v2:0",
				awsRegion: "us-east-1",
			}

			// Mock streaming response with trace but no promptRouter
			mockSend.resolves({
				stream: createMockStream([
					{
						trace: {
							// No promptRouter field
							someOtherField: "value",
						},
					},
					{
						contentBlockStart: {
							start: { text: "Normal response" },
							contentBlockIndex: 0,
						},
					},
				])
			})

			const handler = new AwsBedrockHandler(options)
			const getModelByNameSpy = sinon.spy(handler, 'getModelByName')

			const neutralMessages = [
				{
					role: "user" as const,
					content: [{ type: "text" as const, text: "Test missing promptRouter" }],
				},
			]

			const stream = handler.createMessage("System", neutralMessages)
			const events = []
			
			for await (const event of stream) {
				events.push(event)
			}

			// Should not call getModelByName without promptRouter.invokedModelId
			assert.ok(!getModelByNameSpy.called, "Should not call getModelByName without promptRouter")

			// Should still function normally
			const textEvents = events.filter(event => event.type === "text")
			assert.ok(textEvents.length > 0, "Should emit text events normally")
		})
	})
// Mock cleanup
