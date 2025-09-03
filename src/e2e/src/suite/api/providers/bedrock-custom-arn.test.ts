import { AwsBedrockHandler } from "../bedrock"
import { ApiHandlerOptions } from "../../../shared/api"

// Mock the AWS SDK
// TODO: Mock setup needs manual migration
import * as assert from 'assert'
import * as sinon from 'sinon'
// TODO: Fix mock - needs proxyquire
/*
=> {
	const mockSend = sinon.stub().callsFake(() => {
		return Promise.resolve({
			output: new TextEncoder().encode(JSON.stringify({ content: "Test response" })),

	return {
		BedrockRuntimeClient: sinon.stub().callsFake(() => ({
			send: mockSend,
			config: {
				region: "us-east-1",
			},
		})),
		ConverseCommand: sinon.stub(),
		ConverseStreamCommand: sinon.stub(),

})*/

suite("AwsBedrockHandler with custom ARN", () => {
	const mockOptions: ApiHandlerOptions = {
		apiModelId: "custom-arn",
		awsCustomArn: "arn:aws:bedrock:us-east-1:123456789012:foundation-model/anthropic.claude-3-sonnet-20240229-v1:0",
		awsRegion: "us-east-1",

	test("should use the custom ARN as the model ID", () => {
		const handler = new AwsBedrockHandler(mockOptions)
		const model = handler.getModel()

		assert.strictEqual(model.id, mockOptions.awsCustomArn)
		expect(model.info).toHaveProperty("maxTokens")
		expect(model.info).toHaveProperty("contextWindow")
		expect(model.info).toHaveProperty("supportsPromptCache")

	test("should extract region from ARN and use it for client configuration", () => {
		// Test with matching region
		const handler1 = new AwsBedrockHandler(mockOptions)
		// eslint-disable-next-line @typescript-eslint/no-unsafe-member-access, @typescript-eslint/no-explicit-any
		expect((handler1 as any).client.config.region).toBe("us-east-1")

		// Test with mismatched region
		const mismatchOptions = {
			...mockOptions,
			awsRegion: "us-west-2",

		const handler2 = new AwsBedrockHandler(mismatchOptions)
		// Should use the ARN region, not the provided region
		// eslint-disable-next-line @typescript-eslint/no-unsafe-member-access, @typescript-eslint/no-explicit-any
		expect((handler2 as any).client.config.region).toBe("us-east-1")

	test("should validate ARN format", async () => {
		// Invalid ARN format
		const invalidOptions = {
			...mockOptions,
			awsCustomArn: "invalid-arn-format",

		const handler = new AwsBedrockHandler(invalidOptions)

		// completePrompt should throw an error for invalid ARN
		await expect(handler.completePrompt("test")).rejects.toThrow("Invalid ARN format")

	test("should complete a prompt successfully with valid ARN", async () => {
		const handler = new AwsBedrockHandler(mockOptions)
		const response = await handler.completePrompt("test prompt")

		assert.strictEqual(response, "Test response")
