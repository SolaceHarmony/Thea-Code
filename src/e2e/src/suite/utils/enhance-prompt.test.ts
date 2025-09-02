import { singleCompletionHandler } from "../single-completion-handler"
import { ApiConfiguration } from "../../shared/api"
import { buildApiHandler, SingleCompletionHandler } from "../../api"
import { supportPrompt } from "../../shared/support-prompt"

// Mock the API handler
// TODO: Mock setup needs manual migration
import * as assert from 'assert'
import * as sinon from 'sinon'
// TODO: Mock needs manual migration
// TODO: Implement proper mock with proxyquire

suite("enhancePrompt", () => {
	const mockApiConfig: ApiConfiguration = {
		apiProvider: "openai",
		openAiApiKey: "test-key",
		openAiBaseUrl: "https://api.openai.com/v1",

	setup(() => {
		sinon.restore()

		// Mock the API handler with a completePrompt method
		;(buildApiHandler as sinon.SinonStub).returns({
			completePrompt: sinon.stub().resolves("Enhanced prompt"),
			createMessage: sinon.stub(),
			getModel: sinon.stub().returns({
				id: "test-model",
				info: {
					maxTokens: 4096,
					contextWindow: 8192,
					supportsPromptCache: false,
				},
			}),
		} as unknown as SingleCompletionHandler)

	test("enhances prompt using default enhancement prompt when no custom prompt provided", async () => {
		const result = await singleCompletionHandler(mockApiConfig, "Test prompt")

		assert.strictEqual(result, "Enhanced prompt")
		const handler = buildApiHandler(mockApiConfig) as SingleCompletionHandler
		// eslint-disable-next-line @typescript-eslint/unbound-method
		assert.ok(handler.completePrompt.calledWith(`Test prompt`))

	test("enhances prompt using custom enhancement prompt when provided", async () => {
		const customEnhancePrompt = "You are a custom prompt enhancer"
		const customEnhancePromptWithTemplate = customEnhancePrompt + "\n\n${userInput}"

		const result = await singleCompletionHandler(
			mockApiConfig,
			supportPrompt.create(
				"ENHANCE",
				{
					userInput: "Test prompt",
				},
				{
					ENHANCE: customEnhancePromptWithTemplate,
				},
			),

		assert.strictEqual(result, "Enhanced prompt")
		const handler = buildApiHandler(mockApiConfig) as SingleCompletionHandler
		// eslint-disable-next-line @typescript-eslint/unbound-method
		assert.ok(handler.completePrompt.calledWith(`${customEnhancePrompt}\n\nTest prompt`))

	test("throws error for empty prompt input", async () => {
		await expect(singleCompletionHandler(mockApiConfig, "")).rejects.toThrow("No prompt text provided")

	test("throws error for missing API configuration", async () => {
		await expect(singleCompletionHandler({} as ApiConfiguration, "Test prompt")).rejects.toThrow(
			"No valid API configuration provided",

	test("throws error for API provider that does not support prompt enhancement", async () => {
		;(buildApiHandler as sinon.SinonStub).returns({
			// No completePrompt method
			createMessage: sinon.stub(),
			getModel: sinon.stub().returns({
				id: "test-model",
				info: {
					maxTokens: 4096,
					contextWindow: 8192,
					supportsPromptCache: false,
				},
			}),

		await expect(singleCompletionHandler(mockApiConfig, "Test prompt")).rejects.toThrow(
			"The selected API provider does not support prompt enhancement",

	test("uses appropriate model based on provider", async () => {
		const openRouterConfig: ApiConfiguration = {
			apiProvider: "openrouter",
			openRouterApiKey: "test-key",
			openRouterModelId: "test-model",

		// Mock successful enhancement
		;(buildApiHandler as sinon.SinonStub).returns({
			completePrompt: sinon.stub().resolves("Enhanced prompt"),
			createMessage: sinon.stub(),
			getModel: sinon.stub().returns({
				id: "test-model",
				info: {
					maxTokens: 4096,
					contextWindow: 8192,
					supportsPromptCache: false,
				},
			}),
		} as unknown as SingleCompletionHandler)

		const result = await singleCompletionHandler(openRouterConfig, "Test prompt")

		assert.ok(buildApiHandler.calledWith(openRouterConfig))
		assert.strictEqual(result, "Enhanced prompt")

	test("propagates API errors", async () => {
		;(buildApiHandler as sinon.SinonStub).returns({
			completePrompt: sinon.stub().rejects(new Error("API Error")),
			createMessage: sinon.stub(),
			getModel: sinon.stub().returns({
				id: "test-model",
				info: {
					maxTokens: 4096,
					contextWindow: 8192,
					supportsPromptCache: false,
				},
			}),
		} as unknown as SingleCompletionHandler)

		await expect(singleCompletionHandler(mockApiConfig, "Test prompt")).rejects.toThrow("API Error")
