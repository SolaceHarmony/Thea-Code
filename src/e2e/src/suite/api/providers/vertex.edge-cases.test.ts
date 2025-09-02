import * as assert from 'assert'
import * as sinon from 'sinon'
/**
 * Vertex provider edge case tests as recommended by architect
 * Tests Claude/Gemini paths, thinking variants, completePrompt helpers
 */

import { VertexHandler } from "../vertex"
import { ApiConfiguration } from "../../../types/api"
import { Message } from "../../../shared/message"
import { NeutralVertexClient } from "../../../lib/providers/neutral-vertex-client"

// Mock the NeutralVertexClient
// TODO: Mock setup needs manual migration
suite("Vertex Provider Edge Cases", () => {
	let handler: VertexHandler
	let mockClient: sinon.SinonStubStatic<NeutralVertexClient>
	
	setup(() => {
		// Create mock client
		mockClient = {
			createClaudeMessage: sinon.stub(),
			createGeminiMessage: sinon.stub(),
			completeClaudePrompt: sinon.stub(),
			completeGeminiPrompt: sinon.stub()
		} as any
		
		// Mock the constructor to return our mock
		(NeutralVertexClient as sinon.SinonStub).callsFake(() => mockClient)
		
		handler = new VertexHandler({
			vertexConfig: {
				projectId: "test-project",
				region: "us-central1"

		} as ApiConfiguration)

	teardown(() => {
		sinon.restore()

	suite("Claude path", () => {
		test("should route Claude models correctly", async () => {
			const claudeModels = [
				"claude-3-5-sonnet@20240620",
				"claude-3-opus@20240229",
				"claude-3-sonnet@20240229",
				"claude-3-haiku@20240307"

			for (const modelId of claudeModels) {
				mockClient.createClaudeMessage.callsFake(async function* () {
					yield { type: "text", text: "Claude response" }
				
				const messages: Message[] = [{
					role: "user",
					content: [{ type: "text", text: "Test" }]
				}]
				
				const generator = handler.createMessage({
					systemPrompt: "Test system",
					messages,
					modelId
				
				const chunks = []
				for await (const chunk of generator) {
					chunks.push(chunk)

				// Should call Claude client
				assert.ok(mockClient.createClaudeMessage.called)
				assert.ok(!mockClient.createGeminiMessage.called)
				
				// Clear for next iteration
				sinon.restore()

		test("should handle Claude thinking variants", async () => {
			mockClient.createClaudeMessage.callsFake(async function* () {
				// Emit thinking chunk first
				yield { type: "reasoning", text: "Let me think about this..." }
				yield { type: "text", text: "Here's my response" }
			
			const messages: Message[] = [{
				role: "user",
				content: [{ type: "text", text: "Complex question" }]
			}]
			
			const generator = handler.createMessage({
				systemPrompt: "Think step by step",
				messages,
				modelId: "claude-3-5-sonnet@20240620",
				thinking: true // Enable thinking
			
			const chunks = []
			for await (const chunk of generator) {
				chunks.push(chunk)

			// Should have reasoning chunk
			const reasoningChunk = chunks.find(c => c.type === "reasoning")
			assert.ok(reasoningChunk !== undefined)
			assert.strictEqual(reasoningChunk?.text, "Let me think about this...")
			
			// Should have text chunk
			const textChunk = chunks.find(c => c.type === "text")
			assert.ok(textChunk !== undefined)
			assert.strictEqual(textChunk?.text, "Here's my response")

		test("should normalize Claude thinking output", async () => {
			// Different thinking formats that should be normalized
			const thinkingVariants = [
				{ type: "thinking", content: "Reasoning here" },
				{ type: "reasoning", text: "Reasoning here" },
				{ type: "think", text: "Reasoning here" }

			for (const variant of thinkingVariants) {
				mockClient.createClaudeMessage.callsFake(async function* () {
					yield variant as any
				
				const messages: Message[] = [{
					role: "user",
					content: [{ type: "text", text: "Test" }]
				}]
				
				const generator = handler.createMessage({
					systemPrompt: "Test",
					messages,
					modelId: "claude-3-5-sonnet@20240620",
					thinking: true
				
				const chunks = []
				for await (const chunk of generator) {
					chunks.push(chunk)

				// Should normalize to reasoning type
				assert.strictEqual(chunks[0].type, "reasoning")
				
				sinon.restore()

		test("should handle Claude tool use", async () => {
			mockClient.createClaudeMessage.callsFake(async function* () {
				yield {
					type: "tool_use",
					id: "tool-123",
					name: "calculator",
					input: { expression: "2+2" }

			const messages: Message[] = [{
				role: "user",
				content: [{ type: "text", text: "Calculate 2+2" }]
			}]
			
			const generator = handler.createMessage({
				systemPrompt: "You have access to tools",
				messages,
				modelId: "claude-3-5-sonnet@20240620"
			
			const chunks = []
			for await (const chunk of generator) {
				chunks.push(chunk)

			// Should have tool use chunk
			const toolChunk = chunks.find(c => c.type === "tool_use")
			assert.ok(toolChunk !== undefined)
			assert.strictEqual(toolChunk?.id, "tool-123")
			assert.strictEqual(toolChunk?.name, "calculator")

	suite("Gemini path", () => {
		test("should route Gemini models correctly", async () => {
			const geminiModels = [
				"gemini-1.5-pro",
				"gemini-1.5-pro-001",
				"gemini-1.5-pro-002",
				"gemini-1.5-flash",
				"gemini-1.5-flash-001",
				"gemini-1.5-flash-002",
				"gemini-pro",
				"gemini-1.0-pro"

			for (const modelId of geminiModels) {
				mockClient.createGeminiMessage.callsFake(async function* () {
					yield { type: "text", text: "Gemini response" }
				
				const messages: Message[] = [{
					role: "user",
					content: [{ type: "text", text: "Test" }]
				}]
				
				const generator = handler.createMessage({
					systemPrompt: "Test system",
					messages,
					modelId
				
				const chunks = []
				for await (const chunk of generator) {
					chunks.push(chunk)

				// Should call Gemini client
				assert.ok(mockClient.createGeminiMessage.called)
				assert.ok(!mockClient.createClaudeMessage.called)
				
				// Clear for next iteration
				sinon.restore()

		test("should handle Gemini thinking variants", async () => {
			mockClient.createGeminiMessage.callsFake(async function* () {
				// Gemini might use different thinking format
				yield { type: "thought", content: "Processing..." }
				yield { type: "text", text: "Answer" }
			
			const messages: Message[] = [{
				role: "user",
				content: [{ type: "text", text: "Question" }]
			}]
			
			const generator = handler.createMessage({
				systemPrompt: "Think carefully",
				messages,
				modelId: "gemini-1.5-pro",
				thinking: true
			
			const chunks = []
			for await (const chunk of generator) {
				chunks.push(chunk)

			// Should normalize thinking to reasoning
			const reasoningChunk = chunks.find(c => c.type === "reasoning")
			assert.ok(reasoningChunk !== undefined)

		test("should handle Gemini function calling", async () => {
			mockClient.createGeminiMessage.callsFake(async function* () {
				yield {
					type: "function_call",
					name: "get_weather",
					args: { location: "San Francisco" }

			const messages: Message[] = [{
				role: "user",
				content: [{ type: "text", text: "What's the weather?" }]
			}]
			
			const generator = handler.createMessage({
				systemPrompt: "You can call functions",
				messages,
				modelId: "gemini-1.5-pro"
			
			const chunks = []
			for await (const chunk of generator) {
				chunks.push(chunk)

			// Should convert to tool_use format
			const toolChunk = chunks.find(c => c.type === "tool_use")
			assert.ok(toolChunk !== undefined)

	suite("completePrompt helpers", () => {
		test("should use completeClaudePrompt for non-streaming Claude", async () => {
			mockClient.completeClaudePrompt.resolves({
				content: "Complete Claude response",
				usage: {
					inputTokens: 10,
					outputTokens: 5

			const result = await handler.completePrompt(
				"Test prompt",
				"Test system",
				"claude-3-5-sonnet@20240620"

			assert.ok(mockClient.completeClaudePrompt.calledWith({
				prompt: "Test prompt",
				systemPrompt: "Test system",
				model: "claude-3-5-sonnet@20240620"
			
			assert.strictEqual(result.content, "Complete Claude response")
			assert.strictEqual(result.usage?.inputTokens, 10)
			assert.strictEqual(result.usage?.outputTokens, 5)

		test("should use completeGeminiPrompt for non-streaming Gemini", async () => {
			mockClient.completeGeminiPrompt.resolves({
				content: "Complete Gemini response",
				usage: {
					promptTokens: 8,
					completionTokens: 4,
					totalTokens: 12

			const result = await handler.completePrompt(
				"Test prompt",
				"Test system",
				"gemini-1.5-pro"

			assert.ok(mockClient.completeGeminiPrompt.calledWith({
				prompt: "Test prompt",
				systemPrompt: "Test system",
				model: "gemini-1.5-pro"
			
			assert.strictEqual(result.content, "Complete Gemini response")
			assert.strictEqual(result.usage?.promptTokens, 8)
			assert.strictEqual(result.usage?.completionTokens, 4)

		test("should handle completePrompt errors", async () => {
			mockClient.completeClaudePrompt.rejects(
				new Error("API error: Rate limit exceeded")

			await expect(
				handler.completePrompt(
					"Test prompt",
					"Test system",
					"claude-3-5-sonnet@20240620"

			).rejects.toThrow("Rate limit exceeded")

	suite("Model detection edge cases", () => {
		test("should handle model names with special characters", () => {
			const specialModels = [
				"claude-3-5-sonnet@20240620",
				"gemini-1.5-pro-001",
				"claude-3.5-sonnet",
				"gemini_1_5_pro"

			specialModels.forEach(model => {
				const isClaude = model.toLowerCase().includes("claude")
				const isGemini = model.toLowerCase().includes("gemini")
				
				assert.strictEqual(isClaude || isGemini, true)

		test("should default to Gemini for unknown models", async () => {
			mockClient.createGeminiMessage.callsFake(async function* () {
				yield { type: "text", text: "Default response" }
			
			const messages: Message[] = [{
				role: "user",
				content: [{ type: "text", text: "Test" }]
			}]
			
			const generator = handler.createMessage({
				systemPrompt: "Test",
				messages,
				modelId: "unknown-model-xyz"
			
			const chunks = []
			for await (const chunk of generator) {
				chunks.push(chunk)

			// Should use Gemini as fallback
			assert.ok(mockClient.createGeminiMessage.called)

	suite("Configuration handling", () => {
		test("should handle missing project ID", () => {
			expect(() => {
				new VertexHandler({
					vertexConfig: {
						region: "us-central1"
						// Missing projectId

				} as ApiConfiguration)
			}).toThrow()

		test("should handle missing region", () => {
			expect(() => {
				new VertexHandler({
					vertexConfig: {
						projectId: "test-project"
						// Missing region

				} as ApiConfiguration)
			}).toThrow()

		test("should accept valid regions", () => {
			const validRegions = [
				"us-central1",
				"us-east1",
				"us-west1",
				"europe-west1",
				"europe-west4",
				"asia-northeast1",
				"asia-southeast1"

			validRegions.forEach(region => {
				const h = new VertexHandler({
					vertexConfig: {
						projectId: "test-project",
						region

				} as ApiConfiguration)
				
				assert.ok(h !== undefined)

		test("should handle authentication options", () => {
			// With service account key
			const withKey = new VertexHandler({
				vertexConfig: {
					projectId: "test-project",
					region: "us-central1",
					serviceAccountKey: {
						type: "service_account",
						project_id: "test-project",
						private_key: "-----BEGIN PRIVATE KEY-----",
						client_email: "test@test.iam.gserviceaccount.com"

			} as ApiConfiguration)
			
			assert.ok(withKey !== undefined)
			
			// With default credentials
			const withDefault = new VertexHandler({
				vertexConfig: {
					projectId: "test-project",
					region: "us-central1"
					// Uses application default credentials

			} as ApiConfiguration)
			
			assert.ok(withDefault !== undefined)

	suite("Error handling", () => {
		test("should handle quota exceeded errors", async () => {
			mockClient.createClaudeMessage.callsFake(async function* () {
				throw new Error("Quota exceeded for model claude-3-5-sonnet")
			
			const messages: Message[] = [{
				role: "user",
				content: [{ type: "text", text: "Test" }]
			}]
			
			const generator = handler.createMessage({
				systemPrompt: "Test",
				messages,
				modelId: "claude-3-5-sonnet@20240620"
			
			await expect(async () => {
				for await (const chunk of generator) {
					// Should throw before yielding

			}).rejects.toThrow("Quota exceeded")

		test("should handle model not available in region", async () => {
			mockClient.createClaudeMessage.callsFake(async function* () {
				throw new Error("Model claude-3-opus is not available in region us-central1")
			
			const messages: Message[] = [{
				role: "user",
				content: [{ type: "text", text: "Test" }]
			}]
			
			const generator = handler.createMessage({
				systemPrompt: "Test",
				messages,
				modelId: "claude-3-opus@20240229"
			
			await expect(async () => {
				for await (const chunk of generator) {
					// Should throw

			}).rejects.toThrow("not available in region")

		test("should handle network errors", async () => {
			mockClient.createGeminiMessage.callsFake(async function* () {
				throw new Error("Network error: ECONNREFUSED")
			
			const messages: Message[] = [{
				role: "user",
				content: [{ type: "text", text: "Test" }]
			}]
			
			const generator = handler.createMessage({
				systemPrompt: "Test",
				messages,
				modelId: "gemini-1.5-pro"
			
			await expect(async () => {
				for await (const chunk of generator) {
					// Should throw

			}).rejects.toThrow("ECONNREFUSED")

	suite("Usage tracking", () => {
		test("should emit usage for Claude models", async () => {
			mockClient.createClaudeMessage.callsFake(async function* () {
				yield { type: "text", text: "Response" }
				yield {
					type: "usage",
					inputTokens: 100,
					outputTokens: 50,
					totalTokens: 150

			const messages: Message[] = [{
				role: "user",
				content: [{ type: "text", text: "Test" }]
			}]
			
			const generator = handler.createMessage({
				systemPrompt: "Test",
				messages,
				modelId: "claude-3-5-sonnet@20240620"
			
			const chunks = []
			for await (const chunk of generator) {
				chunks.push(chunk)

			const usageChunk = chunks.find(c => c.type === "usage")
			assert.ok(usageChunk !== undefined)
			assert.strictEqual(usageChunk?.inputTokens, 100)
			assert.strictEqual(usageChunk?.outputTokens, 50)

		test("should emit usage for Gemini models", async () => {
			mockClient.createGeminiMessage.callsFake(async function* () {
				yield { type: "text", text: "Response" }
				yield {
					type: "usage",
					promptTokens: 80,
					completionTokens: 40,
					totalTokens: 120

			const messages: Message[] = [{
				role: "user",
				content: [{ type: "text", text: "Test" }]
			}]
			
			const generator = handler.createMessage({
				systemPrompt: "Test",
				messages,
				modelId: "gemini-1.5-pro"
			
			const chunks = []
			for await (const chunk of generator) {
				chunks.push(chunk)

			const usageChunk = chunks.find(c => c.type === "usage")
			assert.ok(usageChunk !== undefined)
			assert.strictEqual(usageChunk?.promptTokens, 80)
			assert.strictEqual(usageChunk?.completionTokens, 40)
