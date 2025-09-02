// npx jest src/api/transform/__tests__/vscode-lm-format.test.ts

import type { NeutralConversationHistory } from "../../../shared/neutral-history"
import * as vscode from "vscode"

import { convertToVsCodeLmMessages, convertToAnthropicRole } from "../vscode-lm-format"

// Define types for our mocked classes
import * as assert from 'assert'
import * as sinon from 'sinon'
import * as vscode from 'vscode'
interface MockLanguageModelTextPart {
	type: "text"
	value: string

interface MockLanguageModelToolCallPart {
	type: "tool_call"
	callId: string
	name: string
	input: Record<string, unknown>

interface MockLanguageModelToolResultPart {
	type: "tool_result"
	toolUseId: string
	parts: MockLanguageModelTextPart[]

// Mock vscode namespace
// TODO: Mock setup needs manual migration
// TODO: Fix mock - needs proxyquire
/*
=> {
	const LanguageModelChatMessageRole = {
		Assistant: "assistant",
		User: "user",

	class MockLanguageModelTextPart {
		type = "text"
		constructor(public value: string) {}

	class MockLanguageModelToolCallPart {
		type = "tool_call"
		constructor(
			public callId: string,
			public name: string,
			public input: Record<string, unknown>,
		) {}

	class MockLanguageModelToolResultPart {
		type = "tool_result"
		constructor(
			public toolUseId: string,
			public parts: MockLanguageModelTextPart[],
		) {}

	return {
		LanguageModelChatMessage: {
			Assistant: sinon.stub().returns((content: string | MockLanguageModelTextPart[]) => ({
				role: LanguageModelChatMessageRole.Assistant,
				name: "assistant",
				content: Array.isArray(content) ? content : [new MockLanguageModelTextPart(content)],
			})),
			User: sinon.stub().returns((content: string | MockLanguageModelTextPart[]) => ({
				role: LanguageModelChatMessageRole.User,
				name: "user",
				content: Array.isArray(content) ? content : [new MockLanguageModelTextPart(content)],
			})),
		},
		LanguageModelChatMessageRole,
		LanguageModelTextPart: MockLanguageModelTextPart,
		LanguageModelToolCallPart: MockLanguageModelToolCallPart,
		LanguageModelToolResultPart: MockLanguageModelToolResultPart,

})*/

suite("convertToVsCodeLmMessages", () => {
	test("should convert simple string messages", () => {
		const messages: NeutralConversationHistory = [
			{ role: "user", content: "Hello" },
			{ role: "assistant", content: "Hi there" },

		const result = convertToVsCodeLmMessages(messages)

		assert.strictEqual(result.length, 2)
		assert.strictEqual(result[0].role, "user")
		expect((result[0].content[0] as MockLanguageModelTextPart).value).toBe("Hello")
		assert.strictEqual(result[1].role, "assistant")
		expect((result[1].content[0] as MockLanguageModelTextPart).value).toBe("Hi there")

	test("should handle complex user messages with tool results", () => {
		const messages: NeutralConversationHistory = [
			{
				role: "user",
				content: [
					{ type: "text", text: "Here is the result:" },
					{
						type: "tool_result",
						tool_use_id: "tool-1",
						content: [{ type: "text", text: "Tool output" }],
					},
				],
			},

		const result = convertToVsCodeLmMessages(messages)

		assert.strictEqual(result.length, 1)
		assert.strictEqual(result[0].role, "user")
		assert.strictEqual(result[0].content.length, 2)
		const [toolResult, textContent] = result[0].content as [
			MockLanguageModelToolResultPart,
			MockLanguageModelTextPart,

		assert.strictEqual(toolResult.type, "tool_result")
		assert.strictEqual(textContent.type, "text")

	test("should handle complex assistant messages with tool calls", () => {
		const messages: NeutralConversationHistory = [
			{
				role: "assistant",
				content: [
					{ type: "text", text: "Let me help you with that." },
					{
						type: "tool_use",
						id: "tool-1",
						name: "calculator",
						input: { operation: "add", numbers: [2, 2] },
					},
				],
			},

		const result = convertToVsCodeLmMessages(messages)

		assert.strictEqual(result.length, 1)
		assert.strictEqual(result[0].role, "assistant")
		assert.strictEqual(result[0].content.length, 2)
		const [toolCall, textContent] = result[0].content as [MockLanguageModelToolCallPart, MockLanguageModelTextPart]
		assert.strictEqual(toolCall.type, "tool_call")
		assert.strictEqual(textContent.type, "text")

	test("should handle image blocks with appropriate placeholders", () => {
		const messages: NeutralConversationHistory = [
			{
				role: "user",
				content: [
					{ type: "text", text: "Look at this:" },
					{
						type: "image",
						source: {
							type: "base64",
							media_type: "image/png",
							data: "base64data",
						},
					},
				],
			},

		const result = convertToVsCodeLmMessages(messages)

		assert.strictEqual(result.length, 1)
		const imagePlaceholder = result[0].content[1] as MockLanguageModelTextPart
		assert.ok(imagePlaceholder.value.includes("[Image (base64)): image/png not supported by VSCode LM API]")

suite("convertToAnthropicRole", () => {
	test("should convert assistant role correctly", () => {
		const result = convertToAnthropicRole(vscode.LanguageModelChatMessageRole.Assistant)
		assert.strictEqual(result, "assistant")

	test("should convert user role correctly", () => {
		const result = convertToAnthropicRole(vscode.LanguageModelChatMessageRole.User)
		assert.strictEqual(result, "user")

	test("should return null for unknown roles", () => {
		// @ts-expect-error Testing with an invalid role value
		const result = convertToAnthropicRole("unknown")
		assert.strictEqual(result, null)

suite("asObjectSafe via convertToVsCodeLmMessages", () => {
	test("parses JSON strings in tool_use input", () => {
		const messages: NeutralConversationHistory = [
			{
				role: "assistant",
				content: [
					{
						type: "tool_use",
						id: "1",
						name: "test",
						input: { jsonString: '{"foo": "bar"}' },
					},
				],
			},

		const result = convertToVsCodeLmMessages(messages)
		const toolCall = result[0].content[0] as MockLanguageModelToolCallPart
		assert.deepStrictEqual(toolCall.input, { jsonString: '{"foo": "bar"}' })

	test("handles invalid JSON by returning empty object", () => {
		const messages: NeutralConversationHistory = [
			{
				role: "assistant",
				content: [
					{
						type: "tool_use",
						id: "2",
						name: "test",
						input: { invalidJson: "{invalid}" },
					},
				],
			},

		const result = convertToVsCodeLmMessages(messages)
		const toolCall = result[0].content[0] as MockLanguageModelToolCallPart
		assert.deepStrictEqual(toolCall.input, { invalidJson: "{invalid}" })

	test("clones object inputs", () => {
		const obj = { a: 1 }
		const messages: NeutralConversationHistory = [
			{
				role: "assistant",
				content: [{ type: "tool_use", id: "3", name: "test", input: obj }],
			},

		const result = convertToVsCodeLmMessages(messages)
		const toolCall = result[0].content[0] as MockLanguageModelToolCallPart
		assert.deepStrictEqual(toolCall.input, obj)
		assert.notStrictEqual(toolCall.input, obj)
