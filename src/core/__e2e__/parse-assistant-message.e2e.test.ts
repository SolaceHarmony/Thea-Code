import type { AccessMcpResourceToolUse } from "../../assistant-message"
import { parseAssistantMessage } from "../../assistant-message/parse-assistant-message"
import * as assert from 'assert'
import * as sinon from 'sinon'

suite("parseAssistantMessage - access_mcp_resource", () => {
	test("parses access_mcp_resource tool use", () => {
		const msg =
			"Hello <access_mcp_resource><server_name>srv</server_name><uri>/path</uri></access_mcp_resource> World"
		const result = parseAssistantMessage(msg)

		assert.deepStrictEqual(result, [
			{ type: "text", content: "Hello", partial: false },
			{
				type: "tool_use",
				name: "access_mcp_resource",
				params: { server_name: "srv", uri: "/path" },
				partial: false,
			} as AccessMcpResourceToolUse,
			{ type: "text", content: "World", partial: false },
		])
	})
// Mock cleanup
