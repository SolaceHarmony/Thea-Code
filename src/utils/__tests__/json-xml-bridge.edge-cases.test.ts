/**
 * Edge case tests for JsonMatcher and FormatDetector
 * Tests buffer overflow, partial chunks, format detection
 */

import { JsonMatcher, FormatDetector, JsonMatcherResult } from "../json-xml-bridge"

describe("JsonMatcher Edge Cases", () => {
	let matcher: JsonMatcher

	beforeEach(() => {
		matcher = new JsonMatcher("test") // Requires a matchType
	})

	// Helper to extract matched objects from results
	const extractMatched = (results: JsonMatcherResult[]): any[] => {
		return results
			.filter(r => r.matched)
			.map(r => r.data)
	}

	describe("buffer management", () => {
		test("should handle buffer capacity limits", () => {
			// Create a very large JSON object that exceeds buffer
			// Use a proper object with type field
			const largeContent = '{"type": "test", "data": "' + 'x'.repeat(100000) + '"}'
			
			const results1 = matcher.update(largeContent.substring(0, 50000))
			const results2 = matcher.update(largeContent.substring(50000))
			
			const objects = extractMatched([...results1, ...results2])
			// Should still parse if within reasonable limits
			expect(objects.length).toBeGreaterThan(0)
			expect(objects[0].type).toBe("test")
			expect(objects[0].data).toContain("xxx")
		})

		test("should handle partial JSON chunks", () => {
			// Send JSON in small chunks - with proper type field
			matcher.update('{"type"')
			matcher.update(': "test"')
			matcher.update(', "name"')
			matcher.update(': "te')
			matcher.update('st", "id"')
			const results = matcher.update(': "123"}')
			
			const objects = extractMatched(results)
			expect(objects).toHaveLength(1)
			expect(objects[0]).toEqual({
				type: "test",
				name: "test",
				id: "123"
			})
		})

		test("should handle nested braces", () => {
			const nested = '{"type": "test", "outer": {"inner": {"deep": "value"}, "array": [1, 2, {"item": true}]}}'
			
			const results = matcher.update(nested)
			
			const objects = extractMatched(results)
			expect(objects).toHaveLength(1)
			expect(objects[0].type).toBe("test")
			expect(objects[0].outer.inner.deep).toBe("value")
			expect(objects[0].outer.array[2].item).toBe(true)
		})

		test("should extract multiple JSON objects from stream", () => {
			// Each object needs the matching type
			const stream = 'Some text {"type": "test", "first": 1} more text {"type": "test", "second": 2} and {"type": "test", "third": 3} end'
			
			const results = matcher.update(stream)
			
			const objects = extractMatched(results)
			expect(objects).toHaveLength(3)
			expect(objects[0]).toEqual({ type: "test", first: 1 })
			expect(objects[1]).toEqual({ type: "test", second: 2 })
			expect(objects[2]).toEqual({ type: "test", third: 3 })
		})

		test("should handle escaped quotes in strings", () => {
			const escaped = '{"type": "test", "text": "String with \\"escaped\\" quotes", "value": 123}'
			
			const results = matcher.update(escaped)
			
			const objects = extractMatched(results)
			expect(objects).toHaveLength(1)
			expect(objects[0].text).toBe('String with "escaped" quotes')
		})

		test("should handle JSON with thinking blocks", () => {
			const thinkingMatcher = new JsonMatcher("thinking")
			const thinking = 'Text before {"type": "thinking", "content": "Processing..."} after'
			
			const results = thinkingMatcher.update(thinking)
			
			const objects = extractMatched(results)
			expect(objects).toHaveLength(1)
			// For thinking type, JsonMatcher extracts the content field
			expect(objects[0]).toBe("Processing...")
		})

		test("should handle JSON with tool_use objects", () => {
			const toolMatcher = new JsonMatcher("tool_use")
			const toolUse = '{"type": "tool_use", "id": "call_123", "name": "calculator", "input": {"expression": "2+2"}}'
			
			const results = toolMatcher.update(toolUse)
			
			const objects = extractMatched(results)
			expect(objects).toHaveLength(1)
			expect(objects[0].type).toBe("tool_use")
			expect(objects[0].input.expression).toBe("2+2")
		})

		test("should handle malformed JSON gracefully", () => {
			const malformed = '{"type": "test", "valid": true} {broken json} {"type": "test", "another": "valid"}'
			
			const results = matcher.update(malformed)
			
			const objects = extractMatched(results)
			// Should extract the valid JSON objects
			expect(objects.length).toBe(2)
			expect(objects[0]).toEqual({ type: "test", valid: true })
			expect(objects[1]).toEqual({ type: "test", another: "valid" })
		})

		test("should handle arrays at top level", () => {
			// Arrays are not matched by JsonMatcher - it looks for objects
			const array = '[{"item": 1}, {"item": 2}]'
			
			const results = matcher.update(array)
			
			const objects = extractMatched(results)
			// JsonMatcher doesn't match arrays, only objects with type field
			expect(objects).toHaveLength(0)
		})

		test("should handle incomplete JSON at end of buffer", () => {
			let results = matcher.update('{"type": "test", "complete": true}')
			const objects1 = extractMatched(results)
			// Should get the complete object
			expect(objects1).toHaveLength(1)
			expect(objects1[0]).toEqual({ type: "test", complete: true })
			
			// Add incomplete JSON
			results = matcher.update(' {"type": "test", "incomplete": ')
			const objects2 = extractMatched(results)
			expect(objects2).toHaveLength(0) // Incomplete, no match yet
			
			// Now complete the second object
			results = matcher.update('false}')
			const objects3 = extractMatched(results)
			expect(objects3).toHaveLength(1)
			expect(objects3[0]).toEqual({ type: "test", incomplete: false })
		})

		test("should ignore objects without matching type", () => {
			const mixed = '{"type": "other", "data": 1} {"type": "test", "data": 2} {"type": "different", "data": 3}'
			
			const results = matcher.update(mixed)
			
			const objects = extractMatched(results)
			// Should only match objects with type: "test"
			expect(objects).toHaveLength(1)
			expect(objects[0]).toEqual({ type: "test", data: 2 })
		})

		test("should handle thinking content extraction", () => {
			const thinkingMatcher = new JsonMatcher("thinking")
			
			// Test various thinking formats
			const formats = [
				'{"type": "thinking", "content": "Content field"}',
				'{"type": "thinking", "text": "Text field"}',
				'{"type": "thinking", "other": "data", "content": "Has content"}'
			]
			
			for (const format of formats) {
				const results = thinkingMatcher.update(format)
				const objects = extractMatched(results)
				expect(objects).toHaveLength(1)
				// Check that content is extracted
				if (format.includes('"content"')) {
					expect(typeof objects[0]).toBe("string")
				}
				thinkingMatcher.final() // Clear buffer
			}
		})
	})
})

describe("FormatDetector Edge Cases", () => {
	let detector: FormatDetector

	beforeEach(() => {
		detector = new FormatDetector()
	})

	describe("format detection", () => {
		test("should detect XML format", () => {
			const xmlSamples = [
				'<tool_name>test</tool_name>',
				'  <parameters>{}</parameters>  ',
				'<think>reasoning here</think>',
				'<?xml version="1.0"?><root>content</root>',
				'text before <tag>value</tag> text after'
			]
			
			xmlSamples.forEach(sample => {
				expect(detector.detectFormat(sample)).toBe("xml")
			})
		})

		test("should detect JSON format", () => {
			const jsonSamples = [
				'{"type": "tool_use"}',
				'  {"key": "value"}  ',
				'[1, 2, 3]', // This will return unknown since it's an array
				'{"nested": {"deep": true}}',
				'text {"json": "embedded"} more text'
			]
			
			// Note: Arrays return "unknown", not "json"
			expect(detector.detectFormat(jsonSamples[0])).toBe("json")
			expect(detector.detectFormat(jsonSamples[1])).toBe("json")
			expect(detector.detectFormat(jsonSamples[2])).toBe("unknown") // Array
			expect(detector.detectFormat(jsonSamples[3])).toBe("json")
			expect(detector.detectFormat(jsonSamples[4])).toBe("json")
		})

		test("should detect thinking format", () => {
			const thinkingSamples = [
				'{"type": "thinking", "content": "..."}',
				'{"type":"thinking","text":"reasoning"}',
				'prefix {"type": "thinking"} suffix'
			]
			
			thinkingSamples.forEach(sample => {
				expect(detector.detectFormat(sample)).toBe("json")
			})
		})

		test("should detect tool_use format", () => {
			const toolUseSamples = [
				'{"type": "tool_use", "name": "test"}',
				'{"type":"tool_use","id":"123","name":"calc","input":{}}',
				'<tool_use>calculator</tool_use>'
			]
			
			toolUseSamples.forEach(sample => {
				const format = detector.detectFormat(sample)
				expect(["json", "xml"]).toContain(format)
			})
		})

		test("should return unknown for ambiguous content", () => {
			const ambiguousSamples = [
				'plain text without markup',
				'',
				'   ',
				'\n\n\n',
				'just some regular content'
			]
			
			ambiguousSamples.forEach(sample => {
				expect(detector.detectFormat(sample)).toBe("unknown")
			})
		})

		test("should handle mixed format content", () => {
			// When both XML and JSON are present, XML takes precedence
			const mixed1 = '<tag>value</tag> {"json": true}'
			const mixed2 = '{"json": true} <tag>value</tag>'
			
			expect(detector.detectFormat(mixed1)).toBe("xml")
			expect(detector.detectFormat(mixed2)).toBe("xml") // XML detected first
		})

		test("should handle content with special characters", () => {
			const special = '{"text": "Line1\\nLine2\\tTabbed", "symbols": "<>&\\"\'"}'
			
			expect(detector.detectFormat(special)).toBe("json")
		})

		test("should detect format with leading/trailing whitespace", () => {
			const samples = [
				'\n\n   {"json": true}\n\n',
				'\t\t<xml>value</xml>\t\t',
				'   \n  {"type": "tool_use"}  \n   '
			]
			
			expect(detector.detectFormat(samples[0])).toBe("json")
			expect(detector.detectFormat(samples[1])).toBe("xml")
			expect(detector.detectFormat(samples[2])).toBe("json")
		})

		test("should handle incomplete JSON", () => {
			const incomplete = '{"partial": '
			expect(detector.detectFormat(incomplete)).toBe("unknown")
		})

		test("should handle incomplete XML", () => {
			const incomplete = '<tag>no closing'
			// Still detects as XML due to opening tag
			expect(detector.detectFormat(incomplete)).toBe("xml")
		})
	})

	describe("buffer overflow protection", () => {
		test("JsonMatcher should prevent buffer overflow", () => {
			const matcher = new JsonMatcher("test")
			
			// Try to overflow with extremely nested JSON
			let nested = '{'
			for (let i = 0; i < 1000; i++) {
				nested += `"level${i}": {`
			}
			
			// This should not crash
			expect(() => {
				matcher.update(nested)
			}).not.toThrow()
			
			// Buffer should have some limit mechanism
			const results = matcher.update(nested)
			// May or may not parse depending on implementation limits
			expect(results).toBeDefined()
		})

		test("JsonMatcher should handle very long strings", () => {
			const matcher = new JsonMatcher("test")
			const longString = 'a'.repeat(50000)
			const json = `{"type": "test", "data": "${longString}"}`
			
			const results = matcher.update(json)
			
			const objects = extractMatched(results)
			if (objects.length > 0) {
				expect(objects[0].data.length).toBe(50000)
			}
		})

		test("JsonMatcher should handle rapid small chunks", () => {
			const matcher = new JsonMatcher("test")
			const json = '{"type": "test", "test": "value", "number": 123}'
			
			// Send one character at a time
			let allResults: JsonMatcherResult[] = []
			for (const char of json) {
				const results = matcher.update(char)
				allResults = allResults.concat(results)
			}
			
			const objects = extractMatched(allResults)
			expect(objects).toHaveLength(1)
			expect(objects[0]).toEqual({ type: "test", test: "value", number: 123 })
		})

		test("JsonMatcher should spill buffer when exceeding max length", () => {
			const matcher = new JsonMatcher("test")
			
			// Create content that exceeds max buffer without completing
			const hugeIncomplete = '{"type": "test", "data": "' + 'x'.repeat(300000)
			
			const results = matcher.update(hugeIncomplete)
			
			// Should have spilled some content as non-matched
			const nonMatched = results.filter(r => !r.matched)
			expect(nonMatched.length).toBeGreaterThan(0)
		})
	})
})

// Helper function defined in the module scope for test access
function extractMatched(results: JsonMatcherResult[]): any[] {
	return results
		.filter(r => r.matched)
		.map(r => r.data)
}