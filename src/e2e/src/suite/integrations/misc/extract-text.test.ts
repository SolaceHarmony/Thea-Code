import {
import * as assert from 'assert'
	addLineNumbers,
	everyLineHasLineNumbers,
	stripLineNumbers,
	truncateOutput,
	applyRunLengthEncoding,
} from "../extract-text"

suite("addLineNumbers", () => {
	test("should add line numbers starting from 1 by default", () => {
		const input = "line 1\nline 2\nline 3"
		const expected = "1 | line 1\n2 | line 2\n3 | line 3"
		expect(addLineNumbers(input)).toBe(expected)

	test("should add line numbers starting from specified line number", () => {
		const input = "line 1\nline 2\nline 3"
		const expected = "10 | line 1\n11 | line 2\n12 | line 3"
		expect(addLineNumbers(input, 10)).toBe(expected)

	test("should handle empty content", () => {
		expect(addLineNumbers("")).toBe("1 | ")
		expect(addLineNumbers("", 5)).toBe("5 | ")

	test("should handle single line content", () => {
		expect(addLineNumbers("single line")).toBe("1 | single line")
		expect(addLineNumbers("single line", 42)).toBe("42 | single line")

	test("should pad line numbers based on the highest line number", () => {
		const input = "line 1\nline 2"
		// When starting from 99, highest line will be 100, so needs 3 spaces padding
		const expected = " 99 | line 1\n100 | line 2"
		expect(addLineNumbers(input, 99)).toBe(expected)

suite("everyLineHasLineNumbers", () => {
	test("should return true for content with line numbers", () => {
		const input = "1 | line one\n2 | line two\n3 | line three"
		expect(everyLineHasLineNumbers(input)).toBe(true)

	test("should return true for content with padded line numbers", () => {
		const input = "  1 | line one\n  2 | line two\n  3 | line three"
		expect(everyLineHasLineNumbers(input)).toBe(true)

	test("should return false for content without line numbers", () => {
		const input = "line one\nline two\nline three"
		expect(everyLineHasLineNumbers(input)).toBe(false)

	test("should return false for mixed content", () => {
		const input = "1 | line one\nline two\n3 | line three"
		expect(everyLineHasLineNumbers(input)).toBe(false)

	test("should handle empty content", () => {
		expect(everyLineHasLineNumbers("")).toBe(false)

	test("should return false for content with pipe but no line numbers", () => {
		const input = "a | b\nc | d"
		expect(everyLineHasLineNumbers(input)).toBe(false)

suite("stripLineNumbers", () => {
	test("should strip line numbers from content", () => {
		const input = "1 | line one\n2 | line two\n3 | line three"
		const expected = "line one\nline two\nline three"
		expect(stripLineNumbers(input)).toBe(expected)

	test("should strip padded line numbers", () => {
		const input = "  1 | line one\n  2 | line two\n  3 | line three"
		const expected = "line one\nline two\nline three"
		expect(stripLineNumbers(input)).toBe(expected)

	test("should handle content without line numbers", () => {
		const input = "line one\nline two\nline three"
		expect(stripLineNumbers(input)).toBe(input)

	test("should handle empty content", () => {
		expect(stripLineNumbers("")).toBe("")

	test("should preserve content with pipe but no line numbers", () => {
		const input = "a | b\nc | d"
		expect(stripLineNumbers(input)).toBe(input)

	test("should handle windows-style line endings", () => {
		const input = "1 | line one\r\n2 | line two\r\n3 | line three"
		const expected = "line one\r\nline two\r\nline three"
		expect(stripLineNumbers(input)).toBe(expected)

	test("should handle content with varying line number widths", () => {
		const input = "  1 | line one\n 10 | line two\n100 | line three"
		const expected = "line one\nline two\nline three"
		expect(stripLineNumbers(input)).toBe(expected)

suite("truncateOutput", () => {
	test("returns original content when no line limit provided", () => {
		const content = "line1\nline2\nline3"
		expect(truncateOutput(content)).toBe(content)

	test("returns original content when lines are under limit", () => {
		const content = "line1\nline2\nline3"
		expect(truncateOutput(content, 5)).toBe(content)

	test("truncates content with 20/80 split when over limit", () => {
		// Create 25 lines of content
		const lines = Array.from({ length: 25 }, (_, i) => `line${i + 1}`)
		const content = lines.join("\n")

		// Set limit to 10 lines
		const result = truncateOutput(content, 10)

		// Should keep:
		// - First 2 lines (20% of 10)
		// - Last 8 lines (80% of 10)
		// - Omission indicator in between
		const expectedLines = [
			"line1",
			"line2",
			"",
			"[...15 lines omitted...]",
			"",
			"line18",
			"line19",
			"line20",
			"line21",
			"line22",
			"line23",
			"line24",
			"line25",

		assert.strictEqual(result, expectedLines.join("\n"))

	test("handles empty content", () => {
		expect(truncateOutput("", 10)).toBe("")

	test("handles single line content", () => {
		expect(truncateOutput("single line", 10)).toBe("single line")

	test("handles windows-style line endings", () => {
		// Create content with windows line endings
		const lines = Array.from({ length: 15 }, (_, i) => `line${i + 1}`)
		const content = lines.join("\r\n")

		const result = truncateOutput(content, 5)

		// Should keep first line (20% of 5 = 1) and last 4 lines (80% of 5 = 4)
		// Split result by either \r\n or \n to normalize line endings
		const resultLines = result.split(/\r?\n/)
		const expectedLines = ["line1", "", "[...10 lines omitted...]", "", "line12", "line13", "line14", "line15"]
		assert.deepStrictEqual(resultLines, expectedLines)

suite("applyRunLengthEncoding", () => {
	test("should handle empty input", () => {
		expect(applyRunLengthEncoding("")).toBe("")
		expect(applyRunLengthEncoding(null as unknown as string)).toBe(null as unknown as string)
		expect(applyRunLengthEncoding(undefined as unknown as string)).toBe(undefined as unknown as string)

	test("should compress repeated single lines when beneficial", () => {
		const input = "longerline\nlongerline\nlongerline\nlongerline\nlongerline\nlongerline\n"
		const expected = "longerline\n<previous line repeated 5 additional times>\n"
		expect(applyRunLengthEncoding(input)).toBe(expected)

	test("should not compress when not beneficial", () => {
		const input = "y\ny\ny\ny\ny\n"
		expect(applyRunLengthEncoding(input)).toBe(input)
