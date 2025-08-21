import { MultiSearchReplaceDiffStrategy } from "../multi-search-replace"

import * as assert from 'assert'
suite("MultiSearchReplaceDiffStrategy", () => {
	suite("validateMarkerSequencing", () => {
		let strategy: MultiSearchReplaceDiffStrategy

		setup(() => {
			strategy = new MultiSearchReplaceDiffStrategy()

		test("validates correct marker sequence", () => {
			const diff = "<<<<<<< SEARCH\n" + "some content\n" + "=======\n" + "new content\n" + ">>>>>>> REPLACE"
			expect(strategy["validateMarkerSequencing"](diff).success).toBe(true)

		test("validates multiple correct marker sequences", () => {

			const diff =
				"<<<<<<< SEARCH\n" +
				"content1\n" +
				"=======\n" +
				"new1\n" +
				">>>>>>> REPLACE\n\n" +
				"<<<<<<< SEARCH\n" +
				"content2\n" +
				"=======\n" +
				"new2\n" +
				">>>>>>> REPLACE"
			expect(strategy["validateMarkerSequencing"](diff).success).toBe(true)

		test("detects separator before search", () => {

			const diff = "=======\n" + "content\n" + ">>>>>>> REPLACE"
			const result = strategy["validateMarkerSequencing"](diff)
			assert.strictEqual(result.success, false)
			assert.ok(result.error.includes("'=======' found in your diff content"))

		test("detects replace before separator", () => {

			const diff = "<<<<<<< SEARCH\n" + "content\n" + ">>>>>>> REPLACE"
			const result = strategy["validateMarkerSequencing"](diff)
			assert.strictEqual(result.success, false)
			assert.ok(result.error.includes("'>>>>>>> REPLACE' found in your diff content"))

		test("detects incomplete sequence", () => {

			const diff = "<<<<<<< SEARCH\n" + "content\n" + "=======\n" + "new content"
			const result = strategy["validateMarkerSequencing"](diff)
			assert.strictEqual(result.success, false)
			assert.ok(result.error.includes("Expected '>>>>>>> REPLACE' was not found"))

		suite("exact matching", () => {
			let strategy: MultiSearchReplaceDiffStrategy

			setup(() => {
				strategy = new MultiSearchReplaceDiffStrategy(1.0, 5) // Default 1.0 threshold for exact matching, 5 line buffer for tests

			test("should replace matching content", async () => {
				const originalContent = 'function hello() {\n    console.log("hello")\n}\n'
				const diffContent = `test.ts
<<<<<<< SEARCH
function hello() {
    console.log("hello")

=======
function hello() {
    console.log("hello world")

>>>>>>> REPLACE`

				const result = await strategy.applyDiff(originalContent, diffContent)
				assert.strictEqual(result.success, true)
				if (result.success) {
					assert.strictEqual(result.content, 'function hello() {\n    console.log("hello world")\n}\n')

			test("should match content with different surrounding whitespace", async () => {

				const originalContent = "\nfunction example() {\n    return 42;\n}\n\n"
				const diffContent = `test.ts
<<<<<<< SEARCH
function example() {
    return 42;

=======
function example() {
    return 43;

>>>>>>> REPLACE`

				const result = await strategy.applyDiff(originalContent, diffContent)
				assert.strictEqual(result.success, true)
				if (result.success) {
					assert.strictEqual(result.content, "\nfunction example() {\n    return 43;\n}\n\n")

			test("should match content with different indentation in search block", async () => {

				const originalContent = "    function test() {\n        return true;\n    }\n"
				const diffContent = `test.ts
<<<<<<< SEARCH
function test() {
    return true;

=======
function test() {
    return false;

>>>>>>> REPLACE`

				const result = await strategy.applyDiff(originalContent, diffContent)
				assert.strictEqual(result.success, true)
				if (result.success) {
					assert.strictEqual(result.content, "    function test() {\n        return false;\n    }\n")

			test("should handle tab-based indentation", async () => {

				const originalContent = "function test() {\n\treturn true;\n}\n"
				const diffContent = `test.ts
<<<<<<< SEARCH
function test() {
\treturn true;

=======
function test() {
\treturn false;

>>>>>>> REPLACE`

				const result = await strategy.applyDiff(originalContent, diffContent)
				assert.strictEqual(result.success, true)
				if (result.success) {
					assert.strictEqual(result.content, "function test() {\n\treturn false;\n}\n")

			test("should preserve mixed tabs and spaces", async () => {

				const originalContent = "\tclass Example {\n\t    constructor() {\n\t\tthis.value = 0;\n\t    }\n\t}"
				const diffContent = `test.ts
<<<<<<< SEARCH
\tclass Example {
\t    constructor() {
\t\tthis.value = 0;
\t    }
\t}
=======
\tclass Example {
\t    constructor() {
\t\tthis.value = 1;
\t    }
\t}
>>>>>>> REPLACE`

				const result = await strategy.applyDiff(originalContent, diffContent)
				assert.strictEqual(result.success, true)
				if (result.success) {
					assert.strictEqual(result.content, 
						"\tclass Example {\n\t    constructor() {\n\t\tthis.value = 1;\n\t    }\n\t}",

			test("should handle additional indentation with tabs", async () => {

				const originalContent = "\tfunction test() {\n\t\treturn true;\n\t}"
				const diffContent = `test.ts
<<<<<<< SEARCH
function test() {
\treturn true;

=======
function test() {
\t// Add comment
\treturn false;

>>>>>>> REPLACE`

				const result = await strategy.applyDiff(originalContent, diffContent)
				assert.strictEqual(result.success, true)
				if (result.success) {
					assert.strictEqual(result.content, "\tfunction test() {\n\t\t// Add comment\n\t\treturn false;\n\t}")

			test("should preserve exact indentation characters when adding lines", async () => {

				const originalContent = "\tfunction test() {\n\t\treturn true;\n\t}"
				const diffContent = `test.ts
<<<<<<< SEARCH
\tfunction test() {
\t\treturn true;
\t}
=======
\tfunction test() {
\t\t// First comment
\t\t// Second comment
\t\treturn true;
\t}
>>>>>>> REPLACE`

				const result = await strategy.applyDiff(originalContent, diffContent)
				assert.strictEqual(result.success, true)
				if (result.success) {
					assert.strictEqual(result.content, 
						"\tfunction test() {\n\t\t// First comment\n\t\t// Second comment\n\t\treturn true;\n\t}",

			test("should handle Windows-style CRLF line endings", async () => {

				const originalContent = "function test() {\r\n    return true;\r\n}\r\n"
				const diffContent = `test.ts
<<<<<<< SEARCH
function test() {
    return true;

=======
function test() {
    return false;

>>>>>>> REPLACE`

				const result = await strategy.applyDiff(originalContent, diffContent)
				assert.strictEqual(result.success, true)
				if (result.success) {
					assert.strictEqual(result.content, "function test() {\r\n    return false;\r\n}\r\n")

			test("should return false if search content does not match", async () => {

				const originalContent = 'function hello() {\n    console.log("hello")\n}\n'
				const diffContent = `test.ts
<<<<<<< SEARCH
function hello() {
    console.log("wrong")

=======
function hello() {
    console.log("hello world")

>>>>>>> REPLACE`

				const result = await strategy.applyDiff(originalContent, diffContent)
				assert.strictEqual(result.success, false)

			test("should return false if diff format is invalid", async () => {

				const originalContent = 'function hello() {\n    console.log("hello")\n}\n'
				const diffContent = `test.ts\nInvalid diff format`

				const result = await strategy.applyDiff(originalContent, diffContent)
				assert.strictEqual(result.success, false)

			test("should handle multiple lines with proper indentation", async () => {

				const originalContent =
					"class Example {\n    constructor() {\n        this.value = 0\n    }\n\n    getValue() {\n        return this.value\n    }\n}\n"
				const diffContent = `test.ts
<<<<<<< SEARCH
    getValue() {
        return this.value

=======
    getValue() {
        // Add logging
        console.log("Getting value")
        return this.value

>>>>>>> REPLACE`

				const result = await strategy.applyDiff(originalContent, diffContent)
				assert.strictEqual(result.success, true)
				if (result.success) {
					assert.strictEqual(result.content, 
						'class Example {\n    constructor() {\n        this.value = 0\n    }\n\n    getValue() {\n        // Add logging\n        console.log("Getting value")\n        return this.value\n    }\n}\n',

			test("should preserve whitespace exactly in the output", async () => {

				const originalContent = "    indented\n        more indented\n    back\n"
				const diffContent = `test.ts
<<<<<<< SEARCH
    indented
        more indented
    back
=======
    modified
        still indented
    end
>>>>>>> REPLACE`

				const result = await strategy.applyDiff(originalContent, diffContent)
				assert.strictEqual(result.success, true)
				if (result.success) {
					assert.strictEqual(result.content, "    modified\n        still indented\n    end\n")

			test("should preserve indentation when adding new lines after existing content", async () => {

				const originalContent = "				onScroll={() => updateHighlights()}"
				const diffContent = `test.ts
<<<<<<< SEARCH
				onScroll={() => updateHighlights()}
=======
				onScroll={() => updateHighlights()}
				onDragOver={(e) => {
					e.preventDefault()
					e.stopPropagation()
				}}
>>>>>>> REPLACE`

				const result = await strategy.applyDiff(originalContent, diffContent)
				assert.strictEqual(result.success, true)
				if (result.success) {
					assert.strictEqual(result.content, 
						"				onScroll={() => updateHighlights()}\n				onDragOver={(e) => {\n					e.preventDefault()\n					e.stopPropagation()\n				}}",

			test("should handle varying indentation levels correctly", async () => {

				const originalContent = `
class Example {
    constructor() {
        this.value = 0;
        if (true) {
            this.init();

}`.trim()

				const diffContent = `test.ts
<<<<<<< SEARCH
    class Example {
        constructor() {
            this.value = 0;
            if (true) {
                this.init();

=======
    class Example {
        constructor() {
            this.value = 1;
            if (true) {
                this.init();
                this.setup();
                this.validate();

>>>>>>> REPLACE`.trim()

				const result = await strategy.applyDiff(originalContent, diffContent)
				assert.strictEqual(result.success, true)
				if (result.success) {
					assert.strictEqual(result.content, 
						`
class Example {
    constructor() {
        this.value = 1;
        if (true) {
            this.init();
            this.setup();
            this.validate();

}`.trim(),

			test("should handle mixed indentation styles in the same file", async () => {

				const originalContent = `class Example {
    constructor() {
        this.value = 0;
        if (true) {
            this.init();

}`.trim()
				const diffContent = `test.ts
<<<<<<< SEARCH
    constructor() {
        this.value = 0;
        if (true) {
        this.init();

=======
    constructor() {
        this.value = 1;
        if (true) {
        this.init();
        this.validate();

>>>>>>> REPLACE`

				const result = await strategy.applyDiff(originalContent, diffContent)
				assert.strictEqual(result.success, true)
				if (result.success) {
					assert.strictEqual(result.content, `class Example {
    constructor() {
        this.value = 1;
        if (true) {
        this.init();
        this.validate();

}`)

			test("should handle Python-style significant whitespace", async () => {

				const originalContent = `def example():
    if condition:
        do_something()
        for item in items:
            process(item)
    return True`.trim()
				const diffContent = `test.ts
<<<<<<< SEARCH
    if condition:
        do_something()
        for item in items:
            process(item)
=======
    if condition:
        do_something()
        while items:
            item = items.pop()
            process(item)
>>>>>>> REPLACE`

				const result = await strategy.applyDiff(originalContent, diffContent)
				assert.strictEqual(result.success, true)
				if (result.success) {
					assert.strictEqual(result.content, `def example():
    if condition:
        do_something()
        while items:
            item = items.pop()
            process(item)
    return True`)

			test("should preserve empty lines with indentation", async () => {

				const originalContent = `function test() {
    const x = 1;

    if (x) {
        return true;

}`.trim()
				const diffContent = `test.ts
<<<<<<< SEARCH
    const x = 1;

    if (x) {
=======
    const x = 1;

    // Check x
    if (x) {
>>>>>>> REPLACE`

				const result = await strategy.applyDiff(originalContent, diffContent)
				assert.strictEqual(result.success, true)
				if (result.success) {
					assert.strictEqual(result.content, `function test() {
    const x = 1;

    // Check x
    if (x) {
        return true;

}`)

			test("should handle indentation when replacing entire blocks", async () => {

				const originalContent = `class Test {
    method() {
        if (true) {
            console.log("test");

}`.trim()
				const diffContent = `test.ts
<<<<<<< SEARCH
    method() {
        if (true) {
            console.log("test");

=======
    method() {
        try {
            if (true) {
                console.log("test");
} catch (e) {
            console.error(e);

>>>>>>> REPLACE`

				const result = await strategy.applyDiff(originalContent, diffContent)
				assert.strictEqual(result.success, true)
				if (result.success) {
					assert.strictEqual(result.content, `class Test {
    method() {
        try {
            if (true) {
                console.log("test");
} catch (e) {
            console.error(e);

}`)

			test("should handle negative indentation relative to search content", async () => {

				const originalContent = `class Example {
    constructor() {
        if (true) {
            this.init();
            this.setup();

}`.trim()
				const diffContent = `test.ts
<<<<<<< SEARCH
            this.init();
            this.setup();
=======
        this.init();
        this.setup();
>>>>>>> REPLACE`

				const result = await strategy.applyDiff(originalContent, diffContent)
				assert.strictEqual(result.success, true)
				if (result.success) {
					assert.strictEqual(result.content, `class Example {
    constructor() {
        if (true) {
        this.init();
        this.setup();

}`)

			test("should handle extreme negative indentation (no indent)", async () => {

				const originalContent = `class Example {
    constructor() {
        if (true) {
            this.init();

}`.trim()
				const diffContent = `test.ts
<<<<<<< SEARCH
            this.init();
=======
this.init();
>>>>>>> REPLACE`

				const result = await strategy.applyDiff(originalContent, diffContent)
				assert.strictEqual(result.success, true)
				if (result.success) {
					assert.strictEqual(result.content, `class Example {
    constructor() {
        if (true) {
this.init();

}`)

			test("should handle mixed indentation changes in replace block", async () => {

				const originalContent = `class Example {
    constructor() {
        if (true) {
            this.init();
            this.setup();
            this.validate();

}`.trim()
				const diffContent = `test.ts
<<<<<<< SEARCH
            this.init();
            this.setup();
            this.validate();
=======
        this.init();
            this.setup();
    this.validate();
>>>>>>> REPLACE`

				const result = await strategy.applyDiff(originalContent, diffContent)
				assert.strictEqual(result.success, true)
				if (result.success) {
					assert.strictEqual(result.content, `class Example {
    constructor() {
        if (true) {
        this.init();
            this.setup();
    this.validate();

}`)

			test("should find matches from middle out", async () => {

				const originalContent = `
function one() {
    return "target";

function two() {
    return "target";

function three() {
    return "target";

function four() {
    return "target";

function five() {
    return "target";
}`.trim()

				const diffContent = `test.ts
<<<<<<< SEARCH
    return "target";
=======
    return "updated";
>>>>>>> REPLACE`

				// Search around the middle (function three)
				// Even though all functions contain the target text,
				// it should match the one closest to line 9 first
				const result = await strategy.applyDiff(originalContent, diffContent, 9, 9)
				assert.strictEqual(result.success, true)
				if (result.success) {
					assert.strictEqual(result.content, `function one() {
    return "target";

function two() {
    return "target";

function three() {
    return "updated";

function four() {
    return "target";

function five() {
    return "target";
}`)

		suite("line number stripping", () => {
			suite("line number stripping", () => {
				let strategy: MultiSearchReplaceDiffStrategy

				setup(() => {
					strategy = new MultiSearchReplaceDiffStrategy()

				test("should strip line numbers from both search and replace sections", async () => {

					const originalContent = "function test() {\n    return true;\n}\n"
					const diffContent = `test.ts
<<<<<<< SEARCH
1 | function test() {
2 |     return true;
3 | }
=======
1 | function test() {
2 |     return false;
3 | }
>>>>>>> REPLACE`

					const result = await strategy.applyDiff(originalContent, diffContent)
					assert.strictEqual(result.success, true)
					if (result.success) {
						assert.strictEqual(result.content, "function test() {\n    return false;\n}\n")

				test("should strip line numbers with leading spaces", async () => {

					const originalContent = "function test() {\n    return true;\n}\n"
					const diffContent = `test.ts
<<<<<<< SEARCH
 1 | function test() {
 2 |     return true;
 3 | }
=======
 1 | function test() {
 2 |     return false;
 3 | }
>>>>>>> REPLACE`

					const result = await strategy.applyDiff(originalContent, diffContent)
					assert.strictEqual(result.success, true)
					if (result.success) {
						assert.strictEqual(result.content, "function test() {\n    return false;\n}\n")

				test("should not strip when not all lines have numbers in either section", async () => {

					const originalContent = "function test() {\n    return true;\n}\n"
					const diffContent = `test.ts
<<<<<<< SEARCH
1 | function test() {
2 |     return true;
3 | }
=======
1 | function test() {
    return false;
3 | }
>>>>>>> REPLACE`

					const result = await strategy.applyDiff(originalContent, diffContent)
					assert.strictEqual(result.success, false)

				test("should preserve content that naturally starts with pipe", async () => {

					const originalContent = "|header|another|\n|---|---|\n|data|more|\n"
					const diffContent = `test.ts
<<<<<<< SEARCH
1 | |header|another|
2 | |---|---|
3 | |data|more|
=======
1 | |header|another|
2 | |---|---|
3 | |data|updated|
>>>>>>> REPLACE`

					const result = await strategy.applyDiff(originalContent, diffContent)
					assert.strictEqual(result.success, true)
					if (result.success) {
						assert.strictEqual(result.content, "|header|another|\n|---|---|\n|data|updated|\n")

				test("should preserve indentation when stripping line numbers", async () => {

					const originalContent = "    function test() {\n        return true;\n    }\n"
					const diffContent = `test.ts
<<<<<<< SEARCH
1 |     function test() {
2 |         return true;
3 |     }
=======
1 |     function test() {
2 |         return false;
3 |     }
>>>>>>> REPLACE`

					const result = await strategy.applyDiff(originalContent, diffContent)
					assert.strictEqual(result.success, true)
					if (result.success) {
						assert.strictEqual(result.content, "    function test() {\n        return false;\n    }\n")

				test("should handle different line numbers between sections", async () => {

					const originalContent = "function test() {\n    return true;\n}\n"
					const diffContent = `test.ts
<<<<<<< SEARCH
10 | function test() {
11 |     return true;
12 | }
=======
20 | function test() {
21 |     return false;
22 | }
>>>>>>> REPLACE`

					const result = await strategy.applyDiff(originalContent, diffContent)
					assert.strictEqual(result.success, true)
					if (result.success) {
						assert.strictEqual(result.content, "function test() {\n    return false;\n}\n")

				test("detects search marker when expecting replace", () => {

					const diff = "<<<<<<< SEARCH\n" + "content\n" + "=======\n" + "new content\n" + "<<<<<<< SEARCH"
					const result = strategy["validateMarkerSequencing"](diff)
					assert.strictEqual(result.success, false)
					assert.ok(result.error.includes("'<<<<<<< SEARCH' found in your diff content"))

				test("allows escaped search marker in content", () => {

					const diff =
						"<<<<<<< SEARCH\n" +
						"before\n" +
						"\\<<<<<<< SEARCH\n" +
						"after\n" +
						"=======\n" +
						"new content\n" +
						">>>>>>> REPLACE"
					const result = strategy["validateMarkerSequencing"](diff)
					assert.strictEqual(result.success, true)

				test("allows escaped separator in content", () => {

					const diff =
						"<<<<<<< SEARCH\n" +
						"before\n" +
						"\\=======\n" +
						"after\n" +
						"=======\n" +
						"new content\n" +
						">>>>>>> REPLACE"
					const result = strategy["validateMarkerSequencing"](diff)
					assert.strictEqual(result.success, true)

				test("processes escaped search marker in content", async () => {

					const originalContent = "before\n<<<<<<< SEARCH\nafter\n"
					const diffContent =
						"<<<<<<< SEARCH\n" +
						"before\n" +
						"\\<<<<<<< SEARCH\n" +
						"after\n" +
						"=======\n" +
						"unchanged\n" +
						">>>>>>> REPLACE"
					const result = await strategy.applyDiff(originalContent, diffContent)
					assert.strictEqual(result.success, true)
					if (result.success) {
						assert.strictEqual(result.content, "unchanged\n")

				test("processes escaped separator in content", async () => {

					const originalContent = "before\n=======\nafter\n"
					const diffContent =
						"<<<<<<< SEARCH\n" +
						"before\n" +
						"\\=======\n" +
						"after\n" +
						"=======\n" +
						"unchanged\n" +
						">>>>>>> REPLACE"
					const result = await strategy.applyDiff(originalContent, diffContent)
					assert.strictEqual(result.success, true)
					if (result.success) {
						assert.strictEqual(result.content, "unchanged\n")

				test("processes escaped search marker in content", async () => {

					const originalContent = "before\n<<<<<<< SEARCH\nafter\n"
					const diffContent =
						"<<<<<<< SEARCH\n" +
						"before\n" +
						"\\<<<<<<< SEARCH\n" +
						"after\n" +
						"=======\n" +
						"unchanged\n" +
						">>>>>>> REPLACE"
					const result = await strategy.applyDiff(originalContent, diffContent)
					assert.strictEqual(result.success, true)
					if (result.success) {
						assert.strictEqual(result.content, "unchanged\n")

				test("processes escaped separator in content", async () => {

					const originalContent = "before\n=======\nafter\n"
					const diffContent =
						"<<<<<<< SEARCH\n" +
						"before\n" +
						"\\=======\n" +
						"after\n" +
						"=======\n" +
						"unchanged\n" +
						">>>>>>> REPLACE"
					const result = await strategy.applyDiff(originalContent, diffContent)
					assert.strictEqual(result.success, true)
					if (result.success) {
						assert.strictEqual(result.content, "unchanged\n")

				test("processes escaped search marker in content", async () => {

					const originalContent = "before\n<<<<<<< SEARCH\nafter\n"
					const diffContent =
						"test.ts\n" +
						"<<<<<<< SEARCH\n" +
						"before\n" +
						"\\<<<<<<< SEARCH\n" +
						"after\n" +
						"=======\n" +
						"unchanged\n" +
						">>>>>>> REPLACE"
					const result = await strategy.applyDiff(originalContent, diffContent)
					assert.strictEqual(result.success, true)
					if (result.success) {
						assert.strictEqual(result.content, "unchanged\n")

				test("processes escaped search marker in content", async () => {

					const originalContent = "before\n<<<<<<< SEARCH\nafter\n"
					const diffContent =
						"test.ts\n" +
						"<<<<<<< SEARCH\n" +
						"before\n" +
						"\\<<<<<<< SEARCH\n" +
						"after\n" +
						"=======\n" +
						"unchanged\n" +
						">>>>>>> REPLACE"
					const result = await strategy.applyDiff(originalContent, diffContent)
					assert.strictEqual(result.success, true)
					if (result.success) {
						assert.strictEqual(result.content, "unchanged\n")

				test("processes escaped separator in content", async () => {

					const originalContent = "before\n=======\nafter\n"
					const diffContent =
						"<<<<<<< SEARCH\n" +
						"before\n" +
						"\\=======\n" +
						"after\n" +
						"=======\n" +
						"unchanged\n" +
						">>>>>>> REPLACE"
					const result = await strategy.applyDiff(originalContent, diffContent)
					assert.strictEqual(result.success, true)
					if (result.success) {
						assert.strictEqual(result.content, "unchanged\n")

				test("processes escaped search marker in content", async () => {

					const originalContent = "before\n<<<<<<< SEARCH\nafter\n"
					const diffContent =
						"<<<<<<< SEARCH\n" +
						"before\n" +
						"\\<<<<<<< SEARCH\n" +
						"after\n" +
						"=======\n" +
						"unchanged\n" +
						">>>>>>> REPLACE"
					const result = await strategy.applyDiff(originalContent, diffContent)
					assert.strictEqual(result.success, true)
					if (result.success) {
						assert.strictEqual(result.content, "unchanged\n")

				test("processes escaped search marker in content", async () => {

					const originalContent = "before\n<<<<<<< SEARCH\nafter\n"
					const diffContent =
						"<<<<<<< SEARCH\n" +
						"before\n" +
						"\\<<<<<<< SEARCH\n" +
						"after\n" +
						"=======\n" +
						"unchanged\n" +
						">>>>>>> REPLACE"
					const result = await strategy.applyDiff(originalContent, diffContent)
					assert.strictEqual(result.success, true)
					if (result.success) {
						assert.strictEqual(result.content, "unchanged\n")

				test("processes escaped search marker in content", async () => {

					const originalContent = "before\n<<<<<<< SEARCH\nafter\n"
					const diffContent =
						"<<<<<<< SEARCH\n" +
						"before\n" +
						"\\<<<<<<< SEARCH\n" +
						"after\n" +
						"=======\n" +
						"unchanged\n" +
						">>>>>>> REPLACE"
					const result = await strategy.applyDiff(originalContent, diffContent)
					assert.strictEqual(result.success, true)
					if (result.success) {
						assert.strictEqual(result.content, "unchanged\n")

				test("processes escaped separator in content", async () => {

					const originalContent = "before\n=======\nafter\n"
					const diffContent =
						"<<<<<<< SEARCH\n" +
						"before\n" +
						"\\=======\n" +
						"after\n" +
						"=======\n" +
						"unchanged\n" +
						">>>>>>> REPLACE"
					const result = await strategy.applyDiff(originalContent, diffContent)
					assert.strictEqual(result.success, true)
					if (result.success) {
						assert.strictEqual(result.content, "unchanged\n")

				test("allows escaped search marker in content", () => {

					const diff =
						"<<<<<<< SEARCH\n" +
						"before\n" +
						"\\<<<<<<< SEARCH\n" +
						"after\n" +
						"=======\n" +
						"new content\n" +
						">>>>>>> REPLACE"
					expect(strategy["validateMarkerSequencing"](diff).success).toBe(true)

				test("allows escaped separator in content", () => {

					const diff =
						"<<<<<<< SEARCH\n" +
						"before\n" +
						"\\=======\n" +
						"after\n" +
						"=======\n" +
						"new content\n" +
						">>>>>>> REPLACE"
					expect(strategy["validateMarkerSequencing"](diff).success).toBe(true)

				test("allows escaped search marker in content", () => {

					const diff =
						"<<<<<<< SEARCH\n" +
						"before\n" +
						"\\<<<<<<< SEARCH\n" +
						"after\n" +
						"=======\n" +
						"new content\n" +
						">>>>>>> REPLACE"
					expect(strategy["validateMarkerSequencing"](diff).success).toBe(true)

				test("allows escaped search marker in content", () => {

					const diff =
						"<<<<<<< SEARCH\n" +
						"before\n" +
						"\\<<<<<<< SEARCH\n" +
						"after\n" +
						"=======\n" +
						"new content\n" +
						">>>>>>> REPLACE"
					expect(strategy["validateMarkerSequencing"](diff).success).toBe(true)

				test("allows escaped separator in content", () => {

					const diff =
						"<<<<<<< SEARCH\n" +
						"before\n" +
						"\\=======\n" +
						"after\n" +
						"=======\n" +
						"new content\n" +
						">>>>>>> REPLACE"
					expect(strategy["validateMarkerSequencing"](diff).success).toBe(true)

				test("allows escaped replace marker in content", () => {

					const diff =
						"<<<<<<< SEARCH\n" +
						"before\n" +
						"\\>>>>>>> REPLACE\n" +
						"after\n" +
						"=======\n" +
						"new content\n" +
						">>>>>>> REPLACE"
					expect(strategy["validateMarkerSequencing"](diff).success).toBe(true)

				test("allows escaped separator in content", () => {

					const diff =
						"<<<<<<< SEARCH\n" +
						"before\n" +
						"\\=======\n" +
						"after\n" +
						"=======\n" +
						"new content\n" +
						">>>>>>> REPLACE"
					expect(strategy["validateMarkerSequencing"](diff).success).toBe(true)

				test("allows escaped replace marker in content", () => {

					const diff =
						"<<<<<<< SEARCH\n" +
						"before\n" +
						"\\>>>>>>> REPLACE\n" +
						"after\n" +
						"=======\n" +
						"new content\n" +
						">>>>>>> REPLACE"
					expect(strategy["validateMarkerSequencing"](diff).success).toBe(true)

				test("allows escaped replace marker in content", () => {

					const diff =
						"<<<<<<< SEARCH\n" +
						"before\n" +
						"\\>>>>>>> REPLACE\n" +
						"after\n" +
						"=======\n" +
						"new content\n" +
						">>>>>>> REPLACE"
					expect(strategy["validateMarkerSequencing"](diff).success).toBe(true)

				test("processes escaped search marker in content", async () => {

					const originalContent = "before\n<<<<<<< SEARCH\nafter\n"
					const diffContent =
						"<<<<<<< SEARCH\n" +
						"before\n" +
						"\\<<<<<<< SEARCH\n" +
						"after\n" +
						"=======\n" +
						"replaced content\n" +
						">>>>>>> REPLACE"
					const result = await strategy.applyDiff(originalContent, diffContent)
					assert.strictEqual(result.success, true)
					if (result.success) {
						assert.strictEqual(result.content, "replaced content\n")

				test("processes escaped replace marker in content", async () => {

					const originalContent = "before\n>>>>>>> REPLACE\nafter\n"
					const diffContent =
						"<<<<<<< SEARCH\n" +
						"before\n" +
						"\\>>>>>>> REPLACE\n" +
						"after\n" +
						"=======\n" +
						"unchanged\n" +
						">>>>>>> REPLACE"
					const result = await strategy.applyDiff(originalContent, diffContent)
					assert.strictEqual(result.success, true)
					if (result.success) {
						assert.strictEqual(result.content, "unchanged\n")

				test("processes multiple escaped markers in content", async () => {

					const originalContent = "<<<<<<< SEARCH\n=======\n>>>>>>> REPLACE\n"
					const diffContent =
						"<<<<<<< SEARCH\n" +
						"\\<<<<<<< SEARCH\n" +
						"\\=======\n" +
						"\\>>>>>>> REPLACE\n" +
						"=======\n" +
						"unchanged\n" +
						">>>>>>> REPLACE"
					const result = await strategy.applyDiff(originalContent, diffContent)
					assert.strictEqual(result.success, true)
					if (result.success) {
						assert.strictEqual(result.content, "unchanged\n")

				test("processes escaped separator in content", async () => {

					const originalContent = "before\n=======\nafter\n"
					const diffContent =
						"<<<<<<< SEARCH\n" +
						"before\n" +
						"\\=======\n" +
						"after\n" +
						"=======\n" +
						"unchanged\n" +
						">>>>>>> REPLACE"
					const result = await strategy.applyDiff(originalContent, diffContent)
					assert.strictEqual(result.success, true)
					if (result.success) {
						assert.strictEqual(result.content, "unchanged\n")

				test("processes escaped search marker in content", async () => {

					const originalContent = "before\n<<<<<<< SEARCH\nafter\n"
					const diffContent =
						"<<<<<<< SEARCH\n" +
						"before\n" +
						"\\<<<<<<< SEARCH\n" +
						"after\n" +
						"=======\n" +
						"unchanged\n" +
						">>>>>>> REPLACE"
					const result = await strategy.applyDiff(originalContent, diffContent)
					assert.strictEqual(result.success, true)
					if (result.success) {
						assert.strictEqual(result.content, "unchanged\n")

				test("processes escaped replace marker in content", async () => {

					const originalContent = "before\n>>>>>>> REPLACE\nafter\n"
					const diffContent =
						"<<<<<<< SEARCH\n" +
						"before\n" +
						"\\>>>>>>> REPLACE\n" +
						"after\n" +
						"=======\n" +
						"unchanged\n" +
						">>>>>>> REPLACE"
					const result = await strategy.applyDiff(originalContent, diffContent)
					assert.strictEqual(result.success, true)
					if (result.success) {
						assert.strictEqual(result.content, "unchanged\n")

				test("processes escaped separator in content", async () => {

					const originalContent = "before\n=======\nafter\n"
					const diffContent =
						"test.ts\n" +
						"<<<<<<< SEARCH\n" +
						"before\n" +
						"\\=======\n" +
						"after\n" +
						"=======\n" +
						"unchanged\n" +
						">>>>>>> REPLACE"
					const result = await strategy.applyDiff(originalContent, diffContent)
					assert.strictEqual(result.success, true)
					if (result.success) {
						assert.strictEqual(result.content, "unchanged\n")

				test("processes escaped replace marker in content", async () => {

					const originalContent = "before\n>>>>>>> REPLACE\nafter\n"
					const diffContent =
						"<<<<<<< SEARCH\n" +
						"before\n" +
						"\\>>>>>>> REPLACE\n" +
						"after\n" +
						"=======\n" +
						"unchanged\n" +
						">>>>>>> REPLACE"
					const result = await strategy.applyDiff(originalContent, diffContent)
					assert.strictEqual(result.success, true)
					if (result.success) {
						assert.strictEqual(result.content, "unchanged\n")

				test("processes multiple escaped markers in content", async () => {

					const originalContent = "<<<<<<< SEARCH\n=======\n>>>>>>> REPLACE\n"
					const diffContent =
						"<<<<<<< SEARCH\n" +
						"\\<<<<<<< SEARCH\n" +
						"\\=======\n" +
						"\\>>>>>>> REPLACE\n" +
						"=======\n" +
						"unchanged\n" +
						">>>>>>> REPLACE"
					const result = await strategy.applyDiff(originalContent, diffContent)
					assert.strictEqual(result.success, true)
					if (result.success) {
						assert.strictEqual(result.content, "unchanged\n")

				test("processes escaped replace marker in content", async () => {

					const originalContent = "before\n>>>>>>> REPLACE\nafter\n"
					const diffContent =
						"<<<<<<< SEARCH\n" +
						"before\n" +
						"\\>>>>>>> REPLACE\n" +
						"after\n" +
						"=======\n" +
						"unchanged\n" +
						">>>>>>> REPLACE"
					const result = await strategy.applyDiff(originalContent, diffContent)
					assert.strictEqual(result.success, true)
					if (result.success) {
						assert.strictEqual(result.content, "unchanged\n")

				test("processes multiple escaped markers in content", async () => {

					const originalContent = "<<<<<<< SEARCH\n=======\n>>>>>>> REPLACE\n"
					const diffContent =
						"<<<<<<< SEARCH\n" +
						"\\<<<<<<< SEARCH\n" +
						"\\=======\n" +
						"\\>>>>>>> REPLACE\n" +
						"=======\n" +
						"unchanged\n" +
						">>>>>>> REPLACE"
					const result = await strategy.applyDiff(originalContent, diffContent)
					assert.strictEqual(result.success, true)
					if (result.success) {
						assert.strictEqual(result.content, "unchanged\n")

				test("allows escaped replace marker in content", () => {

					const diff =
						"<<<<<<< SEARCH\n" +
						"before\n" +
						"\\>>>>>>> REPLACE\n" +
						"after\n" +
						"=======\n" +
						"new content\n" +
						">>>>>>> REPLACE"
					const result = strategy["validateMarkerSequencing"](diff)
					assert.strictEqual(result.success, true)

				test("allows multiple escaped markers in content", () => {

					const diff =
						"<<<<<<< SEARCH\n" +
						"\\<<<<<<< SEARCH\n" +
						"\\=======\n" +
						"\\>>>>>>> REPLACE\n" +
						"=======\n" +
						"new content\n" +
						">>>>>>> REPLACE"
					const result = strategy["validateMarkerSequencing"](diff)
					assert.strictEqual(result.success, true)

				test("detects separator when expecting replace", () => {

					const diff = "<<<<<<< SEARCH\n" + "content\n" + "=======\n" + "new content\n" + "======="
					const result = strategy["validateMarkerSequencing"](diff)
					assert.strictEqual(result.success, false)
					assert.ok(result.error.includes("'=======' found in your diff content"))

			test("should not strip content that starts with pipe but no line number", async () => {

				const originalContent = "| Pipe\n|---|\n| Data\n"
				const diffContent = `test.ts
<<<<<<< SEARCH
| Pipe
|---|
| Data
=======
| Pipe
|---|
| Updated
>>>>>>> REPLACE`

				const result = await strategy.applyDiff(originalContent, diffContent)
				assert.strictEqual(result.success, true)
				if (result.success) {
					assert.strictEqual(result.content, "| Pipe\n|---|\n| Updated\n")

			test("should handle mix of line-numbered and pipe-only content", async () => {

				const originalContent = "| Pipe\n|---|\n| Data\n"
				const diffContent = `test.ts
<<<<<<< SEARCH
| Pipe
|---|
| Data
=======
1 | | Pipe
2 | |---|
3 | | NewData
>>>>>>> REPLACE`

				const result = await strategy.applyDiff(originalContent, diffContent)
				assert.strictEqual(result.success, true)
				if (result.success) {
					assert.strictEqual(result.content, "1 | | Pipe\n2 | |---|\n3 | | NewData\n")

	suite("insertion/deletion", () => {
		let strategy: MultiSearchReplaceDiffStrategy

		setup(() => {
			strategy = new MultiSearchReplaceDiffStrategy()

		suite("deletion", () => {
			test("should delete code when replace block is empty", async () => {

				const originalContent = `function test() {
    console.log("hello");
    // Comment to remove
    console.log("world");
}`
				const diffContent = `test.ts
<<<<<<< SEARCH
    // Comment to remove
=======
>>>>>>> REPLACE`

				const result = await strategy.applyDiff(originalContent, diffContent)
				assert.strictEqual(result.success, true)
				if (result.success) {
					assert.strictEqual(result.content, `function test() {
    console.log("hello");
    console.log("world");
}`)

			test("should delete multiple lines when replace block is empty", async () => {

				const originalContent = `class Example {
    constructor() {
        // Initialize
        this.value = 0;
        // Set defaults
        this.name = "";
        // End init

}`
				const diffContent = `test.ts
<<<<<<< SEARCH
        // Initialize
        this.value = 0;
        // Set defaults
        this.name = "";
        // End init
=======
>>>>>>> REPLACE`

				const result = await strategy.applyDiff(originalContent, diffContent)
				assert.strictEqual(result.success, true)
				if (result.success) {
					assert.strictEqual(result.content, `class Example {
    constructor() {

}`)

			test("should preserve indentation when deleting nested code", async () => {

				const originalContent = `function outer() {
    if (true) {
        // Remove this
        console.log("test");
        // And this

    return true;
}`
				const diffContent = `test.ts
<<<<<<< SEARCH
        // Remove this
        console.log("test");
        // And this
=======
>>>>>>> REPLACE`

				const result = await strategy.applyDiff(originalContent, diffContent)
				assert.strictEqual(result.success, true)
				if (result.success) {
					assert.strictEqual(result.content, `function outer() {
    if (true) {

    return true;
}`)

		suite("insertion", () => {
			test("should insert code at specified line when search block is empty", async () => {

				const originalContent = `function test() {
    const x = 1;
    return x;
}`
				const diffContent = `test.ts
<<<<<<< SEARCH
:start_line:2
:end_line:2
-------
=======
    console.log("Adding log");
>>>>>>> REPLACE`

				const result = await strategy.applyDiff(originalContent, diffContent, 2, 2)
				assert.strictEqual(result.success, true)
				if (result.success) {
					assert.strictEqual(result.content, `function test() {
    console.log("Adding log");
    const x = 1;
    return x;
}`)

			test("should preserve indentation when inserting at nested location", async () => {

				const originalContent = `function test() {
    if (true) {
        const x = 1;

}`
				const diffContent = `test.ts
<<<<<<< SEARCH
:start_line:3
:end_line:3
-------
=======
        console.log("Before");
        console.log("After");
>>>>>>> REPLACE`

				const result = await strategy.applyDiff(originalContent, diffContent, 3, 3)
				assert.strictEqual(result.success, true)
				if (result.success) {
					assert.strictEqual(result.content, `function test() {
    if (true) {
        console.log("Before");
        console.log("After");
        const x = 1;

}`)

			test("should handle insertion at start of file", async () => {

				const originalContent = `function test() {
    return true;
}`
				const diffContent = `test.ts
<<<<<<< SEARCH
:start_line:1
:end_line:1
-------
=======
// Copyright 2024
// License: MIT

>>>>>>> REPLACE`

				const result = await strategy.applyDiff(originalContent, diffContent, 1, 1)
				assert.strictEqual(result.success, true)
				if (result.success) {
					assert.strictEqual(result.content, `// Copyright 2024
// License: MIT

function test() {
    return true;
}`)

			test("should handle insertion at end of file", async () => {

				const originalContent = `function test() {
    return true;
}`
				const diffContent = `test.ts
<<<<<<< SEARCH
:start_line:4
:end_line:4
-------
=======
// End of file
>>>>>>> REPLACE`

				const result = await strategy.applyDiff(originalContent, diffContent, 4, 4)
				assert.strictEqual(result.success, true)
				if (result.success) {
					assert.strictEqual(result.content, `function test() {
    return true;

// End of file`)

			test("should error if no start_line is provided for insertion", async () => {

				const originalContent = `function test() {
    return true;
}`
				const diffContent = `test.ts
<<<<<<< SEARCH
=======
console.log("test");
>>>>>>> REPLACE`

				const result = await strategy.applyDiff(originalContent, diffContent)
				assert.strictEqual(result.success, false)

	suite("fuzzy matching", () => {
		let strategy: MultiSearchReplaceDiffStrategy
		setup(() => {
			strategy = new MultiSearchReplaceDiffStrategy(0.9, 5) // 90% similarity threshold, 5 line buffer for tests

		test("should match content with small differences (>90% similar)", async () => {

			const originalContent =
				"function getData() {\n    const results = fetchData();\n    return results.filter(Boolean);\n}\n"
			const diffContent = `test.ts
<<<<<<< SEARCH
function getData() {
    const result = fetchData();
    return results.filter(Boolean);

=======
function getData() {
    const data = fetchData();
    return data.filter(Boolean);

>>>>>>> REPLACE`

			strategy = new MultiSearchReplaceDiffStrategy(0.9, 5) // Use 5 line buffer for tests

			const result = await strategy.applyDiff(originalContent, diffContent)
			assert.strictEqual(result.success, true)
			if (result.success) {
				assert.strictEqual(result.content, 
					"function getData() {\n    const data = fetchData();\n    return data.filter(Boolean);\n}\n",

		test("should not match when content is too different (<90% similar)", async () => {

			const originalContent = "function processUsers(data) {\n    return data.map(user => user.name);\n}\n"
			const diffContent = `test.ts
<<<<<<< SEARCH
function handleItems(items) {
    return items.map(item => item.username);

=======
function processData(data) {
    return data.map(d => d.value);

>>>>>>> REPLACE`

			const result = await strategy.applyDiff(originalContent, diffContent)
			assert.strictEqual(result.success, false)

		test("should match content with extra whitespace", async () => {

			const originalContent = "function sum(a, b) {\n    return a + b;\n}"
			const diffContent = `test.ts
<<<<<<< SEARCH
function   sum(a,   b)    {
    return    a + b;

=======
function sum(a, b) {
    return a + b + 1;

>>>>>>> REPLACE`

			const result = await strategy.applyDiff(originalContent, diffContent)
			assert.strictEqual(result.success, true)
			if (result.success) {
				assert.strictEqual(result.content, "function sum(a, b) {\n    return a + b + 1;\n}")

		test("should not exact match empty lines", async () => {

			const originalContent = "function sum(a, b) {\n\n    return a + b;\n}"
			const diffContent = `test.ts
<<<<<<< SEARCH
function sum(a, b) {
=======
import { a } from "a";
function sum(a, b) {
>>>>>>> REPLACE`

			const result = await strategy.applyDiff(originalContent, diffContent)
			assert.strictEqual(result.success, true)
			if (result.success) {
				assert.strictEqual(result.content, 'import { a } from "a";\nfunction sum(a, b) {\n\n    return a + b;\n}')

	suite("line-constrained search", () => {
		let strategy: MultiSearchReplaceDiffStrategy

		setup(() => {
			strategy = new MultiSearchReplaceDiffStrategy(0.9, 5)

		test("should find and replace within specified line range", async () => {

			const originalContent = `
function one() {
    return 1;

function two() {
    return 2;

function three() {
    return 3;

`.trim()
			const diffContent = `test.ts
<<<<<<< SEARCH
function two() {
    return 2;

=======
function two() {
    return "two";

>>>>>>> REPLACE`

			const result = await strategy.applyDiff(originalContent, diffContent, 5, 7)
			assert.strictEqual(result.success, true)
			if (result.success) {
				assert.strictEqual(result.content, `function one() {
    return 1;

function two() {
    return "two";

function three() {
    return 3;
}`)

		test("should find and replace within buffer zone (5 lines before/after)", async () => {

			const originalContent = `
function one() {
    return 1;

function two() {
    return 2;

function three() {
    return 3;

`.trim()
			const diffContent = `test.ts
<<<<<<< SEARCH
function three() {
    return 3;

=======
function three() {
    return "three";

>>>>>>> REPLACE`

			// Even though we specify lines 5-7, it should still find the match at lines 9-11
			// because it's within the 5-line buffer zone
			const result = await strategy.applyDiff(originalContent, diffContent, 5, 7)
			assert.strictEqual(result.success, true)
			if (result.success) {
				assert.strictEqual(result.content, `function one() {
    return 1;

function two() {
    return 2;

function three() {
    return "three";
}`)

		test("should not find matches outside search range and buffer zone", async () => {

			const originalContent = `
function one() {
    return 1;

function two() {
    return 2;

function three() {
    return 3;

function four() {
    return 4;

function five() {
    return 5;

`.trim()
			const diffContent = `test.ts
<<<<<<< SEARCH
:start_line:5
:end_line:7
-------
function five() {
    return 5;

=======
function five() {
    return "five";

>>>>>>> REPLACE`

			// Searching around function two() (lines 5-7)
			// function five() is more than 5 lines away, so it shouldn't match
			const result = await strategy.applyDiff(originalContent, diffContent)
			assert.strictEqual(result.success, false)

		test("should handle search range at start of file", async () => {

			const originalContent = `
function one() {
    return 1;

function two() {
    return 2;

`.trim()
			const diffContent = `test.ts
<<<<<<< SEARCH
function one() {
    return 1;

=======
function one() {
    return "one";

>>>>>>> REPLACE`

			const result = await strategy.applyDiff(originalContent, diffContent, 1, 3)
			assert.strictEqual(result.success, true)
			if (result.success) {
				assert.strictEqual(result.content, `function one() {
    return "one";

function two() {
    return 2;
}`)

		test("should handle search range at end of file", async () => {

			const originalContent = `
function one() {
    return 1;

function two() {
    return 2;

`.trim()
			const diffContent = `test.ts
<<<<<<< SEARCH
function two() {
    return 2;

=======
function two() {
    return "two";

>>>>>>> REPLACE`

			const result = await strategy.applyDiff(originalContent, diffContent, 5, 7)
			assert.strictEqual(result.success, true)
			if (result.success) {
				assert.strictEqual(result.content, `function one() {
    return 1;

function two() {
    return "two";
}`)

		test("should match specific instance of duplicate code using line numbers", async () => {

			const originalContent = `
function processData(data) {
    return data.map(x => x * 2);

function unrelatedStuff() {
    console.log("hello");

// Another data processor
function processData(data) {
    return data.map(x => x * 2);

function moreStuff() {
    console.log("world");

`.trim()
			const diffContent = `test.ts
<<<<<<< SEARCH
function processData(data) {
    return data.map(x => x * 2);

=======
function processData(data) {
    // Add logging
    console.log("Processing data...");
    return data.map(x => x * 2);

>>>>>>> REPLACE`

			// Target the second instance of processData
			const result = await strategy.applyDiff(originalContent, diffContent, 10, 12)
			assert.strictEqual(result.success, true)
			if (result.success) {
				assert.strictEqual(result.content, `function processData(data) {
    return data.map(x => x * 2);

function unrelatedStuff() {
    console.log("hello");

// Another data processor
function processData(data) {
    // Add logging
    console.log("Processing data...");
    return data.map(x => x * 2);

function moreStuff() {
    console.log("world");
}`)

		test("should search from start line to end of file when only start_line is provided", async () => {

			const originalContent = `
function one() {
    return 1;

function two() {
    return 2;

function three() {
    return 3;

`.trim()
			const diffContent = `test.ts
<<<<<<< SEARCH
function three() {
    return 3;

=======
function three() {
    return "three";

>>>>>>> REPLACE`

			// Only provide start_line, should search from there to end of file
			const result = await strategy.applyDiff(originalContent, diffContent, 8)
			assert.strictEqual(result.success, true)
			if (result.success) {
				assert.strictEqual(result.content, `function one() {
    return 1;

function two() {
    return 2;

function three() {
    return "three";
}`)

		test("should search from start of file to end line when only end_line is provided", async () => {

			const originalContent = `
function one() {
    return 1;

function two() {
    return 2;

function three() {
    return 3;

`.trim()
			const diffContent = `test.ts
<<<<<<< SEARCH
function one() {
    return 1;

=======
function one() {
    return "one";

>>>>>>> REPLACE`

			// Only provide end_line, should search from start of file to there
			const result = await strategy.applyDiff(originalContent, diffContent, undefined, 4)
			assert.strictEqual(result.success, true)
			if (result.success) {
				assert.strictEqual(result.content, `function one() {
    return "one";

function two() {
    return 2;

function three() {
    return 3;
}`)

		test("should prioritize exact line match over expanded search", async () => {

			const originalContent = `
function one() {
    return 1;

function process() {
    return "old";

function process() {
    return "old";

function two() {
    return 2;
}`
			const diffContent = `test.ts
<<<<<<< SEARCH
function process() {
    return "old";

=======
function process() {
    return "new";

>>>>>>> REPLACE`

			// Should match the second instance exactly at lines 10-12
			// even though the first instance at 6-8 is within the expanded search range
			const result = await strategy.applyDiff(originalContent, diffContent, 10, 12)
			assert.strictEqual(result.success, true)
			if (result.success) {
				assert.strictEqual(result.content, `
function one() {
    return 1;

function process() {
    return "old";

function process() {
    return "new";

function two() {
    return 2;
}`)

		test("should fall back to expanded search only if exact match fails", async () => {

			const originalContent = `
function one() {
    return 1;

function process() {
    return "target";

function two() {
    return 2;
}`.trim()
			const diffContent = `test.ts
<<<<<<< SEARCH
function process() {
    return "target";

=======
function process() {
    return "updated";

>>>>>>> REPLACE`

			// Specify wrong line numbers (3-5), but content exists at 6-8
			// Should still find and replace it since it's within the expanded range
			const result = await strategy.applyDiff(originalContent, diffContent, 3, 5)
			assert.strictEqual(result.success, true)
			if (result.success) {
				assert.strictEqual(result.content, `function one() {
    return 1;

function process() {
    return "updated";

function two() {
    return 2;
}`)

	suite("getToolDescription", () => {
		let strategy: MultiSearchReplaceDiffStrategy

		setup(() => {
			strategy = new MultiSearchReplaceDiffStrategy()

		test("should include the current working directory", () => {

			const cwd = "/test/dir"
			const description = strategy.getToolDescription({ cwd })
			assert.ok(description.includes(`relative to the current working directory ${cwd}`))

		test("should include required format elements", () => {

			const description = strategy.getToolDescription({ cwd: "/test" })
			assert.ok(description.includes("<<<<<<< SEARCH"))
			assert.ok(description.includes("======="))
			assert.ok(description.includes(">>>>>>> REPLACE"))
			assert.ok(description.includes("<apply_diff>"))
			assert.ok(description.includes("</apply_diff>"))
