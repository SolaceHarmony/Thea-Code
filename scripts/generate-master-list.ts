import ts from "typescript"
import fs from "fs"
import path from "path"

type Entry = {
	type: string
	name: string
	file: string
	hasTest: boolean
}

async function main() {
	const configPath = ts.findConfigFile("./", ts.sys.fileExists, "tsconfig.json")
	if (!configPath) {
		console.error("Could not find tsconfig.json")
		process.exit(1)
	}

	const configFile = ts.readConfigFile(configPath, ts.sys.readFile)
	const config = ts.parseJsonConfigFileContent(configFile.config, ts.sys, "./")
	const sourceFiles = config.fileNames.filter((fn) => fn.startsWith("src/") && fn.endsWith(".ts"))

	const entries: Entry[] = []

	for (const fileName of sourceFiles) {
		const sourceText = await fs.promises.readFile(fileName, "utf8")
		const sf = ts.createSourceFile(fileName, sourceText, ts.ScriptTarget.Latest, true)

		// Check for test files
		const baseName = path.basename(fileName, ".ts")
		const dirName = path.dirname(fileName)
		const testDir = path.join(dirName, "__tests__")
		const e2eDir = path.join("src", "e2e", "src", "suite") // Approximate e2e location

		// Possible test file locations
		const possibleTestFiles = [
			path.join(testDir, `${baseName}.test.ts`),
			path.join(testDir, `${baseName}.mocha.test.ts`),
			// Simple check for e2e tests - this is a heuristic
			path.join(e2eDir, dirName.replace("src/", ""), `${baseName}.test.ts`),
			path.join(e2eDir, dirName.replace("src/", ""), `${baseName}.e2e.test.ts`)
		]

		// Also check for co-located tests if that's a pattern, or other common patterns
		// For now, we'll stick to the explicit ones and maybe a recursive search if needed, 
		// but let's keep it simple first: check if ANY test file exists with the component name in it
		// actually, let's do a smarter check.

		let hasTest = false
		for (const testFile of possibleTestFiles) {
			if (fs.existsSync(testFile)) {
				hasTest = true
				break
			}
		}

		// If not found by direct path, try a broader search in the future, but for now this is better than nothing.
		// Actually, let's try to find *any* file in the project that ends with .test.ts and includes the basename.
		// That might be too slow. Let's stick to the specific paths for now, and maybe add a "manual override" list if needed.

		ts.forEachChild(sf, (node) => {
			if (ts.isFunctionDeclaration(node) && node.name) {
				entries.push({ type: "function", name: node.name.text, file: fileName, hasTest })
			} else if (ts.isClassDeclaration(node) && node.name) {
				entries.push({ type: "class", name: node.name.text, file: fileName, hasTest })
			} else if (ts.isInterfaceDeclaration(node) && node.name) {
				entries.push({ type: "interface", name: node.name.text, file: fileName, hasTest })
			}
		})
	}

	entries.sort((a, b) => a.name.localeCompare(b.name))

	const lines = entries.map((e) => `- [${e.hasTest ? "x" : " "}] ${e.type}: ${e.name} (${e.file})`)
	const header = "# Test Coverage Checklist\n\nGenerated with scripts/generate-master-list.ts\n\nNote: [x] means a test file was found that likely covers this component."
	await fs.promises.writeFile("MASTER_TEST_CHECKLIST.md", `${header}\n\n${lines.join("\n")}\n`)

	console.log(`Generated checklist with ${entries.length} items.`)
}

// Run the async main function with error handling
main().catch((error) => {
	console.error("Error generating master list:", error)
	process.exit(1)
})
