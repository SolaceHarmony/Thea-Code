import * as path from "path"

import { runTests } from "@vscode/test-electron"

async function main() {
	try {
		// The folder containing the Extension Manifest package.json
		// Passed to `--extensionDevelopmentPath`
		const extensionDevelopmentPath = path.resolve(__dirname, "../../")

		// The path to the extension test script
		// Passed to --extensionTestsPath
		const extensionTestsPath = path.resolve(__dirname, "./suite/index")

		console.log("Extension path:", extensionDevelopmentPath)
		console.log("Test path:", extensionTestsPath)
		
		// Download VS Code, unzip it and run the integration test
		await runTests({ 
			extensionDevelopmentPath, 
			extensionTestsPath,
			launchArgs: [
				'--disable-gpu',
				'--no-sandbox',
				'--disable-dev-shm-usage',
				'--log=trace',
				'--disable-extensions',
				'--skip-welcome',
				'--skip-release-notes'
			],
			extensionTestsEnv: {
				THEA_E2E: '1',
				NODE_ENV: 'test'
			}
		})
	} catch {
		console.error("Failed to run tests")
		process.exit(1)
	}
}

void main()
