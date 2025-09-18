import * as fs from "node:fs/promises"
import * as path from "path"

import { build, filesystem, GluegunPrompt } from "gluegun"
import { downloadAndUnzipVSCode, resolveCliArgsFromVSCodeExecutablePath, runTests } from "@vscode/test-electron"
import { spawn } from "child_process"

interface BenchmarkConfig {
	language?: string
	exercise?: string
	runId?: string
}

interface GluegunParameters {
	first?: string
	second?: string
	options: Record<string, string>
}

interface GluegunCommandContext {
	config: BenchmarkConfig
	parameters: GluegunParameters
}

// console.log(__dirname)
// <...>/Thea-Code/benchmark/src

const extensionDevelopmentPath = path.resolve(__dirname, "../../")
const extensionTestsPath = path.resolve(__dirname, "../out/runExercise")
const promptsPath = path.resolve(__dirname, "../prompts")
const exercisesPath = path.resolve(__dirname, "../../../exercises")
const languages = ["cpp", "go", "java", "javascript", "python", "rust"]

async function runAll({ runId, model }: { runId: number; model: string }) {
	for (const language of languages) {
		await runLanguage({ runId, model, language })
	}
}

async function runLanguage({ runId, model, language }: { runId: number; model: string; language: string }) {
	const languagePath = path.resolve(exercisesPath, language)

	try {
		await fs.access(languagePath)
	} catch {
		console.error(`Language directory ${languagePath} does not exist`)
		process.exit(1)
	}

	const exercises = filesystem
		.subdirectories(languagePath)
		.map((exercise) => path.basename(exercise))
		.filter((exercise) => !exercise.startsWith("."))

	for (const exercise of exercises) {
		await runExercise({ runId, model, language, exercise })
	}
}

async function runExercise({
	runId,
	model,
	language,
	exercise,
}: {
	runId: number
	model: string
	language: string
	exercise: string
}) {
	const workspacePath = path.resolve(exercisesPath, language, exercise)
	const promptPath = path.resolve(promptsPath, `${language}.md`)

	const extensionTestsEnv = {
		PROMPT_PATH: promptPath,
		WORKSPACE_PATH: workspacePath,
		OPENROUTER_MODEL_ID: model,
		RUN_ID: runId.toString(),
	}

	try {
		await fs.access(path.resolve(workspacePath, "usage.json"))
		console.log(`Test result exists for ${language} / ${exercise}, skipping`)
		return
	} catch {
		// File doesn't exist, continue with test
	}

	console.log(`Running ${language} / ${exercise}`)

 // Prefer running tests via the VS Code CLI to ensure the host exits cleanly
  try {
    const vscodeExecutablePath = await downloadAndUnzipVSCode({ version: "insiders", extensionDevelopmentPath })
    const [cli] = resolveCliArgsFromVSCodeExecutablePath(vscodeExecutablePath)

    const testRoot = path.resolve(extensionDevelopmentPath, ".vscode-test-benchmark")
    const userDataDir = path.join(testRoot, "user-data")
    const extensionsDir = path.join(testRoot, "extensions")
    try { await fs.mkdir(userDataDir, { recursive: true }) } catch {}
    try { await fs.mkdir(extensionsDir, { recursive: true }) } catch {}

    const args = [
      workspacePath,
      `--user-data-dir=${userDataDir}`,
      `--extensions-dir=${extensionsDir}`,
      "--disable-workspace-trust",
      "--skip-release-notes",
      "--skip-welcome",
      "--disable-updates",
      "--no-sandbox",
      "--disable-gpu-sandbox",
      "--disable-extensions",
      `--extensionDevelopmentPath=${extensionDevelopmentPath}`,
      `--extensionTestsPath=${extensionTestsPath}`,
    ]

    console.log(`[benchmark] Spawning VS Code CLI: ${cli}`)
    const shell = process.platform === "win32"
    await new Promise<void>((resolve, reject) => {
      const child = spawn(shell ? `"${cli}"` : cli, [...args], {
        env: { ...process.env, ...extensionTestsEnv },
        stdio: "inherit",
        shell,
      })

      const killChild = () => {
        try {
          if (!child.killed) {
            if (process.platform === "win32") {
              try { spawn("taskkill", ["/pid", String(child.pid), "/T", "/F"], { stdio: "ignore" }) } catch {}
            } else {
              child.kill("SIGKILL")
            }
          }
        } catch {}
      }
      const onParentExit = () => killChild()
      process.on("exit", onParentExit)
      process.on("SIGINT", onParentExit)
      process.on("SIGTERM", onParentExit)

      child.on("error", (err) => {
        process.off("exit", onParentExit)
        process.off("SIGINT", onParentExit)
        process.off("SIGTERM", onParentExit)
        reject(err)
      })
      child.on("exit", (code, signal) => {
        process.off("exit", onParentExit)
        process.off("SIGINT", onParentExit)
        process.off("SIGTERM", onParentExit)
        console.log(`[benchmark] VS Code exited with ${code ?? signal}`)
        code === 0 ? resolve() : reject(new Error(`VS Code exited with ${code ?? signal}`))
      })
    })
  } catch (cliErr) {
    console.warn(`[benchmark] CLI spawn failed (${cliErr instanceof Error ? cliErr.message : String(cliErr)}); falling back to test-electron`)
    const testRoot = path.resolve(extensionDevelopmentPath, ".vscode-test-benchmark")
    const userDataDir = path.join(testRoot, "user-data")
    const extensionsDir = path.join(testRoot, "extensions")
    try { await fs.mkdir(userDataDir, { recursive: true }) } catch {}
    try { await fs.mkdir(extensionsDir, { recursive: true }) } catch {}

    await runTests({
      extensionDevelopmentPath,
      extensionTestsPath,
      launchArgs: [
        workspacePath,
        `--user-data-dir=${userDataDir}`,
        `--extensions-dir=${extensionsDir}`,
        "--disable-workspace-trust",
        "--skip-release-notes",
        "--skip-welcome",
        "--disable-updates",
        "--no-sandbox",
        "--disable-gpu-sandbox",
        "--disable-extensions",
      ],
      extensionTestsEnv,
    })
  }
}

async function askLanguage(prompt: GluegunPrompt) {
	const languages = filesystem.subdirectories(exercisesPath)

	if (languages.length === 0) {
		throw new Error(`No languages found in ${exercisesPath}`)
	}

	const { language } = await prompt.ask<{ language: string }>({
		type: "select",
		name: "language",
		message: "Which language?",
		choices: languages.map((language) => path.basename(language)).filter((language) => !language.startsWith(".")),
	})

	return language
}

async function askExercise(prompt: GluegunPrompt, language: string) {
	const exercises = filesystem.subdirectories(path.join(exercisesPath, language))

	if (exercises.length === 0) {
		throw new Error(`No exercises found for ${language}`)
	}

	const { exercise } = await prompt.ask<{ exercise: string }>({
		type: "select",
		name: "exercise",
		message: "Which exercise?",
		choices: exercises.map((exercise) => path.basename(exercise)),
	})

	return exercise
}

async function createRun({ model }: { model: string }): Promise<{ id: number; model: string }> {
	const response = await fetch("http://localhost:3000/api/runs", {
		method: "POST",
		body: JSON.stringify({ model }),
	})

	if (!response.ok) {
		throw new Error(`Failed to create run: ${response.statusText}`)
	}

	const {
		run: [run],
	} = (await response.json()) as { run: [{ id: number; model: string }] }
	return run
}

async function main() {
	const cli = build()
		.brand("benchmark-runner")
		.src(__dirname)
		.help()
		.version()
		.command({
			name: "run",
			run: ({ config, parameters }: GluegunCommandContext) => {
				config.language = parameters.first
				config.exercise = parameters.second

				if (parameters.options["runId"]) {
					config.runId = parameters.options["runId"]
				}
			},
		})
		.defaultCommand() // Use the default command if no args.
		.create()

	const { print, prompt, config } = await cli.run(process.argv)
	const benchmarkConfig = config as BenchmarkConfig

	try {
		const model = "anthropic/claude-3.7-sonnet"
		const runId = benchmarkConfig.runId ? Number(benchmarkConfig.runId) : (await createRun({ model })).id

		if (benchmarkConfig.language === "all") {
			console.log("Running all exercises for all languages")
			await runAll({ runId, model })
		} else if (benchmarkConfig.exercise === "all") {
			console.log(`Running all exercises for ${benchmarkConfig.language!}`)
			await runLanguage({ runId, model, language: benchmarkConfig.language! })
		} else {
			const language = benchmarkConfig.language || (await askLanguage(prompt))
			const exercise = benchmarkConfig.exercise || (await askExercise(prompt, language))
			await runExercise({ runId, model, language, exercise })
		}

		process.exit(0)
	} catch (error) {
		print.error(error)
		process.exit(1)
	}
}

void main()
