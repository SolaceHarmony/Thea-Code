import fs from "fs/promises"
import path from "path"
import { Mode } from "../../../shared/modes"
import { fileExistsAtPath } from "../../../utils/fs"
import { EXTENSION_CONFIG_DIR } from "../../../shared/config/thea-config"

/**
 * Safely reads a file, returning an empty string if the file doesn't exist
 */
async function safeReadFile(filePath: string): Promise<string> {
	try {
		const content = await fs.readFile(filePath, "utf-8")
		// When reading with "utf-8" encoding, content should be a string
		return content.trim()
	} catch (err) {
		const e = err as NodeJS.ErrnoException
		const errorCode = e.code
		const message = (e.message || "").toUpperCase()
		if (
			(errorCode && ["ENOENT", "EISDIR"].includes(errorCode)) ||
			message.includes("ENOENT") ||
			message.includes("EISDIR")
		) {
			return ""
		}
		throw err
	}
}

/**
 * Get the path to a system prompt file for a specific mode
 */
export function getSystemPromptFilePath(cwd: string, mode: Mode): string {
	return path.join(cwd, EXTENSION_CONFIG_DIR, `system-prompt-${mode}`)
}

/**
 * Loads custom system prompt from a file at .thea/system-prompt-[mode slug]
 * If the file doesn't exist, returns an empty string
 */
export async function loadSystemPromptFile(cwd: string, mode: Mode): Promise<string> {
	const filePath = getSystemPromptFilePath(cwd, mode)
	return safeReadFile(filePath)
}

/**
 * Ensures the .thea directory exists, creating it if necessary
 */
export async function ensureConfigDirectory(cwd: string): Promise<void> {
	// Rename function
	const configDir = path.join(cwd, EXTENSION_CONFIG_DIR)

	// Check if directory already exists
	if (await fileExistsAtPath(configDir)) {
		return
	}

	// Create the directory
	try {
		await fs.mkdir(configDir, { recursive: true })
	} catch (err) {
		// If directory already exists (race condition), ignore the error
		const errorCode = (err as NodeJS.ErrnoException).code
		if (errorCode !== "EEXIST") {
			throw err
		}
	}
}
