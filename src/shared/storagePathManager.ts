import type * as vscode from 'vscode'
import * as path from "path"
import * as fs from "fs/promises"
import { t } from "../i18n"
import { configSection } from "./config/thea-config"

/**
 * Gets the base storage path for conversations
 * If a custom path is configured, uses that path
 * Otherwise uses the default VSCode extension global storage path
 */
export async function getStorageBasePath(defaultPath: string): Promise<string> {
	// Get user-configured custom storage path
	let customStoragePath = ""

  try {
    const vs = (await import('vscode')) as typeof vscode
    const section = (configSection as () => string)()
    const config = vs.workspace.getConfiguration(section)
    customStoragePath = config.get<string>("customStoragePath", "")
  } catch {
		console.warn("Could not access VSCode configuration - using default path")
		return defaultPath
	}

	// If no custom path is set, use default path
	if (!customStoragePath) {
		return defaultPath
	}

	try {
		// Ensure custom path exists
		await fs.mkdir(customStoragePath, { recursive: true })

		// Test if path is writable
		const testFile = path.join(customStoragePath, ".write_test")
		await fs.writeFile(testFile, "test")
		await fs.rm(testFile)

		return customStoragePath
	} catch (error) {
		// If path is unusable, report error and fall back to default path
		const errorMessage = error instanceof Error ? error.message : String(error)
    console.error(
        t("common:errors.custom_storage_path_unusable_log", { path: customStoragePath, error: errorMessage }),
    )
    try {
      const vs = (await import('vscode')) as typeof vscode
      vs.window.showErrorMessage(t("common:errors.custom_storage_path_unusable", { path: customStoragePath }))
    } catch {
      // ignore if not running in VS Code host
    }
    return defaultPath
  }
}

/**
 * Gets the storage directory path for a task
 */
export async function getTaskDirectoryPath(globalStoragePath: string, taskId: string): Promise<string> {
	const basePath = await getStorageBasePath(globalStoragePath)
	const taskDir = path.join(basePath, "tasks", taskId)
	await fs.mkdir(taskDir, { recursive: true })
	return taskDir
}

/**
 * Gets the settings directory path
 */
export async function getSettingsDirectoryPath(globalStoragePath: string): Promise<string> {
	const basePath = await getStorageBasePath(globalStoragePath)
	const settingsDir = path.join(basePath, "settings")
	await fs.mkdir(settingsDir, { recursive: true })
	return settingsDir
}

/**
 * Gets the cache directory path
 */
export async function getCacheDirectoryPath(globalStoragePath: string): Promise<string> {
	const basePath = await getStorageBasePath(globalStoragePath)
	const cacheDir = path.join(basePath, "cache")
	await fs.mkdir(cacheDir, { recursive: true })
	return cacheDir
}

/**
 * Prompts the user to set a custom storage path
 * Displays an input box allowing the user to enter a custom path
 */
export async function promptForCustomStoragePath(): Promise<void> {
	// VS Code API is expected to be available in an extension context
	// No need for explicit checks for vscode.window or vscode.workspace

  let vs: typeof import('vscode')
  try {
    vs = (await import('vscode')) as typeof vscode
  } catch {
    console.error("promptForCustomStoragePath called outside of VS Code host")
    return
  }
  let currentPath = ""
  try {
    const section = (configSection as () => string)()
    const currentConfig = vs.workspace.getConfiguration(section)
    currentPath = currentConfig.get<string>("customStoragePath", "")
  } catch {
    console.error("Could not access configuration")
    return
  }

  const result = await vs.window.showInputBox({
		value: currentPath,
		placeHolder: t("common:storage.path_placeholder"),
		prompt: t("common:storage.prompt_custom_path"),
		validateInput: (input) => {
			if (!input) {
				return null // Allow empty value (use default path)
			}

			try {
				// Validate path format
				path.parse(input)

				// Check if path is absolute
				if (!path.isAbsolute(input)) {
					return t("common:storage.enter_absolute_path")
				}

				return null // Path format is valid
			} catch {
				return t("common:storage.enter_valid_path")
			}
		},
	})

	// If user canceled the operation, result will be undefined
  if (result !== undefined) {
		try {
      const section = (configSection as () => string)()
      const currentConfig = vs.workspace.getConfiguration(section)
      await currentConfig.update("customStoragePath", result, vs.ConfigurationTarget.Global)

			if (result) {
				try {
					// Test if path is accessible
					await fs.mkdir(result, { recursive: true })
          vs.window.showInformationMessage(t("common:info.custom_storage_path_set", { path: result }))
				} catch (error) {
					const errorMessage = error instanceof Error ? error.message : String(error)
          vs.window.showErrorMessage(
            t("common:errors.cannot_access_path", {
              path: result,
              error: errorMessage,
            }),
          )
        }
      } else {
        vs.window.showInformationMessage(t("common:info.default_storage_path"))
      }
    } catch (error) {
			const errorMessage = error instanceof Error ? error.message : String(error)
			console.error(t("common:errors.failed_to_update_config", { error: errorMessage }))
		}
	}
}
