import * as assert from 'assert'
import { expect } from 'chai'
import * as sinon from 'sinon'
import { EXTENSION_SECRETS_PREFIX } from "../../../shared/config/thea-config"
import type { ExtensionContext } from "vscode"

import { ProviderSettingsManager, ProviderProfiles } from "../../config/ProviderSettingsManager"
import { ProviderSettings } from "../../../schemas"

// Mock VSCode ExtensionContext
const mockSecrets = {
	get: sinon.stub(),
	store: sinon.stub(),
	delete: sinon.stub(),
}

const mockContext = {
	secrets: mockSecrets,
} as unknown as ExtensionContext

suite("ProviderSettingsManager", () => {
	let providerSettingsManager: ProviderSettingsManager

	setup(() => {
		sinon.restore()
		providerSettingsManager = new ProviderSettingsManager(mockContext)
	})

	suite("initialize", () => {
		test("should not write to storage when secrets.get returns null", async () => {
			// Mock readConfig to return null
			mockSecrets.get.mockResolvedValueOnce(null)

			await providerSettingsManager.initialize()

			// Should not write to storage because readConfig returns defaultConfig
			assert.ok(!mockSecrets.store.called)
		})

		test("should not initialize config if it exists", async () => {
			mockSecrets.get.resolves(
				JSON.stringify({
					currentApiConfigName: "default",
					apiConfigs: {
						default: {
							config: {},
							id: "default",
						},
					},
				})

			await providerSettingsManager.initialize()

			assert.ok(!mockSecrets.store.called)
		})

		test("should generate IDs for configs that lack them", async () => {
			// Mock a config with missing IDs
			mockSecrets.get.resolves(
				JSON.stringify({
					currentApiConfigName: "default",
					apiConfigs: {
						default: {
							config: {},
						},
						test: {
							apiProvider: "anthropic",
						},
					},
				})

			await providerSettingsManager.initialize()

			// Should have written the config with new IDs
			assert.ok(mockSecrets.store.called)
			// eslint-disable-next-line @typescript-eslint/no-unsafe-member-access, @typescript-eslint/no-unsafe-argument
			const storedConfig = JSON.parse(mockSecrets.store.mock.calls[0][1]) as ProviderProfiles
			assert.ok(storedConfig.apiConfigs.default.id)
			assert.ok(storedConfig.apiConfigs.test.id)
		})

		test("should throw error if secrets storage fails", async () => {
			mockSecrets.get.rejects(new Error("Storage failed"))

			await expect(providerSettingsManager.initialize()).rejects.toThrow(
				"Failed to initialize config: Error: Failed to read provider profiles from secrets: Error: Storage failed",
			)
		})
	})

	suite("ListConfig", () => {
		test("should list all available configs", async () => {
			const existingConfig: ProviderProfiles = {
				currentApiConfigName: "default",
				apiConfigs: {
					default: {
						id: "default",
					},
					test: {
						apiProvider: "anthropic",
						id: "test-id",
					},
				},
				modeApiConfigs: {
					code: "default",
					architect: "default",
					ask: "default",
				},
			}

			mockSecrets.get.resolves(JSON.stringify(existingConfig))

			const configs = await providerSettingsManager.listConfig()
			assert.deepStrictEqual(configs, [
				{ name: "default", id: "default", apiProvider: undefined },
				{ name: "test", id: "test-id", apiProvider: "anthropic" },
			])
		})

		test("should handle empty config file", async () => {
			const emptyConfig: ProviderProfiles = {
				currentApiConfigName: "default",
				apiConfigs: {},
				modeApiConfigs: {
					code: "default",
					architect: "default",
					ask: "default",
				},
			}

			mockSecrets.get.resolves(JSON.stringify(emptyConfig))

			const configs = await providerSettingsManager.listConfig()
			assert.deepStrictEqual(configs, [])
		})

		test("should throw error if reading from secrets fails", async () => {
			mockSecrets.get.rejects(new Error("Read failed"))

			await expect(providerSettingsManager.listConfig()).rejects.toThrow(
				"Failed to list configs: Error: Failed to read provider profiles from secrets: Error: Read failed",
			)
		})
	})

	suite("SaveConfig", () => {
		test("should save new config", async () => {
			mockSecrets.get.resolves(
				JSON.stringify({
					currentApiConfigName: "default",
					apiConfigs: {
						default: {},
					},
					modeApiConfigs: {
						code: "default",
						architect: "default",
						ask: "default",
					},
				})

			const newConfig: ProviderSettings = {
				apiProvider: "anthropic",
				apiKey: "test-key",
			}

			await providerSettingsManager.saveConfig("test", newConfig)

			// Get the actual stored config to check the generated ID
			// eslint-disable-next-line @typescript-eslint/no-unsafe-member-access, @typescript-eslint/no-unsafe-argument
			const storedConfig = JSON.parse(mockSecrets.store.mock.calls[0][1]) as ProviderProfiles
			const testConfigId = storedConfig.apiConfigs.test.id

			const expectedConfig = {
				currentApiConfigName: "default",
				apiConfigs: {
					default: {},
					test: {
						...newConfig,
						id: testConfigId,
					},
				},
				modeApiConfigs: {
					code: "default",
					architect: "default",
					ask: "default",
				},
			}

			assert.ok(mockSecrets.store.calledWith(
				`${EXTENSION_SECRETS_PREFIX}api_config`,
				JSON.stringify(expectedConfig, null, 2)),
			)
		})

		test("should update existing config", async () => {
			const existingConfig: ProviderProfiles = {
				currentApiConfigName: "default",
				apiConfigs: {
					test: {
						apiProvider: "anthropic",
						apiKey: "old-key",
						id: "test-id",
					},
				},
			}

			mockSecrets.get.resolves(JSON.stringify(existingConfig))

			const updatedConfig: ProviderSettings = {
				apiProvider: "anthropic",
				apiKey: "new-key",
			}

			await providerSettingsManager.saveConfig("test", updatedConfig)

			const expectedConfig = {
				currentApiConfigName: "default",
				apiConfigs: {
					test: {
						apiProvider: "anthropic",
						apiKey: "new-key",
						id: "test-id",
					},
				},
			}

			assert.ok(mockSecrets.store.calledWith(
				`${EXTENSION_SECRETS_PREFIX}api_config`,
				JSON.stringify(expectedConfig, null, 2)),
			)
		})

		test("should throw error if secrets storage fails", async () => {
			mockSecrets.get.resolves(
				JSON.stringify({
					currentApiConfigName: "default",
					apiConfigs: { default: {} },
				})
			mockSecrets.store.mockRejectedValueOnce(new Error("Storage failed"))

			await expect(providerSettingsManager.saveConfig("test", {})).rejects.toThrow(
				"Failed to save config: Error: Failed to write provider profiles to secrets: Error: Storage failed",
			)
		})
	})

	suite("DeleteConfig", () => {
		test("should delete existing config", async () => {
			const existingConfig: ProviderProfiles = {
				currentApiConfigName: "default",
				apiConfigs: {
					default: {
						id: "default",
					},
					test: {
						apiProvider: "anthropic",
						id: "test-id",
					},
				},
			}

			mockSecrets.get.resolves(JSON.stringify(existingConfig))

			await providerSettingsManager.deleteConfig("test")
			// Get the stored config to check the ID
			// eslint-disable-next-line @typescript-eslint/no-unsafe-member-access, @typescript-eslint/no-unsafe-argument
			const storedConfig = JSON.parse(mockSecrets.store.mock.calls[0][1]) as ProviderProfiles
			assert.strictEqual(storedConfig.currentApiConfigName, "default")
			expect(Object.keys(storedConfig.apiConfigs)).toEqual(["default"])
			assert.ok(storedConfig.apiConfigs.default.id)
		})

		test("should throw error when trying to delete non-existent config", async () => {
			mockSecrets.get.resolves(
				JSON.stringify({
					currentApiConfigName: "default",
					apiConfigs: { default: {} },
				})

			await expect(providerSettingsManager.deleteConfig("nonexistent")).rejects.toThrow(
				"Config 'nonexistent' not found",
			)
		})

		test("should throw error when trying to delete last remaining config", async () => {
			mockSecrets.get.resolves(
				JSON.stringify({
					currentApiConfigName: "default",
					apiConfigs: {
						default: {
							id: "default",
						},
					},
				})

			await expect(providerSettingsManager.deleteConfig("default")).rejects.toThrow(
				"Failed to delete config: Error: Cannot delete the last remaining configuration",
			)
		})
	})

	suite("LoadConfig", () => {
		test("should load config and update current config name", async () => {
			const existingConfig: ProviderProfiles = {
				currentApiConfigName: "default",
				apiConfigs: {
					test: {
						apiProvider: "anthropic",
						apiKey: "test-key",
						id: "test-id",
					},
				},
			}

			mockSecrets.get.resolves(JSON.stringify(existingConfig))

			const config = await providerSettingsManager.loadConfig("test")

			assert.deepStrictEqual(config, {
				apiProvider: "anthropic",
				apiKey: "test-key",
				id: "test-id",
			})
			// Get the stored config to check the structure
			// eslint-disable-next-line @typescript-eslint/no-unsafe-member-access, @typescript-eslint/no-unsafe-argument
			const storedConfig = JSON.parse(mockSecrets.store.mock.calls[0][1]) as ProviderProfiles
			assert.strictEqual(storedConfig.currentApiConfigName, "test")
			assert.deepStrictEqual(storedConfig.apiConfigs.test, {
				apiProvider: "anthropic",
				apiKey: "test-key",
				id: "test-id",
			})
		})

		test("should throw error when config does not exist", async () => {
			mockSecrets.get.resolves(
				JSON.stringify({
					currentApiConfigName: "default",
					apiConfigs: {
						default: {
							config: {},
							id: "default",
						},
					},
				})

			await expect(providerSettingsManager.loadConfig("nonexistent")).rejects.toThrow(
				"Config 'nonexistent' not found",
			)
		})

		test("should throw error if secrets storage fails", async () => {
			mockSecrets.get.resolves(
				JSON.stringify({
					currentApiConfigName: "default",
					apiConfigs: {
						test: {
							config: {
								apiProvider: "anthropic",
							},
							id: "test-id",
						},
					},
				})
			mockSecrets.store.mockRejectedValueOnce(new Error("Storage failed"))

			await expect(providerSettingsManager.loadConfig("test")).rejects.toThrow(
				"Failed to load config: Error: Failed to write provider profiles to secrets: Error: Storage failed",
			)
		})
	})

	suite("ResetAllConfigs", () => {
		test("should delete all stored configs", async () => {
			// Setup initial config
			mockSecrets.get.resolves(
				JSON.stringify({
					currentApiConfigName: "test",
					apiConfigs: {
						test: {
							apiProvider: "anthropic",
							id: "test-id",
						},
					},
				})

			await providerSettingsManager.resetAllConfigs()

			// Should have called delete with the correct config key
			assert.ok(mockSecrets.delete.calledWith(`${EXTENSION_SECRETS_PREFIX}api_config`))
		})
	})

	suite("HasConfig", () => {
		test("should return true for existing config", async () => {
			const existingConfig: ProviderProfiles = {
				currentApiConfigName: "default",
				apiConfigs: {
					default: {
						id: "default",
					},
					test: {
						apiProvider: "anthropic",
						id: "test-id",
					},
				},
			}

			mockSecrets.get.resolves(JSON.stringify(existingConfig))

			const hasConfig = await providerSettingsManager.hasConfig("test")
			assert.strictEqual(hasConfig, true)
		})

		test("should return false for non-existent config", async () => {
			mockSecrets.get.resolves(
				JSON.stringify({
					currentApiConfigName: "default",
					apiConfigs: { default: {} },
				})

			const hasConfig = await providerSettingsManager.hasConfig("nonexistent")
			assert.strictEqual(hasConfig, false)
		})

		test("should throw error if secrets storage fails", async () => {
			mockSecrets.get.rejects(new Error("Storage failed"))

			await expect(providerSettingsManager.hasConfig("test")).rejects.toThrow(
				"Failed to check config existence: Error: Failed to read provider profiles from secrets: Error: Storage failed",
			)
		})
	})
// Mock cleanup
})
