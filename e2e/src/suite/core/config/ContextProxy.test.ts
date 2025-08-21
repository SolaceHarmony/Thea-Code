import * as assert from 'assert'
import * as sinon from 'sinon'
/* eslint-disable @typescript-eslint/no-unsafe-assignment */
/* eslint-disable @typescript-eslint/no-unsafe-member-access */
/* eslint-disable @typescript-eslint/no-unsafe-call */
/* eslint-disable @typescript-eslint/no-unsafe-argument */
/* eslint-disable @typescript-eslint/await-thenable */
/* eslint-disable @typescript-eslint/require-await */
/* eslint-disable @typescript-eslint/no-explicit-any */

import * as vscode from "vscode"
import { ContextProxy } from "../ContextProxy"

import { GLOBAL_STATE_KEYS, SECRET_STATE_KEYS } from "../../../schemas"

// TODO: Use proxyquire for module mocking
		// Mock for "vscode" needed here
	Uri: {
		file: sinon.stub((path: string) => ({ path })),
	},
	ExtensionMode: {
		Development: 1,
		Production: 2,
		Test: 3,
	},
// Mock cleanup

suite("ContextProxy", () => {
	let proxy: ContextProxy
	let mockContext: any
	let mockGlobalState: any
	let mockSecrets: any

	setup(async () => {
		// Reset mocks
		sinon.restore()

		// Mock globalState
		mockGlobalState = {
			get: sinon.stub(),
			update: sinon.stub().resolves(undefined),
		}

		// Mock secrets
		mockSecrets = {
			get: sinon.stub().resolves("test-secret"),
			store: sinon.stub().resolves(undefined),
			delete: sinon.stub().resolves(undefined),
		}

		// Mock the extension context
		mockContext = {
			globalState: mockGlobalState,
			secrets: mockSecrets,
			extensionUri: { path: "/test/extension" },
			extensionPath: "/test/extension",
			globalStorageUri: { path: "/test/storage" },
			logUri: { path: "/test/logs" },
			extension: { packageJSON: { version: "1.0.0" } },
			extensionMode: vscode.ExtensionMode.Development,
		}

		// Create proxy instance
		proxy = new ContextProxy(mockContext)
		await proxy.initialize()
	})

	suite("read-only pass-through properties", () => {
		test("should return extension properties from the original context", () => {
			assert.strictEqual(proxy.extensionUri, mockContext.extensionUri)
			assert.strictEqual(proxy.extensionPath, mockContext.extensionPath)
			assert.strictEqual(proxy.globalStorageUri, mockContext.globalStorageUri)
			assert.strictEqual(proxy.logUri, mockContext.logUri)
			assert.strictEqual(proxy.extension, mockContext.extension)
			assert.strictEqual(proxy.extensionMode, mockContext.extensionMode)
		})
	})

	suite("constructor", () => {
		test("should initialize state cache with all global state keys", () => {
			expect(mockGlobalState.get).callCount, GLOBAL_STATE_KEYS.length
			for (const key of GLOBAL_STATE_KEYS) {
				assert.ok(mockGlobalState.get.calledWith(key))
			}
		})

		test("should initialize secret cache with all secret keys", () => {
			expect(mockSecrets.get).callCount, SECRET_STATE_KEYS.length
			for (const key of SECRET_STATE_KEYS) {
				assert.ok(mockSecrets.get.calledWith(key))
			}
		})
	})

	suite("getGlobalState", () => {
		test("should return value from cache when it exists", async () => {
			// Manually set a value in the cache
			await proxy.updateGlobalState("apiProvider", "deepseek")

			// Should return the cached value
			const result = proxy.getGlobalState("apiProvider")
			assert.strictEqual(result, "deepseek")

			// Original context should be called once during updateGlobalState
			expect(mockGlobalState.get).callCount, GLOBAL_STATE_KEYS.length // Only from initialization
		})

		test("should handle default values correctly", async () => {
			// No value in cache
			const result = proxy.getGlobalState("apiProvider", "deepseek")
			assert.strictEqual(result, "deepseek")
		})

		test("should bypass cache for pass-through state keys", async () => {
			// Setup mock return value
			mockGlobalState.get.returns("pass-through-value")

			// Use a pass-through key (taskHistory)
			const result = proxy.getGlobalState("taskHistory")

			// Should get value directly from original context
			assert.strictEqual(result, "pass-through-value")
			assert.ok(mockGlobalState.get.calledWith("taskHistory"))
		})

		test("should respect default values for pass-through state keys", async () => {
			// Setup mock to return undefined
			mockGlobalState.get.returns(undefined)

			// Use a pass-through key with default value
			const historyItems = [
				{
					id: "1",
					number: 1,
					ts: 1,
					task: "test",
					tokensIn: 1,
					tokensOut: 1,
					totalCost: 1,
				},
			]

			const result = proxy.getGlobalState("taskHistory", historyItems)

			// Should return default value when original context returns undefined
			assert.strictEqual(result, historyItems)
		})
	})

	suite("updateGlobalState", () => {
		test("should update state directly in original context", async () => {
			await proxy.updateGlobalState("apiProvider", "deepseek")

			// Should have called original context
			assert.ok(mockGlobalState.update.calledWith("apiProvider", "deepseek"))

			// Should have stored the value in cache
			const storedValue = await proxy.getGlobalState("apiProvider")
			assert.strictEqual(storedValue, "deepseek")
		})

		test("should bypass cache for pass-through state keys", async () => {
			const historyItems = [
				{
					id: "1",
					number: 1,
					ts: 1,
					task: "test",
					tokensIn: 1,
					tokensOut: 1,
					totalCost: 1,
				},
			]

			await proxy.updateGlobalState("taskHistory", historyItems)

			// Should update original context
			assert.ok(mockGlobalState.update.calledWith("taskHistory", historyItems))

			// Setup mock for subsequent get
			mockGlobalState.get.returns(historyItems)

			// Should get fresh value from original context
			const storedValue = proxy.getGlobalState("taskHistory")
			assert.strictEqual(storedValue, historyItems)
			assert.ok(mockGlobalState.get.calledWith("taskHistory"))
		})
	})

	suite("getSecret", () => {
		test("should return value from cache when it exists", async () => {
			// Manually set a value in the cache
			await proxy.storeSecret("apiKey", "cached-secret")

			// Should return the cached value
			const result = proxy.getSecret("apiKey")
			assert.strictEqual(result, "cached-secret")
		})
	})

	suite("storeSecret", () => {
		test("should store secret directly in original context", async () => {
			await proxy.storeSecret("apiKey", "new-secret")

			// Should have called original context
			assert.ok(mockSecrets.store.calledWith("apiKey", "new-secret"))

			// Should have stored the value in cache
			const storedValue = await proxy.getSecret("apiKey")
			assert.strictEqual(storedValue, "new-secret")
		})

		test("should handle undefined value for secret deletion", async () => {
			await proxy.storeSecret("apiKey", undefined)

			// Should have called delete on original context
			assert.ok(mockSecrets.delete.calledWith("apiKey"))

			// Should have stored undefined in cache
			const storedValue = await proxy.getSecret("apiKey")
			assert.strictEqual(storedValue, undefined)
		})
	})

	suite("setValue", () => {
		test("should route secret keys to storeSecret", async () => {
			// Spy on storeSecret
			const storeSecretSpy = sinon.spy(proxy, "storeSecret")

			// Test with a known secret key
			await proxy.setValue("openAiApiKey", "test-api-key")

			// Should have called storeSecret
			assert.ok(storeSecretSpy.calledWith("openAiApiKey", "test-api-key"))

			// Should have stored the value in secret cache
			const storedValue = proxy.getSecret("openAiApiKey")
			assert.strictEqual(storedValue, "test-api-key")
		})

		test("should route global state keys to updateGlobalState", async () => {
			// Spy on updateGlobalState
			const updateGlobalStateSpy = sinon.spy(proxy, "updateGlobalState")

			// Test with a known global state key
			await proxy.setValue("apiModelId", "gpt-4")

			// Should have called updateGlobalState
			assert.ok(updateGlobalStateSpy.calledWith("apiModelId", "gpt-4"))

			// Should have stored the value in state cache
			const storedValue = proxy.getGlobalState("apiModelId")
			assert.strictEqual(storedValue, "gpt-4")
		})
	})

	suite("setValues", () => {
		test("should process multiple values correctly", async () => {
			// Spy on setValue
			const setValueSpy = sinon.spy(proxy, "setValue")

			// Test with multiple values
			await proxy.setValues({
				apiModelId: "gpt-4",
				apiProvider: "openai",
				mode: "test-mode",
			})

			// Should have called setValue for each key
			assert.strictEqual(setValueSpy.callCount, 3)
			assert.ok(setValueSpy.calledWith("apiModelId", "gpt-4"))
			assert.ok(setValueSpy.calledWith("apiProvider", "openai"))
			assert.ok(setValueSpy.calledWith("mode", "test-mode"))

			// Should have stored all values in state cache
			expect(proxy.getGlobalState("apiModelId")).toBe("gpt-4")
			expect(proxy.getGlobalState("apiProvider")).toBe("openai")
			expect(proxy.getGlobalState("mode")).toBe("test-mode")
		})

		test("should handle both secret and global state keys", async () => {
			// Spy on storeSecret and updateGlobalState
			const storeSecretSpy = sinon.spy(proxy, "storeSecret")
			const updateGlobalStateSpy = sinon.spy(proxy, "updateGlobalState")

			// Test with mixed keys
			await proxy.setValues({
				apiModelId: "gpt-4", // global state
				openAiApiKey: "test-api-key", // secret
			})

			// Should have called appropriate methods
			assert.ok(storeSecretSpy.calledWith("openAiApiKey", "test-api-key"))
			assert.ok(updateGlobalStateSpy.calledWith("apiModelId", "gpt-4"))

			// Should have stored values in appropriate caches
			expect(proxy.getSecret("openAiApiKey")).toBe("test-api-key")
			expect(proxy.getGlobalState("apiModelId")).toBe("gpt-4")
		})
	})

	suite("setProviderSettings", () => {
		test("should clear old API configuration values and set new ones", async () => {
			// Set up initial API configuration values
			await proxy.updateGlobalState("apiModelId", "old-model")
			await proxy.updateGlobalState("openAiBaseUrl", "https://old-url.com")
			await proxy.updateGlobalState("modelTemperature", 0.7)

			// Spy on setValues
			const setValuesSpy = sinon.spy(proxy, "setValues")

			// Call setProviderSettings with new configuration
			await proxy.setProviderSettings({
				apiModelId: "new-model",
				apiProvider: "anthropic",
				// Note: openAiBaseUrl is not included in the new config
			})

			// Verify setValues was called with the correct parameters
			// It should include undefined for openAiBaseUrl (to clear it)
			// and the new values for apiModelId and apiProvider
			assert.ok(setValuesSpy.calledWith({
					apiModelId: "new-model",
					apiProvider: "anthropic",
					openAiBaseUrl: undefined,
					modelTemperature: undefined,
				})),
			)

			// Verify the state cache has been updated correctly
			expect(proxy.getGlobalState("apiModelId")).toBe("new-model")
			expect(proxy.getGlobalState("apiProvider")).toBe("anthropic")
			expect(proxy.getGlobalState("openAiBaseUrl")).toBeUndefined()
			expect(proxy.getGlobalState("modelTemperature")).toBeUndefined()
		})

		test("should handle empty API configuration", async () => {
			// Set up initial API configuration values
			await proxy.updateGlobalState("apiModelId", "old-model")
			await proxy.updateGlobalState("openAiBaseUrl", "https://old-url.com")

			// Spy on setValues
			const setValuesSpy = sinon.spy(proxy, "setValues")

			// Call setProviderSettings with empty configuration
			await proxy.setProviderSettings({})

			// Verify setValues was called with undefined for all existing API config keys
			assert.ok(setValuesSpy.calledWith({
					apiModelId: undefined,
					openAiBaseUrl: undefined,
				})),
			)

			// Verify the state cache has been cleared
			expect(proxy.getGlobalState("apiModelId")).toBeUndefined()
			expect(proxy.getGlobalState("openAiBaseUrl")).toBeUndefined()
		})
	})

	suite("resetAllState", () => {
		test("should clear all in-memory caches", async () => {
			// Setup initial state in caches
			await proxy.setValues({
				apiModelId: "gpt-4", // global state
				openAiApiKey: "test-api-key", // secret
			})

			// Verify initial state
			expect(proxy.getGlobalState("apiModelId")).toBe("gpt-4")
			expect(proxy.getSecret("openAiApiKey")).toBe("test-api-key")

			// Reset all state
			await proxy.resetAllState()

			// Caches should be reinitialized with values from the context
			// Since our mock globalState.get returns undefined by default,
			// the cache should now contain undefined values
			expect(proxy.getGlobalState("apiModelId")).toBeUndefined()
		})

		test("should update all global state keys to undefined", async () => {
			// Setup initial state
			await proxy.updateGlobalState("apiModelId", "gpt-4")
			await proxy.updateGlobalState("apiProvider", "openai")

			// Reset all state
			await proxy.resetAllState()

			// Should have called update with undefined for each key
			for (const key of GLOBAL_STATE_KEYS) {
				assert.ok(mockGlobalState.update.calledWith(key, undefined))
			}

			// Total calls should include initial setup + reset operations
			const expectedUpdateCalls = 2 + GLOBAL_STATE_KEYS.length
			expect(mockGlobalState.update).callCount, expectedUpdateCalls
		})

		test("should delete all secrets", async () => {
			// Setup initial secrets
			await proxy.storeSecret("apiKey", "test-api-key")
			await proxy.storeSecret("openAiApiKey", "test-openai-key")

			// Reset all state
			await proxy.resetAllState()

			// Should have called delete for each key
			for (const key of SECRET_STATE_KEYS) {
				assert.ok(mockSecrets.delete.calledWith(key))
			}

			// Total calls should equal the number of secret keys
			expect(mockSecrets.delete).callCount, SECRET_STATE_KEYS.length
		})

		test("should reinitialize caches after reset", async () => {
			// Spy on initialization methods
			const initializeSpy = sinon.spy(proxy as any, "initialize")

			// Reset all state
			await proxy.resetAllState()

			// Should reinitialize caches
			assert.strictEqual(initializeSpy.callCount, 1)
		})
	})
// Mock cleanup
