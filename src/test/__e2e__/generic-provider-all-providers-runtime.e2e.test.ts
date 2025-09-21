import 'ts-node/register'
import * as assert from 'assert'

/* eslint-disable @typescript-eslint/no-unsafe-assignment, @typescript-eslint/no-unsafe-call, @typescript-eslint/no-unsafe-member-access, @typescript-eslint/no-unsafe-argument, @typescript-eslint/no-explicit-any */

import startProviderMocks, { providerMocks as sharedProviderMocks } from '../../../test/generic-provider-mock/setup'
import stopProviderMocks from '../../../test/generic-provider-mock/teardown'

suite('Generic Provider All Providers Runtime', () => {
	let providerPorts: Record<string, number>
	let startedMocks = false

	suiteSetup(async function () {
		this.timeout(20_000)
		if (!(globalThis as any).__PROVIDER_PORTS__) {
			startedMocks = true
			await startProviderMocks()
		}
		providerPorts = (globalThis as any).__PROVIDER_PORTS__ || {}
		assert.ok(Object.keys(providerPorts).length > 0, 'Provider mocks should expose ports during runtime setup')
	})

	suiteTeardown(async () => {
		if (startedMocks) {
			await stopProviderMocks()
		}
	})

	suite('Runtime registration', () => {
		test('should expose provider mocks on the global scope', () => {
			const globalMocks = (globalThis as any).__PROVIDER_MOCKS__
			assert.ok(globalMocks, '__PROVIDER_MOCKS__ should be defined on the global scope')

			const expectedProviders = Object.keys(sharedProviderMocks).sort()
			const activeProviders = Object.keys(globalMocks ?? {}).sort()
			assert.deepStrictEqual(activeProviders, expectedProviders)

			expectedProviders.forEach(provider => {
				const mock = sharedProviderMocks[provider]
				assert.ok(mock, `${provider} mock should be available`)
				assert.strictEqual(mock.getPort(), providerPorts[provider], `${provider} mock should expose the same port via getPort()`)
			})
		})

		test('should set environment variables for accessible providers', () => {
			const envMap: Record<string, string> = {
				openai: 'OPENAI_BASE_URL',
				anthropic: 'ANTHROPIC_BASE_URL',
				bedrock: 'AWS_BEDROCK_ENDPOINT',
				gemini: 'GEMINI_BASE_URL',
				vertex: 'GOOGLE_VERTEX_ENDPOINT',
				mistral: 'MISTRAL_BASE_URL',
				deepseek: 'DEEPSEEK_BASE_URL',
				ollama: 'OLLAMA_BASE_URL',
			}

			for (const [provider, envKey] of Object.entries(envMap)) {
				const port = providerPorts[provider]
				assert.ok(port, `${provider} port should exist for env variable validation`)
				const expected = `http://127.0.0.1:${port}`
				assert.strictEqual(process.env[envKey], expected, `${envKey} should match provider port`)
			}
		})
	})

	suite('Lifecycle management', () => {
		test('should stop and restart providers cleanly', async function () {
			this.timeout(40_000)

			await stopProviderMocks()
			Object.entries(sharedProviderMocks).forEach(([provider, mock]) => {
				assert.strictEqual(mock.getPort(), null, `${provider} port should be cleared after stop`)
			})
			assert.strictEqual((globalThis as any).__PROVIDER_PORTS__, undefined, '__PROVIDER_PORTS__ should be cleared after stop')

			await startProviderMocks()
			providerPorts = (globalThis as any).__PROVIDER_PORTS__ || {}

			Object.entries(sharedProviderMocks).forEach(([provider, mock]) => {
				const port = mock.getPort()
				assert.ok(port && port > 0, `${provider} port should be allocated after restart`)
				assert.strictEqual(providerPorts[provider], port, `${provider} port should match global registry after restart`)
			})

			startedMocks = true
		})
	})
})
