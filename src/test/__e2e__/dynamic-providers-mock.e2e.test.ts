import 'ts-node/register'
import * as assert from 'assert'

/* eslint-disable @typescript-eslint/no-unsafe-assignment, @typescript-eslint/no-unsafe-call, @typescript-eslint/no-unsafe-member-access, @typescript-eslint/no-explicit-any */

import startProviderMocks from '../../../test/generic-provider-mock/setup'
import stopProviderMocks from '../../../test/generic-provider-mock/teardown'

/**
 * Test suite for all dynamic model provider mock servers
 * Verifies that the enhanced mock server can simulate the supported provider APIs
 */

suite('Dynamic Provider Mock Servers', () => {
	let providerPorts: Record<string, number>
	let startedMocks = false

	suiteSetup(async function () {
		this.timeout(20_000)
		if (!(globalThis as any).__PROVIDER_PORTS__) {
			startedMocks = true
			await startProviderMocks()
		}
		providerPorts = (globalThis as any).__PROVIDER_PORTS__ || {}
		assert.ok(Object.keys(providerPorts).length > 0, 'Provider ports should be populated')
		console.log('📋 Available provider ports:', providerPorts)
	})

	suiteTeardown(async () => {
		if (startedMocks) {
			await stopProviderMocks()
		}
	})

	suite('Mock Server Availability', () => {
		test('should have all 8 provider mock servers running', () => {
			assert.ok(providerPorts, 'Provider ports should be defined')

			const expectedProviders = [
				'openai',
				'anthropic',
				'bedrock',
				'gemini',
				'vertex',
				'mistral',
				'deepseek',
				'ollama',
			]

			expectedProviders.forEach(provider => {
				const port = providerPorts[provider]
				assert.ok(port !== undefined, `${provider} mock server should be registered`)
				assert.strictEqual(typeof port, 'number', `${provider} port should be numeric`)
				assert.ok(port > 10000, `${provider} port should look like a test server port`)
			})

			console.log(`✅ All ${expectedProviders.length} provider mock servers are running`)
		})
	})

	suite('OpenAI-Compatible Providers', () => {
		const openaiCompatible = ['openai', 'bedrock', 'gemini', 'mistral', 'deepseek', 'ollama']

		openaiCompatible.forEach(provider => {
			test(`${provider} should respond to /v1/models endpoint`, async () => {
				const port = providerPorts[provider]
				assert.ok(port !== undefined, `${provider} port should be defined`)

				const response = await fetch(`http://127.0.0.1:${port}/v1/models`)
				assert.strictEqual(response.ok, true, `${provider} /v1/models should return 200`)

				const data = await response.json()
				assert.ok(data.hasOwnProperty('data'), `${provider} response should include data array`)
				assert.ok(Array.isArray(data.data), `${provider} data should be an array`)
				assert.ok(data.data.length > 0, `${provider} should return at least one model`)

				console.log(`✅ ${provider}: Found ${data.data.length} models`)
				console.log(`   Sample: ${data.data[0].id}`)
			})

			test(`${provider} should handle chat completions`, async () => {
				const port = providerPorts[provider]
				assert.ok(port !== undefined, `${provider} port should be defined`)

				const response = await fetch(`http://127.0.0.1:${port}/v1/chat/completions`, {
					method: 'POST',
					headers: { 'Content-Type': 'application/json' },
					body: JSON.stringify({
						model: 'test-model',
						messages: [{ role: 'user', content: 'Hello' }],
					}),
				})

				assert.strictEqual(response.ok, true, `${provider} chat completions should return 200`)
				const data = await response.json()
				assert.ok(data.hasOwnProperty('choices'), `${provider} response should include choices`)
				assert.ok(data.choices[0].hasOwnProperty('message'), `${provider} choices should include message`)

				console.log(`✅ ${provider}: Chat completion response received`)
			})
		})
	})

	suite('Anthropic-Compatible Providers', () => {
		const anthropicCompatible = ['anthropic', 'vertex']

		anthropicCompatible.forEach(provider => {
			test(`${provider} should handle /v1/messages endpoint`, async () => {
				const port = providerPorts[provider]
				assert.ok(port !== undefined, `${provider} port should be defined`)

				const response = await fetch(`http://127.0.0.1:${port}/v1/messages`, {
					method: 'POST',
					headers: {
						'Content-Type': 'application/json',
						'x-api-key': 'test-key',
						'anthropic-version': '2023-06-01',
					},
					body: JSON.stringify({
						model: 'claude-3-sonnet-20240229',
						max_tokens: 100,
						messages: [{ role: 'user', content: 'Hello' }],
					}),
				})

				assert.strictEqual(response.ok, true, `${provider} messages should return 200`)
				const data = await response.json()
				assert.ok(data.hasOwnProperty('content'), `${provider} response should include content`)
				assert.strictEqual(data.role, 'assistant', `${provider} response role should be assistant`)

				console.log(`✅ ${provider}: Anthropic messages response received`)
			})
		})
	})

	suite('AWS Bedrock Provider', () => {
		test('should handle ListFoundationModels operation', async () => {
			const port = providerPorts.bedrock
			assert.ok(port !== undefined, 'Bedrock port should be defined')

			const response = await fetch(`http://127.0.0.1:${port}/`, {
				method: 'POST',
				headers: {
					'Content-Type': 'application/x-amz-json-1.1',
					'X-Amz-Target': 'AmazonBedrockControlPlaneService.ListFoundationModels',
				},
				body: JSON.stringify({}),
			})

			assert.strictEqual(response.ok, true, 'Bedrock ListFoundationModels should return 200')
			const data = await response.json()
			assert.ok(data.hasOwnProperty('modelSummaries'), 'Bedrock response should include modelSummaries')
			assert.ok(Array.isArray(data.modelSummaries), 'modelSummaries should be an array')

			console.log(`✅ Bedrock: Found ${data.modelSummaries.length} foundation models`)
		})
	})

	suite('Google Vertex AI Provider', () => {
		test('should handle publisher models endpoint', async () => {
			const port = providerPorts.vertex
			assert.ok(port !== undefined, 'Vertex port should be defined')

			const response = await fetch(`http://127.0.0.1:${port}/v1/projects/test-project/locations/us-central1/publishers/anthropic/models`)
			assert.strictEqual(response.ok, true, 'Vertex publisher models should return 200')

			const data = await response.json()
			assert.ok(data.hasOwnProperty('models'), 'Vertex response should include models')
			assert.ok(Array.isArray(data.models), 'Vertex models should be an array')

			console.log(`✅ Vertex: Found ${data.models.length} publisher models`)
		})

		test('should handle foundation models endpoint', async () => {
			const port = providerPorts.vertex
			assert.ok(port !== undefined, 'Vertex port should be defined')

			const response = await fetch(`http://127.0.0.1:${port}/v1/projects/test-project/locations/us-central1/models`)
			assert.strictEqual(response.ok, true, 'Vertex foundation models should return 200')

			const data = await response.json()
			assert.ok(data.hasOwnProperty('models'), 'Vertex response should include models')
			assert.ok(Array.isArray(data.models), 'Vertex models should be an array')

			console.log(`✅ Vertex: Found ${data.models.length} foundation models`)
		})
	})

	suite('Ollama Provider', () => {
		test('should handle /api/tags endpoint', async () => {
			const port = providerPorts.ollama
			assert.ok(port !== undefined, 'Ollama port should be defined')

			const response = await fetch(`http://127.0.0.1:${port}/api/tags`)
			assert.strictEqual(response.ok, true, 'Ollama /api/tags should return 200')

			const data = await response.json()
			assert.ok(data.hasOwnProperty('models'), 'Ollama response should include models')
			assert.ok(Array.isArray(data.models), 'Ollama models should be an array')

			console.log(`✅ Ollama: Found ${data.models.length} local models`)
		})

		test('should handle /api/chat endpoint', async () => {
			const port = providerPorts.ollama
			assert.ok(port !== undefined, 'Ollama port should be defined')

			const response = await fetch(`http://127.0.0.1:${port}/api/chat`, {
				method: 'POST',
				headers: { 'Content-Type': 'application/json' },
				body: JSON.stringify({
					model: 'llama2',
					messages: [{ role: 'user', content: 'Hello' }],
					stream: false,
				}),
			})

			assert.strictEqual(response.ok, true, 'Ollama /api/chat should return 200')
			const data = await response.json()
			assert.ok(data.hasOwnProperty('message'), 'Ollama /api/chat response should include message')
			assert.strictEqual(data.message.role, 'assistant', 'Ollama message role should be assistant')

			console.log('✅ Ollama: Chat response received')
		})
	})

	suite('Streaming Support', () => {
		test('OpenAI streaming should work', async () => {
			const port = providerPorts.openai
			assert.ok(port !== undefined, 'OpenAI port should be defined')

			const response = await fetch(`http://127.0.0.1:${port}/v1/chat/completions`, {
				method: 'POST',
				headers: { 'Content-Type': 'application/json' },
				body: JSON.stringify({
					model: 'gpt-4',
					messages: [{ role: 'user', content: 'Hello' }],
					stream: true,
				}),
			})

			assert.strictEqual(response.ok, true, 'OpenAI streaming request should return 200')
			const contentType = response.headers.get('content-type')
			assert.ok(contentType && contentType.includes('text/event-stream'), 'OpenAI streaming response should be event stream')

			const text = await response.text()
			assert.ok(text.includes('data:'), 'OpenAI stream should include data chunks')
			assert.ok(text.includes('[DONE]'), 'OpenAI stream should include completion marker')

			console.log('✅ OpenAI streaming works')
		})

		test('Anthropic streaming should work', async () => {
			const port = providerPorts.anthropic
			assert.ok(port !== undefined, 'Anthropic port should be defined')

			const response = await fetch(`http://127.0.0.1:${port}/v1/messages`, {
				method: 'POST',
				headers: {
					'Content-Type': 'application/json',
					'x-api-key': 'test-key',
					'anthropic-version': '2023-06-01',
				},
				body: JSON.stringify({
					model: 'claude-3-sonnet-20240229',
					max_tokens: 100,
					messages: [{ role: 'user', content: 'Hello' }],
					stream: true,
				}),
			})

			assert.strictEqual(response.ok, true, 'Anthropic streaming request should return 200')
			const contentType = response.headers.get('content-type')
			assert.ok(contentType && contentType.includes('text/event-stream'), 'Anthropic streaming response should be event stream')

			const text = await response.text()
			assert.ok(text.includes('data:'), 'Anthropic stream should include data chunks')
			assert.ok(text.includes('[DONE]'), 'Anthropic stream should include completion marker')

			console.log('✅ Anthropic streaming works')
		})
	})

	suite('Performance Test', () => {
		test('should handle concurrent requests to all providers', async () => {
			const startTime = Date.now()
			const providers = Object.keys(providerPorts)

			const results = await Promise.all(
				providers.map(async provider => {
					try {
						const port = providerPorts[provider]
						assert.ok(port !== undefined, `${provider} port should be defined for performance test`)

						const response = await fetch(`http://127.0.0.1:${port}/v1/models`)
						if (!response.ok) {
							return {
								provider,
								success: false,
								status: response.status,
								error: `HTTP ${response.status}`,
							}
						}

						const json = await response.json()
						const modelCount = Array.isArray(json.data) ? json.data.length : 0

						return {
							provider,
							success: true,
							status: response.status,
							modelCount,
						}
					} catch (error) {
						const message = error instanceof Error ? error.message : String(error)
						return {
							provider,
							success: false,
							error: message,
						}
					}
				})
			)

			const endTime = Date.now()
			const totalTime = endTime - startTime

			console.log(`\n🚀 Concurrent Test Results (${totalTime}ms):`)
			results.forEach(result => {
				if (result.success) {
					console.log(`   ✅ ${result.provider}: ${result.modelCount} models`)
				} else {
					console.log(`   ❌ ${result.provider}: ${result.error || `HTTP ${result.status}`}`)
				}
			})

			const successCount = results.filter(result => result.success).length
			assert.strictEqual(successCount, providers.length, 'All providers should respond successfully')
			assert.ok(totalTime < 5000, 'Concurrent request test should finish within 5 seconds')
		})
	})
})
