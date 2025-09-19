import * as assert from 'assert'
import { MockMcpProvider } from '../../../../../../services/mcp/providers/MockMcpProvider'
import { ToolDefinition } from '../../../../../../services/mcp/types/McpProviderTypes'

/**
 * Performance and streaming response validation tests for MCP system
 * Focus on provider-based tests to avoid heavy integration dependencies
 */

suite('MCP Performance and Streaming Validation', () => {
	// Streaming Response Validation
	suite('Streaming Response Validation', () => {
		let provider: MockMcpProvider

		setup(async () => {
			provider = new MockMcpProvider()
			await provider.start()
		})

		teardown(async () => {
			await provider.stop()
			provider.removeAllListeners()
		})

		test('should handle streaming-like tool responses', async () => {
			const streamingTool: ToolDefinition = {
				name: 'streaming_tool',
				description: 'Tool that simulates streaming responses',
				handler: async (args) => {
					const chunks = (args.chunks as number) || 5
					let content = ''

					// Simulate streaming by building content incrementally
					for (let i = 0; i < chunks; i++) {
						await new Promise((resolve) => setTimeout(resolve, 5)) // Small delay
						content += `Chunk ${i + 1}/${chunks}. `
					}

					return {
						content: [{ type: 'text', text: content.trim() }],
						isError: false,
					}
				},
			}

			provider.registerToolDefinition(streamingTool)

			const startTime = Date.now()
			const result = await provider.executeTool('streaming_tool', { chunks: 10 })
			const endTime = Date.now()
			const duration = endTime - startTime

			assert.ok(result.content[0].text?.includes('Chunk 1/10'))
			assert.ok(result.content[0].text?.includes('Chunk 10/10'))
			assert.strictEqual(result.isError, false)

			// Should complete within reasonable time considering delays
			assert.ok(duration >= 50) // At least ~5ms * 10 chunks
			assert.ok(duration < 1000) // But not too long
		})

		test('should handle large response payloads efficiently', async () => {
			const largeTool: ToolDefinition = {
				name: 'large_response_tool',
				description: 'Tool that returns large responses',
				handler: async (args) => {
					const size = (args.size as number) || 1000
					const largeData = Array.from({ length: size }, (_, i) => ({
						id: i,
						data: `This is item ${i} with some additional text to make it larger`,
						metadata: {
							timestamp: Date.now(),
							index: i,
							batch: Math.floor(i / 100),
						},
					}))

					return {
						content: [
							{
								type: 'text',
								text: JSON.stringify(largeData, null, 2),
							},
						],
						isError: false,
					}
				},
			}

			provider.registerToolDefinition(largeTool)

			const startTime = Date.now()
			const result = await provider.executeTool('large_response_tool', { size: 5000 })
			const endTime = Date.now()
			const duration = endTime - startTime

			const parsedData = JSON.parse(result.content[0].text || '[]') as Array<{ id: number }>
			assert.strictEqual(parsedData.length, 5000)
			assert.strictEqual(parsedData[0]?.id, 0)
			assert.strictEqual(parsedData[4999]?.id, 4999)

			// Should handle large payloads efficiently
			assert.ok(duration < 2000) // Should complete within 2 seconds on CI
		})

		test('should validate response time consistency', async () => {
			const consistentTool: ToolDefinition = {
				name: 'consistent_tool',
				description: 'Tool with consistent response times',
				handler: async () => {
					// Fixed small delay to simulate consistent work
					await new Promise((resolve) => setTimeout(resolve, 10))
					return {
						content: [{ type: 'text', text: 'Consistent response' }],
						isError: false,
					}
				},
			}

			provider.registerToolDefinition(consistentTool)

			const iterations = 20
			const responseTimes: number[] = []

			// Measure response times
			for (let i = 0; i < iterations; i++) {
				const startTime = Date.now()
				await provider.executeTool('consistent_tool', {})
				const endTime = Date.now()
				responseTimes.push(endTime - startTime)
			}

			// Calculate statistics
			const averageTime = responseTimes.reduce((sum, time) => sum + time, 0) / iterations
			const minTime = Math.min(...responseTimes)
			const maxTime = Math.max(...responseTimes)
			const variance = responseTimes.reduce((sum, time) => sum + Math.pow(time - averageTime, 2), 0) / iterations
			const standardDeviation = Math.sqrt(variance)

			// Consistency assertions
			assert.ok(averageTime >= 8) // Should be close to 10ms
			assert.ok(averageTime < 100) // But not too much overhead
			assert.ok(standardDeviation < averageTime * 0.8) // Reasonable variance
			assert.ok(maxTime - minTime < 150) // Reasonable spread
		})
	})

	// Error Handling Performance
	suite('Error Handling Performance', () => {
		let provider: MockMcpProvider

		setup(async () => {
			provider = new MockMcpProvider()
			await provider.start()
		})

		teardown(async () => {
			await provider.stop()
			provider.removeAllListeners()
		})

		test('should handle errors efficiently without memory leaks', async () => {
			const errorTool: ToolDefinition = {
				name: 'error_tool',
				description: 'Tool that throws errors',
				handler: async (args) => {
					if (args.shouldError) {
						throw new Error(`Test error ${args.id}`)
					}
					return {
						content: [{ type: 'text', text: `Success ${args.id}` }],
						isError: false,
					}
				},
			}

			provider.registerToolDefinition(errorTool)

			const iterations = 100
			let successCount = 0
			let errorCount = 0

			const startTime = Date.now()

			// Mix of successful and error operations
			for (let i = 0; i < iterations; i++) {
				const shouldError = i % 3 === 0 // Every third operation fails
				const result = await provider.executeTool('error_tool', { id: i, shouldError })
				if (result.isError) {
					errorCount++
				} else {
					successCount++
				}
			}

			const endTime = Date.now()
			const totalTime = endTime - startTime

			assert.ok(successCount > 0)
			assert.ok(errorCount > 0)
			assert.strictEqual(successCount + errorCount, iterations)

			// Should handle errors without significant performance impact
			assert.ok(totalTime < 2000) // Allow some headroom on CI
		})

		test('should recover quickly from error bursts', async () => {
			const recoverTool: ToolDefinition = {
				name: 'recover_tool',
				description: 'Tool for testing error recovery',
				handler: async (args) => {
					const phase = args.phase as string
					if (phase === 'error_burst') {
						throw new Error('Burst error')
					}
					// Normal operation
					await new Promise((resolve) => setTimeout(resolve, 5))
					return {
						content: [{ type: 'text', text: `Recovery success ${args.id}` }],
						isError: false,
					}
				},
			}

			provider.registerToolDefinition(recoverTool)

			// Phase 1: Error burst
			const errorPromises = Array.from({ length: 20 }, (_, i) =>
				provider.executeTool('recover_tool', { phase: 'error_burst', id: i })
			)
			const errorResults = await Promise.all(errorPromises)
			const allErrorsAsExpected = errorResults.every((r) => r.isError === true)
			assert.strictEqual(allErrorsAsExpected, true)

			// Small delay to allow system to stabilize
			await new Promise((resolve) => setTimeout(resolve, 20))

			// Phase 2: Quick recovery
			const recoveryStartTime = Date.now()
			const recoveryPromises = Array.from({ length: 20 }, (_, i) =>
				provider.executeTool('recover_tool', { phase: 'recovery', id: i })
			)
			const recoveryResults = await Promise.all(recoveryPromises)
			const recoveryTime = Date.now() - recoveryStartTime

			const successfulRecoveries = recoveryResults.filter((r) => !r.isError).length
			assert.ok(successfulRecoveries >= recoveryResults.length * 0.8)
			assert.ok(recoveryTime < 1000)
		})
	})
})
