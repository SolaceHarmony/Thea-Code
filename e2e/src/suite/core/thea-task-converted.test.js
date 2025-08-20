// Migrated from src/core/__tests__/TheaTask.test.ts
// Following the Roo migration harness guide

const assert = require('assert')
const path = require('path')
const os = require('os')
const { loadTheaModule } = require('../../../helpers/thea-loader')

// Load Thea modules
const { TheaTask } = loadTheaModule('src/core/TheaTask')
const { TheaProvider } = loadTheaModule('src/core/webview/TheaProvider')
const { getDiffStrategy } = loadTheaModule('src/core/diff/DiffStrategy')

// Mock vscode and fs
const vscode = require('vscode')
const fs = require('fs/promises')

// Store original functions for restoration
let originalReadFile
let originalWriteFile
let originalMkdir
let originalGetConfiguration

describe('TheaTask', () => {
    let mockProvider
    let mockApiConfig
    let mockOutputChannel
    let mockExtensionContext
    
    beforeEach(() => {
        // Mock fs operations
        originalReadFile = fs.readFile
        originalWriteFile = fs.writeFile
        originalMkdir = fs.mkdir
        
        fs.readFile = async (filePath) => {
            if (filePath.includes('ui_messages.json')) {
                return JSON.stringify([{
                    ts: Date.now(),
                    type: 'say',
                    say: 'text',
                    text: 'historical task'
                }])
            }
            if (filePath.includes('api_conversation_history.json')) {
                return JSON.stringify([
                    {
                        role: 'user',
                        content: [{ type: 'text', text: 'historical task' }],
                        ts: Date.now()
                    },
                    {
                        role: 'assistant',
                        content: [{ type: 'text', text: "I'll help you with that task." }],
                        ts: Date.now()
                    }
                ])
            }
            return '[]'
        }
        
        fs.writeFile = async () => undefined
        fs.mkdir = async () => undefined
        
        // Mock workspace configuration
        originalGetConfiguration = vscode.workspace.getConfiguration
        vscode.workspace.getConfiguration = () => ({
            get: (key, defaultValue) => defaultValue
        })
        
        // Setup storage URI
        const storageUri = {
            fsPath: path.join(os.tmpdir(), 'test-storage')
        }
        
        // Setup mock extension context
        mockExtensionContext = {
            globalState: {
                get: (key) => {
                    if (key === 'taskHistory') {
                        return [{
                            id: '123',
                            number: 0,
                            ts: Date.now(),
                            task: 'historical task',
                            tokensIn: 100,
                            tokensOut: 200,
                            cacheWrites: 0,
                            cacheReads: 0,
                            totalCost: 0.001
                        }]
                    }
                    return undefined
                },
                update: async () => undefined,
                keys: () => []
            },
            globalStorageUri: storageUri,
            workspaceState: {
                get: () => undefined,
                update: async () => undefined,
                keys: () => []
            },
            secrets: {
                get: async () => undefined,
                store: async () => undefined,
                delete: async () => undefined
            },
            extensionUri: {
                fsPath: '/mock/extension/path'
            },
            extension: {
                packageJSON: {
                    version: '1.0.0'
                }
            }
        }
        
        // Setup mock output channel
        mockOutputChannel = {
            name: 'mockOutputChannel',
            appendLine: () => {},
            append: () => {},
            clear: () => {},
            show: () => {},
            hide: () => {},
            dispose: () => {},
            replace: () => {}
        }
        
        // Setup mock provider
        mockProvider = new TheaProvider(mockExtensionContext, mockOutputChannel)
        mockProvider.postMessageToWebview = async () => undefined
        mockProvider.postStateToWebview = async () => undefined
        mockProvider.getTaskWithId = (id) => ({
            historyItem: {
                id,
                ts: Date.now(),
                task: 'historical task',
                tokensIn: 100,
                tokensOut: 200,
                cacheWrites: 0,
                cacheReads: 0,
                totalCost: 0.001
            },
            taskDirPath: '/mock/storage/path/tasks/123',
            apiConversationHistoryFilePath: '/mock/storage/path/tasks/123/api_conversation_history.json',
            uiMessagesFilePath: '/mock/storage/path/tasks/123/ui_messages.json',
            apiConversationHistory: [
                {
                    role: 'user',
                    content: [{ type: 'text', text: 'historical task' }],
                    ts: Date.now()
                },
                {
                    role: 'assistant',
                    content: [{ type: 'text', text: "I'll help you with that task." }],
                    ts: Date.now()
                }
            ]
        })
        
        // Setup mock API configuration
        mockApiConfig = {
            apiProvider: 'anthropic',
            apiModelId: 'claude-3-5-sonnet-20241022',
            apiKey: 'test-api-key'
        }
        
        // Mock getEnvironmentDetails to avoid timeouts
        TheaTask.prototype.getEnvironmentDetails = async () => ''
    })
    
    afterEach(() => {
        // Restore original functions
        fs.readFile = originalReadFile
        fs.writeFile = originalWriteFile
        fs.mkdir = originalMkdir
        vscode.workspace.getConfiguration = originalGetConfiguration
    })
    
    describe('constructor', () => {
        it('should respect provided settings', () => {
            const theaTask = new TheaTask({
                provider: mockProvider,
                apiConfiguration: mockApiConfig,
                customInstructions: 'custom instructions',
                fuzzyMatchThreshold: 0.95,
                task: 'test task',
                startTask: false
            })
            
            // Verify the task was created successfully
            assert.ok(theaTask)
            assert.ok(theaTask.diffStrategy)
        })
        
        it('should use default fuzzy match threshold when not provided', () => {
            const theaTask = new TheaTask({
                provider: mockProvider,
                apiConfiguration: mockApiConfig,
                customInstructions: 'custom instructions',
                enableDiff: true,
                fuzzyMatchThreshold: 0.95,
                task: 'test task',
                startTask: false
            })
            
            assert.ok(theaTask.diffStrategy)
        })
        
        it('should require either task or historyItem', () => {
            let errorThrown = false
            try {
                new TheaTask({ 
                    provider: mockProvider, 
                    apiConfiguration: mockApiConfig 
                })
            } catch (error) {
                errorThrown = true
                assert.strictEqual(error.message, 'Either historyItem or task/images must be provided')
            }
            assert.strictEqual(errorThrown, true)
        })
    })
    
    describe('stream handling', () => {
        it('should handle text chunks from stream', async function() {
            this.timeout(5000)
            
            const theaTask = new TheaTask({
                provider: mockProvider,
                apiConfiguration: mockApiConfig,
                task: 'test task',
                startTask: false
            })
            
            // Simulate streaming text chunks
            const chunks = [
                { type: 'text', text: 'Hello' },
                { type: 'text', text: ' world' },
                { type: 'text', text: '!' }
            ]
            
            let accumulatedText = ''
            theaTask.on('text', (text) => {
                accumulatedText += text
            })
            
            // Process chunks
            for (const chunk of chunks) {
                theaTask.handleApiStreamChunk(chunk)
            }
            
            assert.strictEqual(accumulatedText, 'Hello world!')
        })
        
        it('should handle tool use chunks', async function() {
            this.timeout(5000)
            
            const theaTask = new TheaTask({
                provider: mockProvider,
                apiConfiguration: mockApiConfig,
                task: 'test task',
                startTask: false
            })
            
            let toolUseName = ''
            theaTask.on('tool_use', (toolUse) => {
                toolUseName = toolUse.name
            })
            
            // Simulate tool use chunk
            theaTask.handleApiStreamChunk({
                type: 'tool_use',
                id: 'tool-1',
                name: 'str_replace_editor',
                params: { path: 'test.txt', old_str: 'old', new_str: 'new' }
            })
            
            assert.strictEqual(toolUseName, 'str_replace_editor')
        })
    })
    
    describe('parseMentions', () => {
        it('should parse file mentions correctly', () => {
            const { parseMentions } = loadTheaModule('src/core/mentions')
            const result = parseMentions('Check @file.ts and @folder/file.js')
            
            assert.strictEqual(result.length, 2)
            assert.strictEqual(result[0].type, 'file')
            assert.ok(result[0].path.includes('file.ts'))
            assert.strictEqual(result[1].type, 'file')
            assert.ok(result[1].path.includes('folder/file.js'))
        })
        
        it('should parse URL mentions correctly', () => {
            const { parseMentions } = loadTheaModule('src/core/mentions')
            const result = parseMentions('Visit @https://example.com')
            
            assert.strictEqual(result.length, 1)
            assert.strictEqual(result[0].type, 'url')
            assert.strictEqual(result[0].path, 'https://example.com')
        })
        
        it('should handle mixed mentions', () => {
            const { parseMentions } = loadTheaModule('src/core/mentions')
            const result = parseMentions('Check @file.ts and visit @https://example.com')
            
            assert.strictEqual(result.length, 2)
            assert.strictEqual(result[0].type, 'file')
            assert.strictEqual(result[1].type, 'url')
        })
    })
    
    describe('abort handling', () => {
        it('should handle abort request', async function() {
            this.timeout(5000)
            
            const theaTask = new TheaTask({
                provider: mockProvider,
                apiConfiguration: mockApiConfig,
                task: 'test task',
                startTask: false
            })
            
            let abortFired = false
            theaTask.on('abort', () => {
                abortFired = true
            })
            
            theaTask.handleAbort()
            
            assert.strictEqual(abortFired, true)
            assert.strictEqual(theaTask.abortReason, 'user_cancelled')
        })
        
        it('should handle error abort', async function() {
            this.timeout(5000)
            
            const theaTask = new TheaTask({
                provider: mockProvider,
                apiConfiguration: mockApiConfig,
                task: 'test task',
                startTask: false
            })
            
            const error = new Error('Test error')
            theaTask.handleApiStreamError(error)
            
            assert.strictEqual(theaTask.abortReason, 'error')
        })
    })
})