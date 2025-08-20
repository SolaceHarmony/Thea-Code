// Migrated from src/core/webview/__tests__/TheaTaskHistory.test.ts
// Following the Roo migration harness guide

const assert = require('assert')
const path = require('path')
const fs = require('fs/promises')
const { loadTheaModule } = require('../../../helpers/thea-loader')

// Load Thea modules
const { TheaTaskHistory } = loadTheaModule('src/core/webview/history/TheaTaskHistory')

// Mock vscode
const vscode = require('vscode')

describe('TheaTaskHistory', () => {
    let taskHistory
    let mockContext
    let mockContextProxy
    
    // Store original fs functions
    let originalRm
    let originalReadFile
    
    beforeEach(() => {
        // Store original fs functions
        originalRm = fs.rm
        originalReadFile = fs.readFile
        
        // Mock fs methods
        fs.rm = async () => undefined
        fs.readFile = async () => '[]'
        
        // Mock context
        mockContext = {
            extensionPath: '/test/path',
            extensionUri: { fsPath: '/test/extension' },
            globalStorageUri: {
                fsPath: '/test/storage/path'
            }
        }
        
        // Mock contextProxy
        mockContextProxy = {
            getValue: async (key) => {
                if (key === 'taskHistory') return []
                return undefined
            },
            setValue: async () => undefined
        }
        
        // Create instance
        taskHistory = new TheaTaskHistory(mockContext, mockContextProxy)
    })
    
    afterEach(() => {
        // Restore original functions
        fs.rm = originalRm
        fs.readFile = originalReadFile
    })
    
    describe('updateTaskHistory', () => {
        it('adds a new history item when it does not exist', async function() {
            this.timeout(5000)
            
            const mockHistory = []
            let savedHistory = null
            
            mockContextProxy.getValue = async () => mockHistory
            mockContextProxy.setValue = async (key, value) => {
                assert.strictEqual(key, 'taskHistory')
                savedHistory = value
            }
            
            const newHistoryItem = {
                id: 'test-task-id',
                task: 'Test Task',
                ts: 123456789,
                number: 1,
                tokensIn: 100,
                tokensOut: 200,
                totalCost: 0.01
            }
            
            const result = await taskHistory.updateTaskHistory(newHistoryItem)
            
            assert.ok(savedHistory)
            assert.strictEqual(savedHistory.length, 1)
            assert.strictEqual(savedHistory[0].id, 'test-task-id')
            assert.strictEqual(savedHistory[0].task, 'Test Task')
            assert.deepStrictEqual(result, [newHistoryItem])
        })
        
        it('updates an existing history item', async function() {
            this.timeout(5000)
            
            const existingItem = {
                id: 'test-task-id',
                task: 'Original Task',
                ts: 123456789,
                number: 1,
                tokensIn: 100,
                tokensOut: 200,
                totalCost: 0.01
            }
            
            const mockHistory = [existingItem]
            let savedHistory = null
            
            mockContextProxy.getValue = async () => [...mockHistory]
            mockContextProxy.setValue = async (key, value) => {
                assert.strictEqual(key, 'taskHistory')
                savedHistory = value
            }
            
            const updatedItem = {
                id: 'test-task-id',
                task: 'Updated Task',
                ts: 123456789,
                number: 1,
                tokensIn: 150,
                tokensOut: 250,
                totalCost: 0.02
            }
            
            const result = await taskHistory.updateTaskHistory(updatedItem)
            
            assert.ok(savedHistory)
            assert.strictEqual(savedHistory.length, 1)
            assert.strictEqual(savedHistory[0].id, 'test-task-id')
            assert.strictEqual(savedHistory[0].task, 'Updated Task')
            assert.strictEqual(savedHistory[0].tokensIn, 150)
            assert.deepStrictEqual(result, [updatedItem])
        })
        
        it('handles empty history list', async function() {
            this.timeout(5000)
            
            let savedHistory = null
            
            mockContextProxy.getValue = async () => null
            mockContextProxy.setValue = async (key, value) => {
                assert.strictEqual(key, 'taskHistory')
                savedHistory = value
            }
            
            const newHistoryItem = {
                id: 'test-task-id',
                task: 'Test Task',
                ts: 123456789,
                number: 1,
                tokensIn: 100,
                tokensOut: 200,
                totalCost: 0.01
            }
            
            const result = await taskHistory.updateTaskHistory(newHistoryItem)
            
            assert.ok(savedHistory)
            assert.strictEqual(savedHistory.length, 1)
            assert.deepStrictEqual(result, [newHistoryItem])
        })
    })
    
    describe('getTaskWithId', () => {
        it('retrieves task with valid ID', async function() {
            this.timeout(5000)
            
            const mockHistoryItem = {
                id: 'test-id',
                task: 'Test Task',
                ts: Date.now()
            }
            
            mockContextProxy.getValue = async (key) => {
                if (key === 'taskHistory') {
                    return [mockHistoryItem]
                }
                return undefined
            }
            
            // Mock file existence check
            fs.readFile = async (filePath) => {
                if (filePath.includes('api_conversation_history.json')) {
                    return JSON.stringify([
                        { role: 'user', content: [{ type: 'text', text: 'test' }] }
                    ])
                }
                return '[]'
            }
            
            const result = await taskHistory.getTaskWithId('test-id')
            
            assert.ok(result)
            assert.ok(result.historyItem)
            assert.strictEqual(result.historyItem.id, 'test-id')
            assert.ok(result.taskDirPath)
            assert.ok(result.apiConversationHistoryFilePath)
        })
        
        it('returns undefined for non-existent task ID', async function() {
            this.timeout(5000)
            
            mockContextProxy.getValue = async () => []
            
            const result = await taskHistory.getTaskWithId('non-existent')
            
            assert.strictEqual(result, undefined)
        })
    })
    
    describe('deleteTaskWithId', () => {
        it('deletes task from history', async function() {
            this.timeout(5000)
            
            const mockHistory = [
                { id: 'task-1', task: 'Task 1' },
                { id: 'task-2', task: 'Task 2' },
                { id: 'task-3', task: 'Task 3' }
            ]
            
            let savedHistory = null
            let rmCalled = false
            
            mockContextProxy.getValue = async () => [...mockHistory]
            mockContextProxy.setValue = async (key, value) => {
                if (key === 'taskHistory') {
                    savedHistory = value
                }
            }
            
            fs.rm = async (path, options) => {
                rmCalled = true
                assert.ok(path.includes('task-2'))
                assert.strictEqual(options.recursive, true)
                assert.strictEqual(options.force, true)
            }
            
            await taskHistory.deleteTaskWithId('task-2')
            
            assert.ok(savedHistory)
            assert.strictEqual(savedHistory.length, 2)
            assert.strictEqual(savedHistory[0].id, 'task-1')
            assert.strictEqual(savedHistory[1].id, 'task-3')
            assert.strictEqual(rmCalled, true)
        })
        
        it('handles deletion of non-existent task', async function() {
            this.timeout(5000)
            
            const mockHistory = [
                { id: 'task-1', task: 'Task 1' }
            ]
            
            let savedHistory = null
            
            mockContextProxy.getValue = async () => [...mockHistory]
            mockContextProxy.setValue = async (key, value) => {
                if (key === 'taskHistory') {
                    savedHistory = value
                }
            }
            
            await taskHistory.deleteTaskWithId('non-existent')
            
            // History should remain unchanged
            assert.ok(savedHistory)
            assert.strictEqual(savedHistory.length, 1)
            assert.strictEqual(savedHistory[0].id, 'task-1')
        })
    })
    
    describe('exportTaskWithId', () => {
        it('exports task to markdown', async function() {
            this.timeout(5000)
            
            const mockHistoryItem = {
                id: 'export-task',
                task: 'Export Test Task',
                ts: Date.now()
            }
            
            let downloadCalled = false
            
            mockContextProxy.getValue = async () => [mockHistoryItem]
            
            // Mock the downloadTask function
            const { downloadTask } = loadTheaModule('src/integrations/misc/export-markdown')
            const originalDownload = downloadTask
            
            // Override temporarily
            require('../../../helpers/thea-loader').loadTheaModule = (module) => {
                if (module === 'src/integrations/misc/export-markdown') {
                    return {
                        downloadTask: async (history, dirPath) => {
                            downloadCalled = true
                            assert.ok(history)
                            assert.ok(dirPath)
                            return '/exported/path/task.md'
                        }
                    }
                }
                // Return original module loader for other modules
                return require('../../../helpers/thea-loader').loadTheaModule(module)
            }
            
            fs.readFile = async () => JSON.stringify([])
            
            const result = await taskHistory.exportTaskWithId('export-task')
            
            assert.strictEqual(downloadCalled, true)
            assert.ok(result)
        })
    })
})