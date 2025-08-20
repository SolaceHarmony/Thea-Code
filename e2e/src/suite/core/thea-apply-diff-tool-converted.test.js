// Migrated from src/core/tools/__tests__/applyDiffTool.test.ts
// Following the Roo migration harness guide

const assert = require('assert')
const path = require('path')
const { loadTheaModule } = require('../../../helpers/thea-loader')

// Load Thea modules
const { applyDiffTool } = loadTheaModule('src/core/tools/applyDiffTool')

// Mock fs
const fs = require('fs/promises')
let originalReadFile
let originalWriteFile

describe('applyDiffTool', () => {
    let mockTheaTask
    let mockAskApproval
    let mockHandleError
    let mockPushToolResult
    let mockRemoveClosingTag
    
    beforeEach(() => {
        // Store original fs methods
        originalReadFile = fs.readFile
        originalWriteFile = fs.writeFile
        
        // Mock fs methods
        fs.readFile = async () => 'original file content'
        fs.writeFile = async () => undefined
        
        // Setup mock TheaTask
        mockTheaTask = {
            cwd: '/test',
            consecutiveMistakeCount: 0,
            consecutiveMistakeCountForApplyDiff: new Map(),
            webviewCommunicator: {
                ask: async () => undefined,
                say: async () => undefined
            },
            diffViewProvider: {
                open: () => {},
                update: () => {},
                scrollToFirstDiff: () => {},
                revertChanges: () => {},
                saveChanges: async () => ({
                    newProblemsMessage: '',
                    userEdits: undefined,
                    finalContent: 'updated content'
                }),
                reset: () => {}
            },
            diffStrategy: {
                applyDiff: async (originalContent, diff) => ({
                    success: true,
                    content: 'modified content'
                }),
                getProgressStatus: () => null
            },
            theaIgnoreController: {
                validateAccess: () => true
            },
            sayAndCreateMissingParamError: async (tool, param) => {
                return `Missing required parameter: ${param} for tool: ${tool}`
            },
            didEditFile: false
        }
        
        // Setup mock functions
        mockAskApproval = async () => true
        mockHandleError = async () => undefined
        mockPushToolResult = () => {}
        mockRemoveClosingTag = (text) => text || ''
    })
    
    afterEach(() => {
        // Restore original fs methods
        fs.readFile = originalReadFile
        fs.writeFile = originalWriteFile
    })
    
    describe('parameter validation', () => {
        it('handles partial blocks by sending progress update', async function() {
            this.timeout(5000)
            
            let askCalled = false
            mockTheaTask.webviewCommunicator.ask = async () => {
                askCalled = true
            }
            
            const block = {
                type: 'tool_use',
                name: 'apply_diff',
                params: { path: 'file.txt', diff: 'd' },
                partial: true
            }
            
            await applyDiffTool(
                mockTheaTask,
                block,
                mockAskApproval,
                mockHandleError,
                mockPushToolResult,
                mockRemoveClosingTag
            )
            
            assert.strictEqual(askCalled, true)
        })
        
        it('handles missing path parameter', async function() {
            this.timeout(5000)
            
            let errorMessage = null
            let mistakeCount = 0
            
            mockPushToolResult = (message) => {
                errorMessage = message
            }
            
            const block = {
                type: 'tool_use',
                name: 'apply_diff',
                params: { diff: 'some diff' },
                partial: false
            }
            
            await applyDiffTool(
                mockTheaTask,
                block,
                mockAskApproval,
                mockHandleError,
                mockPushToolResult,
                mockRemoveClosingTag
            )
            
            assert.ok(errorMessage)
            assert.ok(errorMessage.includes('Missing required parameter'))
            assert.ok(errorMessage.includes('path'))
            assert.strictEqual(mockTheaTask.consecutiveMistakeCount, 1)
        })
        
        it('handles missing diff parameter', async function() {
            this.timeout(5000)
            
            let errorMessage = null
            
            mockPushToolResult = (message) => {
                errorMessage = message
            }
            
            const block = {
                type: 'tool_use',
                name: 'apply_diff',
                params: { path: 'file.txt' },
                partial: false
            }
            
            await applyDiffTool(
                mockTheaTask,
                block,
                mockAskApproval,
                mockHandleError,
                mockPushToolResult,
                mockRemoveClosingTag
            )
            
            assert.ok(errorMessage)
            assert.ok(errorMessage.includes('Missing required parameter'))
            assert.ok(errorMessage.includes('diff'))
            assert.strictEqual(mockTheaTask.consecutiveMistakeCount, 1)
        })
    })
    
    describe('file operations', () => {
        it('successfully applies diff to existing file', async function() {
            this.timeout(5000)
            
            let diffApplied = false
            let fileWritten = false
            let resultPushed = false
            
            // Mock file exists check
            const { fileExistsAtPath } = loadTheaModule('src/utils/fs')
            const originalFileExists = fileExistsAtPath
            require('../../../helpers/thea-loader').loadTheaModule = (module) => {
                if (module === 'src/utils/fs') {
                    return {
                        fileExistsAtPath: async () => true
                    }
                }
                return require('../../../helpers/thea-loader').loadTheaModule(module)
            }
            
            mockTheaTask.diffStrategy.applyDiff = async (content, diff) => {
                diffApplied = true
                assert.strictEqual(content, 'original file content')
                assert.ok(diff)
                return { success: true, content: 'modified content' }
            }
            
            fs.writeFile = async (filePath, content) => {
                fileWritten = true
                assert.ok(filePath.includes('file.txt'))
                assert.strictEqual(content, 'modified content')
            }
            
            mockPushToolResult = (message) => {
                resultPushed = true
                assert.ok(message.includes('Applied diff'))
            }
            
            const block = {
                type: 'tool_use',
                name: 'apply_diff',
                params: {
                    path: 'file.txt',
                    diff: '--- a/file.txt\n+++ b/file.txt\n@@ -1 +1 @@\n-old\n+new'
                },
                partial: false
            }
            
            await applyDiffTool(
                mockTheaTask,
                block,
                mockAskApproval,
                mockHandleError,
                mockPushToolResult,
                mockRemoveClosingTag
            )
            
            assert.strictEqual(diffApplied, true)
            assert.strictEqual(fileWritten, true)
            assert.strictEqual(resultPushed, true)
            assert.strictEqual(mockTheaTask.didEditFile, true)
        })
        
        it('handles non-existent file by creating it', async function() {
            this.timeout(5000)
            
            let fileCreated = false
            let approvalAsked = false
            
            // Mock file doesn't exist
            require('../../../helpers/thea-loader').loadTheaModule = (module) => {
                if (module === 'src/utils/fs') {
                    return {
                        fileExistsAtPath: async () => false
                    }
                }
                return require('../../../helpers/thea-loader').loadTheaModule(module)
            }
            
            mockAskApproval = async (message) => {
                approvalAsked = true
                assert.ok(message.includes('create'))
                return true
            }
            
            fs.writeFile = async (filePath, content) => {
                fileCreated = true
                assert.ok(filePath.includes('new-file.txt'))
            }
            
            const block = {
                type: 'tool_use',
                name: 'apply_diff',
                params: {
                    path: 'new-file.txt',
                    diff: '--- /dev/null\n+++ b/new-file.txt\n@@ -0,0 +1 @@\n+new file content'
                },
                partial: false
            }
            
            await applyDiffTool(
                mockTheaTask,
                block,
                mockAskApproval,
                mockHandleError,
                mockPushToolResult,
                mockRemoveClosingTag
            )
            
            assert.strictEqual(approvalAsked, true)
            assert.strictEqual(fileCreated, true)
        })
    })
    
    describe('error handling', () => {
        it('handles diff application failure', async function() {
            this.timeout(5000)
            
            let errorHandled = false
            
            // Mock file exists
            require('../../../helpers/thea-loader').loadTheaModule = (module) => {
                if (module === 'src/utils/fs') {
                    return {
                        fileExistsAtPath: async () => true
                    }
                }
                return require('../../../helpers/thea-loader').loadTheaModule(module)
            }
            
            mockTheaTask.diffStrategy.applyDiff = async () => ({
                success: false,
                error: 'Failed to apply diff'
            })
            
            mockHandleError = async (error) => {
                errorHandled = true
                assert.ok(error.includes('Failed to apply diff'))
            }
            
            const block = {
                type: 'tool_use',
                name: 'apply_diff',
                params: {
                    path: 'file.txt',
                    diff: 'invalid diff'
                },
                partial: false
            }
            
            await applyDiffTool(
                mockTheaTask,
                block,
                mockAskApproval,
                mockHandleError,
                mockPushToolResult,
                mockRemoveClosingTag
            )
            
            assert.strictEqual(errorHandled, true)
            assert.strictEqual(mockTheaTask.consecutiveMistakeCount, 1)
        })
        
        it('handles access denied by ignore controller', async function() {
            this.timeout(5000)
            
            let errorPushed = false
            
            mockTheaTask.theaIgnoreController.validateAccess = () => false
            
            mockPushToolResult = (message) => {
                errorPushed = true
                assert.ok(message.includes('Access denied'))
            }
            
            const block = {
                type: 'tool_use',
                name: 'apply_diff',
                params: {
                    path: 'ignored-file.txt',
                    diff: 'some diff'
                },
                partial: false
            }
            
            await applyDiffTool(
                mockTheaTask,
                block,
                mockAskApproval,
                mockHandleError,
                mockPushToolResult,
                mockRemoveClosingTag
            )
            
            assert.strictEqual(errorPushed, true)
        })
    })
})