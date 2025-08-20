// Migrated from src/integrations/terminal/__tests__/TerminalProcess.test.ts
// Following the Roo migration harness guide

const assert = require('assert')
const { loadTheaModule } = require('../../../helpers/thea-loader')

// Load Thea modules
const { TerminalProcess, mergePromise } = loadTheaModule('src/integrations/terminal/TerminalProcess')
const { Terminal } = loadTheaModule('src/integrations/terminal/Terminal')
const { TerminalRegistry } = loadTheaModule('src/integrations/terminal/TerminalRegistry')
const { EXTENSION_DISPLAY_NAME } = loadTheaModule('src/shared/config/thea-config')

describe('TerminalProcess', () => {
    let terminalProcess
    let mockTerminal
    let mockTerminalInfo
    let mockExecution
    let mockStream
    
    // Mock vscode module
    const vscode = require('vscode')
    const originalCreateTerminal = vscode.window.createTerminal
    
    beforeEach(() => {
        // Create mock terminal with shell integration
        mockTerminal = {
            shellIntegration: {
                executeCommand: () => mockExecution
            },
            name: EXTENSION_DISPLAY_NAME,
            processId: Promise.resolve(123),
            creationOptions: {},
            exitStatus: undefined,
            state: { isInteractedWith: true },
            dispose: () => {},
            hide: () => {},
            show: () => {},
            sendText: () => {}
        }
        
        // Stub createTerminal
        vscode.window.createTerminal = () => mockTerminal
        
        mockTerminalInfo = new Terminal(1, mockTerminal, './')
        terminalProcess = new TerminalProcess(mockTerminalInfo)
        
        TerminalRegistry.terminals.push(mockTerminalInfo)
        
        // Reset event listeners
        terminalProcess.removeAllListeners()
    })
    
    afterEach(() => {
        // Restore original function
        vscode.window.createTerminal = originalCreateTerminal
        // Clear registry
        TerminalRegistry.terminals.length = 0
    })
    
    describe('run', () => {
        it('handles shell integration commands correctly', async function() {
            this.timeout(5000)
            
            let lines = []
            
            terminalProcess.on('completed', (output) => {
                if (output) {
                    lines = output.split('\n')
                }
            })
            
            // Mock stream data with shell integration sequences
            mockStream = (function* () {
                yield '\x1b]633;C\x07' // Command start sequence
                yield 'Initial output\n'
                yield 'More output\n'
                yield 'Final output'
                yield '\x1b]633;D\x07' // Command end sequence
                terminalProcess.emit('shell_execution_complete', { exitCode: 0 })
            })()
            
            mockExecution = {
                read: () => mockStream
            }
            
            mockTerminal.shellIntegration.executeCommand = () => mockExecution
            
            const runPromise = terminalProcess.run('test command')
            terminalProcess.emit('stream_available', mockStream)
            await runPromise
            
            assert.deepStrictEqual(lines, ['Initial output', 'More output', 'Final output'])
            assert.strictEqual(terminalProcess.isHot, false)
        })
        
        it('handles terminals without shell integration', async function() {
            this.timeout(5000)
            
            // Create terminal without shell integration
            const noShellTerminal = {
                sendText: (text, addNewline) => {
                    assert.strictEqual(text, 'test command')
                    assert.strictEqual(addNewline, true)
                },
                shellIntegration: undefined,
                name: 'No Shell Terminal',
                processId: Promise.resolve(456),
                creationOptions: {},
                exitStatus: undefined,
                state: { isInteractedWith: true },
                dispose: () => {},
                hide: () => {},
                show: () => {}
            }
            
            const noShellTerminalInfo = new Terminal(2, noShellTerminal, './')
            const noShellProcess = new TerminalProcess(noShellTerminalInfo)
            
            // Track events
            let noShellIntegrationFired = false
            let completedFired = false
            let continueFired = false
            
            noShellProcess.once('no_shell_integration', () => {
                noShellIntegrationFired = true
            })
            noShellProcess.once('completed', () => {
                completedFired = true
            })
            noShellProcess.once('continue', () => {
                continueFired = true
            })
            
            await noShellProcess.run('test command')
            
            // Give events time to fire
            await new Promise(resolve => setTimeout(resolve, 100))
            
            assert.strictEqual(noShellIntegrationFired, true)
            assert.strictEqual(completedFired, true)
            assert.strictEqual(continueFired, true)
        })
        
        it('sets hot state for compiling commands', async function() {
            this.timeout(5000)
            
            let lines = []
            
            terminalProcess.on('completed', (output) => {
                if (output) {
                    lines = output.split('\n')
                }
            })
            
            let shellExecutionCompleteFired = false
            terminalProcess.on('shell_execution_complete', () => {
                shellExecutionCompleteFired = true
            })
            
            mockStream = (function* () {
                yield '\x1b]633;C\x07'
                yield 'compiling...\n'
                yield 'still compiling...\n'
                yield 'done'
                yield '\x1b]633;D\x07'
                terminalProcess.emit('shell_execution_complete', { exitCode: 0 })
            })()
            
            mockExecution = {
                read: () => mockStream
            }
            
            mockTerminal.shellIntegration.executeCommand = () => mockExecution
            
            const runPromise = terminalProcess.run('npm run build')
            terminalProcess.emit('stream_available', mockStream)
            
            assert.strictEqual(terminalProcess.isHot, true)
            await runPromise
            
            assert.deepStrictEqual(lines, ['compiling...', 'still compiling...', 'done'])
            
            // Wait for shell execution complete
            await new Promise(resolve => setTimeout(resolve, 100))
            assert.strictEqual(shellExecutionCompleteFired, true)
            assert.strictEqual(terminalProcess.isHot, false)
        })
    })
    
    describe('continue', () => {
        it('stops listening and emits continue event', () => {
            let continueCalled = false
            terminalProcess.on('continue', () => {
                continueCalled = true
            })
            
            terminalProcess.continue()
            
            assert.strictEqual(continueCalled, true)
            assert.strictEqual(terminalProcess.isListening, false)
        })
    })
    
    describe('getUnretrievedOutput', () => {
        it('returns and clears unretrieved output', () => {
            // Access private fields directly
            terminalProcess.fullOutput = '\x1b]633;C\x07previous\nnew output\x1b]633;D\x07'
            terminalProcess.lastRetrievedIndex = 17 // After "previous\n"
            
            const unretrieved = terminalProcess.getUnretrievedOutput()
            assert.strictEqual(unretrieved, 'new output')
            
            const expectedIndex = terminalProcess.fullOutput.length - 'previous'.length
            assert.strictEqual(terminalProcess.lastRetrievedIndex, expectedIndex)
        })
    })
    
    describe('interpretExitCode', () => {
        it('handles undefined exit code', () => {
            const result = TerminalProcess.interpretExitCode(undefined)
            assert.deepStrictEqual(result, { exitCode: undefined })
        })
        
        it('handles normal exit codes (0-128)', () => {
            const result0 = TerminalProcess.interpretExitCode(0)
            assert.deepStrictEqual(result0, { exitCode: 0 })
            
            const result1 = TerminalProcess.interpretExitCode(1)
            assert.deepStrictEqual(result1, { exitCode: 1 })
            
            const result128 = TerminalProcess.interpretExitCode(128)
            assert.deepStrictEqual(result128, { exitCode: 128 })
        })
        
        it('interprets signal exit codes (>128)', () => {
            // SIGTERM (15) -> 128 + 15 = 143
            const resultTerm = TerminalProcess.interpretExitCode(143)
            assert.deepStrictEqual(resultTerm, {
                exitCode: 143,
                signal: 15,
                signalName: 'SIGTERM',
                coreDumpPossible: false
            })
            
            // SIGSEGV (11) -> 128 + 11 = 139
            const resultSegv = TerminalProcess.interpretExitCode(139)
            assert.deepStrictEqual(resultSegv, {
                exitCode: 139,
                signal: 11,
                signalName: 'SIGSEGV',
                coreDumpPossible: true
            })
        })
        
        it('handles unknown signals', () => {
            const result = TerminalProcess.interpretExitCode(255)
            assert.deepStrictEqual(result, {
                exitCode: 255,
                signal: 127,
                signalName: 'Unknown Signal (127)',
                coreDumpPossible: false
            })
        })
    })
    
    describe('mergePromise', () => {
        it('merges promise methods with terminal process', async () => {
            const process = new TerminalProcess(mockTerminalInfo)
            const promise = Promise.resolve()
            
            const merged = mergePromise(process, promise)
            
            assert.strictEqual(typeof merged.then, 'function')
            assert.strictEqual(typeof merged.catch, 'function')
            assert.strictEqual(typeof merged.finally, 'function')
            assert.strictEqual(merged instanceof TerminalProcess, true)
            
            await merged // Should resolve without error
        })
    })
})