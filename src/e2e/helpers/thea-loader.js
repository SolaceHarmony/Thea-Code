// Thea module loader for test harness
// Following the Roo migration harness guide

const path = require('path')
const esbuild = require('esbuild')
const fs = require('fs')

// Cache for compiled modules
const moduleCache = new Map()

/**
 * Load a TypeScript module from the Thea codebase
 * @param {string} modulePath - Path to the module relative to src/ (e.g., 'src/core/TheaTask')
 * @returns {any} The loaded module exports
 */
function loadTheaModule(modulePath) {
    // Check cache first
    if (moduleCache.has(modulePath)) {
        return moduleCache.get(modulePath)
    }
    
    // Resolve the full path
    const fullPath = path.resolve(__dirname, '../../', modulePath)
    const tsPath = fullPath.endsWith('.ts') ? fullPath : `${fullPath}.ts`
    
    try {
        // Bundle the TypeScript module
        const result = esbuild.buildSync({
            entryPoints: [tsPath],
            bundle: true,
            write: false,
            platform: 'node',
            target: 'node16',
            format: 'cjs',
            external: [
                'vscode',
                'child_process',
                'fs',
                'fs/promises',
                'path',
                'os',
                'util',
                'crypto',
                'stream',
                'events',
                'net',
                'http',
                'https',
                'url',
                'querystring',
                'zlib',
                'assert',
                'buffer',
                'timers',
                'process',
                'console',
                'tcp-port-used',
                'delay',
                'p-wait-for',
                'execa',
                'find-up',
                'globby',
                'strip-ansi',
                'ansi-colors',
                'nanoid',
                'diff',
                'jsdom',
                'marked',
                'eventsource-parser',
                'zod',
                'anthropic',
                'openai',
                '@anthropic-ai/sdk',
                '@aws-sdk/*',
                'aws-sdk',
                'google-auth-library',
                '@google-cloud/*'
            ],
            loader: {
                '.ts': 'ts',
                '.tsx': 'tsx',
                '.js': 'js',
                '.jsx': 'jsx',
                '.json': 'json'
            },
            logLevel: 'silent'
        })
        
        // Execute the bundled code
        const code = result.outputFiles[0].text
        const moduleExports = {}
        const moduleRequire = (id) => {
            // Handle special cases
            if (id === 'vscode') {
                return require('../node_modules/vscode')
            }
            return require(id)
        }
        
        // Create a function wrapper
        const wrapper = new Function('exports', 'require', 'module', '__filename', '__dirname', code)
        const moduleObj = { exports: moduleExports }
        
        // Execute in module context
        wrapper(moduleExports, moduleRequire, moduleObj, tsPath, path.dirname(tsPath))
        
        // Cache the result
        const finalExports = moduleObj.exports
        moduleCache.set(modulePath, finalExports)
        
        return finalExports
    } catch (error) {
        console.error(`Failed to load module ${modulePath}:`, error.message)
        
        // Try direct require as fallback for JS files
        if (fs.existsSync(fullPath.replace('.ts', '.js'))) {
            const jsModule = require(fullPath.replace('.ts', '.js'))
            moduleCache.set(modulePath, jsModule)
            return jsModule
        }
        
        throw error
    }
}

module.exports = { loadTheaModule }