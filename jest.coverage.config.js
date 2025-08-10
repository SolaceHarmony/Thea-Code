/**
 * Jest configuration with coverage enforcement
 * As recommended by architect: 80% for MCP core/transports/providers, 70% overall
 */

const baseConfig = require('./jest.config.js');

module.exports = {
	...baseConfig,
	collectCoverage: true,
	coverageDirectory: "coverage",
	coverageReporters: ["text", "lcov", "html", "json-summary"],
	collectCoverageFrom: [
		"src/**/*.{ts,tsx}",
		"!src/**/*.d.ts",
		"!src/**/*.test.{ts,tsx}",
		"!src/**/__tests__/**",
		"!src/**/__mocks__/**",
		"!src/test/**",
		"!src/extension.ts", // Entry point
		"!webview-ui/**" // Frontend tested separately
	],
	coverageThreshold: {
		global: {
			branches: 70,
			functions: 70,
			lines: 70,
			statements: 70
		},
		// MCP core components - higher threshold
		"src/services/mcp/core/**/*.ts": {
			branches: 80,
			functions: 80,
			lines: 80,
			statements: 80
		},
		// MCP transports - higher threshold
		"src/services/mcp/transport/**/*.ts": {
			branches: 80,
			functions: 80,
			lines: 80,
			statements: 80
		},
		// MCP providers - higher threshold
		"src/services/mcp/providers/**/*.ts": {
			branches: 80,
			functions: 80,
			lines: 80,
			statements: 80
		},
		// API providers - higher threshold
		"src/api/providers/**/*.ts": {
			branches: 75,
			functions: 75,
			lines: 75,
			statements: 75
		}
	},
	// Update projects to include coverage settings
	projects: baseConfig.projects.map(project => ({
		...project,
		collectCoverageFrom: project.displayName === 'backend' ? [
			"<rootDir>/src/**/*.ts",
			"!<rootDir>/src/**/*.d.ts",
			"!<rootDir>/src/**/*.test.ts",
			"!<rootDir>/src/**/__tests__/**",
			"!<rootDir>/src/**/__mocks__/**",
			"!<rootDir>/src/test/**",
			"!<rootDir>/src/extension.ts"
		] : project.displayName === 'frontend' ? [
			"<rootDir>/webview-ui/src/**/*.{ts,tsx}",
			"!<rootDir>/webview-ui/src/**/*.d.ts",
			"!<rootDir>/webview-ui/src/**/*.test.{ts,tsx}",
			"!<rootDir>/webview-ui/src/**/__tests__/**",
			"!<rootDir>/webview-ui/src/**/__mocks__/**",
			"!<rootDir>/webview-ui/src/setupTests.ts"
		] : project.collectCoverageFrom
	}))
};