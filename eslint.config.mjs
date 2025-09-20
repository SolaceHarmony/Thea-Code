import globals from "globals"
import tseslintPlugin from "@typescript-eslint/eslint-plugin"
import tseslintParser from "@typescript-eslint/parser"
import reactPlugin from "eslint-plugin-react"
import reactHooksPlugin from "eslint-plugin-react-hooks"
import * as espree from "espree"
import { globalIgnores } from "eslint/config"

const tsconfigRootDir = new URL('.', import.meta.url).pathname

const commonTsConfig = {
	languageOptions: {
		parser: tseslintParser,
		parserOptions: {
			ecmaVersion: 2022,
			sourceType: "module",
			ecmaFeatures: {},
		},
		globals: {
			...globals.node,
        
			...globals.mocha,
		},
	},
	plugins: {
		"@typescript-eslint": tseslintPlugin,
	},
	rules: {
		...tseslintPlugin.configs["recommended-type-checked"].rules,
		"@typescript-eslint/no-base-to-string": "off",
		"@typescript-eslint/no-explicit-any": "warn",
		"@typescript-eslint/no-duplicate-type-constituents": "warn",
		"@typescript-eslint/no-confusing-void-expression": [
			"warn",
			{ "ignoreArrowShorthand": true, "ignoreVoidOperator": true }
		],
	},
}

	export default [
	{
		files: ["src/__mocks__/**/*.{ts,tsx,js,jsx}"],
		languageOptions: {
			parser: espree, // Use espree for mock files
			parserOptions: {
				ecmaVersion: 2022,
				sourceType: "module",
				ecmaFeatures: {
					jsx: true,
				},
			},
			globals: {
				...globals.browser,
				...globals.node,
            
				...globals.mocha,
			},
		},
		plugins: {
			react: reactPlugin,
			"react-hooks": reactHooksPlugin,
		},
		rules: {},
	},
	globalIgnores([
		// Generated or vendor outputs only. Keep lint signal in source trees.
		"node_modules/",
		"dist/",
		"build/",
		"coverage/",
		"coverage-report/",
		"webview-ui/build/**",
		"webview-ui/dist/**",
		"test/**/.cache/**",
		"benchmark/",
		// VS Code test harness shims that confuse parsers
		"src/e2e/.vscode-test.mjs",
		"**/.vscode-test.mjs",
	]),
	{
		files: [
			"src/**/*.{ts,tsx}",
			"!src/**/*.js",
			"!src/__mocks__/**/*",
		],
		...commonTsConfig,
		languageOptions: {
			...commonTsConfig.languageOptions,
			parserOptions: {
				...commonTsConfig.languageOptions.parserOptions,
				ecmaFeatures: {
					jsx: true,
				},
				project: "./tsconfig.eslint.json",
				tsconfigRootDir,
			},
		},
		rules: {
			...commonTsConfig.rules,
			"@typescript-eslint/no-misused-promises": [
				"error",
				{ checksVoidReturn: { attributes: false } }
			],
			"@typescript-eslint/no-unsafe-return": "error",
			"@typescript-eslint/no-unsafe-argument": "error",
			"@typescript-eslint/no-unsafe-assignment": "error",
			"@typescript-eslint/no-unsafe-call": "error",
			"@typescript-eslint/no-unsafe-member-access": "error",
			"@typescript-eslint/restrict-plus-operands": "error",
			"@typescript-eslint/restrict-template-expressions": [
				"error",
				{ allowNumber: true }
			],
			"@typescript-eslint/unbound-method": "error",
			"@typescript-eslint/no-floating-promises": "error",
			"@typescript-eslint/no-unused-vars": [
				"error",
				{ argsIgnorePattern: "^_", varsIgnorePattern: "^_" }
			],
			"@typescript-eslint/require-await": "error",
			"@typescript-eslint/no-explicit-any": "warn",
		},
	},
  {
    files: [
      "src/shared/__tests__/array.test.ts",
      "src/shared/__tests__/formatPath.test.ts",
      "src/shared/__tests__/language.test.ts",
      "src/shared/__tests__/vsCodeSelectorUtils.test.ts",
      "src/shared/__tests__/experiments.test.ts",
    ],
		...commonTsConfig,
		languageOptions: {
			...commonTsConfig.languageOptions,
			parserOptions: {
				...commonTsConfig.languageOptions.parserOptions,
				ecmaFeatures: {
					jsx: true,
				},
				project: "./tsconfig.eslint.json",
				tsconfigRootDir,
			},
		},
 	},
	{
		files: [
			"src/**/__tests__/**/*.{ts,tsx}",
			"src/**/*.test.{ts,tsx}",
			"test/**/*.{ts,tsx}",
		],
		...commonTsConfig,
		languageOptions: {
			...commonTsConfig.languageOptions,
			parserOptions: {
				...commonTsConfig.languageOptions.parserOptions,
				ecmaFeatures: { jsx: true },
				project: "./tsconfig.eslint.json",
				tsconfigRootDir,
			},
			globals: {
				...commonTsConfig.languageOptions.globals,
				...globals.mocha,
				...globals.node,
			},
		},
		rules: {
			// Allow BDD-style assertions like `expect(foo).to.be.true`
			"@typescript-eslint/no-unused-expressions": "off",
		},
	},
	// E2E co-located tests: relax a couple of rules for pragmatic test code
	{
		files: [
			"src/**/__e2e__/**/*.{ts,tsx}",
		],
		...commonTsConfig,
		languageOptions: {
			...commonTsConfig.languageOptions,
			parserOptions: {
				...commonTsConfig.languageOptions.parserOptions,
				project: "./tsconfig.eslint.json",
				tsconfigRootDir,
			},
		},
		rules: {
			"@typescript-eslint/no-unnecessary-type-assertion": "off",
			"@typescript-eslint/no-explicit-any": "off",
		},
	},
	{
		files: [
			"src/e2e/src/launch.ts",
			"src/e2e/src/runTest.ts",
			"src/e2e/src/suite/index.ts",
			"src/e2e/src/suite/setup.test.ts",
			"src/e2e/src/suite/selected/**/*.{ts,tsx}",
		],
		...commonTsConfig,
		languageOptions: {
			...commonTsConfig.languageOptions,
			parserOptions: {
				...commonTsConfig.languageOptions.parserOptions,
				ecmaFeatures: {
					jsx: true,
				},
				project: "./src/e2e/tsconfig.json",
				tsconfigRootDir,
			},
		},
	},
	{
		files: ["test/**/*.{ts,tsx}", "!test/benchmark/**", "!test/e2e/**", "!test/**/*.js"],
		...commonTsConfig,
		languageOptions: {
			...commonTsConfig.languageOptions,
			parserOptions: {
				...commonTsConfig.languageOptions.parserOptions,
				ecmaFeatures: {
					jsx: true,
				},
				project: "./tsconfig.eslint.json",
				tsconfigRootDir,
			},
		},
		rules: {
			...commonTsConfig.rules,
			"@typescript-eslint/no-misused-promises": [
				"error",
				{ checksVoidReturn: { attributes: false } }
			],
			"@typescript-eslint/no-unsafe-return": "error",
			"@typescript-eslint/no-unsafe-argument": "error",
			"@typescript-eslint/no-unsafe-assignment": "error",
			"@typescript-eslint/no-unsafe-call": "error",
			"@typescript-eslint/no-unsafe-member-access": "error",
			"@typescript-eslint/restrict-plus-operands": "error",
			"@typescript-eslint/restrict-template-expressions": [
				"error",
				{ allowNumber: true }
			],
			"@typescript-eslint/unbound-method": "error",
			"@typescript-eslint/no-floating-promises": "error",
			"@typescript-eslint/no-unused-vars": [
				"error",
				{ argsIgnorePattern: "^_", varsIgnorePattern: "^_" }
			],
			"@typescript-eslint/require-await": "error",
			"@typescript-eslint/no-explicit-any": "warn",
		},
	},
	{
		files: ["scripts/**/*.{ts,js}", "*.{cjs,mjs,js}"],
		languageOptions: {
			parser: espree,
			ecmaVersion: 2022,
			sourceType: "module",
			globals: { ...globals.node },
		},
		rules: {
			// Ensure TS type-aware rules are not applied to plain JS scripts
			"@typescript-eslint/await-thenable": "off",
			"@typescript-eslint/no-array-delete": "off",
			"@typescript-eslint/no-misused-promises": "off",
			"@typescript-eslint/no-floating-promises": "off",
			"@typescript-eslint/no-unsafe-assignment": "off",
			"@typescript-eslint/no-unsafe-member-access": "off",
			"@typescript-eslint/no-unsafe-call": "off",
			"@typescript-eslint/no-unsafe-argument": "off",
			"@typescript-eslint/no-unsafe-return": "off",
			"@typescript-eslint/unbound-method": "off",
			"@typescript-eslint/restrict-plus-operands": "off",
			"@typescript-eslint/restrict-template-expressions": "off"
		},
	},
	{
		files: ["**/*.{js,mjs,cjs,jsx}"],
		languageOptions: {
			parser: espree,
			ecmaVersion: 2022,
			sourceType: "module",
			globals: {
				...globals.browser,
				...globals.node,
			        
				...globals.mocha,
			},
		},
		plugins: {
			react: reactPlugin,
			"react-hooks": reactHooksPlugin,
		},
 		rules: {
				// Keep JS/JSX rules minimal to avoid cross-plugin issues
				"no-undef": "error",
				"no-import-assign": "error",
				// Ensure TS type-aware rules are not applied to JS files
				"@typescript-eslint/await-thenable": "off",
				"@typescript-eslint/no-array-delete": "off",
				"@typescript-eslint/no-misused-promises": "off",
				"@typescript-eslint/no-floating-promises": "off",
				"@typescript-eslint/no-unsafe-assignment": "off",
				"@typescript-eslint/no-unsafe-member-access": "off",
				"@typescript-eslint/no-unsafe-call": "off",
				"@typescript-eslint/no-unsafe-argument": "off",
				"@typescript-eslint/no-unsafe-return": "off",
				"@typescript-eslint/unbound-method": "off",
				"@typescript-eslint/restrict-plus-operands": "off",
				"@typescript-eslint/restrict-template-expressions": "off"
			}
		},
	// Webview UI: use its own tsconfig for type-aware TS rules
	{
		files: ["webview-ui/**/*.{ts,tsx}"],
		...commonTsConfig,
		languageOptions: {
			...commonTsConfig.languageOptions,
			parserOptions: {
				...commonTsConfig.languageOptions.parserOptions,
				ecmaFeatures: { jsx: true },
				project: "./webview-ui/tsconfig.json",
				tsconfigRootDir,
			},
		},
		rules: {
			...commonTsConfig.rules,
			"@typescript-eslint/no-misused-promises": [
				"error",
				{ checksVoidReturn: { attributes: false } }
			],
			"@typescript-eslint/no-unsafe-return": "error",
			"@typescript-eslint/no-unsafe-argument": "error",
			"@typescript-eslint/no-unsafe-assignment": "error",
			"@typescript-eslint/no-unsafe-call": "error",
			"@typescript-eslint/no-unsafe-member-access": "error",
			"@typescript-eslint/restrict-plus-operands": "error",
			"@typescript-eslint/restrict-template-expressions": [
				"error",
				{ allowNumber: true }
			],
			"@typescript-eslint/unbound-method": "error",
			"@typescript-eslint/no-floating-promises": "error",
			"@typescript-eslint/no-unused-vars": [
				"error",
				{ argsIgnorePattern: "^_", varsIgnorePattern: "^_" }
			],
			"@typescript-eslint/require-await": "error",
			"@typescript-eslint/no-explicit-any": "warn",
		},
	},
]
