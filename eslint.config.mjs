import globals from "globals"
import pluginJs from "@eslint/js"
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
		"webview-ui/build/**",
		"webview-ui/dist/**",
		"test/**/.cache/**", // Ignore test cache artifacts
		"src/e2e/.vscode-test.mjs", // Ignore e2e harness file that triggers TS rules under espree
		"node_modules/",
		"dist/",
		"build/",
		"coverage/",
		"coverage-report/",
		"*.cjs",
		"src/__mocks__/**/*",
		// "webview-ui/", // Removed from ignores so subproject linting works
		// "e2e/", // Removed from ignores so subproject linting works
		"jest.config.js",
		"eslint.config.mjs",
		".eslintrc.js",
		"temp_similarity_check.js",
		"benchmark/", // Re-enabled ignore to fix ESLint parsing issues
		"src/**/*.js", // Ignore stray compiled JS under src to avoid espree+TS rule conflicts
		"src/e2e/src/**",
		"src/e2e/.vscode-test/**",
		"**/*.md",
		"**/*.json",
		"**/*.yaml",
		"**/*.mmd",
		"**/*.log",
		"**/*.bak",
		".clinerules",
		".clinerules-code",
		".dockerignore",
		".env.sample",
		".git-blame-ignore-revs",
		".gitattributes",
		".gitconfig",
		"LICENSE",
		"assets/",
		"audio/",
		"cline_docs/",
		"locales/",
		"mock/",
		"package.json",
		"package-lock.json",
		// branding.json removed; branding is now static in src/shared/config/thea-config.ts
		"knip.json",
		"tsconfig.json",
		"transformer_architecture.md",
		"xlstm_vs_transformers_comparison.md",
		"PRIVACY.md",
		"README.md",
		"CHANGELOG.md",
		"CODE_OF_CONDUCT.md",
		"CONTRIBUTING.md",
		"ellipsis.yaml",
		".idea/",
		".changeset/",
		".gitignore",
		".idx/",
		".npmrc",
		".nvmrc",
		".prettierignore",
		".rooignore",
		".roomodes",
		".vscodeignore",
		"flake.lock",
		"flake.nix",
		"**/*.gitkeep",
		"**/*.snap",
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
			"@typescript-eslint/no-unused-expressions": "off",
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
		rules: {},
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
				...pluginJs.configs.recommended.rules,
				...reactPlugin.configs.recommended.rules,
				...reactHooksPlugin.configs.recommended.rules,
				"no-undef": "error",
				"no-import-assign": "error",
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
