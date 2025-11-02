import { EventEmitter } from "events"
import { z } from "zod"

/**
 * ProviderSettings
 */
declare const providerSettingsSchema: z.ZodObject<
	{
		apiProvider: z.ZodOptional<
			z.ZodEnum<{
				anthropic: "anthropic"
				glama: "glama"
				openrouter: "openrouter"
				bedrock: "bedrock"
				vertex: "vertex"
				openai: "openai"
				ollama: "ollama"
				"vscode-lm": "vscode-lm"
				lmstudio: "lmstudio"
				gemini: "gemini"
				"openai-native": "openai-native"
				mistral: "mistral"
				deepseek: "deepseek"
				unbound: "unbound"
				requesty: "requesty"
				"human-relay": "human-relay"
				"fake-ai": "fake-ai"
			}>
		>
		apiModelId: z.ZodOptional<z.ZodString>
		apiKey: z.ZodOptional<z.ZodString>
		anthropicBaseUrl: z.ZodOptional<z.ZodString>
		anthropicModelId: z.ZodOptional<z.ZodString>
		anthropicModelInfo: z.ZodOptional<
			z.ZodNullable<
				z.ZodObject<
					{
						maxTokens: z.ZodOptional<z.ZodNullable<z.ZodNumber>>
						contextWindow: z.ZodNumber
						supportsImages: z.ZodOptional<z.ZodBoolean>
						supportsComputerUse: z.ZodOptional<z.ZodBoolean>
						supportsPromptCache: z.ZodBoolean
						supportsTemperature: z.ZodOptional<z.ZodBoolean>
						supportsTopP: z.ZodOptional<z.ZodBoolean>
						supportsSystemInstructions: z.ZodOptional<z.ZodBoolean>
						supportsAssistantTool: z.ZodOptional<z.ZodBoolean>
						reasoningTokens: z.ZodOptional<z.ZodBoolean>
						temperature: z.ZodOptional<z.ZodNumber>
						topP: z.ZodOptional<z.ZodNumber>
						topK: z.ZodOptional<z.ZodNumber>
						inputPrice: z.ZodOptional<z.ZodNumber>
						outputPrice: z.ZodOptional<z.ZodNumber>
						cacheWritesPrice: z.ZodOptional<z.ZodNumber>
						cacheReadsPrice: z.ZodOptional<z.ZodNumber>
						description: z.ZodOptional<z.ZodString>
						reasoningEffort: z.ZodOptional<
							z.ZodEnum<{
								low: "low"
								medium: "medium"
								high: "high"
							}>
						>
						thinking: z.ZodOptional<z.ZodBoolean>
					},
					z.core.$strip
				>
			>
		>
		glamaModelId: z.ZodOptional<z.ZodString>
		glamaModelInfo: z.ZodOptional<
			z.ZodNullable<
				z.ZodObject<
					{
						maxTokens: z.ZodOptional<z.ZodNullable<z.ZodNumber>>
						contextWindow: z.ZodNumber
						supportsImages: z.ZodOptional<z.ZodBoolean>
						supportsComputerUse: z.ZodOptional<z.ZodBoolean>
						supportsPromptCache: z.ZodBoolean
						supportsTemperature: z.ZodOptional<z.ZodBoolean>
						supportsTopP: z.ZodOptional<z.ZodBoolean>
						supportsSystemInstructions: z.ZodOptional<z.ZodBoolean>
						supportsAssistantTool: z.ZodOptional<z.ZodBoolean>
						reasoningTokens: z.ZodOptional<z.ZodBoolean>
						temperature: z.ZodOptional<z.ZodNumber>
						topP: z.ZodOptional<z.ZodNumber>
						topK: z.ZodOptional<z.ZodNumber>
						inputPrice: z.ZodOptional<z.ZodNumber>
						outputPrice: z.ZodOptional<z.ZodNumber>
						cacheWritesPrice: z.ZodOptional<z.ZodNumber>
						cacheReadsPrice: z.ZodOptional<z.ZodNumber>
						description: z.ZodOptional<z.ZodString>
						reasoningEffort: z.ZodOptional<
							z.ZodEnum<{
								low: "low"
								medium: "medium"
								high: "high"
							}>
						>
						thinking: z.ZodOptional<z.ZodBoolean>
					},
					z.core.$strip
				>
			>
		>
		glamaApiKey: z.ZodOptional<z.ZodString>
		openRouterApiKey: z.ZodOptional<z.ZodString>
		openRouterModelId: z.ZodOptional<z.ZodString>
		openRouterModelInfo: z.ZodOptional<
			z.ZodNullable<
				z.ZodObject<
					{
						maxTokens: z.ZodOptional<z.ZodNullable<z.ZodNumber>>
						contextWindow: z.ZodNumber
						supportsImages: z.ZodOptional<z.ZodBoolean>
						supportsComputerUse: z.ZodOptional<z.ZodBoolean>
						supportsPromptCache: z.ZodBoolean
						supportsTemperature: z.ZodOptional<z.ZodBoolean>
						supportsTopP: z.ZodOptional<z.ZodBoolean>
						supportsSystemInstructions: z.ZodOptional<z.ZodBoolean>
						supportsAssistantTool: z.ZodOptional<z.ZodBoolean>
						reasoningTokens: z.ZodOptional<z.ZodBoolean>
						temperature: z.ZodOptional<z.ZodNumber>
						topP: z.ZodOptional<z.ZodNumber>
						topK: z.ZodOptional<z.ZodNumber>
						inputPrice: z.ZodOptional<z.ZodNumber>
						outputPrice: z.ZodOptional<z.ZodNumber>
						cacheWritesPrice: z.ZodOptional<z.ZodNumber>
						cacheReadsPrice: z.ZodOptional<z.ZodNumber>
						description: z.ZodOptional<z.ZodString>
						reasoningEffort: z.ZodOptional<
							z.ZodEnum<{
								low: "low"
								medium: "medium"
								high: "high"
							}>
						>
						thinking: z.ZodOptional<z.ZodBoolean>
					},
					z.core.$strip
				>
			>
		>
		openRouterBaseUrl: z.ZodOptional<z.ZodString>
		openRouterSpecificProvider: z.ZodOptional<z.ZodString>
		openRouterUseMiddleOutTransform: z.ZodOptional<z.ZodBoolean>
		awsAccessKey: z.ZodOptional<z.ZodString>
		awsSecretKey: z.ZodOptional<z.ZodString>
		awsSessionToken: z.ZodOptional<z.ZodString>
		awsRegion: z.ZodOptional<z.ZodString>
		awsUseCrossRegionInference: z.ZodOptional<z.ZodBoolean>
		awsUsePromptCache: z.ZodOptional<z.ZodBoolean>
		awspromptCacheId: z.ZodOptional<z.ZodString>
		awsProfile: z.ZodOptional<z.ZodString>
		awsUseProfile: z.ZodOptional<z.ZodBoolean>
		awsCustomArn: z.ZodOptional<z.ZodString>
		vertexKeyFile: z.ZodOptional<z.ZodString>
		vertexJsonCredentials: z.ZodOptional<z.ZodString>
		vertexProjectId: z.ZodOptional<z.ZodString>
		vertexRegion: z.ZodOptional<z.ZodString>
		openAiBaseUrl: z.ZodOptional<z.ZodString>
		openAiApiKey: z.ZodOptional<z.ZodString>
		openAiR1FormatEnabled: z.ZodOptional<z.ZodBoolean>
		openAiModelId: z.ZodOptional<z.ZodString>
		openAiCustomModelInfo: z.ZodOptional<
			z.ZodNullable<
				z.ZodObject<
					{
						maxTokens: z.ZodOptional<z.ZodNullable<z.ZodNumber>>
						contextWindow: z.ZodNumber
						supportsImages: z.ZodOptional<z.ZodBoolean>
						supportsComputerUse: z.ZodOptional<z.ZodBoolean>
						supportsPromptCache: z.ZodBoolean
						supportsTemperature: z.ZodOptional<z.ZodBoolean>
						supportsTopP: z.ZodOptional<z.ZodBoolean>
						supportsSystemInstructions: z.ZodOptional<z.ZodBoolean>
						supportsAssistantTool: z.ZodOptional<z.ZodBoolean>
						reasoningTokens: z.ZodOptional<z.ZodBoolean>
						temperature: z.ZodOptional<z.ZodNumber>
						topP: z.ZodOptional<z.ZodNumber>
						topK: z.ZodOptional<z.ZodNumber>
						inputPrice: z.ZodOptional<z.ZodNumber>
						outputPrice: z.ZodOptional<z.ZodNumber>
						cacheWritesPrice: z.ZodOptional<z.ZodNumber>
						cacheReadsPrice: z.ZodOptional<z.ZodNumber>
						description: z.ZodOptional<z.ZodString>
						reasoningEffort: z.ZodOptional<
							z.ZodEnum<{
								low: "low"
								medium: "medium"
								high: "high"
							}>
						>
						thinking: z.ZodOptional<z.ZodBoolean>
					},
					z.core.$strip
				>
			>
		>
		openAiUseAzure: z.ZodOptional<z.ZodBoolean>
		azureApiVersion: z.ZodOptional<z.ZodString>
		openAiStreamingEnabled: z.ZodOptional<z.ZodBoolean>
		ollamaModelId: z.ZodOptional<z.ZodString>
		ollamaBaseUrl: z.ZodOptional<z.ZodString>
		vsCodeLmModelSelector: z.ZodOptional<
			z.ZodObject<
				{
					vendor: z.ZodOptional<z.ZodString>
					family: z.ZodOptional<z.ZodString>
					version: z.ZodOptional<z.ZodString>
					id: z.ZodOptional<z.ZodString>
				},
				z.core.$strip
			>
		>
		lmStudioModelId: z.ZodOptional<z.ZodString>
		lmStudioBaseUrl: z.ZodOptional<z.ZodString>
		lmStudioDraftModelId: z.ZodOptional<z.ZodString>
		lmStudioSpeculativeDecodingEnabled: z.ZodOptional<z.ZodBoolean>
		geminiApiKey: z.ZodOptional<z.ZodString>
		googleGeminiBaseUrl: z.ZodOptional<z.ZodString>
		openAiNativeApiKey: z.ZodOptional<z.ZodString>
		mistralApiKey: z.ZodOptional<z.ZodString>
		mistralCodestralUrl: z.ZodOptional<z.ZodString>
		deepSeekBaseUrl: z.ZodOptional<z.ZodString>
		deepSeekApiKey: z.ZodOptional<z.ZodString>
		unboundApiKey: z.ZodOptional<z.ZodString>
		unboundModelId: z.ZodOptional<z.ZodString>
		unboundModelInfo: z.ZodOptional<
			z.ZodNullable<
				z.ZodObject<
					{
						maxTokens: z.ZodOptional<z.ZodNullable<z.ZodNumber>>
						contextWindow: z.ZodNumber
						supportsImages: z.ZodOptional<z.ZodBoolean>
						supportsComputerUse: z.ZodOptional<z.ZodBoolean>
						supportsPromptCache: z.ZodBoolean
						supportsTemperature: z.ZodOptional<z.ZodBoolean>
						supportsTopP: z.ZodOptional<z.ZodBoolean>
						supportsSystemInstructions: z.ZodOptional<z.ZodBoolean>
						supportsAssistantTool: z.ZodOptional<z.ZodBoolean>
						reasoningTokens: z.ZodOptional<z.ZodBoolean>
						temperature: z.ZodOptional<z.ZodNumber>
						topP: z.ZodOptional<z.ZodNumber>
						topK: z.ZodOptional<z.ZodNumber>
						inputPrice: z.ZodOptional<z.ZodNumber>
						outputPrice: z.ZodOptional<z.ZodNumber>
						cacheWritesPrice: z.ZodOptional<z.ZodNumber>
						cacheReadsPrice: z.ZodOptional<z.ZodNumber>
						description: z.ZodOptional<z.ZodString>
						reasoningEffort: z.ZodOptional<
							z.ZodEnum<{
								low: "low"
								medium: "medium"
								high: "high"
							}>
						>
						thinking: z.ZodOptional<z.ZodBoolean>
					},
					z.core.$strip
				>
			>
		>
		requestyApiKey: z.ZodOptional<z.ZodString>
		requestyModelId: z.ZodOptional<z.ZodString>
		requestyModelInfo: z.ZodOptional<
			z.ZodNullable<
				z.ZodObject<
					{
						maxTokens: z.ZodOptional<z.ZodNullable<z.ZodNumber>>
						contextWindow: z.ZodNumber
						supportsImages: z.ZodOptional<z.ZodBoolean>
						supportsComputerUse: z.ZodOptional<z.ZodBoolean>
						supportsPromptCache: z.ZodBoolean
						supportsTemperature: z.ZodOptional<z.ZodBoolean>
						supportsTopP: z.ZodOptional<z.ZodBoolean>
						supportsSystemInstructions: z.ZodOptional<z.ZodBoolean>
						supportsAssistantTool: z.ZodOptional<z.ZodBoolean>
						reasoningTokens: z.ZodOptional<z.ZodBoolean>
						temperature: z.ZodOptional<z.ZodNumber>
						topP: z.ZodOptional<z.ZodNumber>
						topK: z.ZodOptional<z.ZodNumber>
						inputPrice: z.ZodOptional<z.ZodNumber>
						outputPrice: z.ZodOptional<z.ZodNumber>
						cacheWritesPrice: z.ZodOptional<z.ZodNumber>
						cacheReadsPrice: z.ZodOptional<z.ZodNumber>
						description: z.ZodOptional<z.ZodString>
						reasoningEffort: z.ZodOptional<
							z.ZodEnum<{
								low: "low"
								medium: "medium"
								high: "high"
							}>
						>
						thinking: z.ZodOptional<z.ZodBoolean>
					},
					z.core.$strip
				>
			>
		>
		modelTemperature: z.ZodOptional<z.ZodNullable<z.ZodNumber>>
		modelMaxTokens: z.ZodOptional<z.ZodNumber>
		modelMaxThinkingTokens: z.ZodOptional<z.ZodNumber>
		includeMaxTokens: z.ZodOptional<z.ZodBoolean>
		fakeAi: z.ZodOptional<z.ZodUnknown>
	},
	z.core.$strip
>
type ProviderSettings = z.infer<typeof providerSettingsSchema>
/**
 * GlobalSettings
 */
declare const globalSettingsSchema: z.ZodObject<
	{
		currentApiConfigName: z.ZodOptional<z.ZodString>
		listApiConfigMeta: z.ZodOptional<
			z.ZodArray<
				z.ZodObject<
					{
						id: z.ZodString
						name: z.ZodString
						apiProvider: z.ZodOptional<
							z.ZodEnum<{
								anthropic: "anthropic"
								glama: "glama"
								openrouter: "openrouter"
								bedrock: "bedrock"
								vertex: "vertex"
								openai: "openai"
								ollama: "ollama"
								"vscode-lm": "vscode-lm"
								lmstudio: "lmstudio"
								gemini: "gemini"
								"openai-native": "openai-native"
								mistral: "mistral"
								deepseek: "deepseek"
								unbound: "unbound"
								requesty: "requesty"
								"human-relay": "human-relay"
								"fake-ai": "fake-ai"
							}>
						>
					},
					z.core.$strip
				>
			>
		>
		pinnedApiConfigs: z.ZodOptional<z.ZodRecord<z.ZodString, z.ZodBoolean>>
		lastShownAnnouncementId: z.ZodOptional<z.ZodString>
		customInstructions: z.ZodOptional<z.ZodString>
		taskHistory: z.ZodOptional<
			z.ZodArray<
				z.ZodObject<
					{
						id: z.ZodString
						number: z.ZodNumber
						ts: z.ZodNumber
						task: z.ZodString
						tokensIn: z.ZodNumber
						tokensOut: z.ZodNumber
						cacheWrites: z.ZodOptional<z.ZodNumber>
						cacheReads: z.ZodOptional<z.ZodNumber>
						totalCost: z.ZodNumber
						size: z.ZodOptional<z.ZodNumber>
					},
					z.core.$strip
				>
			>
		>
		autoApprovalEnabled: z.ZodOptional<z.ZodBoolean>
		alwaysAllowReadOnly: z.ZodOptional<z.ZodBoolean>
		alwaysAllowReadOnlyOutsideWorkspace: z.ZodOptional<z.ZodBoolean>
		alwaysAllowWrite: z.ZodOptional<z.ZodBoolean>
		alwaysAllowWriteOutsideWorkspace: z.ZodOptional<z.ZodBoolean>
		writeDelayMs: z.ZodOptional<z.ZodNumber>
		alwaysAllowBrowser: z.ZodOptional<z.ZodBoolean>
		alwaysApproveResubmit: z.ZodOptional<z.ZodBoolean>
		requestDelaySeconds: z.ZodOptional<z.ZodNumber>
		alwaysAllowMcp: z.ZodOptional<z.ZodBoolean>
		alwaysAllowModeSwitch: z.ZodOptional<z.ZodBoolean>
		alwaysAllowSubtasks: z.ZodOptional<z.ZodBoolean>
		alwaysAllowExecute: z.ZodOptional<z.ZodBoolean>
		allowedCommands: z.ZodOptional<z.ZodArray<z.ZodString>>
		browserToolEnabled: z.ZodOptional<z.ZodBoolean>
		browserViewportSize: z.ZodOptional<z.ZodString>
		screenshotQuality: z.ZodOptional<z.ZodNumber>
		remoteBrowserEnabled: z.ZodOptional<z.ZodBoolean>
		remoteBrowserHost: z.ZodOptional<z.ZodString>
		cachedChromeHostUrl: z.ZodOptional<z.ZodString>
		enableCheckpoints: z.ZodOptional<z.ZodBoolean>
		checkpointStorage: z.ZodOptional<
			z.ZodEnum<{
				task: "task"
				workspace: "workspace"
			}>
		>
		ttsEnabled: z.ZodOptional<z.ZodBoolean>
		ttsSpeed: z.ZodOptional<z.ZodNumber>
		soundEnabled: z.ZodOptional<z.ZodBoolean>
		soundVolume: z.ZodOptional<z.ZodNumber>
		maxOpenTabsContext: z.ZodOptional<z.ZodNumber>
		maxWorkspaceFiles: z.ZodOptional<z.ZodNumber>
		showTheaIgnoredFiles: z.ZodOptional<z.ZodBoolean>
		maxReadFileLine: z.ZodOptional<z.ZodNumber>
		terminalOutputLineLimit: z.ZodOptional<z.ZodNumber>
		terminalShellIntegrationTimeout: z.ZodOptional<z.ZodNumber>
		rateLimitSeconds: z.ZodOptional<z.ZodNumber>
		diffEnabled: z.ZodOptional<z.ZodBoolean>
		fuzzyMatchThreshold: z.ZodOptional<z.ZodNumber>
		experiments: z.ZodOptional<
			z.ZodObject<
				{
					search_and_replace: z.ZodBoolean
					experimentalDiffStrategy: z.ZodBoolean
					insert_content: z.ZodBoolean
					powerSteering: z.ZodBoolean
				},
				z.core.$strip
			>
		>
		language: z.ZodOptional<
			z.ZodEnum<{
				ca: "ca"
				de: "de"
				en: "en"
				es: "es"
				fr: "fr"
				hi: "hi"
				it: "it"
				ja: "ja"
				ko: "ko"
				pl: "pl"
				"pt-BR": "pt-BR"
				tr: "tr"
				vi: "vi"
				"zh-CN": "zh-CN"
				"zh-TW": "zh-TW"
			}>
		>
		telemetrySetting: z.ZodOptional<
			z.ZodEnum<{
				unset: "unset"
				enabled: "enabled"
				disabled: "disabled"
			}>
		>
		mcpEnabled: z.ZodOptional<z.ZodBoolean>
		enableMcpServerCreation: z.ZodOptional<z.ZodBoolean>
		mode: z.ZodOptional<z.ZodString>
		modeApiConfigs: z.ZodOptional<z.ZodRecord<z.ZodString, z.ZodString>>
		customModes: z.ZodOptional<
			z.ZodArray<
				z.ZodObject<
					{
						slug: z.ZodString
						name: z.ZodString
						roleDefinition: z.ZodString
						customInstructions: z.ZodOptional<z.ZodString>
						groups: z.ZodArray<
							z.ZodUnion<
								readonly [
									z.ZodEnum<{
										read: "read"
										edit: "edit"
										browser: "browser"
										command: "command"
										mcp: "mcp"
										modes: "modes"
									}>,
									z.ZodTuple<
										[
											z.ZodEnum<{
												read: "read"
												edit: "edit"
												browser: "browser"
												command: "command"
												mcp: "mcp"
												modes: "modes"
											}>,
											z.ZodObject<
												{
													fileRegex: z.ZodOptional<z.ZodString>
													description: z.ZodOptional<z.ZodString>
												},
												z.core.$strip
											>,
										],
										null
									>,
								]
							>
						>
						source: z.ZodOptional<
							z.ZodEnum<{
								global: "global"
								project: "project"
							}>
						>
					},
					z.core.$strip
				>
			>
		>
		customModePrompts: z.ZodOptional<
			z.ZodRecord<
				z.ZodString,
				z.ZodOptional<
					z.ZodObject<
						{
							roleDefinition: z.ZodOptional<z.ZodString>
							customInstructions: z.ZodOptional<z.ZodString>
						},
						z.core.$strip
					>
				>
			>
		>
		customSupportPrompts: z.ZodOptional<z.ZodRecord<z.ZodString, z.ZodOptional<z.ZodString>>>
		enhancementApiConfigId: z.ZodOptional<z.ZodString>
	},
	z.core.$strip
>
type GlobalSettings = z.infer<typeof globalSettingsSchema>
/**
 * TheaMessage // Renamed
 */
declare const clineMessageSchema: z.ZodObject<
	{
		ts: z.ZodNumber
		type: z.ZodUnion<readonly [z.ZodLiteral<"ask">, z.ZodLiteral<"say">]>
		ask: z.ZodOptional<
			z.ZodEnum<{
				command: "command"
				followup: "followup"
				command_output: "command_output"
				completion_result: "completion_result"
				tool: "tool"
				api_req_failed: "api_req_failed"
				resume_task: "resume_task"
				resume_completed_task: "resume_completed_task"
				mistake_limit_reached: "mistake_limit_reached"
				browser_action_launch: "browser_action_launch"
				use_mcp_server: "use_mcp_server"
				finishTask: "finishTask"
			}>
		>
		say: z.ZodOptional<
			z.ZodEnum<{
				command: "command"
				task: "task"
				error: "error"
				command_output: "command_output"
				completion_result: "completion_result"
				tool: "tool"
				api_req_started: "api_req_started"
				api_req_finished: "api_req_finished"
				api_req_retried: "api_req_retried"
				api_req_retry_delayed: "api_req_retry_delayed"
				api_req_deleted: "api_req_deleted"
				text: "text"
				reasoning: "reasoning"
				user_feedback: "user_feedback"
				user_feedback_diff: "user_feedback_diff"
				shell_integration_warning: "shell_integration_warning"
				browser_action: "browser_action"
				browser_action_result: "browser_action_result"
				mcp_server_request_started: "mcp_server_request_started"
				mcp_server_response: "mcp_server_response"
				new_task_started: "new_task_started"
				new_task: "new_task"
				checkpoint_saved: "checkpoint_saved"
				theaignore_error: "theaignore_error"
			}>
		>
		text: z.ZodOptional<z.ZodString>
		images: z.ZodOptional<z.ZodArray<z.ZodString>>
		partial: z.ZodOptional<z.ZodBoolean>
		reasoning: z.ZodOptional<z.ZodString>
		conversationHistoryIndex: z.ZodOptional<z.ZodNumber>
		checkpoint: z.ZodOptional<z.ZodRecord<z.ZodString, z.ZodUnknown>>
		progressStatus: z.ZodOptional<
			z.ZodObject<
				{
					icon: z.ZodOptional<z.ZodString>
					text: z.ZodOptional<z.ZodString>
				},
				z.core.$strip
			>
		>
	},
	z.core.$strip
>
type TheaMessage = z.infer<typeof clineMessageSchema>
/**
 * TokenUsage
 */
declare const tokenUsageSchema: z.ZodObject<
	{
		totalTokensIn: z.ZodNumber
		totalTokensOut: z.ZodNumber
		totalCacheWrites: z.ZodOptional<z.ZodNumber>
		totalCacheReads: z.ZodOptional<z.ZodNumber>
		totalCost: z.ZodNumber
		contextTokens: z.ZodNumber
	},
	z.core.$strip
>
type TokenUsage = z.infer<typeof tokenUsageSchema>

type TheaCodeSettings = GlobalSettings & ProviderSettings

interface TheaCodeEvents {
	message: [
		{
			taskId: string
			action: "created" | "updated"
			message: TheaMessage
		},
	]
	taskCreated: [taskId: string]
	taskStarted: [taskId: string]
	taskPaused: [taskId: string]
	taskUnpaused: [taskId: string]
	taskAskResponded: [taskId: string]
	taskAborted: [taskId: string]
	taskSpawned: [taskId: string, childTaskId: string]
	taskCompleted: [taskId: string, usage: TokenUsage]
	taskTokenUsageUpdated: [taskId: string, usage: TokenUsage]
}
interface TheaCodeAPI extends EventEmitter<TheaCodeEvents> {
	/**
	 * Starts a new task with an optional initial message and images.
	 * @param task Optional initial task message.
	 * @param images Optional array of image data URIs (e.g., "data:image/webp;base64,...").
	 * @returns The ID of the new task.
	 */
	startNewTask(task?: string, images?: string[]): Promise<string>
	/**
	 * Returns the current task stack.
	 * @returns An array of task IDs.
	 */
	getCurrentTaskStack(): string[]
	/**
	 * Clears the current task.
	 */
	clearCurrentTask(lastMessage?: string): Promise<void>
	/**
	 * Cancels the current task.
	 */
	cancelCurrentTask(): Promise<void>
	/**
	 * Sends a message to the current task.
	 * @param message Optional message to send.
	 * @param images Optional array of image data URIs (e.g., "data:image/webp;base64,...").
	 */
	sendMessage(message?: string, images?: string[]): Promise<void>
	/**
	 * Simulates pressing the primary button in the chat interface.
	 */
	pressPrimaryButton(): Promise<void>
	/**
	 * Simulates pressing the secondary button in the chat interface.
	 */
	pressSecondaryButton(): Promise<void>
	/**
	 * Returns the current configuration.
	 * @returns The current configuration.
	 */
	getConfiguration(): TheaCodeSettings
	/**
	 * Returns the value of a configuration key.
	 * @param key The key of the configuration value to return.
	 * @returns The value of the configuration key.
	 */
	getConfigurationValue<K extends keyof TheaCodeSettings>(key: K): TheaCodeSettings[K]
	/**
	 * Sets the configuration for the current task.
	 * @param values An object containing key-value pairs to set.
	 */
	setConfiguration(values: TheaCodeSettings): Promise<void>
	/**
	 * Sets the value of a configuration key.
	 * @param key The key of the configuration value to set.
	 * @param value The value to set.
	 */
	setConfigurationValue<K extends keyof TheaCodeSettings>(key: K, value: TheaCodeSettings[K]): Promise<void>
	/**
	 * Returns true if the API is ready to use.
	 */
	isReady(): boolean
	/**
	 * Returns the messages for a given task.
	 * @param taskId The ID of the task.
	 * @returns An array of TheaMessage objects.
	 */
	getMessages(taskId: string): TheaMessage[]
	/**
	 * Returns the token usage for a given task.
	 * @param taskId The ID of the task.
	 * @returns A TokenUsage object.
	 */
	getTokenUsage(taskId: string): TokenUsage
	/**
	 * Logs a message to the output channel.
	 * @param message The message to log.
	 */
	log(message: string): void
}

export type { GlobalSettings, ProviderSettings, TheaCodeAPI, TheaCodeEvents, TheaCodeSettings, TheaMessage, TokenUsage }
