import React, { memo, useCallback, useEffect, useMemo, useState } from "react"
import { useAppTranslation } from "@/i18n/TranslationContext"
import { Trans } from "react-i18next"
import { getRequestyAuthUrl, getOpenRouterAuthUrl, getGlamaAuthUrl } from "@/oauth/urls.ts"
import { useDebounce, useEvent } from "react-use"
import { LanguageModelChatSelector } from "vscode"
import { Checkbox } from "vscrui"
import { Button } from "@/components/ui"
import { VSCodeTextField, VSCodeLink } from "../ui/vscode-components"
// Simple replacements for VSCodeRadio and VSCodeRadioGroup
const VSCodeRadio: React.FC<{ value: string; children: React.ReactNode }> = ({ value, children, ...props }) => (
	<label style={{ display: "inline-flex", alignItems: "center", margin: "4px 0" }}>
		<input type="radio" value={value} {...props} />
		<span style={{ marginLeft: "4px" }}>{children}</span>
	</label>
)
const VSCodeRadioGroup: React.FC<{
	onChange?: (e: React.ChangeEvent<HTMLInputElement>) => void
	children: React.ReactNode
}> = ({ onChange, children, ...props }) => (
	<div role="radiogroup" onChange={onChange} {...props}>
		{children}
	</div>
)
import { ExternalLinkIcon } from "@radix-ui/react-icons"

import {
	ApiConfiguration,
	ModelInfo,
	anthropicDefaultModelId,
	anthropicModels,
	azureOpenAiDefaultApiVersion,
	bedrockDefaultModelId,
	bedrockModels,
	deepSeekDefaultModelId,
	deepSeekModels,
	geminiDefaultModelId,
	geminiModels,
	glamaDefaultModelId,
	glamaDefaultModelInfo,
	mistralDefaultModelId,
	mistralModels,
	openAiModelInfoSaneDefaults,
	openAiNativeDefaultModelId,
	openAiNativeModels,
	openRouterDefaultModelId,
	openRouterDefaultModelInfo,
	vertexDefaultModelId,
	vertexModels,
	unboundDefaultModelId,
	unboundDefaultModelInfo,
	requestyDefaultModelId,
	requestyDefaultModelInfo,
	ApiProvider,
} from "../../../../src/shared/api"
import { ExtensionMessage } from "../../../../src/shared/ExtensionMessage"

import { vscode } from "@/utils/vscode"
import { validateApiConfiguration, validateModelId, validateBedrockArn } from "@/utils/validate"
import {
	useOpenRouterModelProviders,
	OPENROUTER_DEFAULT_PROVIDER_NAME,
} from "@/components/ui/hooks/useOpenRouterModelProviders"
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue, SelectSeparator } from "@/components/ui"

import { MODELS_BY_PROVIDER, PROVIDERS, AWS_REGIONS, VERTEX_REGIONS } from "./constants"
import { VSCodeButtonLink } from "../common/VSCodeButtonLink"
import { ModelInfoView } from "./ModelInfoView"
import { ModelPicker } from "./ModelPicker"
import { TemperatureControl } from "./TemperatureControl"
import { ApiErrorMessage } from "./ApiErrorMessage"
import { ThinkingBudget } from "./ThinkingBudget"
import { R1FormatSetting } from "./R1FormatSetting"
import { OpenRouterBalanceDisplay } from "./OpenRouterBalanceDisplay"
import { RequestyBalanceDisplay } from "./RequestyBalanceDisplay"

interface ApiOptionsProps {
	uriScheme: string | undefined
	apiConfiguration: ApiConfiguration
	setApiConfigurationField: <K extends keyof ApiConfiguration>(field: K, value: ApiConfiguration[K]) => void
	fromWelcomeView?: boolean
	errorMessage: string | undefined
	setErrorMessage: React.Dispatch<React.SetStateAction<string | undefined>>
}

const ApiOptions = ({
	uriScheme,
	apiConfiguration,
	setApiConfigurationField,
	fromWelcomeView,
	errorMessage,
	setErrorMessage,
}: ApiOptionsProps) => {
	const { t } = useAppTranslation()

	const [ollamaModels, setOllamaModels] = useState<string[]>([])
	const [lmStudioModels, setLmStudioModels] = useState<string[]>([])
	const [vsCodeLmModels, setVsCodeLmModels] = useState<LanguageModelChatSelector[]>([])

	const [openRouterModels, setOpenRouterModels] = useState<Record<string, ModelInfo>>({
		[openRouterDefaultModelId]: openRouterDefaultModelInfo,
	})

	const [glamaModels, setGlamaModels] = useState<Record<string, ModelInfo>>({
		[glamaDefaultModelId]: glamaDefaultModelInfo,
	})

	const [unboundModels, setUnboundModels] = useState<Record<string, ModelInfo>>({
		[unboundDefaultModelId]: unboundDefaultModelInfo,
	})

	const [requestyModels, setRequestyModels] = useState<Record<string, ModelInfo>>({
		[requestyDefaultModelId]: requestyDefaultModelInfo,
	})

	const [openAiModels, setOpenAiModels] = useState<Record<string, ModelInfo> | null>(null)
	
	const [anthropicModels, setAnthropicModels] = useState<Record<string, ModelInfo> | null>(null)
	const [bedrockModels, setBedrockModels] = useState<Record<string, ModelInfo> | null>(null)  
	const [vertexModels, setVertexModels] = useState<Record<string, ModelInfo> | null>(null)
	const [geminiModels, setGeminiModels] = useState<Record<string, ModelInfo> | null>(null)
	const [mistralModels, setMistralModels] = useState<Record<string, ModelInfo> | null>(null)
	const [deepseekModels, setDeepseekModels] = useState<Record<string, ModelInfo> | null>(null)

	const [anthropicBaseUrlSelected, setAnthropicBaseUrlSelected] = useState(!!apiConfiguration?.anthropicBaseUrl)
	const [azureApiVersionSelected, setAzureApiVersionSelected] = useState(!!apiConfiguration?.azureApiVersion)
	const [openRouterBaseUrlSelected, setOpenRouterBaseUrlSelected] = useState(!!apiConfiguration?.openRouterBaseUrl)
	const [googleGeminiBaseUrlSelected, setGoogleGeminiBaseUrlSelected] = useState(
		!!apiConfiguration?.googleGeminiBaseUrl,
	)
	const [isDescriptionExpanded, setIsDescriptionExpanded] = useState(false)
	const noTransform = <T,>(value: T) => value

	const inputEventTransform = <E,>(event: E): string => (event as { target: HTMLInputElement })?.target?.value ?? ""

	const handleInputChange = useCallback(
		<K extends keyof ApiConfiguration, E>(
			field: K,
			transform: (event: E) => ApiConfiguration[K] = inputEventTransform as (event: E) => ApiConfiguration[K],
		) =>
			(event: E | Event) => {
				setApiConfigurationField(field, transform(event as E))
			},
		[setApiConfigurationField],
	)

	const { selectedProvider, selectedModelId, selectedModelInfo } = useMemo(
		() => normalizeApiConfiguration(apiConfiguration, { 
			anthropicModels,
			bedrockModels,
			vertexModels,
			geminiModels,
			mistralModels,
			deepseekModels,
		}),
		[apiConfiguration, anthropicModels, bedrockModels, vertexModels, geminiModels, mistralModels, deepseekModels],
	)

	// Debounced refresh model updates, only executed 250ms after the user
	// stops typing.
	useDebounce(
		() => {
			switch(selectedProvider) {
				case "anthropic":
					vscode.postMessage({ type: "refreshAnthropicModels" })
					break
				case "bedrock":
					vscode.postMessage({ type: "refreshBedrockModels" })
					break
				case "vertex":
					vscode.postMessage({ type: "refreshVertexModels" })
					break
				case "gemini":
					vscode.postMessage({ type: "refreshGeminiModels" })
					break
				case "mistral":
					vscode.postMessage({ type: "refreshMistralModels" })
					break
				case "deepseek":
					vscode.postMessage({ type: "refreshDeepSeekModels" })
					break
				case "openrouter":
					vscode.postMessage({ type: "refreshOpenRouterModels" })
					break
				case "glama":
					vscode.postMessage({ type: "refreshGlamaModels" })
					break
				case "unbound":
					vscode.postMessage({ type: "refreshUnboundModels" })
					break
				case "requesty":
					vscode.postMessage({
						type: "refreshRequestyModels",
						values: { apiKey: apiConfiguration?.requestyApiKey },
					})
					break
				case "openai":
					vscode.postMessage({
						type: "refreshOpenAiModels",
						values: { baseUrl: apiConfiguration?.openAiBaseUrl, apiKey: apiConfiguration?.openAiApiKey },
					})
					break
				case "ollama":
					vscode.postMessage({ type: "requestOllamaModels", text: apiConfiguration?.ollamaBaseUrl })
					break
				case "lmstudio":
					vscode.postMessage({ type: "requestLmStudioModels", text: apiConfiguration?.lmStudioBaseUrl })
					break
				case "vscode-lm":
					vscode.postMessage({ type: "requestVsCodeLmModels" })
					break
			}
		},
		250,
		[
			selectedProvider,
			apiConfiguration?.requestyApiKey,
			apiConfiguration?.openAiBaseUrl,
			apiConfiguration?.openAiApiKey,
			apiConfiguration?.ollamaBaseUrl,
			apiConfiguration?.lmStudioBaseUrl,
		],
	)

	useEffect(() => {
		const apiValidationResult =
			validateApiConfiguration(apiConfiguration) ||
			validateModelId(apiConfiguration, glamaModels, openRouterModels, unboundModels, requestyModels)

		setErrorMessage(apiValidationResult)
	}, [apiConfiguration, glamaModels, openRouterModels, setErrorMessage, unboundModels, requestyModels])

	const { data: openRouterModelProviders } = useOpenRouterModelProviders(apiConfiguration?.openRouterModelId, {
		enabled:
			selectedProvider === "openrouter" &&
			!!apiConfiguration?.openRouterModelId &&
			apiConfiguration.openRouterModelId in openRouterModels,
	})

	const onMessage = useCallback((event: MessageEvent) => {
		const message: ExtensionMessage = event.data

		switch (message.type) {
			case "openRouterModels": {
				const updatedModels = message.openRouterModels ?? {}
				setOpenRouterModels({ [openRouterDefaultModelId]: openRouterDefaultModelInfo, ...updatedModels })
				break
			}
			case "glamaModels": {
				const updatedModels = message.glamaModels ?? {}
				setGlamaModels({ [glamaDefaultModelId]: glamaDefaultModelInfo, ...updatedModels })
				break
			}
			case "unboundModels": {
				const updatedModels = message.unboundModels ?? {}
				setUnboundModels({ [unboundDefaultModelId]: unboundDefaultModelInfo, ...updatedModels })
				break
			}
			case "requestyModels": {
				const updatedModels = message.requestyModels ?? {}
				setRequestyModels({ [requestyDefaultModelId]: requestyDefaultModelInfo, ...updatedModels })
				break
			}
			case "openAiModels": {
				const updatedModels = message.openAiModels ?? []
				setOpenAiModels(Object.fromEntries(updatedModels.map((item) => [item, openAiModelInfoSaneDefaults])))
				break
			}
			case "anthropicModels": {
				const updatedModels = message.anthropicModels ?? {}
				setAnthropicModels(updatedModels)
				break
			}
			case "bedrockModels": {
				const updatedModels = message.bedrockModels ?? {}
				setBedrockModels(updatedModels)
				break
			}
			case "vertexModels": {
				const updatedModels = message.vertexModels ?? {}
				setVertexModels(updatedModels)
				break
			}
			case "geminiModels": {
				const updatedModels = message.geminiModels ?? {}
				setGeminiModels(updatedModels)
				break
			}
			case "mistralModels": {
				const updatedModels = message.mistralModels ?? {}
				setMistralModels(updatedModels)
				break
			}
			case "deepseekModels": {
				const updatedModels = message.deepseekModels ?? {}
				setDeepseekModels(updatedModels)
				break
			}
			case "ollamaModels":
				{
					const newModels = message.ollamaModels ?? []
					setOllamaModels(newModels)
				}
				break
			case "lmStudioModels":
				{
					const newModels = message.lmStudioModels ?? []
					setLmStudioModels(newModels)
				}
				break
			case "vsCodeLmModels":
				{
					const newModels = message.vsCodeLmModels ?? []
					setVsCodeLmModels(newModels)
				}
				break
		}
	}, [])

	useEvent("message", onMessage)

	const selectedProviderModelOptions = useMemo(
		() => {
			// Map of providers to their dynamic models
			const dynamicProviders = {
				anthropic: anthropicModels,
				bedrock: bedrockModels,
				vertex: vertexModels,
				gemini: geminiModels,
				mistral: mistralModels,
				deepseek: deepseekModels,
			}
			
			// Check if the current provider has dynamic models available
			if (selectedProvider in dynamicProviders) {
				const models = dynamicProviders[selectedProvider as keyof typeof dynamicProviders]
				if (models && Object.keys(models).length > 0) {
					return Object.keys(models).map((modelId) => ({
						value: modelId,
						label: modelId,
					}))
				}
			}
			
			// Fallback to static models (for providers that haven't been migrated yet)
			return MODELS_BY_PROVIDER[selectedProvider]
				? Object.keys(MODELS_BY_PROVIDER[selectedProvider]).map((modelId) => ({
						value: modelId,
						label: modelId,
					}))
				: []
		},
		[selectedProvider, anthropicModels, bedrockModels, vertexModels, geminiModels, mistralModels, deepseekModels],
	)

	// Base URL for provider documentation
	const DOC_BASE_URL = "https://docs.thea-placeholder.com/providers"

	// Custom URL path mappings for providers with different slugs
	const providerUrlSlugs: Record<string, string> = {
		"openai-native": "openai",
		openai: "openai-compatible",
	}

	// Helper function to get provider display name from PROVIDERS constant
	const getProviderDisplayName = (providerKey: string): string | undefined => {
		const provider = PROVIDERS.find((p) => p.value === providerKey)
		return provider?.label
	}

	// Helper function to get the documentation URL and name for the currently selected provider
	const getSelectedProviderDocUrl = (): { url: string; name: string } | undefined => {
		const displayName = getProviderDisplayName(selectedProvider)
		if (!displayName) {
			return undefined
		}

		// Get the URL slug - use custom mapping if available, otherwise use the provider key
		const urlSlug = providerUrlSlugs[selectedProvider] || selectedProvider

		return {
			url: `${DOC_BASE_URL}/${urlSlug}`,
			name: displayName,
		}
	}

	return (
		<div className="flex flex-col gap-3">
			<div className="flex flex-col gap-1 relative">
				<div className="flex justify-between items-center">
					<label className="block font-medium mb-1">{t("settings:providers.apiProvider")}</label>
					{getSelectedProviderDocUrl() && (
						<div className="text-xs text-vscode-descriptionForeground">
							<VSCodeLink
								href={getSelectedProviderDocUrl()!.url}
								className="hover:text-vscode-foreground">
								{t("settings:providers.providerDocumentation", {
									provider: getSelectedProviderDocUrl()!.name,
								})}
							</VSCodeLink>
						</div>
					)}
				</div>
				<Select
					value={selectedProvider}
					onValueChange={(value) => setApiConfigurationField("apiProvider", value as ApiProvider)}>
					<SelectTrigger className="w-full">
						<SelectValue placeholder={t("settings:common.select")} />
					</SelectTrigger>
					<SelectContent>
						<SelectItem value="openrouter">OpenRouter</SelectItem>
						<SelectSeparator />
						{PROVIDERS.filter((p) => p.value !== "openrouter").map(({ value, label }) => (
							<SelectItem key={value} value={value}>
								{label}
							</SelectItem>
						))}
					</SelectContent>
				</Select>
			</div>

			{errorMessage && <ApiErrorMessage errorMessage={errorMessage} />}

			{selectedProvider === "openrouter" && (
				<>
					<VSCodeTextField
						value={apiConfiguration?.openRouterApiKey || ""}
						type="password"
						onInput={handleInputChange("openRouterApiKey")}
						placeholder={t("settings:placeholders.apiKey")}
						className="w-full">
						<div className="flex justify-between items-center mb-1">
							<label className="block font-medium">{t("settings:providers.openRouterApiKey")}</label>
							{apiConfiguration?.openRouterApiKey && (
								<OpenRouterBalanceDisplay
									apiKey={apiConfiguration.openRouterApiKey}
									baseUrl={apiConfiguration.openRouterBaseUrl}
								/>
							)}
						</div>
					</VSCodeTextField>
					<div className="text-sm text-vscode-descriptionForeground -mt-2">
						{t("settings:providers.apiKeyStorageNotice")}
					</div>
					{!apiConfiguration?.openRouterApiKey && (
						<VSCodeButtonLink href={getOpenRouterAuthUrl(uriScheme)} style={{ width: "100%" }}>
							{t("settings:providers.getOpenRouterApiKey")}
						</VSCodeButtonLink>
					)}
					{!fromWelcomeView && (
						<>
							<div>
								<Checkbox
									checked={openRouterBaseUrlSelected}
									onChange={(checked: boolean) => {
										setOpenRouterBaseUrlSelected(checked)

										if (!checked) {
											setApiConfigurationField("openRouterBaseUrl", "")
										}
									}}>
									{t("settings:providers.useCustomBaseUrl")}
								</Checkbox>
								{openRouterBaseUrlSelected && (
									<VSCodeTextField
										value={apiConfiguration?.openRouterBaseUrl || ""}
										type="url"
										onInput={handleInputChange("openRouterBaseUrl")}
										placeholder="Default: https://openrouter.ai/api/v1"
										className="w-full mt-1"
									/>
								)}
							</div>
							<Checkbox
								checked={apiConfiguration?.openRouterUseMiddleOutTransform ?? true}
								onChange={handleInputChange("openRouterUseMiddleOutTransform", noTransform)}>
								<Trans
									i18nKey="settings:providers.openRouterTransformsText"
									components={{
										a: (
											<a
												href="https://openrouter.ai/docs/transforms"
												target="_blank"
												rel="noopener noreferrer"
											/>
										),
									}}
								/>
							</Checkbox>
						</>
					)}
				</>
			)}

			{selectedProvider === "anthropic" && (
				<>
					<VSCodeTextField
						value={apiConfiguration?.apiKey || ""}
						type="password"
						onInput={handleInputChange("apiKey")}
						placeholder={t("settings:placeholders.apiKey")}
						className="w-full">
						<label className="block font-medium mb-1">{t("settings:providers.anthropicApiKey")}</label>
					</VSCodeTextField>
					<div className="text-sm text-vscode-descriptionForeground -mt-2">
						{t("settings:providers.apiKeyStorageNotice")}
					</div>
					{!apiConfiguration?.apiKey && (
						<VSCodeButtonLink href="https://console.anthropic.com/settings/keys">
							{t("settings:providers.getAnthropicApiKey")}
						</VSCodeButtonLink>
					)}
					<div>
						<Checkbox
							checked={anthropicBaseUrlSelected}
							onChange={(checked: boolean) => {
								setAnthropicBaseUrlSelected(checked)

								if (!checked) {
									setApiConfigurationField("anthropicBaseUrl", "")
								}
							}}>
							{t("settings:providers.useCustomBaseUrl")}
						</Checkbox>
						{anthropicBaseUrlSelected && (
							<VSCodeTextField
								value={apiConfiguration?.anthropicBaseUrl || ""}
								type="url"
								onInput={handleInputChange("anthropicBaseUrl")}
								placeholder="https://api.anthropic.com"
								className="w-full mt-1"
							/>
						)}
					</div>
				</>
			)}

			{selectedProvider === "glama" && (
				<>
					<VSCodeTextField
						value={apiConfiguration?.glamaApiKey || ""}
						type="password"
						onInput={handleInputChange("glamaApiKey")}
						placeholder={t("settings:placeholders.apiKey")}
						className="w-full">
						<label className="block font-medium mb-1">{t("settings:providers.glamaApiKey")}</label>
					</VSCodeTextField>
					<div className="text-sm text-vscode-descriptionForeground -mt-2">
						{t("settings:providers.apiKeyStorageNotice")}
					</div>
					{!apiConfiguration?.glamaApiKey && (
						<VSCodeButtonLink href={getGlamaAuthUrl(uriScheme)} style={{ width: "100%" }}>
							{t("settings:providers.getGlamaApiKey")}
						</VSCodeButtonLink>
					)}
				</>
			)}

			{selectedProvider === "requesty" && (
				<>
					<VSCodeTextField
						value={apiConfiguration?.requestyApiKey || ""}
						type="password"
						onInput={handleInputChange("requestyApiKey")}
						placeholder={t("settings:providers.getRequestyApiKey")}
						className="w-full">
						<div className="flex justify-between items-center mb-1">
							<label className="block font-medium">{t("settings:providers.requestyApiKey")}</label>
							{apiConfiguration?.requestyApiKey && (
								<RequestyBalanceDisplay apiKey={apiConfiguration.requestyApiKey} />
							)}
						</div>
					</VSCodeTextField>
					<div className="text-sm text-vscode-descriptionForeground -mt-2">
						{t("settings:providers.apiKeyStorageNotice")}
					</div>
					{!apiConfiguration?.requestyApiKey && (
						<VSCodeButtonLink href={getRequestyAuthUrl(uriScheme)} style={{ width: "100%" }}>
							{t("settings:providers.getRequestyApiKey")}
						</VSCodeButtonLink>
					)}
				</>
			)}

			{selectedProvider === "openai-native" && (
				<>
					<VSCodeTextField
						value={apiConfiguration?.openAiNativeApiKey || ""}
						type="password"
						onInput={handleInputChange("openAiNativeApiKey")}
						placeholder={t("settings:placeholders.apiKey")}
						className="w-full">
						<label className="block font-medium mb-1">{t("settings:providers.openAiApiKey")}</label>
					</VSCodeTextField>
					<div className="text-sm text-vscode-descriptionForeground -mt-2">
						{t("settings:providers.apiKeyStorageNotice")}
					</div>
					{!apiConfiguration?.openAiNativeApiKey && (
						<VSCodeButtonLink href="https://platform.openai.com/api-keys">
							{t("settings:providers.getOpenAiApiKey")}
						</VSCodeButtonLink>
					)}
				</>
			)}

			{selectedProvider === "mistral" && (
				<>
					<VSCodeTextField
						value={apiConfiguration?.mistralApiKey || ""}
						type="password"
						onInput={handleInputChange("mistralApiKey")}
						placeholder={t("settings:placeholders.apiKey")}
						className="w-full">
						<span className="font-medium">{t("settings:providers.mistralApiKey")}</span>
					</VSCodeTextField>
					<div className="text-sm text-vscode-descriptionForeground -mt-2">
						{t("settings:providers.apiKeyStorageNotice")}
					</div>
					{!apiConfiguration?.mistralApiKey && (
						<VSCodeButtonLink href="https://console.mistral.ai/">
							{t("settings:providers.getMistralApiKey")}
						</VSCodeButtonLink>
					)}
					{(apiConfiguration?.apiModelId?.startsWith("codestral-") ||
						(!apiConfiguration?.apiModelId && mistralDefaultModelId.startsWith("codestral-"))) && (
						<>
							<VSCodeTextField
								value={apiConfiguration?.mistralCodestralUrl || ""}
								type="url"
								onInput={handleInputChange("mistralCodestralUrl")}
								placeholder="https://codestral.mistral.ai"
								className="w-full">
								<label className="block font-medium mb-1">
									{t("settings:providers.codestralBaseUrl")}
								</label>
							</VSCodeTextField>
							<div className="text-sm text-vscode-descriptionForeground -mt-2">
								{t("settings:providers.codestralBaseUrlDesc")}
							</div>
						</>
					)}
				</>
			)}

			{selectedProvider === "bedrock" && (
				<>
					{" "}
					<VSCodeRadioGroup
						onChange={handleInputChange(
							"awsUseProfile",
							(e) => (e.target as HTMLInputElement).value === "profile",
						)}>
						<VSCodeRadio value="credentials">{t("settings:providers.awsCredentials")}</VSCodeRadio>
						<VSCodeRadio value="profile">{t("settings:providers.awsProfile")}</VSCodeRadio>
					</VSCodeRadioGroup>
					<div className="text-sm text-vscode-descriptionForeground -mt-3">
						{t("settings:providers.apiKeyStorageNotice")}
					</div>
					{apiConfiguration?.awsUseProfile ? (
						<VSCodeTextField
							value={apiConfiguration?.awsProfile || ""}
							onInput={handleInputChange("awsProfile")}
							placeholder={t("settings:placeholders.profileName")}
							className="w-full">
							<label className="block font-medium mb-1">{t("settings:providers.awsProfileName")}</label>
						</VSCodeTextField>
					) : (
						<>
							<VSCodeTextField
								value={apiConfiguration?.awsAccessKey || ""}
								type="password"
								onInput={handleInputChange("awsAccessKey")}
								placeholder={t("settings:placeholders.accessKey")}
								className="w-full">
								<label className="block font-medium mb-1">{t("settings:providers.awsAccessKey")}</label>
							</VSCodeTextField>
							<VSCodeTextField
								value={apiConfiguration?.awsSecretKey || ""}
								type="password"
								onInput={handleInputChange("awsSecretKey")}
								placeholder={t("settings:placeholders.secretKey")}
								className="w-full">
								<label className="block font-medium mb-1">{t("settings:providers.awsSecretKey")}</label>
							</VSCodeTextField>
							<VSCodeTextField
								value={apiConfiguration?.awsSessionToken || ""}
								type="password"
								onInput={handleInputChange("awsSessionToken")}
								placeholder={t("settings:placeholders.sessionToken")}
								className="w-full">
								<label className="block font-medium mb-1">
									{t("settings:providers.awsSessionToken")}
								</label>
							</VSCodeTextField>
						</>
					)}
					<div>
						<label className="block font-medium mb-1">{t("settings:providers.awsRegion")}</label>
						<Select
							value={apiConfiguration?.awsRegion || ""}
							onValueChange={(value) => setApiConfigurationField("awsRegion", value)}>
							<SelectTrigger className="w-full">
								<SelectValue placeholder={t("settings:common.select")} />
							</SelectTrigger>
							<SelectContent>
								{AWS_REGIONS.map(({ value, label }) => (
									<SelectItem key={value} value={value}>
										{label}
									</SelectItem>
								))}
							</SelectContent>
						</Select>
					</div>
					<Checkbox
						checked={apiConfiguration?.awsUseCrossRegionInference || false}
						onChange={handleInputChange("awsUseCrossRegionInference", noTransform)}>
						{t("settings:providers.awsCrossRegion")}
					</Checkbox>
				</>
			)}

			{selectedProvider === "vertex" && (
				<>
					<div className="text-sm text-vscode-descriptionForeground">
						<div>{t("settings:providers.googleCloudSetup.title")}</div>
						<div>
							<VSCodeLink
								href="https://cloud.google.com/vertex-ai/generative-ai/docs/partner-models/use-claude#before_you_begin"
								className="text-sm">
								{t("settings:providers.googleCloudSetup.step1")}
							</VSCodeLink>
						</div>
						<div>
							<VSCodeLink
								href="https://cloud.google.com/docs/authentication/provide-credentials-adc#google-idp"
								className="text-sm">
								{t("settings:providers.googleCloudSetup.step2")}
							</VSCodeLink>
						</div>
						<div>
							<VSCodeLink
								href="https://developers.google.com/workspace/guides/create-credentials?hl=en#service-account"
								className="text-sm">
								{t("settings:providers.googleCloudSetup.step3")}
							</VSCodeLink>
						</div>
					</div>
					<VSCodeTextField
						value={apiConfiguration?.vertexJsonCredentials || ""}
						onInput={handleInputChange("vertexJsonCredentials")}
						placeholder={t("settings:placeholders.credentialsJson")}
						className="w-full">
						<label className="block font-medium mb-1">
							{t("settings:providers.googleCloudCredentials")}
						</label>
					</VSCodeTextField>
					<VSCodeTextField
						value={apiConfiguration?.vertexKeyFile || ""}
						onInput={handleInputChange("vertexKeyFile")}
						placeholder={t("settings:placeholders.keyFilePath")}
						className="w-full">
						<label className="block font-medium mb-1">{t("settings:providers.googleCloudKeyFile")}</label>
					</VSCodeTextField>
					<VSCodeTextField
						value={apiConfiguration?.vertexProjectId || ""}
						onInput={handleInputChange("vertexProjectId")}
						placeholder={t("settings:placeholders.projectId")}
						className="w-full">
						<label className="block font-medium mb-1">{t("settings:providers.googleCloudProjectId")}</label>
					</VSCodeTextField>
					<div>
						<label className="block font-medium mb-1">{t("settings:providers.googleCloudRegion")}</label>
						<Select
							value={apiConfiguration?.vertexRegion || ""}
							onValueChange={(value) => setApiConfigurationField("vertexRegion", value)}>
							<SelectTrigger className="w-full">
								<SelectValue placeholder={t("settings:common.select")} />
							</SelectTrigger>
							<SelectContent>
								{VERTEX_REGIONS.map(({ value, label }) => (
									<SelectItem key={value} value={value}>
										{label}
									</SelectItem>
								))}
							</SelectContent>
						</Select>
					</div>
				</>
			)}

			{selectedProvider === "gemini" && (
				<>
					<VSCodeTextField
						value={apiConfiguration?.geminiApiKey || ""}
						type="password"
						onInput={handleInputChange("geminiApiKey")}
						placeholder={t("settings:placeholders.apiKey")}
						className="w-full">
						<label className="block font-medium mb-1">{t("settings:providers.geminiApiKey")}</label>
					</VSCodeTextField>
					<div className="text-sm text-vscode-descriptionForeground -mt-2">
						{t("settings:providers.apiKeyStorageNotice")}
					</div>
					{!apiConfiguration?.geminiApiKey && (
						<VSCodeButtonLink href="https://ai.google.dev/">
							{t("settings:providers.getGeminiApiKey")}
						</VSCodeButtonLink>
					)}
					<div>
						<Checkbox
							checked={googleGeminiBaseUrlSelected}
							onChange={(checked: boolean) => {
								setGoogleGeminiBaseUrlSelected(checked)

								if (!checked) {
									setApiConfigurationField("googleGeminiBaseUrl", "")
								}
							}}>
							{t("settings:providers.useCustomBaseUrl")}
						</Checkbox>
						{googleGeminiBaseUrlSelected && (
							<VSCodeTextField
								value={apiConfiguration?.googleGeminiBaseUrl || ""}
								type="url"
								onInput={handleInputChange("googleGeminiBaseUrl")}
								placeholder={t("settings:defaults.geminiUrl")}
								className="w-full mt-1"
							/>
						)}
					</div>
				</>
			)}

			{selectedProvider === "openai" && (
				<>
					<VSCodeTextField
						value={apiConfiguration?.openAiBaseUrl || ""}
						type="url"
						onInput={handleInputChange("openAiBaseUrl")}
						placeholder={t("settings:placeholders.baseUrl")}
						className="w-full">
						<label className="block font-medium mb-1">{t("settings:providers.openAiBaseUrl")}</label>
					</VSCodeTextField>
					<VSCodeTextField
						value={apiConfiguration?.openAiApiKey || ""}
						type="password"
						onInput={handleInputChange("openAiApiKey")}
						placeholder={t("settings:placeholders.apiKey")}
						className="w-full">
						<label className="block font-medium mb-1">{t("settings:providers.openAiApiKey")}</label>
					</VSCodeTextField>
					<ModelPicker
						apiConfiguration={apiConfiguration}
						setApiConfigurationField={setApiConfigurationField}
						defaultModelId=""
						defaultModelInfo={openAiModelInfoSaneDefaults}
						models={openAiModels}
						modelIdKey="openAiModelId"
						modelInfoKey="openAiCustomModelInfo"
						serviceName="OpenAI"
						serviceUrl="https://platform.openai.com"
					/>
					<R1FormatSetting
						onChange={handleInputChange("openAiR1FormatEnabled", noTransform)}
						openAiR1FormatEnabled={apiConfiguration?.openAiR1FormatEnabled ?? false}
					/>
					<Checkbox
						checked={apiConfiguration?.openAiStreamingEnabled ?? true}
						onChange={handleInputChange("openAiStreamingEnabled", noTransform)}>
						{t("settings:modelInfo.enableStreaming")}
					</Checkbox>
					<Checkbox
						checked={apiConfiguration?.openAiUseAzure ?? false}
						onChange={handleInputChange("openAiUseAzure", noTransform)}>
						{t("settings:modelInfo.useAzure")}
					</Checkbox>
					<div>
						<Checkbox
							checked={azureApiVersionSelected}
							onChange={(checked: boolean) => {
								setAzureApiVersionSelected(checked)

								if (!checked) {
									setApiConfigurationField("azureApiVersion", "")
								}
							}}>
							{t("settings:modelInfo.azureApiVersion")}
						</Checkbox>
						{azureApiVersionSelected && (
							<VSCodeTextField
								value={apiConfiguration?.azureApiVersion || ""}
								onInput={handleInputChange("azureApiVersion")}
								placeholder={`Default: ${azureOpenAiDefaultApiVersion}`}
								className="w-full mt-1"
							/>
						)}
					</div>

					<div className="flex flex-col gap-3">
						<div className="text-sm text-vscode-descriptionForeground">
							{t("settings:providers.customModel.capabilities")}
						</div>

						<div>
							<VSCodeTextField
								value={
									apiConfiguration?.openAiCustomModelInfo?.maxTokens?.toString() ||
									openAiModelInfoSaneDefaults.maxTokens?.toString() ||
									""
								}
								type="text"
								style={{
									borderColor: (() => {
										const value = apiConfiguration?.openAiCustomModelInfo?.maxTokens

										if (!value) {
											return "var(--vscode-input-border)"
										}

										return value > 0
											? "var(--vscode-charts-green)"
											: "var(--vscode-errorForeground)"
									})(),
								}}
								title={t("settings:providers.customModel.maxTokens.description")}
								onInput={handleInputChange("openAiCustomModelInfo", (e) => {
									const value = parseInt((e.target as HTMLInputElement).value)

									return {
										...(apiConfiguration?.openAiCustomModelInfo || openAiModelInfoSaneDefaults),
										maxTokens: isNaN(value) ? undefined : value,
									}
								})}
								placeholder={t("settings:placeholders.numbers.maxTokens")}
								className="w-full">
								<label className="block font-medium mb-1">
									{t("settings:providers.customModel.maxTokens.label")}
								</label>
							</VSCodeTextField>
							<div className="text-sm text-vscode-descriptionForeground">
								{t("settings:providers.customModel.maxTokens.description")}
							</div>
						</div>

						<div>
							<VSCodeTextField
								value={
									apiConfiguration?.openAiCustomModelInfo?.contextWindow?.toString() ||
									openAiModelInfoSaneDefaults.contextWindow?.toString() ||
									""
								}
								type="text"
								style={{
									borderColor: (() => {
										const value = apiConfiguration?.openAiCustomModelInfo?.contextWindow

										if (!value) {
											return "var(--vscode-input-border)"
										}

										return value > 0
											? "var(--vscode-charts-green)"
											: "var(--vscode-errorForeground)"
									})(),
								}}
								title={t("settings:providers.customModel.contextWindow.description")}
								onInput={handleInputChange("openAiCustomModelInfo", (e) => {
									const value = (e.target as HTMLInputElement).value
									const parsed = parseInt(value)

									return {
										...(apiConfiguration?.openAiCustomModelInfo || openAiModelInfoSaneDefaults),
										contextWindow: isNaN(parsed)
											? openAiModelInfoSaneDefaults.contextWindow
											: parsed,
									}
								})}
								placeholder={t("settings:placeholders.numbers.contextWindow")}
								className="w-full">
								<label className="block font-medium mb-1">
									{t("settings:providers.customModel.contextWindow.label")}
								</label>
							</VSCodeTextField>
							<div className="text-sm text-vscode-descriptionForeground">
								{t("settings:providers.customModel.contextWindow.description")}
							</div>
						</div>

						<div>
							<div className="flex items-center gap-1">
								<Checkbox
									checked={
										apiConfiguration?.openAiCustomModelInfo?.supportsImages ??
										openAiModelInfoSaneDefaults.supportsImages
									}
									onChange={handleInputChange("openAiCustomModelInfo", (checked) => {
										return {
											...(apiConfiguration?.openAiCustomModelInfo || openAiModelInfoSaneDefaults),
											supportsImages: checked,
										}
									})}>
									<span className="font-medium">
										{t("settings:providers.customModel.imageSupport.label")}
									</span>
								</Checkbox>
								<i
									className="codicon codicon-info text-vscode-descriptionForeground"
									title={t("settings:providers.customModel.imageSupport.description")}
									style={{ fontSize: "12px" }}
								/>
							</div>
							<div className="text-sm text-vscode-descriptionForeground pt-1">
								{t("settings:providers.customModel.imageSupport.description")}
							</div>
						</div>

						<div>
							<div className="flex items-center gap-1">
								<Checkbox
									checked={apiConfiguration?.openAiCustomModelInfo?.supportsComputerUse ?? false}
									onChange={handleInputChange("openAiCustomModelInfo", (checked) => {
										return {
											...(apiConfiguration?.openAiCustomModelInfo || openAiModelInfoSaneDefaults),
											supportsComputerUse: checked,
										}
									})}>
									<span className="font-medium">
										{t("settings:providers.customModel.computerUse.label")}
									</span>
								</Checkbox>
								<i
									className="codicon codicon-info text-vscode-descriptionForeground"
									title={t("settings:providers.customModel.computerUse.description")}
									style={{ fontSize: "12px" }}
								/>
							</div>
							<div className="text-sm text-vscode-descriptionForeground pt-1">
								{t("settings:providers.customModel.computerUse.description")}
							</div>
						</div>

						<div>
							<div className="flex items-center gap-1">
								<Checkbox
									checked={apiConfiguration?.openAiCustomModelInfo?.supportsPromptCache ?? false}
									onChange={handleInputChange("openAiCustomModelInfo", (checked) => {
										return {
											...(apiConfiguration?.openAiCustomModelInfo || openAiModelInfoSaneDefaults),
											supportsPromptCache: checked,
										}
									})}>
									<span className="font-medium">
										{t("settings:providers.customModel.promptCache.label")}
									</span>
								</Checkbox>
								<i
									className="codicon codicon-info text-vscode-descriptionForeground"
									title={t("settings:providers.customModel.promptCache.description")}
									style={{ fontSize: "12px" }}
								/>
							</div>
							<div className="text-sm text-vscode-descriptionForeground pt-1">
								{t("settings:providers.customModel.promptCache.description")}
							</div>
						</div>

						<div>
							<VSCodeTextField
								value={
									apiConfiguration?.openAiCustomModelInfo?.inputPrice?.toString() ??
									openAiModelInfoSaneDefaults.inputPrice?.toString() ??
									""
								}
								type="text"
								style={{
									borderColor: (() => {
										const value = apiConfiguration?.openAiCustomModelInfo?.inputPrice

										if (!value && value !== 0) {
											return "var(--vscode-input-border)"
										}

										return value >= 0
											? "var(--vscode-charts-green)"
											: "var(--vscode-errorForeground)"
									})(),
								}}
								onChange={handleInputChange("openAiCustomModelInfo", (e) => {
									const value = (e.target as HTMLInputElement).value
									const parsed = parseFloat(value)

									return {
										...(apiConfiguration?.openAiCustomModelInfo ?? openAiModelInfoSaneDefaults),
										inputPrice: isNaN(parsed) ? openAiModelInfoSaneDefaults.inputPrice : parsed,
									}
								})}
								placeholder={t("settings:placeholders.numbers.inputPrice")}
								className="w-full">
								<div className="flex items-center gap-1">
									<label className="block font-medium mb-1">
										{t("settings:providers.customModel.pricing.input.label")}
									</label>
									<i
										className="codicon codicon-info text-vscode-descriptionForeground"
										title={t("settings:providers.customModel.pricing.input.description")}
										style={{ fontSize: "12px" }}
									/>
								</div>
							</VSCodeTextField>
						</div>

						<div>
							<VSCodeTextField
								value={
									apiConfiguration?.openAiCustomModelInfo?.outputPrice?.toString() ||
									openAiModelInfoSaneDefaults.outputPrice?.toString() ||
									""
								}
								type="text"
								style={{
									borderColor: (() => {
										const value = apiConfiguration?.openAiCustomModelInfo?.outputPrice

										if (!value && value !== 0) {
											return "var(--vscode-input-border)"
										}

										return value >= 0
											? "var(--vscode-charts-green)"
											: "var(--vscode-errorForeground)"
									})(),
								}}
								onChange={handleInputChange("openAiCustomModelInfo", (e) => {
									const value = (e.target as HTMLInputElement).value
									const parsed = parseFloat(value)

									return {
										...(apiConfiguration?.openAiCustomModelInfo || openAiModelInfoSaneDefaults),
										outputPrice: isNaN(parsed) ? openAiModelInfoSaneDefaults.outputPrice : parsed,
									}
								})}
								placeholder={t("settings:placeholders.numbers.outputPrice")}
								className="w-full">
								<div className="flex items-center gap-1">
									<label className="block font-medium mb-1">
										{t("settings:providers.customModel.pricing.output.label")}
									</label>
									<i
										className="codicon codicon-info text-vscode-descriptionForeground"
										title={t("settings:providers.customModel.pricing.output.description")}
										style={{ fontSize: "12px" }}
									/>
								</div>
							</VSCodeTextField>
						</div>

						{apiConfiguration?.openAiCustomModelInfo?.supportsPromptCache && (
							<>
								<div>
									<VSCodeTextField
										value={
											apiConfiguration?.openAiCustomModelInfo?.cacheReadsPrice?.toString() ?? "0"
										}
										type="text"
										style={{
											borderColor: (() => {
												const value = apiConfiguration?.openAiCustomModelInfo?.cacheReadsPrice

												if (!value && value !== 0) {
													return "var(--vscode-input-border)"
												}

												return value >= 0
													? "var(--vscode-charts-green)"
													: "var(--vscode-errorForeground)"
											})(),
										}}
										onChange={handleInputChange("openAiCustomModelInfo", (e) => {
											const value = (e.target as HTMLInputElement).value
											const parsed = parseFloat(value)

											return {
												...(apiConfiguration?.openAiCustomModelInfo ??
													openAiModelInfoSaneDefaults),
												cacheReadsPrice: isNaN(parsed) ? 0 : parsed,
											}
										})}
										placeholder={t("settings:placeholders.numbers.inputPrice")}
										className="w-full">
										<div className="flex items-center gap-1">
											<span className="font-medium">
												{t("settings:providers.customModel.pricing.cacheReads.label")}
											</span>
											<i
												className="codicon codicon-info text-vscode-descriptionForeground"
												title={t(
													"settings:providers.customModel.pricing.cacheReads.description",
												)}
												style={{ fontSize: "12px" }}
											/>
										</div>
									</VSCodeTextField>
								</div>
								<div>
									<VSCodeTextField
										value={
											apiConfiguration?.openAiCustomModelInfo?.cacheWritesPrice?.toString() ?? "0"
										}
										type="text"
										style={{
											borderColor: (() => {
												const value = apiConfiguration?.openAiCustomModelInfo?.cacheWritesPrice

												if (!value && value !== 0) {
													return "var(--vscode-input-border)"
												}

												return value >= 0
													? "var(--vscode-charts-green)"
													: "var(--vscode-errorForeground)"
											})(),
										}}
										onChange={handleInputChange("openAiCustomModelInfo", (e) => {
											const value = (e.target as HTMLInputElement).value
											const parsed = parseFloat(value)

											return {
												...(apiConfiguration?.openAiCustomModelInfo ??
													openAiModelInfoSaneDefaults),
												cacheWritesPrice: isNaN(parsed) ? 0 : parsed,
											}
										})}
										placeholder={t("settings:placeholders.numbers.cacheWritePrice")}
										className="w-full">
										<div className="flex items-center gap-1">
											<label className="block font-medium mb-1">
												{t("settings:providers.customModel.pricing.cacheWrites.label")}
											</label>
											<i
												className="codicon codicon-info text-vscode-descriptionForeground"
												title={t(
													"settings:providers.customModel.pricing.cacheWrites.description",
												)}
												style={{ fontSize: "12px" }}
											/>
										</div>
									</VSCodeTextField>
								</div>
							</>
						)}

						<Button
							variant="secondary"
							onClick={() =>
								setApiConfigurationField("openAiCustomModelInfo", openAiModelInfoSaneDefaults)
							}>
							{t("settings:providers.customModel.resetDefaults")}
						</Button>
					</div>
				</>
			)}

			{selectedProvider === "lmstudio" && (
				<>
					<VSCodeTextField
						value={apiConfiguration?.lmStudioBaseUrl || ""}
						type="url"
						onInput={handleInputChange("lmStudioBaseUrl")}
						placeholder={t("settings:defaults.lmStudioUrl")}
						className="w-full">
						<label className="block font-medium mb-1">{t("settings:providers.lmStudio.baseUrl")}</label>
					</VSCodeTextField>
					<VSCodeTextField
						value={apiConfiguration?.lmStudioModelId || ""}
						onInput={handleInputChange("lmStudioModelId")}
						placeholder={t("settings:placeholders.modelId.lmStudio")}
						className="w-full">
						<label className="block font-medium mb-1">{t("settings:providers.lmStudio.modelId")}</label>
					</VSCodeTextField>
					{lmStudioModels.length > 0 && (
						<VSCodeRadioGroup onChange={handleInputChange("lmStudioModelId")}>
							{lmStudioModels.map((model) => (
								<VSCodeRadio key={model} value={model}>
									{model}
								</VSCodeRadio>
							))}
						</VSCodeRadioGroup>
					)}
					<Checkbox
						checked={apiConfiguration?.lmStudioSpeculativeDecodingEnabled === true}
						onChange={(checked) => {
							setApiConfigurationField("lmStudioSpeculativeDecodingEnabled", checked)
						}}>
						{t("settings:providers.lmStudio.speculativeDecoding")}
					</Checkbox>
					{apiConfiguration?.lmStudioSpeculativeDecodingEnabled && (
						<>
							<div>
								<VSCodeTextField
									value={apiConfiguration?.lmStudioDraftModelId || ""}
									onInput={handleInputChange("lmStudioDraftModelId")}
									placeholder={t("settings:placeholders.modelId.lmStudioDraft")}
									className="w-full">
									<label className="block font-medium mb-1">
										{t("settings:providers.lmStudio.draftModelId")}
									</label>
								</VSCodeTextField>
								<div className="text-sm text-vscode-descriptionForeground">
									{t("settings:providers.lmStudio.draftModelDesc")}
								</div>
							</div>
							{lmStudioModels.length > 0 && (
								<>
									<div className="font-medium">
										{t("settings:providers.lmStudio.selectDraftModel")}
									</div>
									<VSCodeRadioGroup onChange={handleInputChange("lmStudioDraftModelId")}>
										{lmStudioModels.map((model) => (
											<VSCodeRadio key={`draft-${model}`} value={model}>
												{model}
											</VSCodeRadio>
										))}
									</VSCodeRadioGroup>
									{lmStudioModels.length === 0 && (
										<div
											className="text-sm rounded-xs p-2"
											style={{
												backgroundColor: "var(--vscode-inputValidation-infoBackground)",
												border: "1px solid var(--vscode-inputValidation-infoBorder)",
												color: "var(--vscode-inputValidation-infoForeground)",
											}}>
											{t("settings:providers.lmStudio.noModelsFound")}
										</div>
									)}
								</>
							)}
						</>
					)}
					<div className="text-sm text-vscode-descriptionForeground">
						<Trans
							i18nKey="settings:providers.lmStudio.description"
							components={{
								a: <VSCodeLink href="https://lmstudio.ai/docs" />,
								b: <VSCodeLink href="https://lmstudio.ai/docs/basics/server" />,
								span: (
									<span className="text-vscode-errorForeground ml-1">
										<span className="font-medium">Note:</span>
									</span>
								),
							}}
						/>
					</div>
				</>
			)}

			{selectedProvider === "deepseek" && (
				<>
					<VSCodeTextField
						value={apiConfiguration?.deepSeekApiKey || ""}
						type="password"
						onInput={handleInputChange("deepSeekApiKey")}
						placeholder={t("settings:placeholders.apiKey")}
						className="w-full">
						<label className="block font-medium mb-1">{t("settings:providers.deepSeekApiKey")}</label>
					</VSCodeTextField>
					<div className="text-sm text-vscode-descriptionForeground -mt-2">
						{t("settings:providers.apiKeyStorageNotice")}
					</div>
					{!apiConfiguration?.deepSeekApiKey && (
						<VSCodeButtonLink href="https://platform.deepseek.com/">
							{t("settings:providers.getDeepSeekApiKey")}
						</VSCodeButtonLink>
					)}
				</>
			)}

			{selectedProvider === "vscode-lm" && (
				<>
					<div>
						<label className="block font-medium mb-1">{t("settings:providers.vscodeLmModel")}</label>
						{vsCodeLmModels.length > 0 ? (
							<Select
								value={
									apiConfiguration?.vsCodeLmModelSelector
										? `${apiConfiguration.vsCodeLmModelSelector.vendor ?? ""}/${apiConfiguration.vsCodeLmModelSelector.family ?? ""}`
										: ""
								}
								onValueChange={handleInputChange("vsCodeLmModelSelector", (value) => {
									const [vendor, family] = value.split("/")
									return { vendor, family }
								})}>
								<SelectTrigger className="w-full">
									<SelectValue placeholder={t("settings:common.select")} />
								</SelectTrigger>
								<SelectContent>
									{vsCodeLmModels.map((model) => (
										<SelectItem
											key={`${model.vendor}/${model.family}`}
											value={`${model.vendor}/${model.family}`}>
											{`${model.vendor} - ${model.family}`}
										</SelectItem>
									))}
								</SelectContent>
							</Select>
						) : (
							<div className="text-sm text-vscode-descriptionForeground">
								{t("settings:providers.vscodeLmDescription")}
							</div>
						)}
					</div>
					<div className="text-sm text-vscode-errorForeground">{t("settings:providers.vscodeLmWarning")}</div>
				</>
			)}

			{selectedProvider === "ollama" && (
				<>
					<VSCodeTextField
						value={apiConfiguration?.ollamaBaseUrl || ""}
						type="url"
						onInput={handleInputChange("ollamaBaseUrl")}
						placeholder={t("settings:defaults.ollamaUrl")}
						className="w-full">
						<label className="block font-medium mb-1">{t("settings:providers.ollama.baseUrl")}</label>
					</VSCodeTextField>
					<VSCodeTextField
						value={apiConfiguration?.ollamaModelId || ""}
						onInput={handleInputChange("ollamaModelId")}
						placeholder={t("settings:placeholders.modelId.ollama")}
						className="w-full">
						<label className="block font-medium mb-1">{t("settings:providers.ollama.modelId")}</label>
					</VSCodeTextField>
					{ollamaModels.length > 0 && (
						<VSCodeRadioGroup onChange={handleInputChange("ollamaModelId")}>
							{ollamaModels.map((model) => (
								<VSCodeRadio key={model} value={model}>
									{model}
								</VSCodeRadio>
							))}
						</VSCodeRadioGroup>
					)}
					<div className="text-sm text-vscode-descriptionForeground">
						{t("settings:providers.ollama.description")}
						<span className="text-vscode-errorForeground ml-1">
							{t("settings:providers.ollama.warning")}
						</span>
					</div>
				</>
			)}

			{selectedProvider === "unbound" && (
				<>
					<VSCodeTextField
						value={apiConfiguration?.unboundApiKey || ""}
						type="password"
						onInput={handleInputChange("unboundApiKey")}
						placeholder={t("settings:placeholders.apiKey")}
						className="w-full">
						<label className="block font-medium mb-1">{t("settings:providers.unboundApiKey")}</label>
					</VSCodeTextField>
					<div className="text-sm text-vscode-descriptionForeground -mt-2">
						{t("settings:providers.apiKeyStorageNotice")}
					</div>
					{!apiConfiguration?.unboundApiKey && (
						<VSCodeButtonLink href="https://gateway.getunbound.ai">
							{t("settings:providers.getUnboundApiKey")}
						</VSCodeButtonLink>
					)}
				</>
			)}

			{selectedProvider === "human-relay" && (
				<>
					<div className="text-sm text-vscode-descriptionForeground">
						{t("settings:providers.humanRelay.description")}
					</div>
					<div className="text-sm text-vscode-descriptionForeground">
						{t("settings:providers.humanRelay.instructions")}
					</div>
				</>
			)}

			{/* Model Pickers */}

			{selectedProvider === "openrouter" && (
				<ModelPicker
					apiConfiguration={apiConfiguration}
					setApiConfigurationField={setApiConfigurationField}
					defaultModelId={openRouterDefaultModelId}
					defaultModelInfo={openRouterDefaultModelInfo}
					models={openRouterModels}
					modelIdKey="openRouterModelId"
					modelInfoKey="openRouterModelInfo"
					serviceName="OpenRouter"
					serviceUrl="https://openrouter.ai/models"
				/>
			)}

			{selectedProvider === "openrouter" &&
				openRouterModelProviders &&
				Object.keys(openRouterModelProviders).length > 0 && (
					<div>
						<div className="flex items-center gap-1">
							<label className="block font-medium mb-1">
								{t("settings:providers.openRouter.providerRouting.title")}
							</label>
							<a href={`https://openrouter.ai/${selectedModelId}/providers`}>
								<ExternalLinkIcon className="w-4 h-4" />
							</a>
						</div>
						<Select
							value={apiConfiguration?.openRouterSpecificProvider || OPENROUTER_DEFAULT_PROVIDER_NAME}
							onValueChange={(value) => {
								if (openRouterModelProviders[value]) {
									setApiConfigurationField("openRouterModelInfo", {
										...apiConfiguration.openRouterModelInfo,
										...openRouterModelProviders[value],
									})
								}

								setApiConfigurationField("openRouterSpecificProvider", value)
							}}>
							<SelectTrigger className="w-full">
								<SelectValue placeholder={t("settings:common.select")} />
							</SelectTrigger>
							<SelectContent>
								<SelectItem value={OPENROUTER_DEFAULT_PROVIDER_NAME}>
									{OPENROUTER_DEFAULT_PROVIDER_NAME}
								</SelectItem>
								{Object.entries(openRouterModelProviders).map(([value, { label }]) => (
									<SelectItem key={value} value={value}>
										{label}
									</SelectItem>
								))}
							</SelectContent>
						</Select>
						<div className="text-sm text-vscode-descriptionForeground mt-1">
							{t("settings:providers.openRouter.providerRouting.description")}{" "}
							<a href="https://openrouter.ai/docs/features/provider-routing">
								{t("settings:providers.openRouter.providerRouting.learnMore")}.
							</a>
						</div>
					</div>
				)}

			{selectedProvider === "glama" && (
				<ModelPicker
					apiConfiguration={apiConfiguration}
					setApiConfigurationField={setApiConfigurationField}
					defaultModelId={glamaDefaultModelId}
					defaultModelInfo={glamaDefaultModelInfo}
					models={glamaModels}
					modelInfoKey="glamaModelInfo"
					modelIdKey="glamaModelId"
					serviceName="Glama"
					serviceUrl="https://glama.ai/models"
				/>
			)}

			{selectedProvider === "unbound" && (
				<ModelPicker
					apiConfiguration={apiConfiguration}
					defaultModelId={unboundDefaultModelId}
					defaultModelInfo={unboundDefaultModelInfo}
					models={unboundModels}
					modelInfoKey="unboundModelInfo"
					modelIdKey="unboundModelId"
					serviceName="Unbound"
					serviceUrl="https://api.getunbound.ai/models"
					setApiConfigurationField={setApiConfigurationField}
				/>
			)}

			{selectedProvider === "requesty" && (
				<ModelPicker
					apiConfiguration={apiConfiguration}
					setApiConfigurationField={setApiConfigurationField}
					defaultModelId={requestyDefaultModelId}
					defaultModelInfo={requestyDefaultModelInfo}
					models={requestyModels}
					modelIdKey="requestyModelId"
					modelInfoKey="requestyModelInfo"
					serviceName="Requesty"
					serviceUrl="https://requesty.ai"
				/>
			)}

			{selectedProviderModelOptions.length > 0 && (
				<>
					<div>
						<label className="block font-medium mb-1">{t("settings:providers.model")}</label>

						<Select
							value={selectedModelId === "custom-arn" ? "custom-arn" : selectedModelId}
							onValueChange={(value) => {
								setApiConfigurationField("apiModelId", value)

								// Clear custom ARN if not using custom ARN option.
								if (value !== "custom-arn" && selectedProvider === "bedrock") {
									setApiConfigurationField("awsCustomArn", "")
								}
							}}>
							<SelectTrigger className="w-full">
								<SelectValue placeholder={t("settings:common.select")} />
							</SelectTrigger>
							<SelectContent>
								{selectedProviderModelOptions.map((option) => (
									<SelectItem key={option.value} value={option.value}>
										{option.label}
									</SelectItem>
								))}
								{selectedProvider === "bedrock" && (
									<SelectItem value="custom-arn">{t("settings:labels.useCustomArn")}</SelectItem>
								)}
							</SelectContent>
						</Select>
					</div>

					{selectedProvider === "bedrock" && selectedModelId === "custom-arn" && (
						<>
							<VSCodeTextField
								value={apiConfiguration?.awsCustomArn || ""}
								onInput={(e) => {
									const value = (e.target as HTMLInputElement).value
									setApiConfigurationField("awsCustomArn", value)
								}}
								placeholder={t("settings:placeholders.customArn")}
								className="w-full">
								<label className="block font-medium mb-1">{t("settings:labels.customArn")}</label>
							</VSCodeTextField>
							<div className="text-sm text-vscode-descriptionForeground -mt-2">
								{t("settings:providers.awsCustomArnUse")}
								<ul className="list-disc pl-5 mt-1">
									<li>
										arn:aws:bedrock:us-east-1:123456789012:foundation-model/anthropic.claude-3-sonnet-20240229-v1:0
									</li>
									<li>
										arn:aws:bedrock:us-west-2:123456789012:provisioned-model/my-provisioned-model
									</li>
									<li>
										arn:aws:bedrock:us-east-1:123456789012:default-prompt-router/anthropic.claude:1
									</li>
								</ul>
								{t("settings:providers.awsCustomArnDesc")}
							</div>
							{apiConfiguration?.awsCustomArn &&
								(() => {
									const validation = validateBedrockArn(
										apiConfiguration.awsCustomArn,
										apiConfiguration.awsRegion,
									)

									if (!validation.isValid) {
										return (
											<div className="text-sm text-vscode-errorForeground mt-2">
												{validation.errorMessage || t("settings:providers.invalidArnFormat")}
											</div>
										)
									}

									if (validation.errorMessage) {
										return (
											<div className="text-sm text-vscode-errorForeground mt-2">
												{validation.errorMessage}
											</div>
										)
									}

									return null
								})()}
						</>
					)}
					<ModelInfoView
						selectedModelId={selectedModelId}
						modelInfo={selectedModelInfo}
						isDescriptionExpanded={isDescriptionExpanded}
						setIsDescriptionExpanded={setIsDescriptionExpanded}
					/>
					<ThinkingBudget
						key={`${selectedProvider}-${selectedModelId}`}
						apiConfiguration={apiConfiguration}
						setApiConfigurationField={setApiConfigurationField}
						modelInfo={selectedModelInfo}
					/>
				</>
			)}

			{!fromWelcomeView && selectedModelInfo?.supportsTemperature !== false && (
				<TemperatureControl
					value={apiConfiguration?.modelTemperature}
					onChange={handleInputChange("modelTemperature", noTransform)}
					maxValue={2}
				/>
			)}
		</div>
	)
}

export function normalizeApiConfiguration(
	apiConfiguration?: ApiConfiguration,
	dynamicModels?: {
		anthropicModels?: Record<string, ModelInfo> | null
		bedrockModels?: Record<string, ModelInfo> | null
		vertexModels?: Record<string, ModelInfo> | null
		geminiModels?: Record<string, ModelInfo> | null
		mistralModels?: Record<string, ModelInfo> | null
		deepseekModels?: Record<string, ModelInfo> | null
	}
) {
	const provider = apiConfiguration?.apiProvider || "anthropic"
	const modelId = apiConfiguration?.apiModelId

	const getProviderData = (models: Record<string, ModelInfo>, defaultId: string) => {
		let selectedModelId: string
		let selectedModelInfo: ModelInfo

		if (modelId && modelId in models) {
			selectedModelId = modelId
			selectedModelInfo = models[modelId]
		} else if (defaultId && defaultId in models) {
			selectedModelId = defaultId
			selectedModelInfo = models[defaultId]
		} else {
			// If no default or default not found, use first available model
			const modelKeys = Object.keys(models)
			selectedModelId = modelKeys[0] || ""
			selectedModelInfo = models[selectedModelId] || openAiModelInfoSaneDefaults
		}

		return { selectedProvider: provider, selectedModelId, selectedModelInfo }
	}

	switch (provider) {
		case "anthropic":
			// Use dynamic models if available, otherwise fall back to static
			{ const anthropicModelsToUse = dynamicModels?.anthropicModels || anthropicModels
			return getProviderData(anthropicModelsToUse, anthropicDefaultModelId) }
		case "bedrock":
			// Special case for custom ARN
			{ if (modelId === "custom-arn") {
				return {
					selectedProvider: provider,
					selectedModelId: "custom-arn",
					selectedModelInfo: {
						maxTokens: 5000,
						contextWindow: 128_000,
						supportsPromptCache: false,
						supportsImages: true,
					},
				}
			}
			const bedrockModelsToUse = dynamicModels?.bedrockModels || bedrockModels
			return getProviderData(bedrockModelsToUse, bedrockDefaultModelId) }
		case "vertex":
			{ const vertexModelsToUse = dynamicModels?.vertexModels || vertexModels
			return getProviderData(vertexModelsToUse, vertexDefaultModelId) }
		case "gemini":
			{ const geminiModelsToUse = dynamicModels?.geminiModels || geminiModels
			return getProviderData(geminiModelsToUse, geminiDefaultModelId) }
		case "mistral":
			{ const mistralModelsToUse = dynamicModels?.mistralModels || mistralModels
			return getProviderData(mistralModelsToUse, mistralDefaultModelId) }
		case "deepseek":
			{ const deepseekModelsToUse = dynamicModels?.deepseekModels || deepSeekModels
			return getProviderData(deepseekModelsToUse, deepSeekDefaultModelId) }
		case "openai-native":
			return getProviderData(openAiNativeModels, openAiNativeDefaultModelId)
		case "openrouter":
			return {
				selectedProvider: provider,
				selectedModelId: apiConfiguration?.openRouterModelId || openRouterDefaultModelId,
				selectedModelInfo: apiConfiguration?.openRouterModelInfo || openRouterDefaultModelInfo,
			}
		case "glama":
			return {
				selectedProvider: provider,
				selectedModelId: apiConfiguration?.glamaModelId || glamaDefaultModelId,
				selectedModelInfo: apiConfiguration?.glamaModelInfo || glamaDefaultModelInfo,
			}
		case "unbound":
			return {
				selectedProvider: provider,
				selectedModelId: apiConfiguration?.unboundModelId || unboundDefaultModelId,
				selectedModelInfo: apiConfiguration?.unboundModelInfo || unboundDefaultModelInfo,
			}
		case "requesty":
			return {
				selectedProvider: provider,
				selectedModelId: apiConfiguration?.requestyModelId || requestyDefaultModelId,
				selectedModelInfo: apiConfiguration?.requestyModelInfo || requestyDefaultModelInfo,
			}
		case "openai":
			return {
				selectedProvider: provider,
				selectedModelId: apiConfiguration?.openAiModelId || "",
				selectedModelInfo: apiConfiguration?.openAiCustomModelInfo || openAiModelInfoSaneDefaults,
			}
		case "ollama":
			return {
				selectedProvider: provider,
				selectedModelId: apiConfiguration?.ollamaModelId || "",
				selectedModelInfo: openAiModelInfoSaneDefaults,
			}
		case "lmstudio":
			return {
				selectedProvider: provider,
				selectedModelId: apiConfiguration?.lmStudioModelId || "",
				selectedModelInfo: openAiModelInfoSaneDefaults,
			}
		case "vscode-lm":
			return {
				selectedProvider: provider,
				selectedModelId: apiConfiguration?.vsCodeLmModelSelector
					? `${apiConfiguration.vsCodeLmModelSelector.vendor}/${apiConfiguration.vsCodeLmModelSelector.family}`
					: "",
				selectedModelInfo: {
					...openAiModelInfoSaneDefaults,
					supportsImages: false, // VSCode LM API currently doesn't support images.
				},
			}
		default:
			return getProviderData(anthropicModels, anthropicDefaultModelId)
	}
}

export default memo(ApiOptions)
