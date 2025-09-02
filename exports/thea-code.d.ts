// Re-export central TheaCode types so e2e bootstrap imports resolve.
// Use explicit named type exports and omit the .d.ts suffix for robust resolution.
export type {
	TheaMessage,
	GlobalSettings,
	ProviderSettings,
	TheaCodeAPI,
	TheaCodeEvents,
	TheaCodeSettings,
	TokenUsage,
} from "../src/exports/thea-code"
