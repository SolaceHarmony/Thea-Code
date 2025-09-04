export interface LanguageModelChatSelectorLike {
  vendor?: string
  family?: string
  version?: string
  id?: string
}

export const SELECTOR_SEPARATOR = "/"

export function stringifyVsCodeLmModelSelector(selector: LanguageModelChatSelectorLike): string {
	return [selector.vendor, selector.family, selector.version, selector.id].filter(Boolean).join(SELECTOR_SEPARATOR)
}
