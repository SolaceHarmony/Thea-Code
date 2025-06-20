import i18next from "./setup"

/**
 * Initialize i18next with the specified language
 *
 * @param language The language code to use
 */
export async function initializeI18n(language: string): Promise<void> {
	await i18next.changeLanguage(language)
}

/**
 * Get the current language
 *
 * @returns The current language code
 */
export function getCurrentLanguage(): string {
	return i18next.language
}

/**
 * Change the current language
 *
 * @param language The language code to change to
 */
export async function changeLanguage(language: string): Promise<void> {
	await i18next.changeLanguage(language)
}

/**
 * Translate a string using i18next
 *
 * @param key The translation key, can use namespace with colon, e.g. "common:welcome"
 * @param options Options for interpolation or pluralization
 * @returns The translated string
 */
export function t(key: string, options?: Record<string, unknown>): string {
	return i18next.t(key, options)
}

export default i18next
