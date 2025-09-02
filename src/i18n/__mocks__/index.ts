// Mock implementation for i18n (Mocha + Sinon-friendly)
import sinon from 'sinon'

const i18nMock = {
	t: sinon.stub().callsFake((key: string, _options?: Record<string, unknown>): string => {
 		void _options
 		const translationMap: Record<string, string> = {
 			"common:confirmation.reset_state": "Are you sure you want to reset all state?",
 			"common:answers.yes": "Yes",
 		}
 		return translationMap[key] || key
 	}),
 	changeLanguage: sinon.stub(),
 	getCurrentLanguage: sinon.stub().returns('en'),
 	initializeI18n: sinon.stub(),
}

export const t = i18nMock.t
export const changeLanguage = i18nMock.changeLanguage
export const getCurrentLanguage = i18nMock.getCurrentLanguage
export const initializeI18n = i18nMock.initializeI18n

export default i18nMock
