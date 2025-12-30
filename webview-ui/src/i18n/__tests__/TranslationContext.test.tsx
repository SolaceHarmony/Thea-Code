import React from "react"
import { render } from "@testing-library/react"
import "@testing-library/jest-dom"
import sinon from "sinon"
import proxyquire from "proxyquire"
import { setupI18nForTests } from "../test-utils"

const proxyquireStrict = proxyquire.noPreserveCache()

// Mock component that uses the translation context
const buildTestComponent = (useAppTranslation: typeof import("../TranslationContext").useAppTranslation) =>
	function TestComponent() {
		const { t } = useAppTranslation()
		return (
			<div>
				<h1 data-testid="translation-test">{t("settings.autoApprove.title")}</h1>
				<p data-testid="translation-interpolation">{t("notifications.error", { message: "Test error" })}</p>
			</div>
		)
	}

describe("TranslationContext", () => {
	let sandbox: sinon.SinonSandbox
	let TranslationProvider: typeof import("../TranslationContext").default
	let useAppTranslation: typeof import("../TranslationContext").useAppTranslation

	beforeEach(() => {
		sandbox = sinon.createSandbox()

		const translationModule = proxyquireStrict("../TranslationContext", {
			"@/context/ExtensionStateContext": {
				useExtensionState: sandbox.stub().returns({
					language: "en",
				}),
			},
		}) as typeof import("../TranslationContext")

		TranslationProvider = translationModule.default
		useAppTranslation = translationModule.useAppTranslation
	})

	afterEach(() => {
		sandbox.restore()
	})

	beforeAll(() => {
		// Initialize i18next with test translations
		setupI18nForTests()
	})

	it("should provide translations via context", () => {
		const TestComponent = buildTestComponent(useAppTranslation)
		const { getByTestId } = render(
			<TranslationProvider>
				<TestComponent />
			</TranslationProvider>,
		)

		// Check if translation is provided correctly
		expect(getByTestId("translation-test")).toHaveTextContent("Auto-Approve")
	})

	it("should handle interpolation correctly", () => {
		const TestComponent = buildTestComponent(useAppTranslation)
		const { getByTestId } = render(
			<TranslationProvider>
				<TestComponent />
			</TranslationProvider>,
		)

		// Check if interpolation works
		expect(getByTestId("translation-interpolation")).toHaveTextContent("Operation failed: Test error")
	})
})
