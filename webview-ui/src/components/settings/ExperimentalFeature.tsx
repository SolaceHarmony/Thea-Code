import React from "react"
import { VSCodeCheckbox } from "@/components/ui/vscode-components"
import { useAppTranslation } from "@/i18n/TranslationContext"

interface ExperimentalFeatureProps {
	enabled: boolean
	onChange: (value: boolean) => void
	// Additional property to identify the experiment
	experimentKey?: string
}

export const ExperimentalFeature = ({ enabled, onChange, experimentKey }: ExperimentalFeatureProps) => {
	const { t } = useAppTranslation()

	// Generate translation keys based on experiment key
	const nameKey = experimentKey ? `settings:experimental.${experimentKey}.name` : ""
	const descriptionKey = experimentKey ? `settings:experimental.${experimentKey}.description` : ""

	return (
		<div>
			<div className="flex items-center gap-2">
				<span className="text-vscode-errorForeground">{t("settings:experimental.warning")}</span>
				<VSCodeCheckbox
					checked={enabled}
					onChange={(checked: boolean) => onChange(checked)}>
					<span className="font-medium">{t(nameKey)}</span>
				</VSCodeCheckbox>
			</div>
			<p className="text-vscode-descriptionForeground text-sm mt-0">{t(descriptionKey)}</p>
		</div>
	)
}
