import { StrictMode } from "react"
import { createRoot } from "react-dom/client"

import { allComponents, provideVSCodeDesignSystem } from "@vscode/webview-ui-toolkit"

import "./index.css"
import App from "./App"
import "../../node_modules/@vscode/codicons/dist/codicon.css"

provideVSCodeDesignSystem().register(allComponents)

createRoot(document.getElementById("root")!).render(
	<StrictMode>
		<App />
	</StrictMode>,
)
