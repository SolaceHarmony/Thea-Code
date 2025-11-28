import { StrictMode } from "react"
import { createRoot } from "react-dom/client"
import { vscode } from "./utils/vscode";

window.onerror = (message, source, lineno, colno, error) => {
	vscode.postMessage({
		type: "error",
		value: `Webview Error: ${message} at ${source}:${lineno}:${colno}\nStack: ${error?.stack}`
	} as any);
};

import "./index.css"
import App from "./App"
import "../../node_modules/@vscode/codicons/dist/codicon.css"

createRoot(document.getElementById("root")!).render(
	<StrictMode>
		<App />
	</StrictMode>,
)
