import sinon from 'sinon'


export class McpHub {
	connections = []
	isConnecting = false

	constructor() {
		this.toggleToolAlwaysAllow = sinon.stub()
		this.callTool = sinon.stub()

	}

	async toggleToolAlwaysAllow(_serverName: string, _toolName: string, _shouldAllow: boolean): Promise<void> {
		return Promise.resolve()
	}

	async callTool(_serverName: string, _toolName: string, _toolArguments?: Record<string, unknown>): Promise<Record<string, string>> {
		return Promise.resolve({ result: "success" })
	}
}
