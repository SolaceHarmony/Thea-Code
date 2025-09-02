import sinon from 'sinon'

export class McpHub {
	connections = []
	isConnecting = false

	constructor() {
		this.toggleToolAlwaysAllow = sinon.stub()
		this.callTool = sinon.stub()
	}

	async toggleToolAlwaysAllow(serverName: string, toolName: string, shouldAllow: boolean): Promise<void> {
		return Promise.resolve()
	}

	async callTool(serverName: string, toolName: string, toolArguments?: Record<string, unknown>): Promise<any> {
		return Promise.resolve({ result: "success" })
	}
}
