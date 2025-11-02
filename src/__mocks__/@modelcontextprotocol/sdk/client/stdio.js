class StdioClientTransport {
	constructor() {
		this.start = () => Promise.resolve(undefined)
		this.close = () => Promise.resolve(undefined)
		this.stderr = {
			on: () => {},
		}
	}
}

class StdioServerParameters {
	constructor() {
		this.command = ""
		this.args = []
		this.env = {}
	}
}

export { StdioClientTransport, StdioServerParameters }
