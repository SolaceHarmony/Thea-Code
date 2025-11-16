class SSEClientTransport {
	constructor(url, options = {}) {
		this.url = url
		this.options = options
		this.onerror = null
		this.connect = () => Promise.resolve()
		this.close = () => Promise.resolve()
		this.start = () => Promise.resolve()
	}
}

export { SSEClientTransport }
