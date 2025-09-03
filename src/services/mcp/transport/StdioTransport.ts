import { IMcpTransport } from "../types/McpTransportTypes"
import { StdioTransportConfig } from "../types/McpTransportTypes"

/**
 * Mock implementation of the StdioServerTransport for type compatibility
 * This will be replaced with the actual implementation when the MCP SDK is installed
 */
class MockStdioServerTransport {
    private readonly _stderr: { on?: (event: string, handler: (data: Buffer) => void) => void }
    constructor() {
        // Provide a minimal event emitter-ish object to satisfy tests reading stderr
        this._stderr = { on: () => void 0 }
    }
    start(): Promise<void> {
        return Promise.resolve()
    }
    close(): Promise<void> {
        return Promise.resolve()
    }
    get stderr(): unknown {
        return this._stderr
    }
    onerror?: (error: Error) => void
    onclose?: () => void

}

/**
 * StdioTransport provides an implementation of the MCP transport using stdio.
 */
interface StdioServerTransportLike {
	start(): Promise<void>
	close(): Promise<void>
	readonly stderr?: unknown
	onerror?: (error: Error) => void
	onclose?: () => void
}

export class StdioTransport implements IMcpTransport {
	private transport?: StdioServerTransportLike
	private _stderr?: unknown
	private readonly options: StdioTransportConfig

	constructor(options: StdioTransportConfig) {
		this.options = options
	}

    private async initTransport(): Promise<void> {
        if (this.transport) {
            return
        }
        try {
            if (process.env.THEA_DISABLE_MCP_SDK === '1') {
                throw new Error('MCP SDK disabled via THEA_DISABLE_MCP_SDK=1')
            }
            const mod = await import("@modelcontextprotocol/sdk/server/stdio.js")
            const Transport = mod.StdioServerTransport as unknown as new (
                opts: StdioTransportConfig & { stderr: string },
            ) => StdioServerTransportLike
            this.transport = new Transport({
                command: this.options.command,
                args: this.options.args,
                env: {
                    ...this.options.env,
                    ...(process.env.PATH ? { PATH: process.env.PATH } : {}),
                },
                stderr: "pipe",
            })
        } catch (e) {
            console.warn("MCP SDK not found or disabled, using mock implementation", e instanceof Error ? e.message : String(e))
            this.transport = new MockStdioServerTransport()
        }
    }

	async start(): Promise<void> {
		await this.initTransport()
		if (this.transport) {
			await this.transport.start()
			this._stderr = this.transport.stderr
		}
	}

	async close(): Promise<void> {
		if (this.transport) {
			await this.transport.close()
		}
	}

	getPort(): number | undefined {
		return undefined
	}

	get stderr(): unknown {
		return this._stderr
	}

	set onerror(handler: (error: Error) => void) {
		if (this.transport) {
			this.transport.onerror = handler
		}
	}

	set onclose(handler: () => void) {
		if (this.transport) {
			this.transport.onclose = handler
		}
	}
}
