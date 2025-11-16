const CallToolResultSchema = {
	parse: () => ({}),
}

const ListToolsResultSchema = {
	parse: () => ({
		tools: [],
	}),
}

const ListResourcesResultSchema = {
	parse: () => ({
		resources: [],
	}),
}

const ListResourceTemplatesResultSchema = {
	parse: () => ({
		resourceTemplates: [],
	}),
}

const ReadResourceResultSchema = {
	parse: () => ({
		contents: [],
	}),
}

const ErrorCode = {
	InvalidRequest: "InvalidRequest",
	MethodNotFound: "MethodNotFound",
	InvalidParams: "InvalidParams",
	InternalError: "InternalError",
}

class McpError extends Error {
	constructor(code, message) {
		super(message)
		this.code = code
	}
}

export { CallToolResultSchema, ListToolsResultSchema, ListResourcesResultSchema, ListResourceTemplatesResultSchema, ReadResourceResultSchema, ErrorCode, McpError }
