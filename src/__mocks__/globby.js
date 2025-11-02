function globby(_patterns, _options) {
	return Promise.resolve([])
}

globby.sync = function (_patterns, _options) {
	return []
}

export default globby
