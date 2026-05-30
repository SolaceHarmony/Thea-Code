// PNPM hook to ensure serialize-javascript is updated to fix CVE
// See: https://github.com/advisories/GHSA-5c6j-r48x-rmvq

function readPackage(pkg) {
	// Override serialize-javascript version for all packages that depend on it
	if (pkg.dependencies && pkg.dependencies['serialize-javascript']) {
		pkg.dependencies['serialize-javascript'] = '>=7.0.5'
	}
	if (pkg.devDependencies && pkg.devDependencies['serialize-javascript']) {
		pkg.devDependencies['serialize-javascript'] = '>=7.0.5'
	}
	return pkg
}

module.exports = {
	hooks: {
		readPackage,
	},
}
