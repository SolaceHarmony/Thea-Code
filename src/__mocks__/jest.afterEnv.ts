// Flag active test phase so console wrapper only drops logs after teardown or after each test
type GlobalWithTestFlag = typeof globalThis & { __MOCHA_ACTIVE_TEST__?: boolean }

suiteSetup(() => {
	(globalThis as GlobalWithTestFlag).__MOCHA_ACTIVE_TEST__ = true
})

teardown(() => {
	// If a test completes, mark not active to reduce post-test logging noise between tests
	(globalThis as GlobalWithTestFlag).__MOCHA_ACTIVE_TEST__ = false
})

setup(() => {
	// Re-enable during each test start
	(globalThis as GlobalWithTestFlag).__MOCHA_ACTIVE_TEST__ = true

})
