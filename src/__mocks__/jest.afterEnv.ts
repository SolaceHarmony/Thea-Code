// Flag active test phase so console wrapper only drops logs after teardown or after each test
type GlobalWithJestFlag = typeof globalThis & { __JEST_ACTIVE_TEST__?: boolean }

beforeAll(() => {
	(globalThis as GlobalWithJestFlag).__JEST_ACTIVE_TEST__ = true
})

afterEach(() => {
	// If a test completes, mark not active to reduce post-test logging noise between tests
	(globalThis as GlobalWithJestFlag).__JEST_ACTIVE_TEST__ = false
})

beforeEach(() => {
	// Re-enable during each test start
	(globalThis as GlobalWithJestFlag).__JEST_ACTIVE_TEST__ = true
})
