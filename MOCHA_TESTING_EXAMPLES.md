# Mocha Testing: Real Classes vs Mocks - Practical Examples

## Quick Reference: Mocha vs Jest

```typescript
// JEST                          // MOCHA (TDD)
describe("Suite", () => {        // suite("Suite", () => {
  beforeEach(() => {})           //   setup(() => {})
  afterEach(() => {})            //   teardown(() => {})
  beforeAll(() => {})            //   suiteSetup(() => {})
  afterAll(() => {})             //   suiteTeardown(() => {})
  
  it("test", () => {})           //   test("test", () => {})
  it.skip("test", () => {})      //   test.skip("test", () => {})
  it.only("test", () => {})      //   test.only("test", () => {})
})                              // })
```

---

## Pattern 1: Real Classes (Default)

Use this pattern when testing actual functionality without external dependencies.

### ✅ Good Example
```typescript
import * as assert from 'assert'
import * as sinon from 'sinon'
import { OpenAiHandler } from '../openai'

suite('OpenAiHandler', () => {
  let handler: OpenAiHandler
  let sandbox: sinon.SinonSandbox

  setup(() => {
    sandbox = sinon.createSandbox()
    // Initialize REAL handler with real options
    handler = new OpenAiHandler({
      openAiApiKey: 'test-key',
      openAiModelId: 'gpt-4',
      openAiBaseUrl: 'https://api.openai.com/v1'
    })
  })

  test('should initialize correctly', () => {
    // Test REAL object behavior
    assert.ok(handler instanceof OpenAiHandler)
    assert.strictEqual(handler.getModel().id, 'gpt-4')
  })

  teardown(() => {
    sandbox.restore()
  })
})
```

---

## Pattern 2: Mocking External Dependencies

Use this pattern when you need to mock HTTP calls or external services.

### ✅ Good Example
```typescript
import * as assert from 'assert'
import * as sinon from 'sinon'
import * as https from 'https'
import { ApiClient } from '../api-client'

suite('ApiClient', () => {
  let client: ApiClient
  let sandbox: sinon.SinonSandbox

  setup(() => {
    sandbox = sinon.createSandbox()
    client = new ApiClient()
    
    // MOCK ONLY the external HTTP call
    sandbox.stub(https, 'request').callsFake((options, callback) => {
      // Simulate response
      callback(null, { statusCode: 200, data: { success: true } })
    })
  })

  test('should handle API response', async () => {
    const result = await client.fetch('/endpoint')
    
    // Test with mocked HTTP
    assert.strictEqual(result.success, true)
  })

  teardown(() => {
    sandbox.restore()
  })
})
```

---

## Pattern 3: Spying on Real Classes

Use this pattern to verify real classes were called correctly.

### ✅ Good Example
```typescript
import * as assert from 'assert'
import * as sinon from 'sinon'
import { Logger } from '../logger'
import { Application } from '../application'

suite('Application Logging', () => {
  let app: Application
  let logger: Logger
  let sandbox: sinon.SinonSandbox

  setup(() => {
    sandbox = sinon.createSandbox()
    logger = new Logger()
    app = new Application(logger)
    
    // SPY on the logger to track calls
    sandbox.spy(logger, 'info')
  })

  test('should log on startup', () => {
    app.start()
    
    // Verify the REAL logger was called
    assert.ok(logger.info.calledOnce)
    assert.ok(logger.info.calledWith('Application started'))
  })

  teardown(() => {
    sandbox.restore()
  })
})
```

---

## Pattern 4: Partial Mocking (Real + Stub)

Use this pattern when testing a real class that depends on external services.

### ✅ Good Example
```typescript
import * as assert from 'assert'
import * as sinon from 'sinon'
import { PaymentProcessor } from '../payment'
import { BillingService } from '../billing'

suite('PaymentProcessor', () => {
  let processor: PaymentProcessor
  let billing: BillingService
  let sandbox: sinon.SinonSandbox

  setup(() => {
    sandbox = sinon.createSandbox()
    
    // Create REAL processor
    processor = new PaymentProcessor()
    
    // Create REAL billing service
    billing = new BillingService()
    
    // Mock ONLY the payment gateway (external)
    sandbox.stub(processor, 'chargeCard').resolves({ success: true })
  })

  test('should process payment with real logic', async () => {
    // REAL processor logic runs
    // But external chargeCard call is mocked
    const result = await processor.processPayment(100)
    
    assert.strictEqual(result.success, true)
  })

  teardown(() => {
    sandbox.restore()
  })
})
```

---

## Pattern 5: Error Testing with Mocks

Use this pattern to test error handling without triggering real errors.

### ✅ Good Example
```typescript
import * as assert from 'assert'
import * as sinon from 'sinon'
import { FileReader } from '../file-reader'
import * as fs from 'fs'

suite('FileReader Error Handling', () => {
  let reader: FileReader
  let sandbox: sinon.SinonSandbox

  setup(() => {
    sandbox = sinon.createSandbox()
    reader = new FileReader()
  })

  test('should handle file not found', async () => {
    // Mock fs.readFile to simulate error
    sandbox.stub(fs, 'readFile').rejects(new Error('ENOENT: File not found'))
    
    // Test REAL error handling in FileReader
    assert.rejects(
      async () => await reader.read('/nonexistent.txt'),
      (err) => err instanceof Error && err.message.includes('ENOENT')
    )
  })

  test('should handle permission denied', async () => {
    // Mock different error condition
    sandbox.stub(fs, 'readFile').rejects(new Error('EACCES: Permission denied'))
    
    assert.rejects(
      async () => await reader.read('/protected.txt'),
      (err) => err instanceof Error && err.message.includes('EACCES')
    )
  })

  teardown(() => {
    sandbox.restore()
  })
})
```

---

## Pattern 6: Type-Safe Configuration Testing

Use this pattern for testing configuration handling.

### ✅ Good Example
```typescript
import * as assert from 'assert'
import { Configuration } from '../config'

suite('Configuration', () => {
  let config: Configuration

  setup(() => {
    // REAL config with test values
    config = new Configuration({
      apiKey: 'test-key',
      timeout: 5000,
      retries: 3
    })
  })

  test('should validate api key', () => {
    const apiKey = config.get('apiKey')
    
    // Type-safe assertion
    assert.ok(apiKey === undefined || typeof apiKey === 'string')
    assert.strictEqual(apiKey, 'test-key')
  })

  test('should apply defaults', () => {
    const unsetValue = config.get('nonExistent')
    
    assert.strictEqual(unsetValue, undefined)
  })
})
```

---

## Pattern 7: Async Testing

Use this pattern for async operations.

### ✅ Good Example
```typescript
import * as assert from 'assert'
import { AsyncService } from '../async-service'

suite('AsyncService', () => {
  let service: AsyncService

  setup(() => {
    service = new AsyncService()
  })

  test('should process data asynchronously', async function() {
    // Set custom timeout if needed
    this.timeout(5000)
    
    const result = await service.processData({ value: 42 })
    
    assert.strictEqual(result.processed, true)
    assert.strictEqual(result.value, 42)
  })

  test('should handle rejections', async () => {
    assert.rejects(
      async () => await service.processData(null),
      (err) => err instanceof Error
    )
  })
})
```

---

## Anti-Patterns (What NOT to Do)

### ❌ Don't: Over-mock real classes
```typescript
// BAD - Mocking everything defeats the purpose of testing
suite('MyService', () => {
  let service: MyService
  let sandbox: sinon.SinonSandbox

  setup(() => {
    sandbox = sinon.createSandbox()
    
    // DON'T do this - defeats the test purpose
    sandbox.stub(MyService.prototype, 'doSomething').returns('mocked')
    service = new MyService()
  })
})
```

### ❌ Don't: Use `any` type
```typescript
// BAD - No type safety
let api: any = extension.exports  // DON'T
await api.someMethod()  // Unsafe

// GOOD - Type safe
let api: unknown = extension.exports  // Good
if (typeof api === 'object' && api !== null && 'someMethod' in api) {
  const safeApi = api as Record<string, unknown>
  if (typeof safeApi.someMethod === 'function') {
    await (safeApi.someMethod as () => Promise<void>)()
  }
}
```

### ❌ Don't: Mix Jest and Mocha syntax
```typescript
// BAD - Mixing patterns
suite('Test', () => {
  beforeEach(() => {})  // Jest pattern
  setup(() => {})       // Mocha pattern - confusing!
})

// GOOD - Consistent Mocha
suite('Test', () => {
  setup(() => {})       // Only use Mocha patterns
  teardown(() => {})
})
```

### ❌ Don't: Forget to restore sandbox
```typescript
// BAD - Sandbox leaks to next test
suite('Test', () => {
  setup(() => {
    sandbox = sinon.createSandbox()
  })
  
  test('test 1', () => {
    sandbox.stub(something, 'method')
    // BUG: Never restored!
  })
})

// GOOD - Always restore
suite('Test', () => {
  setup(() => {
    sandbox = sinon.createSandbox()
  })
  
  teardown(() => {
    sandbox.restore()  // Clean up!
  })
})
```

---

## Checklist for New Tests

- [ ] Use Mocha TDD style (suite/test/setup/teardown)
- [ ] Start with real classes, not mocks
- [ ] Use `unknown` instead of `any`
- [ ] Mock only external dependencies
- [ ] Create sandbox in `setup()`, restore in `teardown()`
- [ ] Use assertions from `assert/strict`
- [ ] Handle async with `async function` or `Promise`
- [ ] Set timeouts with `this.timeout()` if needed
- [ ] Document why any mocks exist
- [ ] Run linting before committing (`npm run lint`)

---

## Useful Sinon Patterns

```typescript
// Create sandbox
const sandbox = sinon.createSandbox()

// Stub a method
const stub = sandbox.stub(obj, 'method').returns('value')
stub.resolves(Promise.resolve('value'))  // For async
stub.rejects(new Error('Failed'))        // For errors

// Spy on a method
const spy = sandbox.spy(obj, 'method')
spy.calledOnce       // Check call count
spy.calledWith(arg)  // Check arguments
spy.callCount        // Get exact count
spy.restore()        // Restore single spy

// Restore all stubs/spies
sandbox.restore()    // Clean up everything

// Verify then restore
sandbox.assert.calledOnce(spy)
sandbox.restore()
```

---

## References

- Project Conventions: `/docs/TESTING_LESSONS_LEARNED.md`
- Mocha Documentation: https://mochajs.org/
- Sinon Documentation: https://sinonjs.org/
- Node Assert: https://nodejs.org/api/assert.html
