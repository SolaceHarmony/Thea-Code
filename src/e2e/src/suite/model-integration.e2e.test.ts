import * as assert from "assert"
import * as vscode from "vscode"
import { EXTENSION_ID } from "../thea-constants"
import type { TheaCodeAPI } from "../../../exports/thea-code"
import GenericProviderMock, { PROVIDER_CONFIGS } from "../../../../test/generic-provider-mock/server"

suite("Model Integration E2E", () => {
    let extension: vscode.Extension<any> | undefined
    let api: TheaCodeAPI | undefined
    let mockServer: GenericProviderMock | undefined
    let mockPort: number | undefined

    suiteSetup(async function () {
        this.timeout(60_000)

        // Start mock server
        mockServer = new GenericProviderMock(PROVIDER_CONFIGS.openai)
        mockPort = await mockServer.start()
        console.log(`Mock server started on port ${mockPort}`)

        // Get extension and API
        extension = vscode.extensions.getExtension(EXTENSION_ID)
        assert.ok(extension, `Extension ${EXTENSION_ID} should be found`)

        if (!extension.isActive) {
            await extension.activate()
        }

        // Get API from exports or global (setup.ts should have set global.api)
        const exp = extension.exports
        if (exp && typeof exp === "object") {
            api = (exp as any).api || exp
        }

        if (!api && (global as any).api) {
            api = (global as any).api
        }

        assert.ok(api, "TheaCode API should be available")
    })

    suiteTeardown(async () => {
        if (mockServer) {
            await mockServer.stop()
        }
    })

    test("Should communicate with mock model", async function () {
        this.timeout(30_000)
        assert.ok(api, "API must be available")
        assert.ok(mockPort, "Mock server must be running")

        // Configure extension to use mock server
        await api.setConfigurationValue("apiProvider", "openai")
        await api.setConfigurationValue("openAiBaseUrl", `http://127.0.0.1:${mockPort}/v1`)
        await api.setConfigurationValue("openAiApiKey", "sk-test-key")
        await api.setConfigurationValue("openAiModelId", "gpt-4")

        // Start a new task
        const taskId = await api.startNewTask("Hello, are you there?")
        assert.ok(taskId, "Task ID should be returned")

        // Wait for response
        // We can check message history or wait for a specific event
        // For simplicity, we poll the message history

        let responseReceived = false
        let attempts = 0
        while (!responseReceived && attempts < 20) {
            await new Promise(resolve => setTimeout(resolve, 1000))
            const messages = api.getMessages(taskId)

            // Check for assistant response
            const lastMessage = messages[messages.length - 1]
            if (lastMessage && lastMessage.say === "text" && lastMessage.text) {
                responseReceived = true
                console.log("Received response:", lastMessage.text)
            }
            attempts++
        }

        assert.ok(responseReceived, "Should receive response from mock model")

        // Verify request log on mock server
        const logs = mockServer?.getRequestLog()
        assert.ok(logs && logs.length > 0, "Mock server should have received requests")

        const chatRequest = logs?.find(log => log.path === "/v1/chat/completions")
        assert.ok(chatRequest, "Should have sent chat completion request")
        assert.strictEqual(chatRequest.body.messages[0].content, "Hello, are you there?", "Request content should match")
    })
})
