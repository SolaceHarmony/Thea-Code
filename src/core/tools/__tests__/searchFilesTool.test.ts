import assert from "assert"
import sinon from "sinon"
import { searchFilesTool } from "../searchFilesTool"
import * as pathUtils from "../../../utils/path"
import * as ripgrep from "../../../services/ripgrep"

describe("searchFilesTool", () => {
    let mockTheaTask: any
    let mockAskApproval: sinon.SinonStub
    let mockHandleError: sinon.SinonStub
    let mockPushToolResult: sinon.SinonStub
    let mockRemoveClosingTag: sinon.SinonStub
    let mockDependencies: any

    beforeEach(() => {
        sinon.restore()

        mockAskApproval = sinon.stub().resolves(true)
        mockHandleError = sinon.stub()
        mockPushToolResult = sinon.stub()
        mockRemoveClosingTag = sinon.stub().callsFake((tag, content) => content)

        mockTheaTask = {
            cwd: "/test/workspace",
            webviewCommunicator: {
                ask: sinon.stub().resolves(),
            },
            sayAndCreateMissingParamError: sinon.stub().resolves("Missing param error"),
            consecutiveMistakeCount: 0,
            theaIgnoreController: {},
        }

        mockDependencies = {
            path: {
                getReadablePath: sinon.stub().callsFake((cwd, p) => p || ""),
            },
            ripgrep: {
                regexSearchFiles: sinon.stub().resolves("search results"),
            },
        }
    })

    afterEach(() => {
        sinon.restore()
    })

    it("should fail if path is missing", async () => {
        await searchFilesTool(
            mockTheaTask,
            { params: { regex: "pattern" } } as any,
            mockAskApproval,
            mockHandleError,
            mockPushToolResult,
            mockRemoveClosingTag,
            mockDependencies,
        )

        assert(mockTheaTask.sayAndCreateMissingParamError.calledWith("search_files", "path"))
        assert(mockPushToolResult.calledWith("Missing param error"))
    })

    it("should fail if regex is missing", async () => {
        await searchFilesTool(
            mockTheaTask,
            { params: { path: "." } } as any,
            mockAskApproval,
            mockHandleError,
            mockPushToolResult,
            mockRemoveClosingTag,
            mockDependencies,
        )

        assert(mockTheaTask.sayAndCreateMissingParamError.calledWith("search_files", "regex"))
        assert(mockPushToolResult.calledWith("Missing param error"))
    })

    it("should handle partial updates", async () => {
        await searchFilesTool(
            mockTheaTask,
            { params: { path: ".", regex: "pattern" }, partial: true } as any,
            mockAskApproval,
            mockHandleError,
            mockPushToolResult,
            mockRemoveClosingTag,
            mockDependencies,
        )

        assert(mockTheaTask.webviewCommunicator.ask.called)
        assert(mockPushToolResult.notCalled)
    })

    it("should perform search and return results", async () => {
        await searchFilesTool(
            mockTheaTask,
            { params: { path: ".", regex: "pattern" } } as any,
            mockAskApproval,
            mockHandleError,
            mockPushToolResult,
            mockRemoveClosingTag,
            mockDependencies,
        )

        assert(mockDependencies.ripgrep.regexSearchFiles.called)
        assert(mockAskApproval.called)
        assert(mockPushToolResult.calledWith("search results"))
    })

    it("should handle user rejection", async () => {
        mockAskApproval.resolves(false)

        await searchFilesTool(
            mockTheaTask,
            { params: { path: ".", regex: "pattern" } } as any,
            mockAskApproval,
            mockHandleError,
            mockPushToolResult,
            mockRemoveClosingTag,
            mockDependencies,
        )

        assert(mockDependencies.ripgrep.regexSearchFiles.called)
        assert(mockAskApproval.called)
        assert(mockPushToolResult.notCalled)
    })
})
