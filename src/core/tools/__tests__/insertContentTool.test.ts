import assert from "assert"
import sinon from "sinon"
import { insertContentTool } from "../insertContentTool"
import * as pathUtils from "../../../utils/path"
import * as fsUtils from "../../../utils/fs"
import * as insertGroupsModule from "../../diff/insert-groups"

describe("insertContentTool", () => {
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
                ask: sinon.stub().resolves({ response: "yesButtonClicked" }),
                say: sinon.stub(),
            },
            sayAndCreateMissingParamError: sinon.stub().resolves("Missing param error"),
            consecutiveMistakeCount: 0,
            diffViewProvider: {
                editType: undefined,
                originalContent: undefined,
                isEditing: false,
                open: sinon.stub(),
                update: sinon.stub(),
                scrollToFirstDiff: sinon.stub(),
                revertChanges: sinon.stub(),
                saveChanges: sinon.stub().resolves({
                    newProblemsMessage: "",
                    userEdits: undefined,
                    finalContent: "final content",
                }),
                reset: sinon.stub(),
            },
            didEditFile: false,
        }

        mockDependencies = {
            path: {
                getReadablePath: sinon.stub().callsFake((cwd, p) => p || ""),
            },
            fs: {
                fileExistsAtPath: sinon.stub().resolves(true),
                readFile: sinon.stub().resolves("line1\nline2\nline3"),
            },
            insertGroups: sinon.stub().returns(["line1", "inserted", "line2", "line3"]),
        }
    })

    afterEach(() => {
        sinon.restore()
    })

    it("should fail if path is missing", async () => {
        await insertContentTool(
            mockTheaTask,
            { params: { operations: "[]" } } as any,
            mockAskApproval,
            mockHandleError,
            mockPushToolResult,
            mockRemoveClosingTag,
            mockDependencies,
        )

        assert(mockTheaTask.sayAndCreateMissingParamError.calledWith("insert_content", "path"))
        assert(mockPushToolResult.calledWith("Missing param error"))
    })

    it("should fail if operations is missing", async () => {
        await insertContentTool(
            mockTheaTask,
            { params: { path: "file.txt" } } as any,
            mockAskApproval,
            mockHandleError,
            mockPushToolResult,
            mockRemoveClosingTag,
            mockDependencies,
        )

        assert(mockTheaTask.sayAndCreateMissingParamError.calledWith("insert_content", "operations"))
        assert(mockPushToolResult.calledWith("Missing param error"))
    })

    it("should fail if file does not exist", async () => {
        mockDependencies.fs.fileExistsAtPath.resolves(false)

        await insertContentTool(
            mockTheaTask,
            { params: { path: "file.txt", operations: "[]" } } as any,
            mockAskApproval,
            mockHandleError,
            mockPushToolResult,
            mockRemoveClosingTag,
            mockDependencies,
        )

        assert(mockTheaTask.webviewCommunicator.say.calledWith("error"))
        assert(mockPushToolResult.called)
        assert(mockPushToolResult.firstCall.args[0].includes("File does not exist"))
    })

    it("should fail if operations is invalid JSON", async () => {
        await insertContentTool(
            mockTheaTask,
            { params: { path: "file.txt", operations: "invalid" } } as any,
            mockAskApproval,
            mockHandleError,
            mockPushToolResult,
            mockRemoveClosingTag,
            mockDependencies,
        )

        assert(mockTheaTask.webviewCommunicator.say.calledWith("error"))
        assert(mockPushToolResult.called)
    })

    it("should handle partial updates", async () => {
        await insertContentTool(
            mockTheaTask,
            { params: { path: "file.txt", operations: "[]" }, partial: true } as any,
            mockAskApproval,
            mockHandleError,
            mockPushToolResult,
            mockRemoveClosingTag,
            mockDependencies,
        )

        assert(mockTheaTask.webviewCommunicator.ask.called)
        assert(mockPushToolResult.notCalled)
    })

    it("should insert content successfully", async () => {
        const operations = JSON.stringify([{ start_line: 2, content: "inserted" }])
        await insertContentTool(
            mockTheaTask,
            { params: { path: "file.txt", operations } } as any,
            mockAskApproval,
            mockHandleError,
            mockPushToolResult,
            mockRemoveClosingTag,
            mockDependencies,
        )

        assert(mockDependencies.fs.readFile.called)
        assert(mockDependencies.insertGroups.called)
        assert(mockTheaTask.diffViewProvider.open.called)
        assert(mockTheaTask.diffViewProvider.update.called)
        assert(mockTheaTask.webviewCommunicator.ask.called)
        assert(mockTheaTask.diffViewProvider.saveChanges.called)
        assert(mockPushToolResult.called)
        assert(mockPushToolResult.firstCall.args[0].includes("successfully inserted"))
    })

    it("should handle user rejection", async () => {
        mockTheaTask.webviewCommunicator.ask.resolves({ response: "cancelButtonClicked" })
        const operations = JSON.stringify([{ start_line: 2, content: "inserted" }])

        await insertContentTool(
            mockTheaTask,
            { params: { path: "file.txt", operations } } as any,
            mockAskApproval,
            mockHandleError,
            mockPushToolResult,
            mockRemoveClosingTag,
            mockDependencies,
        )

        assert(mockTheaTask.diffViewProvider.revertChanges.called)
        assert(mockPushToolResult.calledWith("Changes were rejected by the user."))
    })
})
