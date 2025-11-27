import assert from "assert"
import sinon from "sinon"
import { writeToFileTool } from "../writeToFileTool"
import * as vscode from "vscode"

describe("writeToFileTool", () => {
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
            api: {
                getModel: sinon.stub().returns({ id: "claude-3-sonnet" }),
            },
            theaIgnoreController: {
                validateAccess: sinon.stub().returns(true),
            },
            webviewCommunicator: {
                say: sinon.stub(),
                ask: sinon.stub(),
            },
            diffViewProvider: {
                editType: undefined,
                isEditing: false,
                open: sinon.stub(),
                update: sinon.stub(),
                scrollToFirstDiff: sinon.stub(),
                revertChanges: sinon.stub(),
                reset: sinon.stub(),
                saveChanges: sinon.stub().resolves({
                    newProblemsMessage: "",
                    userEdits: undefined,
                    finalContent: "new content",
                }),
                originalContent: "original content",
            },
            sayAndCreateMissingParamError: sinon.stub().resolves("Missing param error"),
            consecutiveMistakeCount: 0,
            didEditFile: false,
        };

        // Reset vscode stubs
        (vscode.window.showWarningMessage as unknown as sinon.SinonStub).reset();
        (vscode.env.openExternal as unknown as sinon.SinonStub).reset();
        (vscode.workspace as any).workspaceFolders = [{ uri: { fsPath: "/test/workspace" } }];

        mockDependencies = {
            vscode,
            fs: {
                fileExistsAtPath: sinon.stub().resolves(false),
            },
            path: {
                getReadablePath: sinon.stub().callsFake((cwd, p) => p || ""),
            },
            extractText: {
                addLineNumbers: sinon.stub().callsFake((content) => content),
                stripLineNumbers: sinon.stub().callsFake((content) => content),
                everyLineHasLineNumbers: sinon.stub().returns(false),
            },
            detectOmission: {
                detectCodeOmission: sinon.stub().returns(false),
            },
        }
    })

    afterEach(() => {
        sinon.restore()
    })

    it("should fail if path is missing", async () => {
        await writeToFileTool(
            mockTheaTask,
            { params: { content: "content", line_count: "10" } } as any,
            mockAskApproval,
            mockHandleError,
            mockPushToolResult,
            mockRemoveClosingTag,
            mockDependencies,
        )

        assert(mockTheaTask.sayAndCreateMissingParamError.calledWith("write_to_file", "path"))
        assert(mockPushToolResult.calledWith("Missing param error"))
    })

    it("should fail if content is missing", async () => {
        await writeToFileTool(
            mockTheaTask,
            { params: { path: "test.txt", line_count: "10" } } as any,
            mockAskApproval,
            mockHandleError,
            mockPushToolResult,
            mockRemoveClosingTag,
            mockDependencies,
        )

        assert(mockTheaTask.sayAndCreateMissingParamError.calledWith("write_to_file", "content"))
        assert(mockPushToolResult.calledWith("Missing param error"))
    })

    it("should fail if line_count is missing", async () => {
        await writeToFileTool(
            mockTheaTask,
            { params: { path: "test.txt", content: "content" } } as any,
            mockAskApproval,
            mockHandleError,
            mockPushToolResult,
            mockRemoveClosingTag,
            mockDependencies,
        )

        assert(mockTheaTask.sayAndCreateMissingParamError.calledWith("write_to_file", "line_count"))
        assert(mockPushToolResult.calledWith("Missing param error"))
    })

    it("should check for file access", async () => {
        mockTheaTask.theaIgnoreController.validateAccess.returns(false)

        await writeToFileTool(
            mockTheaTask,
            { params: { path: "ignored.txt", content: "content", line_count: "10" } } as any,
            mockAskApproval,
            mockHandleError,
            mockPushToolResult,
            mockRemoveClosingTag,
            mockDependencies,
        )

        assert(mockTheaTask.webviewCommunicator.say.calledWith("theaignore_error", "ignored.txt"))
        assert(mockPushToolResult.called)
    })

    it("should handle new file creation", async () => {
        mockDependencies.fs.fileExistsAtPath.resolves(false)

        await writeToFileTool(
            mockTheaTask,
            { params: { path: "new.txt", content: "new content", line_count: "1" } } as any,
            mockAskApproval,
            mockHandleError,
            mockPushToolResult,
            mockRemoveClosingTag,
            mockDependencies,
        )

        assert(mockTheaTask.diffViewProvider.open.calledWith("new.txt"))
        assert(mockTheaTask.diffViewProvider.update.calledWith("new content", true))
        assert(mockAskApproval.called)
        assert(mockTheaTask.diffViewProvider.saveChanges.called)
        assert(mockPushToolResult.firstCall.args[0].includes("successfully saved"))
    })

    it("should handle existing file modification", async () => {
        mockDependencies.fs.fileExistsAtPath.resolves(true)

        await writeToFileTool(
            mockTheaTask,
            { params: { path: "existing.txt", content: "modified content", line_count: "1" } } as any,
            mockAskApproval,
            mockHandleError,
            mockPushToolResult,
            mockRemoveClosingTag,
            mockDependencies,
        )

        assert(mockTheaTask.diffViewProvider.open.calledWith("existing.txt"))
        assert(mockTheaTask.diffViewProvider.update.calledWith("modified content", true))
        assert(mockAskApproval.called)
        assert(mockTheaTask.diffViewProvider.saveChanges.called)
        assert(mockPushToolResult.firstCall.args[0].includes("successfully saved"))
    })
})
