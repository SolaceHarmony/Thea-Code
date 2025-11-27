
import sinon from "sinon"

export const workspace = {
    workspaceFolders: [],
    fs: {
        readFile: sinon.stub(),
        writeFile: sinon.stub(),
    },
    getConfiguration: sinon.stub().returns({
        get: sinon.stub(),
        update: sinon.stub(),
    }),
}

export const window = {
    showInformationMessage: sinon.stub(),
    showWarningMessage: sinon.stub(),
    showErrorMessage: sinon.stub(),
    createOutputChannel: sinon.stub().returns({
        append: sinon.stub(),
        appendLine: sinon.stub(),
        clear: sinon.stub(),
        show: sinon.stub(),
        dispose: sinon.stub(),
    }),
}

export const Uri = {
    parse: sinon.stub().callsFake((path) => ({ path, fsPath: path, scheme: 'file' })),
    file: sinon.stub().callsFake((path) => ({ path, fsPath: path, scheme: 'file' })),
    joinPath: sinon.stub(),
}

export const env = {
    openExternal: sinon.stub(),
}

export const Range = class {
    constructor(public startLine: number, public startChar: number, public endLine: number, public endChar: number) { }
}

export const Position = class {
    constructor(public line: number, public character: number) { }
}

export const Diagnostic = class {
    constructor(public range: any, public message: string, public severity: any) { }
}

export const DiagnosticSeverity = {
    Error: 0,
    Warning: 1,
    Information: 2,
    Hint: 3,
}

export const EventEmitter = class {
    event = sinon.stub()
    fire = sinon.stub()
    dispose = sinon.stub()
}

export const Disposable = class {
    static from(...disposables: any[]) {
        return { dispose: () => { } }
    }
    dispose() { }
}

export const ThemeColor = class {
    constructor(public id: string) { }
}

export const ThemeIcon = class {
    constructor(public id: string) { }
}

export enum ViewColumn {
    Active = -1,
    Beside = -2,
    One = 1,
    Two = 2,
    Three = 3,
    Four = 4,
    Five = 5,
    Six = 6,
    Seven = 7,
    Eight = 8,
    Nine = 9,
}

export default {
    workspace,
    window,
    Uri,
    env,
    Range,
    Position,
    Diagnostic,
    DiagnosticSeverity,
    EventEmitter,
    Disposable,
    ThemeColor,
    ThemeIcon,
    ViewColumn,
}
