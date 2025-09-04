import * as vscode from 'vscode'
import { getShell as getShellCore } from './shell'

export function getShell(): string {
  return getShellCore(vscode)
}

