"use strict";
var __create = Object.create;
var __defProp = Object.defineProperty;
var __getOwnPropDesc = Object.getOwnPropertyDescriptor;
var __getOwnPropNames = Object.getOwnPropertyNames;
var __getProtoOf = Object.getPrototypeOf;
var __hasOwnProp = Object.prototype.hasOwnProperty;
var __export = (target, all) => {
  for (var name in all)
    __defProp(target, name, { get: all[name], enumerable: true });
};
var __copyProps = (to, from, except, desc) => {
  if (from && typeof from === "object" || typeof from === "function") {
    for (let key of __getOwnPropNames(from))
      if (!__hasOwnProp.call(to, key) && key !== except)
        __defProp(to, key, { get: () => from[key], enumerable: !(desc = __getOwnPropDesc(from, key)) || desc.enumerable });
  }
  return to;
};
var __toESM = (mod, isNodeMode, target) => (target = mod != null ? __create(__getProtoOf(mod)) : {}, __copyProps(
  // If the importer is in node compatibility mode or this is not an ESM
  // file that has been converted to a CommonJS file using a Babel-
  // compatible transform (i.e. "__esModule" has not been set), then set
  // "default" to the CommonJS "module.exports" for node compatibility.
  isNodeMode || !mod || !mod.__esModule ? __defProp(target, "default", { value: mod, enumerable: true }) : target,
  mod
));
var __toCommonJS = (mod) => __copyProps(__defProp({}, "__esModule", { value: true }), mod);

// src/utils/path.ts
var path_exports = {};
__export(path_exports, {
  arePathsEqual: () => arePathsEqual,
  getReadablePath: () => getReadablePath,
  getWorkspacePath: () => getWorkspacePath,
  toRelativePath: () => toRelativePath
});
module.exports = __toCommonJS(path_exports);
var path = __toESM(require("path"));
var import_os = __toESM(require("os"));
var vscode = __toESM(require("vscode"));

// src/shared/formatPath.ts
function formatPath(path2, os2, handleSpace = true) {
  let formattedPath = path2;
  if (os2 === "win32") {
    formattedPath = formattedPath.startsWith("\\") ? formattedPath : `\\${formattedPath}`;
  } else {
    formattedPath = formattedPath.startsWith("/") ? formattedPath : `/${formattedPath}`;
  }
  if (handleSpace) {
    formattedPath = formattedPath.replaceAll(" ", os2 === "win32" ? "/ " : "\\ ");
  }
  return formattedPath;
}

// src/utils/path.ts
function toPosixPath(p) {
  const isExtendedLengthPath = p.startsWith("\\\\?\\");
  if (isExtendedLengthPath) {
    return p;
  }
  return p.replace(/\\/g, "/");
}
String.prototype.toPosix = function() {
  return toPosixPath(this);
};
function arePathsEqual(path1, path2) {
  if (!path1 && !path2) {
    return true;
  }
  if (!path1 || !path2) {
    return false;
  }
  path1 = normalizePath(path1);
  path2 = normalizePath(path2);
  if (process.platform === "win32") {
    return path1.toLowerCase() === path2.toLowerCase();
  }
  return path1 === path2;
}
function normalizePath(p) {
  let normalized = path.normalize(p);
  if (normalized.length > 1 && (normalized.endsWith("/") || normalized.endsWith("\\"))) {
    normalized = normalized.slice(0, -1);
  }
  return normalized;
}
function getBasenameBrowser(p) {
  p = p.replace(/[\\/]$/, "");
  const lastSlashIndex = Math.max(p.lastIndexOf("/"), p.lastIndexOf("\\"));
  return lastSlashIndex === -1 ? p : p.slice(lastSlashIndex + 1);
}
function getReadablePath(cwd, relPath) {
  relPath = relPath || "";
  const absolutePath = path.resolve(cwd, relPath);
  if (arePathsEqual(cwd, path.join(import_os.default.homedir(), "Desktop"))) {
    return absolutePath.toPosix();
  }
  if (arePathsEqual(path.normalize(absolutePath), path.normalize(cwd))) {
    return getBasenameBrowser(absolutePath).toPosix();
  } else {
    const normalizedRelPath = path.relative(cwd, absolutePath);
    if (absolutePath.includes(cwd)) {
      return normalizedRelPath.toPosix();
    } else {
      return absolutePath.toPosix();
    }
  }
}
var toRelativePath = (filePath, cwd) => {
  const relativePath = path.relative(cwd, filePath);
  const pathWithTrailingSlash = filePath.endsWith("/") || filePath.endsWith("\\") ? relativePath + (process.platform === "win32" ? "\\" : "/") : relativePath;
  return formatPath(pathWithTrailingSlash, process.platform);
};
var getWorkspacePath = (defaultCwdPath = "") => {
  const cwdPath = vscode.workspace.workspaceFolders?.map((folder) => folder.uri.fsPath).at(0) || defaultCwdPath;
  const currentFileUri = vscode.window.activeTextEditor?.document.uri;
  if (currentFileUri) {
    const workspaceFolder = vscode.workspace.getWorkspaceFolder(currentFileUri);
    return workspaceFolder?.uri.fsPath || cwdPath;
  }
  return cwdPath;
};
// Annotate the CommonJS export names for ESM import in node:
0 && (module.exports = {
  arePathsEqual,
  getReadablePath,
  getWorkspacePath,
  toRelativePath
});
