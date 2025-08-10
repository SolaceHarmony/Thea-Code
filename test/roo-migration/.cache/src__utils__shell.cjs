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

// src/utils/shell.ts
var shell_exports = {};
__export(shell_exports, {
  getShell: () => getShell
});
module.exports = __toCommonJS(shell_exports);
var vscode = __toESM(require("vscode"));
var import_os = require("os");
var SHELL_PATHS = {
  // Windows paths
  POWERSHELL_7: "C:\\Program Files\\PowerShell\\7\\pwsh.exe",
  POWERSHELL_LEGACY: "C:\\Windows\\System32\\WindowsPowerShell\\v1.0\\powershell.exe",
  CMD: "C:\\Windows\\System32\\cmd.exe",
  WSL_BASH: "/bin/bash",
  // Unix paths
  MAC_DEFAULT: "/bin/zsh",
  LINUX_DEFAULT: "/bin/bash",
  CSH: "/bin/csh",
  BASH: "/bin/bash",
  KSH: "/bin/ksh",
  SH: "/bin/sh",
  ZSH: "/bin/zsh",
  DASH: "/bin/dash",
  TCSH: "/bin/tcsh",
  FALLBACK: "/bin/sh"
};
function getWindowsTerminalConfig() {
  try {
    const config = vscode.workspace.getConfiguration("terminal.integrated");
    const defaultProfileName = config.get("defaultProfile.windows");
    const profiles = config.get("profiles.windows") || {};
    return { defaultProfileName, profiles };
  } catch {
    return { defaultProfileName: null, profiles: {} };
  }
}
function getMacTerminalConfig() {
  try {
    const config = vscode.workspace.getConfiguration("terminal.integrated");
    const defaultProfileName = config.get("defaultProfile.osx");
    const profiles = config.get("profiles.osx") || {};
    return { defaultProfileName, profiles };
  } catch {
    return { defaultProfileName: null, profiles: {} };
  }
}
function getLinuxTerminalConfig() {
  try {
    const config = vscode.workspace.getConfiguration("terminal.integrated");
    const defaultProfileName = config.get("defaultProfile.linux");
    const profiles = config.get("profiles.linux") || {};
    return { defaultProfileName, profiles };
  } catch {
    return { defaultProfileName: null, profiles: {} };
  }
}
function getWindowsShellFromVSCode() {
  const { defaultProfileName, profiles } = getWindowsTerminalConfig();
  if (!defaultProfileName) {
    return null;
  }
  const profile = profiles[defaultProfileName];
  if (defaultProfileName.toLowerCase().includes("powershell")) {
    if (profile?.path) {
      return profile.path;
    } else if (profile?.source === "PowerShell") {
      return SHELL_PATHS.POWERSHELL_7;
    }
    return SHELL_PATHS.POWERSHELL_LEGACY;
  }
  if (profile?.path) {
    return profile.path;
  }
  if (profile?.source === "WSL" || defaultProfileName.toLowerCase().includes("wsl")) {
    return SHELL_PATHS.WSL_BASH;
  }
  return SHELL_PATHS.CMD;
}
function getMacShellFromVSCode() {
  const { defaultProfileName, profiles } = getMacTerminalConfig();
  if (!defaultProfileName) {
    return null;
  }
  const profile = profiles[defaultProfileName];
  return profile?.path || null;
}
function getLinuxShellFromVSCode() {
  const { defaultProfileName, profiles } = getLinuxTerminalConfig();
  if (!defaultProfileName) {
    return null;
  }
  const profile = profiles[defaultProfileName];
  return profile?.path || null;
}
function getShellFromUserInfo() {
  try {
    const { shell } = (0, import_os.userInfo)();
    return shell || null;
  } catch {
    return null;
  }
}
function getShellFromEnv() {
  const { env } = process;
  if (process.platform === "win32") {
    return env.COMSPEC || "C:\\Windows\\System32\\cmd.exe";
  }
  if (process.platform === "darwin") {
    return env.SHELL || "/bin/zsh";
  }
  if (process.platform === "linux") {
    return env.SHELL || "/bin/bash";
  }
  return null;
}
function getShell() {
  if (process.platform === "win32") {
    const windowsShell = getWindowsShellFromVSCode();
    if (windowsShell) {
      return windowsShell;
    }
  } else if (process.platform === "darwin") {
    const macShell = getMacShellFromVSCode();
    if (macShell) {
      return macShell;
    }
  } else if (process.platform === "linux") {
    const linuxShell = getLinuxShellFromVSCode();
    if (linuxShell) {
      return linuxShell;
    }
  }
  const userInfoShell = getShellFromUserInfo();
  if (userInfoShell) {
    return userInfoShell;
  }
  const envShell = getShellFromEnv();
  if (envShell) {
    return envShell;
  }
  if (process.platform === "win32") {
    return SHELL_PATHS.CMD;
  }
  return SHELL_PATHS.FALLBACK;
}
// Annotate the CommonJS export names for ESM import in node:
0 && (module.exports = {
  getShell
});
