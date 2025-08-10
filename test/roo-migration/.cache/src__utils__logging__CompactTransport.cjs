"use strict";
var __defProp = Object.defineProperty;
var __getOwnPropDesc = Object.getOwnPropertyDescriptor;
var __getOwnPropNames = Object.getOwnPropertyNames;
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
var __toCommonJS = (mod) => __copyProps(__defProp({}, "__esModule", { value: true }), mod);

// src/utils/logging/CompactTransport.ts
var CompactTransport_exports = {};
__export(CompactTransport_exports, {
  CompactTransport: () => CompactTransport
});
module.exports = __toCommonJS(CompactTransport_exports);
var import_fs = require("fs");
var import_path = require("path");

// src/utils/logging/types.ts
var LOG_LEVELS = ["debug", "info", "warn", "error", "fatal"];

// src/utils/logging/CompactTransport.ts
var DEFAULT_CONFIG = {
  level: "debug",
  fileOutput: {
    enabled: true,
    path: "./logs/app.log"
  }
};
function isLevelEnabled(configLevel, entryLevel) {
  const configIdx = LOG_LEVELS.indexOf(configLevel);
  const entryIdx = LOG_LEVELS.indexOf(entryLevel);
  return entryIdx >= configIdx;
}
var CompactTransport = class {
  /**
   * Creates a new CompactTransport instance
   * @param config - Optional transport configuration
   */
  constructor(config = DEFAULT_CONFIG) {
    this.config = config;
    this.sessionStart = Date.now();
    this.lastTimestamp = this.sessionStart;
    if (config.fileOutput?.enabled) {
      this.filePath = config.fileOutput.path;
    }
  }
  sessionStart;
  lastTimestamp;
  filePath;
  initialized = false;
  /**
   * Ensures the log file is initialized with proper directory structure and session start marker
   * @private
   * @throws {Error} If file initialization fails
   */
  ensureInitialized() {
    if (this.initialized || !this.filePath) return;
    try {
      (0, import_fs.mkdirSync)((0, import_path.dirname)(this.filePath), { recursive: true });
      (0, import_fs.writeFileSync)(this.filePath, "", { flag: "w" });
      const sessionStart = {
        t: 0,
        l: "info",
        m: "Log session started",
        d: { timestamp: new Date(this.sessionStart).toISOString() }
      };
      (0, import_fs.writeFileSync)(this.filePath, JSON.stringify(sessionStart) + "\n", { flag: "w" });
      this.initialized = true;
    } catch (err) {
      throw new Error(`Failed to initialize log file: ${err.message}`);
    }
  }
  /**
   * Writes a log entry to configured outputs (console and/or file)
   * @param entry - The log entry to write
   */
  write(entry) {
    const deltaT = entry.t - this.lastTimestamp;
    this.lastTimestamp = entry.t;
    const compact = {
      ...entry,
      t: deltaT
    };
    const output = JSON.stringify(compact) + "\n";
    if (this.config.level && isLevelEnabled(this.config.level, entry.l)) {
      process.stdout.write(output);
    }
    if (this.filePath) {
      this.ensureInitialized();
      (0, import_fs.writeFileSync)(this.filePath, output, { flag: "a" });
    }
  }
  /**
   * Closes the transport and writes session end marker
   */
  close() {
    if (this.filePath && this.initialized) {
      const sessionEnd = {
        t: Date.now() - this.lastTimestamp,
        l: "info",
        m: "Log session ended",
        d: { timestamp: (/* @__PURE__ */ new Date()).toISOString() }
      };
      (0, import_fs.writeFileSync)(this.filePath, JSON.stringify(sessionEnd) + "\n", { flag: "a" });
    }
  }
};
// Annotate the CommonJS export names for ESM import in node:
0 && (module.exports = {
  CompactTransport
});
