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

// src/utils/port-utils.ts
var port_utils_exports = {};
__export(port_utils_exports, {
  findAvailablePort: () => findAvailablePort,
  isPortAvailable: () => isPortAvailable,
  waitForPortAvailable: () => waitForPortAvailable,
  waitForPortInUse: () => waitForPortInUse
});
module.exports = __toCommonJS(port_utils_exports);
var tcpPortUsedModule = __toESM(require("tcp-port-used"));

// src/utils/logging/CompactTransport.ts
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

// src/utils/logging/CompactLogger.ts
var CompactLogger = class _CompactLogger {
  transport;
  parentMeta;
  /**
   * Creates a new CompactLogger instance
   * @param transport - Optional custom transport instance
   * @param parentMeta - Optional parent metadata for hierarchical logging
   */
  constructor(transport, parentMeta) {
    this.transport = transport ?? new CompactTransport();
    this.parentMeta = parentMeta;
  }
  /**
   * Logs a debug level message
   * @param message - The message to log
   * @param meta - Optional metadata to include
   */
  debug(message, meta) {
    this.log("debug", message, this.combineMeta(meta));
  }
  /**
   * Logs an info level message
   * @param message - The message to log
   * @param meta - Optional metadata to include
   */
  info(message, meta) {
    this.log("info", message, this.combineMeta(meta));
  }
  /**
   * Logs a warning level message
   * @param message - The message to log
   * @param meta - Optional metadata to include
   */
  warn(message, meta) {
    this.log("warn", message, this.combineMeta(meta));
  }
  /**
   * Logs an error level message
   * @param message - The error message or Error object
   * @param meta - Optional metadata to include
   */
  error(message, meta) {
    this.handleErrorLog("error", message, meta);
  }
  /**
   * Logs a fatal level message
   * @param message - The error message or Error object
   * @param meta - Optional metadata to include
   */
  fatal(message, meta) {
    this.handleErrorLog("fatal", message, meta);
  }
  /**
   * Creates a child logger inheriting this logger's metadata
   * @param meta - Additional metadata for the child logger
   * @returns A new logger instance with combined metadata
   */
  child(meta) {
    const combinedMeta = this.parentMeta ? { ...this.parentMeta, ...meta } : meta;
    return new _CompactLogger(this.transport, combinedMeta);
  }
  /**
   * Closes the logger and its transport
   */
  close() {
    this.transport.close();
  }
  /**
   * Handles logging of error and fatal messages with special error object processing
   * @private
   * @param level - The log level (error or fatal)
   * @param message - The message or Error object to log
   * @param meta - Optional metadata to include
   */
  handleErrorLog(level, message, meta) {
    if (message instanceof Error) {
      const errorMeta = {
        ...meta,
        ctx: meta?.ctx ?? level,
        error: {
          name: message.name,
          message: message.message,
          stack: message.stack
        }
      };
      this.log(level, message.message, this.combineMeta(errorMeta));
    } else {
      this.log(level, message, this.combineMeta(meta));
    }
  }
  /**
   * Combines parent and current metadata with proper context handling
   * @private
   * @param meta - The current metadata to combine with parent metadata
   * @returns Combined metadata or undefined if no metadata exists
   */
  combineMeta(meta) {
    if (!this.parentMeta) {
      return meta;
    }
    if (!meta) {
      return this.parentMeta;
    }
    return {
      ...this.parentMeta,
      ...meta,
      ctx: meta.ctx || this.parentMeta.ctx
    };
  }
  /**
   * Core logging function that processes and writes log entries
   * @private
   * @param level - The log level
   * @param message - The message to log
   * @param meta - Optional metadata to include
   */
  log(level, message, meta) {
    const entry = {
      t: Date.now(),
      l: level,
      m: message,
      c: meta?.ctx,
      d: meta ? (({ ctx, ...rest }) => (void ctx, Object.keys(rest).length > 0 ? rest : void 0))(meta) : void 0
    };
    this.transport.write(entry);
  }
};

// src/utils/logging/index.ts
var noopLogger = {
  debug: () => {
  },
  info: () => {
  },
  warn: () => {
  },
  error: () => {
  },
  fatal: () => {
  },
  child: () => noopLogger,
  close: () => {
  }
};
var logger = process.env.JEST_WORKER_ID !== void 0 ? noopLogger : new CompactLogger();

// src/utils/port-utils.ts
var tcpPortUsed = tcpPortUsedModule;
async function isPortAvailable(port, host = "localhost") {
  try {
    const inUse = await tcpPortUsed.check(port, host);
    return !inUse;
  } catch (error) {
    console.error(`Error checking port ${port} availability:`, error);
    return false;
  }
}
async function findAvailablePort(startPort = 3e3, host = "localhost", preferredRanges, maxAttempts = 100, silent = false) {
  if (preferredRanges && preferredRanges.length > 0) {
    for (const [rangeStart, rangeEnd] of preferredRanges) {
      if (rangeStart < 1024 || rangeEnd > 65535 || rangeStart > rangeEnd) {
        if (!silent) console.warn(`Invalid port range [${rangeStart}, ${rangeEnd}], skipping`);
        continue;
      }
      if (!silent) console.log(`Trying preferred port range [${rangeStart}, ${rangeEnd}]`);
      for (let i = 0; i < 10; i++) {
        const randomPort = Math.floor(Math.random() * (rangeEnd - rangeStart + 1)) + rangeStart;
        const available = await isPortAvailable(randomPort, host);
        if (available) {
          if (!silent) console.log(`Found available port ${randomPort} in preferred range`);
          return randomPort;
        }
      }
    }
    if (!silent) console.log("No available ports found in preferred ranges, trying sequential search");
  }
  let port = startPort;
  const maxPort = 65535;
  let attempts = 0;
  while (port <= maxPort && attempts < maxAttempts) {
    attempts++;
    const available = await isPortAvailable(port, host);
    if (available) {
      if (!silent) console.log(`Found available port ${port} after ${attempts} attempts`);
      return port;
    }
    port++;
    if (attempts % 20 === 0) {
      const randomPort = Math.floor(Math.random() * (maxPort - 1024 + 1)) + 1024;
      port = randomPort;
      if (!silent) console.log(`Switching to random port search at port ${port}`);
    }
  }
  if (!silent) console.log("Trying last resort ports...");
  const lastResortPorts = [8080, 8081, 8888, 9e3, 9090, 1e4, 12345, 19999, 2e4, 3e4];
  for (const lastResortPort of lastResortPorts) {
    if (lastResortPort >= startPort) {
      const available = await isPortAvailable(lastResortPort, host);
      if (available) {
        if (!silent) console.log(`Found available last resort port ${lastResortPort}`);
        return lastResortPort;
      }
    }
  }
  throw new Error(`No available ports found after ${attempts} attempts`);
}
async function waitForPortAvailable(port, host = "localhost", retryTimeMs = 200, timeOutMs = 3e4, resourceName, maxRetries = 10, signal) {
  const isTestEnv = !!process.env.JEST_WORKER_ID || process.env.NODE_ENV === "test" || typeof globalThis.jest !== "undefined";
  if (isTestEnv) return;
  const resourceDesc = resourceName ? `${resourceName} on port ${port}` : `port ${port}`;
  logger.info(`Waiting for ${resourceDesc} to become available...`, { ctx: "ports" });
  let currentRetry = 0;
  let currentRetryTime = retryTimeMs;
  while (currentRetry < maxRetries) {
    if (signal?.aborted) return;
    try {
      const attemptTimeout = Math.min(timeOutMs / 3, 1e4);
      await tcpPortUsed.waitUntilFree(port, host, currentRetryTime, attemptTimeout);
      logger.info(`${resourceDesc} is now available`, { ctx: "ports" });
      return;
    } catch (error) {
      currentRetry++;
      if (currentRetry >= maxRetries) {
        const errorMsg = `Timeout waiting for ${resourceDesc} to become available after ${maxRetries} attempts`;
        logger.error(errorMsg, { ctx: "ports", error });
        throw new Error(errorMsg);
      }
      const jitter = Math.random() * 100;
      currentRetryTime = Math.min(currentRetryTime * 1.5 + jitter, 2e3);
      logger.warn(`Retry ${currentRetry}/${maxRetries} for ${resourceDesc} (next retry in ${Math.round(currentRetryTime)}ms)`, { ctx: "ports" });
      await new Promise((resolve) => {
        const t = setTimeout(resolve, currentRetryTime);
        if (signal) {
          const onAbort = () => {
            clearTimeout(t);
            resolve();
          };
          if (signal.aborted) {
            clearTimeout(t);
            resolve();
            return;
          }
          signal.addEventListener("abort", onAbort, { once: true });
        }
      });
    }
  }
  throw new Error(`Failed to wait for ${resourceDesc} to become available after ${maxRetries} attempts`);
}
async function waitForPortInUse(port, host = "localhost", retryTimeMs = 200, timeOutMs = 3e4, serverName, maxRetries = 10, silent = false, signal) {
  const isTestEnv = !!process.env.JEST_WORKER_ID || process.env.NODE_ENV === "test" || typeof globalThis.jest !== "undefined";
  if (isTestEnv) return;
  const serverDesc = serverName ? `${serverName} on port ${port}` : `port ${port}`;
  if (!silent) logger.info(`Waiting for ${serverDesc} to be ready...`, { ctx: "ports" });
  if (silent && maxRetries <= 2) {
    try {
      await tcpPortUsed.waitUntilUsed(port, host, retryTimeMs, Math.min(timeOutMs, 250));
    } catch {
    }
    return;
  }
  let currentRetry = 0;
  let currentRetryTime = retryTimeMs;
  while (currentRetry < maxRetries) {
    if (signal?.aborted) return;
    try {
      const attemptTimeout = Math.min(timeOutMs / 3, 1e4);
      await tcpPortUsed.waitUntilUsed(port, host, currentRetryTime, attemptTimeout);
      if (!silent) logger.info(`${serverDesc} is now ready`, { ctx: "ports" });
      return;
    } catch (error) {
      currentRetry++;
      if (currentRetry >= maxRetries) {
        const errorMsg = `Timeout waiting for ${serverDesc} to be ready after ${maxRetries} attempts`;
        if (!silent) logger.error(errorMsg, { ctx: "ports", error });
        throw new Error(errorMsg);
      }
      const jitter = Math.random() * 100;
      currentRetryTime = Math.min(currentRetryTime * 1.5 + jitter, 2e3);
      if (!silent) logger.warn(`Retry ${currentRetry}/${maxRetries} for ${serverDesc} (next retry in ${Math.round(currentRetryTime)}ms)`, { ctx: "ports" });
      await new Promise((resolve) => {
        const t = setTimeout(resolve, currentRetryTime);
        if (signal) {
          const onAbort = () => {
            clearTimeout(t);
            resolve();
          };
          if (signal.aborted) {
            clearTimeout(t);
            resolve();
            return;
          }
          signal.addEventListener("abort", onAbort, { once: true });
        }
      });
    }
  }
  throw new Error(`Failed to connect to ${serverDesc} after ${maxRetries} attempts`);
}
// Annotate the CommonJS export names for ESM import in node:
0 && (module.exports = {
  findAvailablePort,
  isPortAvailable,
  waitForPortAvailable,
  waitForPortInUse
});
