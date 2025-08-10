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

// src/integrations/misc/line-counter.ts
var line_counter_exports = {};
__export(line_counter_exports, {
  countFileLines: () => countFileLines
});
module.exports = __toCommonJS(line_counter_exports);
var import_fs = __toESM(require("fs"));
var import_fs2 = require("fs");
var import_readline = require("readline");
async function countFileLines(filePath) {
  try {
    await import_fs.default.promises.access(filePath, import_fs.default.constants.F_OK);
  } catch {
    throw new Error(`File not found: ${filePath}`);
  }
  return new Promise((resolve, reject) => {
    let lineCount = 0;
    const readStream = (0, import_fs2.createReadStream)(filePath);
    const rl = (0, import_readline.createInterface)({
      input: readStream,
      crlfDelay: Infinity
    });
    rl.on("line", () => {
      lineCount++;
    });
    rl.on("close", () => {
      resolve(lineCount);
    });
    rl.on("error", (err) => {
      reject(err instanceof Error ? err : new Error(String(err)));
    });
    readStream.on("error", (err) => {
      reject(err instanceof Error ? err : new Error(String(err)));
    });
  });
}
// Annotate the CommonJS export names for ESM import in node:
0 && (module.exports = {
  countFileLines
});
