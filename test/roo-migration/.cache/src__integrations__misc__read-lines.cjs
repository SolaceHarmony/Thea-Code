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

// src/integrations/misc/read-lines.ts
var read_lines_exports = {};
__export(read_lines_exports, {
  readLines: () => readLines
});
module.exports = __toCommonJS(read_lines_exports);
var import_fs = require("fs");
var import_readline = require("readline");
var outOfRangeError = (filepath, n) => {
  return new RangeError(`Line with index ${n} does not exist in '${filepath}'. Note that line indexing is zero-based`);
};
function readLines(filepath, endLine, startLine) {
  return new Promise((resolve, reject) => {
    if (startLine !== void 0 && (startLine < 0 || startLine % 1 !== 0)) {
      return reject(
        new RangeError(`Invalid startLine: ${startLine}. Line numbers must be non-negative integers.`)
      );
    }
    if (endLine !== void 0 && (endLine < 0 || endLine % 1 !== 0)) {
      return reject(new RangeError(`Invalid endLine: ${endLine}. Line numbers must be non-negative integers.`));
    }
    const effectiveStartLine = startLine === void 0 ? 0 : startLine;
    if (endLine !== void 0 && effectiveStartLine > endLine) {
      return reject(
        new RangeError(`startLine (${effectiveStartLine}) must be less than or equal to endLine (${endLine})`)
      );
    }
    let cursor = 0;
    const lines = [];
    const input = (0, import_fs.createReadStream)(filepath);
    const rl = (0, import_readline.createInterface)({ input });
    rl.on("line", (line) => {
      if (cursor >= effectiveStartLine && (endLine === void 0 || cursor <= endLine)) {
        lines.push(line);
      }
      if (endLine !== void 0 && cursor === endLine) {
        rl.close();
        input.close();
        resolve(lines.join("\n"));
      }
      cursor++;
    });
    rl.on("error", reject);
    input.on("end", () => {
      if (lines.length > 0) {
        resolve(lines.join("\n"));
      } else {
        reject(outOfRangeError(filepath, effectiveStartLine));
      }
    });
  });
}
// Annotate the CommonJS export names for ESM import in node:
0 && (module.exports = {
  readLines
});
