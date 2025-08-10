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

// src/api/transform/neutral-ollama-format.ts
var neutral_ollama_format_exports = {};
__export(neutral_ollama_format_exports, {
  convertToNeutralHistoryFromOllama: () => convertToNeutralHistoryFromOllama,
  convertToOllamaContentBlocks: () => convertToOllamaContentBlocks,
  convertToOllamaHistory: () => convertToOllamaHistory
});
module.exports = __toCommonJS(neutral_ollama_format_exports);
function convertToOllamaHistory(neutralHistory) {
  return neutralHistory.map((neutralMessage) => {
    let ollamaMessage;
    switch (neutralMessage.role) {
      case "user":
        ollamaMessage = {
          role: "user",
          content: ""
          // Empty string instead of null
        };
        break;
      case "assistant":
        ollamaMessage = {
          role: "assistant",
          content: ""
          // Empty string instead of null
        };
        break;
      case "system":
        ollamaMessage = {
          role: "system",
          content: ""
          // Empty string instead of null
        };
        break;
      default:
        console.warn(`Unknown role type: ${neutralMessage.role}, defaulting to 'user'`);
        ollamaMessage = {
          role: "user",
          content: ""
          // Empty string instead of null
        };
    }
    if (typeof neutralMessage.content === "string") {
      ollamaMessage.content = neutralMessage.content;
    } else if (Array.isArray(neutralMessage.content)) {
      const textBlocks = neutralMessage.content.filter((block) => block.type === "text").map((block) => block.text);
      ollamaMessage.content = textBlocks.join("\n\n");
      if (neutralMessage.content.some((block) => block.type !== "text")) {
        console.warn("Ollama does not support non-text content. Some content may be lost.");
      }
    }
    return ollamaMessage;
  });
}
function convertToOllamaContentBlocks(neutralContent) {
  if (typeof neutralContent === "string") {
    return neutralContent;
  }
  const textBlocks = neutralContent.filter((block) => block.type === "text").map((block) => block.text);
  return textBlocks.join("\n\n");
}
function convertToNeutralHistoryFromOllama(ollamaHistory) {
  return ollamaHistory.map((ollamaMessage) => {
    const neutralMessage = {
      role: mapRoleFromOllama(ollamaMessage.role),
      content: []
      // Will be populated below
    };
    if (typeof ollamaMessage.content === "string") {
      neutralMessage.content = [
        {
          type: "text",
          text: ollamaMessage.content
        }
      ];
    } else if (Array.isArray(ollamaMessage.content)) {
      neutralMessage.content = ollamaMessage.content.map((part) => {
        if (typeof part === "string") {
          return {
            type: "text",
            text: part
          };
        } else {
          return {
            type: "text",
            text: JSON.stringify(part)
          };
        }
      });
    } else if (ollamaMessage.content !== null && ollamaMessage.content !== void 0) {
      neutralMessage.content = [
        {
          type: "text",
          text: JSON.stringify(ollamaMessage.content)
        }
      ];
    }
    return neutralMessage;
  });
}
function mapRoleFromOllama(role) {
  switch (role) {
    case "user":
      return "user";
    case "assistant":
      return "assistant";
    case "system":
      return "system";
    default:
      console.warn(`Unknown Ollama role: ${role}, defaulting to 'user'`);
      return "user";
  }
}
// Annotate the CommonJS export names for ESM import in node:
0 && (module.exports = {
  convertToNeutralHistoryFromOllama,
  convertToOllamaContentBlocks,
  convertToOllamaHistory
});
