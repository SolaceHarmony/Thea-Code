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

// src/api/transform/vscode-lm-format.ts
var vscode_lm_format_exports = {};
__export(vscode_lm_format_exports, {
  convertToAnthropicRole: () => convertToAnthropicRole,
  convertToVsCodeLmMessages: () => convertToVsCodeLmMessages
});
module.exports = __toCommonJS(vscode_lm_format_exports);
var vscode = __toESM(require("vscode"));
function asObjectSafe(value) {
  if (!value) {
    return {};
  }
  try {
    if (typeof value === "string") {
      return JSON.parse(value);
    }
    if (typeof value === "object") {
      return Object.assign({}, value);
    }
    return {};
  } catch (error) {
    console.warn("Thea Code <Language Model API>: Failed to parse object:", error);
    return {};
  }
}
function convertToVsCodeLmMessages(neutralHistory) {
  const vsCodeLmMessages = [];
  for (const anthropicMessage of neutralHistory) {
    if (typeof anthropicMessage.content === "string") {
      vsCodeLmMessages.push(
        anthropicMessage.role === "assistant" ? vscode.LanguageModelChatMessage.Assistant(anthropicMessage.content) : vscode.LanguageModelChatMessage.User(anthropicMessage.content)
      );
      continue;
    }
    switch (anthropicMessage.role) {
      case "user": {
        const { nonToolMessages, toolMessages } = anthropicMessage.content.reduce(
          (acc, part) => {
            if (part.type === "tool_result") {
              acc.toolMessages.push(part);
            } else if (part.type === "text" || part.type === "image") {
              acc.nonToolMessages.push(part);
            }
            return acc;
          },
          { nonToolMessages: [], toolMessages: [] }
        );
        const contentParts = [
          // Convert tool messages to ToolResultParts
          ...toolMessages.map((toolMessage) => {
            const toolContentParts = typeof toolMessage.content === "string" ? [new vscode.LanguageModelTextPart(toolMessage.content)] : toolMessage.content?.map((part) => {
              if (part.type === "image") {
                return new vscode.LanguageModelTextPart(
                  `[Image (${part.source?.type || "Unknown source-type"}): ${part.source?.type === "base64" ? part.source.media_type : "media-type not applicable for URL source"} not supported by VSCode LM API]`
                );
              }
              if (part.type === "text") {
                return new vscode.LanguageModelTextPart(part.text);
              }
              return new vscode.LanguageModelTextPart("");
            }) ?? [new vscode.LanguageModelTextPart("")];
            return new vscode.LanguageModelToolResultPart(toolMessage.tool_use_id, toolContentParts);
          }),
          // Convert non-tool messages to TextParts after tool messages
          ...nonToolMessages.map((part) => {
            if (part.type === "image") {
              return new vscode.LanguageModelTextPart(
                `[Image (${part.source?.type || "Unknown source-type"}): ${part.source?.type === "base64" ? part.source.media_type : "media-type not applicable for URL source"} not supported by VSCode LM API]`
              );
            }
            if (part.type === "text") {
              return new vscode.LanguageModelTextPart(part.text);
            }
            return new vscode.LanguageModelTextPart("");
          })
        ];
        vsCodeLmMessages.push(vscode.LanguageModelChatMessage.User(contentParts));
        break;
      }
      case "assistant": {
        const { nonToolMessages, toolMessages } = anthropicMessage.content.reduce(
          (acc, part) => {
            if (part.type === "tool_use") {
              acc.toolMessages.push(part);
            } else if (part.type === "text" || part.type === "image") {
              acc.nonToolMessages.push(part);
            }
            return acc;
          },
          { nonToolMessages: [], toolMessages: [] }
        );
        const contentParts = [
          // Convert tool messages to ToolCallParts first
          ...toolMessages.map(
            (toolMessage) => new vscode.LanguageModelToolCallPart(
              toolMessage.id,
              toolMessage.name,
              asObjectSafe(toolMessage.input)
            )
          ),
          // Convert non-tool messages to TextParts after tool messages
          ...nonToolMessages.map((part) => {
            if (part.type === "image") {
              return new vscode.LanguageModelTextPart("[Image generation not supported by VSCode LM API]");
            }
            if (part.type === "text") {
              return new vscode.LanguageModelTextPart(part.text);
            }
            return new vscode.LanguageModelTextPart("");
          })
        ];
        vsCodeLmMessages.push(vscode.LanguageModelChatMessage.Assistant(contentParts));
        break;
      }
    }
  }
  return vsCodeLmMessages;
}
function convertToAnthropicRole(vsCodeLmMessageRole) {
  switch (vsCodeLmMessageRole) {
    case vscode.LanguageModelChatMessageRole.Assistant:
      return "assistant";
    case vscode.LanguageModelChatMessageRole.User:
      return "user";
    default:
      return null;
  }
}
// Annotate the CommonJS export names for ESM import in node:
0 && (module.exports = {
  convertToAnthropicRole,
  convertToVsCodeLmMessages
});
