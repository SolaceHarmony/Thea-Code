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

// src/api/transform/simple-format.ts
var simple_format_exports = {};
__export(simple_format_exports, {
  convertToSimpleContent: () => convertToSimpleContent,
  convertToSimpleMessages: () => convertToSimpleMessages
});
module.exports = __toCommonJS(simple_format_exports);
function isTextBlock(block) {
  return block.type === "text" && "text" in block;
}
function isImageBlock(block) {
  return (block.type === "image" || block.type === "image_url" || block.type === "image_base64") && "source" in block;
}
function isToolUseBlock(block) {
  return block.type === "tool_use" && "name" in block;
}
function isToolResultBlock(block) {
  return block.type === "tool_result" && "content" in block;
}
function convertToSimpleContent(content) {
  if (typeof content === "string") {
    return content;
  }
  if (content.length === 0) {
    return "";
  }
  return content.map((block) => {
    if (isTextBlock(block)) {
      return block.text;
    }
    if (isImageBlock(block)) {
      const mediaType = block.source.type === "base64" ? block.source.media_type : "image";
      return `[Image: ${mediaType}]`;
    }
    if (isToolUseBlock(block)) {
      return `[Tool Use: ${block.name}]`;
    }
    if (isToolResultBlock(block)) {
      return block.content.map((part) => {
        if (isTextBlock(part)) {
          return part.text;
        }
        if (isImageBlock(part)) {
          const mediaType = part.source.type === "base64" ? part.source.media_type : "image";
          return `[Image: ${mediaType}]`;
        }
        return "";
      }).join("\n");
    }
    return `[Unknown content type: ${block.type}]`;
  }).filter(Boolean).join("\n");
}
function convertToSimpleMessages(messages) {
  return messages.map(
    (message) => ({
      role: message.role,
      content: convertToSimpleContent(message.content)
    })
  );
}
// Annotate the CommonJS export names for ESM import in node:
0 && (module.exports = {
  convertToSimpleContent,
  convertToSimpleMessages
});
