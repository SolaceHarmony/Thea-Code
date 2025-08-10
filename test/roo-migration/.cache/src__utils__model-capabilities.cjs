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

// src/utils/model-capabilities.ts
var model_capabilities_exports = {};
__export(model_capabilities_exports, {
  getContextWindowSize: () => getContextWindowSize,
  getMaxTokens: () => getMaxTokens,
  getReasoningEffort: () => getReasoningEffort,
  hasCapability: () => hasCapability,
  supportsComputerUse: () => supportsComputerUse,
  supportsImages: () => supportsImages,
  supportsPromptCaching: () => supportsPromptCaching,
  supportsTemperature: () => supportsTemperature,
  supportsThinking: () => supportsThinking
});
module.exports = __toCommonJS(model_capabilities_exports);
function supportsComputerUse(modelInfo) {
  return !!modelInfo.supportsComputerUse;
}
function supportsPromptCaching(modelInfo) {
  return !!modelInfo.supportsPromptCache;
}
function supportsImages(modelInfo) {
  return !!modelInfo.supportsImages;
}
function supportsThinking(modelInfo) {
  return !!modelInfo.thinking;
}
function supportsTemperature(modelInfo) {
  return modelInfo.supportsTemperature !== false;
}
function getMaxTokens(modelInfo, defaultMaxTokens = 4096) {
  return modelInfo.maxTokens ?? defaultMaxTokens;
}
function getReasoningEffort(modelInfo) {
  return modelInfo.reasoningEffort;
}
function hasCapability(modelInfo, capability) {
  switch (capability) {
    case "computerUse":
      return supportsComputerUse(modelInfo);
    case "promptCache":
      return supportsPromptCaching(modelInfo);
    case "images":
      return supportsImages(modelInfo);
    case "thinking":
      return supportsThinking(modelInfo);
    case "temperature":
      return supportsTemperature(modelInfo);
    default:
      return false;
  }
}
function getContextWindowSize(modelInfo, defaultContextWindow = 8192) {
  return modelInfo.contextWindow ?? defaultContextWindow;
}
// Annotate the CommonJS export names for ESM import in node:
0 && (module.exports = {
  getContextWindowSize,
  getMaxTokens,
  getReasoningEffort,
  hasCapability,
  supportsComputerUse,
  supportsImages,
  supportsPromptCaching,
  supportsTemperature,
  supportsThinking
});
