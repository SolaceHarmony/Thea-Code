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

// src/utils/model-pattern-detection.ts
var model_pattern_detection_exports = {};
__export(model_pattern_detection_exports, {
  getBaseModelId: () => getBaseModelId,
  isClaude35Model: () => isClaude35Model,
  isClaude37Model: () => isClaude37Model,
  isClaude3SonnetModel: () => isClaude3SonnetModel,
  isClaudeHaikuModel: () => isClaudeHaikuModel,
  isClaudeModel: () => isClaudeModel,
  isClaudeOpusModel: () => isClaudeOpusModel,
  isDeepSeekR1Model: () => isDeepSeekR1Model,
  isO3MiniModel: () => isO3MiniModel,
  isThinkingModel: () => isThinkingModel,
  setCapabilitiesFromModelId: () => setCapabilitiesFromModelId
});
module.exports = __toCommonJS(model_pattern_detection_exports);
function isClaudeModel(modelId) {
  return modelId.includes("claude");
}
function isClaude37Model(modelId) {
  return modelId.includes("claude-3.7");
}
function isClaude35Model(modelId) {
  return modelId.includes("claude-3.5");
}
function isClaudeOpusModel(modelId) {
  return modelId.includes("opus");
}
function isClaudeHaikuModel(modelId) {
  return modelId.includes("haiku");
}
function isClaude3SonnetModel(modelId) {
  return modelId.includes("sonnet");
}
function isThinkingModel(modelId) {
  return modelId.includes(":thinking") || modelId.endsWith("-thinking");
}
function isDeepSeekR1Model(modelId) {
  return modelId.startsWith("deepseek/deepseek-r1") || modelId === "perplexity/sonar-reasoning";
}
function isO3MiniModel(modelId) {
  return modelId.startsWith("o3-mini") || modelId.includes("openai/o3-mini");
}
function setCapabilitiesFromModelId(modelId, modelInfo) {
  const updatedModelInfo = { ...modelInfo };
  if (isThinkingModel(modelId)) {
    updatedModelInfo.thinking = true;
  }
  if (isClaudeModel(modelId)) {
    updatedModelInfo.supportsPromptCache = true;
    if (isClaudeOpusModel(modelId)) {
      updatedModelInfo.cacheWritesPrice = 18.75;
      updatedModelInfo.cacheReadsPrice = 1.5;
    } else if (isClaudeHaikuModel(modelId)) {
      updatedModelInfo.cacheWritesPrice = 1.25;
      updatedModelInfo.cacheReadsPrice = 0.1;
    } else {
      updatedModelInfo.cacheWritesPrice = 3.75;
      updatedModelInfo.cacheReadsPrice = 0.3;
    }
    if (isClaude3SonnetModel(modelId) && !modelId.includes("20240620")) {
      updatedModelInfo.supportsComputerUse = true;
    }
    if (isClaude37Model(modelId)) {
      updatedModelInfo.maxTokens = updatedModelInfo.thinking ? 64e3 : 8192;
      updatedModelInfo.supportsComputerUse = true;
    } else if (isClaude35Model(modelId)) {
      updatedModelInfo.maxTokens = 8192;
    }
  }
  if (isO3MiniModel(modelId)) {
    updatedModelInfo.supportsTemperature = false;
  }
  if (isDeepSeekR1Model(modelId)) {
    updatedModelInfo.reasoningEffort = "high";
  }
  return updatedModelInfo;
}
function getBaseModelId(modelId) {
  if (modelId.includes(":thinking")) {
    return modelId.split(":")[0];
  }
  return modelId;
}
// Annotate the CommonJS export names for ESM import in node:
0 && (module.exports = {
  getBaseModelId,
  isClaude35Model,
  isClaude37Model,
  isClaude3SonnetModel,
  isClaudeHaikuModel,
  isClaudeModel,
  isClaudeOpusModel,
  isDeepSeekR1Model,
  isO3MiniModel,
  isThinkingModel,
  setCapabilitiesFromModelId
});
