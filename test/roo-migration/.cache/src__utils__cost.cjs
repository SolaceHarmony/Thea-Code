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

// src/utils/cost.ts
var cost_exports = {};
__export(cost_exports, {
  calculateApiCostAnthropic: () => calculateApiCostAnthropic,
  calculateApiCostOpenAI: () => calculateApiCostOpenAI,
  parseApiPrice: () => parseApiPrice
});
module.exports = __toCommonJS(cost_exports);
function calculateApiCostInternal(modelInfo, inputTokens, outputTokens, cacheCreationInputTokens, cacheReadInputTokens) {
  const cacheWritesCost = (modelInfo.cacheWritesPrice || 0) / 1e6 * cacheCreationInputTokens;
  const cacheReadsCost = (modelInfo.cacheReadsPrice || 0) / 1e6 * cacheReadInputTokens;
  const baseInputCost = (modelInfo.inputPrice || 0) / 1e6 * inputTokens;
  const outputCost = (modelInfo.outputPrice || 0) / 1e6 * outputTokens;
  const totalCost = cacheWritesCost + cacheReadsCost + baseInputCost + outputCost;
  return totalCost;
}
function calculateApiCostAnthropic(modelInfo, inputTokens, outputTokens, cacheCreationInputTokens, cacheReadInputTokens) {
  const cacheCreationInputTokensNum = cacheCreationInputTokens || 0;
  const cacheReadInputTokensNum = cacheReadInputTokens || 0;
  return calculateApiCostInternal(
    modelInfo,
    inputTokens,
    outputTokens,
    cacheCreationInputTokensNum,
    cacheReadInputTokensNum
  );
}
function calculateApiCostOpenAI(modelInfo, inputTokens, outputTokens, cacheCreationInputTokens, cacheReadInputTokens) {
  const cacheCreationInputTokensNum = cacheCreationInputTokens || 0;
  const cacheReadInputTokensNum = cacheReadInputTokens || 0;
  const nonCachedInputTokens = Math.max(0, inputTokens - cacheCreationInputTokensNum - cacheReadInputTokensNum);
  return calculateApiCostInternal(
    modelInfo,
    nonCachedInputTokens,
    outputTokens,
    cacheCreationInputTokensNum,
    cacheReadInputTokensNum
  );
}
var parseApiPrice = (price) => price ? parseFloat(price) * 1e6 : void 0;
// Annotate the CommonJS export names for ESM import in node:
0 && (module.exports = {
  calculateApiCostAnthropic,
  calculateApiCostOpenAI,
  parseApiPrice
});
