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

// src/api/transform/neutral-openai-format.ts
var neutral_openai_format_exports = {};
__export(neutral_openai_format_exports, {
  convertToNeutralHistoryFromOpenAi: () => convertToNeutralHistoryFromOpenAi,
  convertToOpenAiContentBlocks: () => convertToOpenAiContentBlocks,
  convertToOpenAiHistory: () => convertToOpenAiHistory
});
module.exports = __toCommonJS(neutral_openai_format_exports);
function convertToOpenAiHistory(neutralHistory) {
  return neutralHistory.map((neutralMessage) => {
    let openAiMessage;
    const hasToolResults = Array.isArray(neutralMessage.content) && neutralMessage.content.some((block) => block.type === "tool_result");
    const effectiveRole = hasToolResults ? "tool" : neutralMessage.role;
    switch (effectiveRole) {
      case "user":
        openAiMessage = {
          role: "user",
          content: ""
          // Empty string instead of null
        };
        break;
      case "assistant":
        openAiMessage = {
          role: "assistant",
          content: ""
          // Empty string instead of null
        };
        break;
      case "system":
        openAiMessage = {
          role: "system",
          content: ""
          // Empty string instead of null
        };
        break;
      case "tool":
        openAiMessage = {
          role: "tool",
          content: "",
          // Empty string instead of null
          tool_call_id: ""
          // Will be populated later
        };
        break;
      default:
        console.warn(`Unknown role type: ${String(neutralMessage.role)}, defaulting to 'user'`);
        openAiMessage = {
          role: "user",
          content: ""
          // Empty string instead of null
        };
    }
    if (typeof neutralMessage.content === "string") {
      openAiMessage.content = neutralMessage.content;
    } else if (Array.isArray(neutralMessage.content)) {
      const contentBlocks = convertToOpenAiContentBlocks(neutralMessage.content);
      const toolUseBlocks = neutralMessage.content.filter((block) => block.type === "tool_use");
      if (toolUseBlocks.length > 0 && neutralMessage.role === "assistant") {
        ;
        openAiMessage.tool_calls = toolUseBlocks.map((block) => ({
          id: block.id,
          type: "function",
          function: {
            name: block.name,
            arguments: JSON.stringify(block.input)
          }
        }));
        const textBlocks = neutralMessage.content.filter((block) => block.type === "text");
        if (textBlocks.length > 0) {
          openAiMessage.content = textBlocks.map((block) => block.text).join("\n\n");
        } else {
          openAiMessage.content = null;
        }
      } else if (effectiveRole === "tool") {
        const toolResultBlock = neutralMessage.content.find(
          (block) => block.type === "tool_result"
        );
        if (toolResultBlock) {
          ;
          openAiMessage.tool_call_id = toolResultBlock.tool_use_id;
          const textContent = toolResultBlock.content.filter((block) => block.type === "text").map((block) => block.text).join("\n\n");
          openAiMessage.content = textContent;
        }
      } else {
        if (contentBlocks.length === 1 && typeof contentBlocks[0] === "string") {
          openAiMessage.content = contentBlocks[0];
        } else if (contentBlocks.length > 0) {
          openAiMessage.content = contentBlocks;
        }
      }
    }
    return openAiMessage;
  });
}
function convertToOpenAiContentBlocks(neutralContent) {
  if (typeof neutralContent === "string") {
    return neutralContent;
  }
  if (neutralContent.length === 1 && neutralContent[0].type === "text") {
    return neutralContent[0].text;
  }
  const result = [];
  for (const block of neutralContent) {
    if (block.type === "text") {
      result.push({
        type: "text",
        text: block.text
      });
    } else if (block.type === "image") {
      if (block.source.type === "base64") {
        result.push({
          type: "image_url",
          image_url: {
            url: `data:${block.source.media_type};base64,${block.source.data}`
          }
        });
      } else {
        result.push({
          type: "image_url",
          image_url: {
            url: block.source.url
          }
        });
      }
    } else if (block.type === "tool_use" || block.type === "tool_result") {
      result.push({
        type: "text",
        text: `[${block.type} - handled separately]`
      });
    } else {
      const unknownBlock = block;
      console.warn(`Unsupported block type: ${unknownBlock.type}`);
      result.push({
        type: "text",
        text: `[Unsupported block type: ${unknownBlock.type}]`
      });
    }
  }
  return result;
}
function convertToNeutralHistoryFromOpenAi(openAiHistory) {
  return openAiHistory.map((openAiMessage) => {
    const neutralMessage = {
      role: mapRoleFromOpenAi(openAiMessage.role),
      content: []
      // Will be populated below
    };
    if (typeof openAiMessage.content === "string") {
      neutralMessage.content = [
        {
          type: "text",
          text: openAiMessage.content
        }
      ];
    } else if (Array.isArray(openAiMessage.content)) {
      neutralMessage.content = openAiMessage.content.flatMap((part) => {
        if (part.type === "text") {
          return {
            type: "text",
            text: part.text
          };
        } else if (part.type === "image_url") {
          const url = typeof part.image_url === "string" ? part.image_url : part.image_url.url;
          if (url.startsWith("data:")) {
            const matches = url.match(/^data:([^;]+);base64,(.+)$/);
            if (matches && matches.length === 3) {
              return {
                type: "image",
                source: {
                  type: "base64",
                  media_type: matches[1],
                  data: matches[2]
                }
              };
            }
          }
          console.warn("Non-base64 image URLs not fully supported");
          return {
            type: "text",
            text: `[Image: ${url}]`
          };
        }
        console.warn(`Unsupported OpenAI content part type: ${part.type}`);
        return {
          type: "text",
          text: `[Unsupported content type: ${part.type}]`
        };
      });
    }
    const assistantWithTools = openAiMessage;
    if (assistantWithTools.tool_calls && openAiMessage.role === "assistant") {
      assistantWithTools.tool_calls.forEach(
        (toolCall) => {
          if (toolCall.type === "function") {
            let args = {};
            try {
              args = JSON.parse(toolCall.function.arguments);
            } catch (e) {
              console.warn("Failed to parse tool call arguments:", e);
              args = { raw: toolCall.function.arguments };
            }
            ;
            neutralMessage.content.push({
              type: "tool_use",
              id: toolCall.id,
              name: toolCall.function.name,
              input: args
            });
          }
        }
      );
    }
    const toolMessage = openAiMessage;
    if (openAiMessage.role === "tool" && toolMessage.tool_call_id) {
      const toolResult = {
        type: "tool_result",
        tool_use_id: toolMessage.tool_call_id,
        content: [
          {
            type: "text",
            text: typeof openAiMessage.content === "string" ? openAiMessage.content : JSON.stringify(openAiMessage.content)
          }
        ]
      };
      neutralMessage.content = [toolResult];
    }
    return neutralMessage;
  });
}
function mapRoleFromOpenAi(role) {
  switch (role) {
    case "user":
      return "user";
    case "assistant":
      return "assistant";
    case "system":
      return "system";
    case "tool":
      return "tool";
    default:
      console.warn(`Unknown OpenAI role: ${role}, defaulting to 'user'`);
      return "user";
  }
}
// Annotate the CommonJS export names for ESM import in node:
0 && (module.exports = {
  convertToNeutralHistoryFromOpenAi,
  convertToOpenAiContentBlocks,
  convertToOpenAiHistory
});
