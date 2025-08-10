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

// src/api/transform/neutral-mistral-format.ts
var neutral_mistral_format_exports = {};
__export(neutral_mistral_format_exports, {
  convertToMistralContent: () => convertToMistralContent,
  convertToMistralMessages: () => convertToMistralMessages,
  convertToNeutralHistoryFromMistral: () => convertToNeutralHistoryFromMistral
});
module.exports = __toCommonJS(neutral_mistral_format_exports);
function convertToMistralMessages(neutralHistory) {
  const mistralMessages = [];
  for (const neutralMessage of neutralHistory) {
    if (typeof neutralMessage.content === "string") {
      mistralMessages.push({
        role: neutralMessage.role,
        content: neutralMessage.content
      });
    } else {
      if (neutralMessage.role === "user") {
        const { nonToolMessages } = neutralMessage.content.reduce(
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
        if (nonToolMessages.length > 0) {
          mistralMessages.push({
            role: "user",
            content: nonToolMessages.map((part) => {
              if (part.type === "image") {
                if (part.source.type === "base64") {
                  return {
                    type: "image_url",
                    imageUrl: {
                      url: `data:${part.source.media_type};base64,${part.source.data}`
                    }
                  };
                } else {
                  return {
                    type: "text",
                    text: "[Image content not supported in this format]"
                  };
                }
              }
              if (part.type === "text" && typeof part.text === "string") {
                return { type: "text", text: part.text };
              }
              return { type: "text", text: "[Text content could not be processed]" };
            })
          });
        }
      } else if (neutralMessage.role === "assistant") {
        const { nonToolMessages } = neutralMessage.content.reduce(
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
        let content;
        if (nonToolMessages.length > 0) {
          content = nonToolMessages.map((part) => {
            if (part.type === "image") {
              return "";
            }
            if (part.type === "text" && typeof part.text === "string") {
              return part.text;
            }
            return "[Text content could not be processed]";
          }).join("\n");
        }
        mistralMessages.push({
          role: "assistant",
          content
        });
      } else if (neutralMessage.role === "system") {
        const textContent = neutralMessage.content.filter((block) => block.type === "text").map((block) => block.text).join("\n");
        mistralMessages.push({
          role: "system",
          content: textContent
        });
      }
    }
  }
  return mistralMessages;
}
function convertToMistralContent(neutralContent) {
  if (typeof neutralContent === "string") {
    return neutralContent;
  }
  return neutralContent.map((block) => {
    if (block.type === "text") {
      return { type: "text", text: block.text };
    } else if (block.type === "image") {
      if (block.source.type === "base64") {
        return {
          type: "image_url",
          imageUrl: {
            url: `data:${block.source.media_type};base64,${block.source.data}`
          }
        };
      } else {
        return {
          type: "text",
          text: "[Image content not supported in this format]"
        };
      }
    }
    return { type: "text", text: `[Unsupported content type: ${block.type}]` };
  });
}
function convertToNeutralHistoryFromMistral(mistralMessages) {
  return mistralMessages.map((mistralMessage) => {
    const neutralMessage = {
      role: mistralMessage.role,
      content: []
    };
    if (typeof mistralMessage.content === "string") {
      neutralMessage.content = [{ type: "text", text: mistralMessage.content }];
    } else if (Array.isArray(mistralMessage.content)) {
      neutralMessage.content = mistralMessage.content.map((part) => {
        if (part.type === "text") {
          return { type: "text", text: part.text };
        } else if (part.type === "image_url") {
          const url = typeof part.imageUrl === "string" ? part.imageUrl : part.imageUrl.url;
          const match = url.match(/^data:([^;]+);base64,(.+)$/);
          if (match) {
            const [, media_type, data] = match;
            return {
              type: "image",
              source: {
                type: "base64",
                media_type,
                data
              }
            };
          }
        }
        return {
          type: "text",
          text: "[Unsupported Mistral content type]"
        };
      });
    }
    return neutralMessage;
  });
}
// Annotate the CommonJS export names for ESM import in node:
0 && (module.exports = {
  convertToMistralContent,
  convertToMistralMessages,
  convertToNeutralHistoryFromMistral
});
