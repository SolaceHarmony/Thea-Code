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

// src/api/transform/r1-format.ts
var r1_format_exports = {};
__export(r1_format_exports, {
  convertToR1Format: () => convertToR1Format
});
module.exports = __toCommonJS(r1_format_exports);

// src/api/transform/neutral-openai-format.ts
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

// src/api/transform/r1-format.ts
function convertToR1Format(neutralHistory) {
  const messages = convertToOpenAiHistory(neutralHistory);
  return messages.reduce((merged, message) => {
    const lastMessage = merged[merged.length - 1];
    let messageContent = "";
    let hasImages = false;
    if (Array.isArray(message.content)) {
      const textParts = [];
      const imageParts = [];
      const unknownPartsAsText = [];
      message.content.forEach((part) => {
        if (part.type === "text") {
          textParts.push(part.text);
        } else if (part.type === "image_url") {
          hasImages = true;
          imageParts.push(part);
        } else {
          try {
            if (typeof part === "object" && part !== null) {
              const maybeText = part.text;
              if (typeof maybeText === "string") {
                unknownPartsAsText.push(maybeText);
              } else {
                unknownPartsAsText.push(JSON.stringify(part));
              }
            } else {
              unknownPartsAsText.push(String(part));
            }
          } catch {
            unknownPartsAsText.push("[unsupported content]");
          }
        }
      });
      if (hasImages) {
        const parts = [];
        if (textParts.length > 0) {
          parts.push({ type: "text", text: textParts.concat(unknownPartsAsText).join("\n") });
        }
        parts.push(...imageParts);
        messageContent = parts;
      } else {
        messageContent = textParts.concat(unknownPartsAsText).join("\n");
      }
    } else {
      messageContent = message.content || "";
    }
    if (lastMessage?.role === message.role) {
      if (typeof lastMessage.content === "string" && typeof messageContent === "string") {
        lastMessage.content += `
${messageContent}`;
      } else {
        const lastContent = Array.isArray(lastMessage.content) ? lastMessage.content : [{ type: "text", text: lastMessage.content || "" }];
        const newContent = Array.isArray(messageContent) ? messageContent : [{ type: "text", text: messageContent }];
        if (message.role === "assistant") {
          const mergedContent = [...lastContent, ...newContent];
          lastMessage.content = mergedContent;
        } else {
          const mergedContent = [...lastContent, ...newContent];
          lastMessage.content = mergedContent;
        }
      }
    } else {
      if (message.role === "assistant") {
        const newMessage = {
          role: "assistant",
          content: messageContent
        };
        merged.push(newMessage);
      } else {
        const newMessage = {
          role: "user",
          content: messageContent
        };
        merged.push(newMessage);
      }
    }
    return merged;
  }, []);
}
// Annotate the CommonJS export names for ESM import in node:
0 && (module.exports = {
  convertToR1Format
});
