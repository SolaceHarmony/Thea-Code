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

// src/utils/json-xml-bridge.ts
var json_xml_bridge_exports = {};
__export(json_xml_bridge_exports, {
  FormatDetector: () => FormatDetector,
  HybridMatcher: () => HybridMatcher,
  JsonMatcher: () => JsonMatcher,
  ToolResultMatcher: () => ToolResultMatcher,
  ToolUseMatcher: () => ToolUseMatcher,
  jsonThinkingToXml: () => jsonThinkingToXml,
  jsonToolResultToXml: () => jsonToolResultToXml,
  jsonToolUseToXml: () => jsonToolUseToXml,
  neutralToolUseToOpenAiFunctionCall: () => neutralToolUseToOpenAiFunctionCall,
  openAiFunctionCallToNeutralToolUse: () => openAiFunctionCallToNeutralToolUse,
  xmlThinkingToJson: () => xmlThinkingToJson,
  xmlToolResultToJson: () => xmlToolResultToJson,
  xmlToolUseToJson: () => xmlToolUseToJson
});
module.exports = __toCommonJS(json_xml_bridge_exports);

// src/utils/xml-matcher.ts
var XmlMatcher = class {
  constructor(tagName, transform, position = 0) {
    this.tagName = tagName;
    this.transform = transform;
    this.position = position;
  }
  index = 0;
  chunks = [];
  cached = [];
  matched = false;
  state = "TEXT";
  depth = 0;
  pointer = 0;
  collect() {
    if (!this.cached.length) {
      return;
    }
    const last = this.chunks.at(-1);
    const data = this.cached.join("");
    const matched = this.matched;
    if (last?.matched === matched) {
      last.data += data;
    } else {
      this.chunks.push({
        data,
        matched
      });
    }
    this.cached = [];
  }
  pop() {
    const chunks = this.chunks;
    this.chunks = [];
    if (!this.transform) {
      return chunks;
    }
    return chunks.map(this.transform);
  }
  _update(chunk) {
    for (let i = 0; i < chunk.length; i++) {
      const char = chunk[i];
      this.cached.push(char);
      this.pointer++;
      if (this.state === "TEXT") {
        if (char === "<" && (this.pointer <= this.position + 1 || this.matched)) {
          this.state = "TAG_OPEN";
          this.index = 0;
        } else {
          this.collect();
        }
      } else if (this.state === "TAG_OPEN") {
        if (char === ">" && this.index === this.tagName.length) {
          this.state = "TEXT";
          if (!this.matched) {
            this.cached = [];
          }
          this.depth++;
          this.matched = true;
        } else if (this.index === 0 && char === "/") {
          this.state = "TAG_CLOSE";
        } else if (char === " " && (this.index === 0 || this.index === this.tagName.length)) {
          continue;
        } else if (this.tagName[this.index] === char) {
          this.index++;
        } else {
          this.state = "TEXT";
          this.collect();
        }
      } else if (this.state === "TAG_CLOSE") {
        if (char === ">" && this.index === this.tagName.length) {
          this.state = "TEXT";
          this.depth--;
          this.matched = this.depth > 0;
          if (!this.matched) {
            this.cached = [];
          }
        } else if (char === " " && (this.index === 0 || this.index === this.tagName.length)) {
          continue;
        } else if (this.tagName[this.index] === char) {
          this.index++;
        } else {
          this.state = "TEXT";
          this.collect();
        }
      }
    }
  }
  final(chunk) {
    if (chunk) {
      this._update(chunk);
    }
    this.collect();
    return this.pop();
  }
  update(chunk) {
    this._update(chunk);
    return this.pop();
  }
};

// src/utils/json-xml-bridge.ts
function safeStringify(value) {
  if (typeof value === "string") {
    return value;
  }
  if (value === null || value === void 0) {
    return "";
  }
  if (typeof value === "number" || typeof value === "boolean") {
    return String(value);
  }
  if (typeof value === "object") {
    try {
      return JSON.stringify(value);
    } catch {
      return "[object Object]";
    }
  }
  return "[unknown]";
}
var JsonMatcher = class _JsonMatcher {
  // 256 KB
  /**
   * Create a new JsonMatcher
   *
   * @param matchType The type of JSON object to match (e.g., "thinking", "tool_use")
   */
  constructor(matchType) {
    this.matchType = matchType;
  }
  buffer = "";
  objectDepth = 0;
  inString = false;
  escapeNext = false;
  // Cap buffer growth to prevent unbounded memory usage when JSON never completes
  static MAX_BUFFER_LENGTH = 256 * 1024;
  /**
   * Update the matcher with a new chunk of text
   *
   * @param chunk New text chunk to process
   * @returns Array of matched results
   */
  update(chunk) {
    this.buffer += chunk;
    if (this.buffer.length > _JsonMatcher.MAX_BUFFER_LENGTH) {
      const overflow = this.buffer.length - _JsonMatcher.MAX_BUFFER_LENGTH;
      const spill = this.buffer.slice(0, overflow);
      this.buffer = this.buffer.slice(overflow);
      const spillResult = { matched: false, data: spill };
      return [spillResult, ...this.processBuffer()];
    }
    return this.processBuffer();
  }
  /**
   * Process any remaining content and return final results
   *
   * @param chunk Optional final chunk to process
   * @returns Array of matched results
   */
  final(chunk) {
    if (chunk) {
      this.buffer += chunk;
    }
    const results = this.processBuffer();
    if (this.buffer.trim()) {
      const textResult = {
        matched: false,
        data: this.buffer
      };
      this.buffer = "";
      return [textResult];
    }
    return results;
  }
  /**
   * Process the current buffer to extract JSON objects
   *
   * @returns Array of matched results
   */
  processBuffer() {
    const results = [];
    let startIndex = 0;
    while (startIndex < this.buffer.length) {
      const objectStart = this.buffer.indexOf("{", startIndex);
      if (objectStart === -1) {
        if (startIndex < this.buffer.length) {
          const text = this.buffer.substring(startIndex);
          const textResult = {
            matched: false,
            data: text
          };
          results.push(textResult);
          this.buffer = "";
        }
        break;
      }
      if (objectStart > startIndex) {
        const text = this.buffer.substring(startIndex, objectStart);
        const textResult = {
          matched: false,
          data: text
        };
        results.push(textResult);
      }
      const objectEnd = this.findObjectEnd(objectStart);
      if (objectEnd === -1) {
        this.buffer = this.buffer.substring(startIndex);
        break;
      }
      const jsonStr = this.buffer.substring(objectStart, objectEnd + 1);
      try {
        const jsonObj = JSON.parse(jsonStr);
        if (jsonObj.type === this.matchType) {
          const matchedResult = {
            matched: true,
            data: this.matchType === "thinking" ? typeof jsonObj.content === "string" || typeof jsonObj.content === "object" && jsonObj.content !== null ? jsonObj.content : typeof jsonObj.text === "string" || typeof jsonObj.text === "object" && jsonObj.text !== null ? jsonObj.text : jsonObj : jsonObj,
            type: this.matchType
          };
          results.push(matchedResult);
        } else {
          const textResult = {
            matched: false,
            data: jsonStr
          };
          results.push(textResult);
        }
      } catch {
        const textResult = {
          matched: false,
          data: jsonStr
        };
        results.push(textResult);
      }
      startIndex = objectEnd + 1;
    }
    this.buffer = this.buffer.substring(startIndex);
    return results;
  }
  /**
   * Find the matching closing brace for a JSON object
   *
   * @param start Starting index of the opening brace
   * @returns Index of the matching closing brace, or -1 if not found
   */
  findObjectEnd(start) {
    let depth = 0;
    let inString = false;
    let escapeNext = false;
    if (this.buffer[start] === "{") {
      depth = 1;
    } else {
      return -1;
    }
    for (let i = start + 1; i < this.buffer.length; i++) {
      const char = this.buffer[i];
      if (escapeNext) {
        escapeNext = false;
        continue;
      }
      if (char === "\\") {
        escapeNext = true;
        continue;
      }
      if (char === '"') {
        inString = !inString;
      } else if (!inString) {
        if (char === "{") {
          depth++;
        } else if (char === "}") {
          depth--;
          if (depth === 0) {
            return i;
          }
        }
      }
    }
    return -1;
  }
};
var FormatDetector = class {
  /**
   * Detect the format of a text chunk
   *
   * @param content Text content to analyze
   * @returns Format type: 'json', 'xml', or 'unknown'
   */
  detectFormat(content) {
    if (content.includes("<think>") || content.match(/<\w+>/) || content.includes("<tool_result>")) {
      return "xml";
    }
    if (content.includes("{") && content.includes("}")) {
      try {
        const startIndex = content.indexOf("{");
        const endIndex = content.lastIndexOf("}");
        if (startIndex !== -1 && endIndex !== -1 && endIndex > startIndex) {
          const sample = content.substring(startIndex, endIndex + 1);
          if (sample.startsWith("{") && sample.endsWith("}")) {
            const jsonObj = JSON.parse(sample);
            if (jsonObj.type === "tool_use" || jsonObj.type === "tool_result" || jsonObj.type === "thinking") {
              return "json";
            }
            if (jsonObj.tool_calls || jsonObj.function_call) {
              return "json";
            }
            return "json";
          }
        }
      } catch {
      }
    }
    return "unknown";
  }
};
function jsonThinkingToXml(jsonObj) {
  if (typeof jsonObj === "object" && jsonObj.type === "thinking" && jsonObj.content) {
    return `<think>${safeStringify(jsonObj.content)}</think>`;
  }
  return JSON.stringify(jsonObj);
}
function xmlThinkingToJson(xmlContent) {
  const thinkRegex = /<think>(.*?)<\/think>/s;
  const match = thinkRegex.exec(xmlContent);
  if (match && match[1]) {
    return JSON.stringify({
      type: "thinking",
      content: match[1]
    });
  }
  return xmlContent;
}
function jsonToolUseToXml(jsonObj) {
  if (typeof jsonObj === "object" && jsonObj.type === "tool_use" && jsonObj.name) {
    const toolName = safeStringify(jsonObj.name);
    let xml = `<${toolName}>
`;
    if (jsonObj.input && typeof jsonObj.input === "object" && jsonObj.input !== null) {
      for (const [key, value] of Object.entries(jsonObj.input)) {
        let stringValue;
        if (typeof value === "object" && value !== null) {
          stringValue = JSON.stringify(value);
        } else {
          stringValue = String(value);
        }
        xml += `<${key}>${stringValue}</${key}>
`;
      }
    }
    xml += `</${toolName}>`;
    return xml;
  }
  return JSON.stringify(jsonObj);
}
function xmlToolUseToJson(xmlContent) {
  const toolNameRegex = /<(\w+)>/;
  const toolNameMatch = toolNameRegex.exec(xmlContent);
  if (toolNameMatch && toolNameMatch[1]) {
    const toolName = toolNameMatch[1];
    let outerContent = xmlContent;
    outerContent = outerContent.replace(new RegExp(`<${toolName}>\\s*`), "");
    outerContent = outerContent.replace(new RegExp(`\\s*</${toolName}>`), "");
    const params = {};
    const paramRegex = /<(\w+)>([\s\S]*?)<\/\1>/g;
    let match;
    while ((match = paramRegex.exec(outerContent)) !== null) {
      const paramName = match[1];
      const paramValue = match[2].trim();
      if (paramName !== toolName) {
        if (paramValue.startsWith("{") || paramValue.startsWith("[")) {
          try {
            params[paramName] = JSON.parse(paramValue);
          } catch {
            params[paramName] = paramValue;
          }
        } else if (paramValue === "true" || paramValue === "false") {
          params[paramName] = paramValue === "true";
        } else if (/^-?\d+$/.test(paramValue)) {
          params[paramName] = parseInt(paramValue, 10);
        } else if (/^-?\d+\.\d+$/.test(paramValue)) {
          params[paramName] = parseFloat(paramValue);
        } else {
          params[paramName] = paramValue;
        }
      }
    }
    const jsonObj = {
      type: "tool_use",
      name: toolName,
      id: `${toolName}-${Date.now()}`,
      // Generate a unique ID
      input: params
    };
    return JSON.stringify(jsonObj);
  }
  return xmlContent;
}
function jsonToolResultToXml(jsonObj) {
  if (typeof jsonObj === "object" && jsonObj.type === "tool_result" && jsonObj.tool_use_id) {
    let xml = `<tool_result tool_use_id="${safeStringify(jsonObj.tool_use_id)}"`;
    if (jsonObj.status) {
      xml += ` status="${safeStringify(jsonObj.status)}"`;
    }
    xml += ">\n";
    if (Array.isArray(jsonObj.content)) {
      for (const item of jsonObj.content) {
        if (typeof item === "object" && item !== null) {
          const typedItem = item;
          if (typedItem.type === "text") {
            xml += `${safeStringify(typedItem.text)}
`;
          } else if (typedItem.type === "image") {
            if (typedItem.source) {
              xml += `<image type="${safeStringify(typedItem.source.media_type)}" data="${safeStringify(typedItem.source.data)}" />
`;
            }
          }
        }
      }
    }
    if (jsonObj.error && typeof jsonObj.error === "object") {
      const errorObj = jsonObj.error;
      xml += `<error message="${errorObj.message ?? ""}"`;
      if (errorObj.details) {
        const escapedDetails = JSON.stringify(errorObj.details).replace(/"/g, "&quot;");
        xml += ` details="${escapedDetails}"`;
      }
      xml += " />\n";
    }
    xml += "</tool_result>";
    return xml;
  }
  return JSON.stringify(jsonObj);
}
function xmlToolResultToJson(xmlContent) {
  const toolResultRegex = /<tool_result\s+tool_use_id="([^"]+)"(?:\s+status="([^"]+)")?>/;
  const toolResultMatch = toolResultRegex.exec(xmlContent);
  if (toolResultMatch) {
    const toolUseId = toolResultMatch[1];
    const status = toolResultMatch[2] || "success";
    const contentRegex = /<tool_result[^>]*>([\s\S]*?)<\/tool_result>/;
    const contentMatch = contentRegex.exec(xmlContent);
    let content = [];
    if (contentMatch && contentMatch[1]) {
      const textContent = contentMatch[1].replace(/<[^>]*>/g, "").trim();
      if (textContent) {
        content.push({
          type: "text",
          text: textContent
        });
      }
      const imageRegex = /<image\s+type="([^"]+)"\s+data="([^"]+)"\s*\/>/g;
      let imageMatch;
      while ((imageMatch = imageRegex.exec(contentMatch[1])) !== null) {
        content.push({
          type: "image",
          source: {
            type: "base64",
            media_type: imageMatch[1],
            data: imageMatch[2]
          }
        });
      }
    }
    const errorRegex = /<error\s+message="([^"]+)"(?:\s+details="([^"]+)")?\s*\/>/;
    const errorMatch = errorRegex.exec(xmlContent);
    let error = void 0;
    if (errorMatch) {
      error = {
        message: errorMatch[1]
      };
      if (errorMatch[2]) {
        const decodeEntities = (s) => s.replace(/&quot;/g, '"').replace(/&apos;/g, "'").replace(/&lt;/g, "<").replace(/&gt;/g, ">").replace(/&amp;/g, "&");
        try {
          const decoded = decodeEntities(errorMatch[2]);
          const parsed = JSON.parse(decoded);
          error.details = parsed;
        } catch {
          error.details = decodeEntities(errorMatch[2]);
        }
      }
    }
    const jsonObj = {
      type: "tool_result",
      tool_use_id: toolUseId,
      content,
      status
    };
    if (error) {
      jsonObj.error = error;
    }
    return JSON.stringify(jsonObj);
  }
  return xmlContent;
}
function openAiFunctionCallToNeutralToolUse(openAiFunctionCall) {
  if (openAiFunctionCall.function_call) {
    let args;
    if (openAiFunctionCall.function_call.arguments === null) {
      return {
        type: "tool_use",
        id: openAiFunctionCall.function_call.id || `function-${Date.now()}`,
        name: openAiFunctionCall.function_call.name,
        input: null
      };
    }
    if (typeof openAiFunctionCall.function_call.arguments === "undefined") {
      return {
        type: "tool_use",
        id: openAiFunctionCall.function_call.id || `function-${Date.now()}`,
        name: openAiFunctionCall.function_call.name,
        input: { raw: void 0 }
      };
    }
    try {
      const parsed = JSON.parse(openAiFunctionCall.function_call.arguments);
      if (typeof parsed === "object" && parsed !== null) {
        args = parsed;
      } else {
        args = { raw: openAiFunctionCall.function_call.arguments };
      }
    } catch {
      args = { raw: openAiFunctionCall.function_call.arguments };
    }
    return {
      type: "tool_use",
      id: openAiFunctionCall.function_call.id || `function-${Date.now()}`,
      name: openAiFunctionCall.function_call.name,
      input: args
    };
  } else if (openAiFunctionCall.tool_calls && Array.isArray(openAiFunctionCall.tool_calls)) {
    for (const toolCall of openAiFunctionCall.tool_calls) {
      if (toolCall.type === "function" && toolCall.function) {
        try {
          let args;
          const parsed = JSON.parse(toolCall.function.arguments);
          if (typeof parsed === "object" && parsed !== null) {
            args = parsed;
          } else {
            args = { raw: toolCall.function.arguments };
          }
          return {
            type: "tool_use",
            id: toolCall.id || `function-${Date.now()}`,
            name: toolCall.function.name,
            input: args
          };
        } catch {
          return {
            type: "tool_use",
            id: toolCall.id || `function-${Date.now()}`,
            name: toolCall.function.name,
            input: { raw: toolCall.function.arguments }
          };
        }
      }
    }
  }
  return null;
}
function neutralToolUseToOpenAiFunctionCall(neutralToolUse) {
  if (neutralToolUse.type === "tool_use" && neutralToolUse.name) {
    return {
      function_call: {
        id: neutralToolUse.id || `function-${Date.now()}`,
        name: neutralToolUse.name,
        arguments: JSON.stringify(neutralToolUse.input)
      }
    };
  }
  return null;
}
var ToolUseMatcher = class {
  /**
   * Create a new ToolUseMatcher
   *
   * @param transform Transform function for matched results
   */
  constructor(transform) {
    this.transform = transform;
    this.xmlMatcher = new XmlMatcher("");
    this.jsonMatcher = new JsonMatcher("tool_use");
    this.formatDetector = new FormatDetector();
  }
  xmlMatcher;
  jsonMatcher;
  formatDetector;
  detectedFormat = "unknown";
  toolUseIds = /* @__PURE__ */ new Map();
  /**
   * Update the matcher with a new chunk of text
   *
   * @param chunk New text chunk to process
   * @returns Array of matched results
   */
  update(chunk) {
    if (this.detectedFormat === "unknown") {
      this.detectedFormat = this.formatDetector.detectFormat(chunk);
    }
    let results = [];
    if (this.detectedFormat === "xml" || this.detectedFormat === "unknown") {
      const re = /<([A-Za-z_][\w-]*)>[\s\S]*?<\/\1>/g;
      let m;
      let found = false;
      while ((m = re.exec(chunk)) !== null) {
        const tag = m[1];
        if (tag === "think" || tag === "tool_result") continue;
        found = true;
        const block = m[0];
        try {
          const jsonStr = xmlToolUseToJson(block);
          const obj = JSON.parse(jsonStr);
          const toolId = obj.id || `${obj.name}-${Date.now()}`;
          this.toolUseIds.set(obj.name, toolId);
          results.push(this.transformResult({ matched: true, data: obj, type: "tool_use" }));
        } catch {
          results.push(this.transformResult({ matched: true, data: block, type: "tool_use" }));
        }
      }
      if (!found) {
        results.push(...this.xmlMatcher.update(chunk).map((r) => this.transformResult(r)));
      }
    } else if (this.detectedFormat === "json") {
      const jsonResults = this.jsonMatcher.update(chunk);
      for (const result of jsonResults) {
        if (result.matched && typeof result.data === "object" && result.data.type === "tool_use") {
          const toolUseObj = result.data;
          const toolId = toolUseObj.id || `${toolUseObj.name}-${Date.now()}`;
          this.toolUseIds.set(toolUseObj.name, toolId);
          results.push(
            this.transformResult({
              matched: true,
              data: toolUseObj,
              type: "tool_use"
            })
          );
        } else {
          results.push(this.transformResult(result));
        }
      }
    }
    return results;
  }
  /**
   * Process any remaining content and return final results
   *
   * @param chunk Optional final chunk to process
   * @returns Array of matched results
   */
  final(chunk) {
    if (chunk) {
      if (this.detectedFormat === "unknown") {
        this.detectedFormat = this.formatDetector.detectFormat(chunk);
      }
    }
    if (this.detectedFormat === "json") {
      return this.jsonMatcher.final(chunk).map((r) => this.transformResult(r));
    } else {
      return this.xmlMatcher.final(chunk).map((r) => this.transformResult(r));
    }
  }
  /**
   * Apply the transform function to a result
   *
   * @param result Result to transform
   * @returns Transformed result
   */
  transformResult(result) {
    if (!this.transform) {
      return result;
    }
    return this.transform(result);
  }
  /**
   * Get the map of tool use IDs
   *
   * @returns Map of tool name to tool use ID
   */
  getToolUseIds() {
    return new Map(this.toolUseIds);
  }
};
var ToolResultMatcher = class {
  /**
   * Create a new ToolResultMatcher
   *
   * @param toolUseIds Map of tool name to tool use ID
   * @param transform Transform function for matched results
   */
  constructor(toolUseIds, transform) {
    this.toolUseIds = toolUseIds;
    this.transform = transform;
    this.xmlMatcher = new XmlMatcher("tool_result");
    this.jsonMatcher = new JsonMatcher("tool_result");
    this.formatDetector = new FormatDetector();
  }
  xmlMatcher;
  jsonMatcher;
  formatDetector;
  detectedFormat = "unknown";
  /**
   * Update the matcher with a new chunk of text
   *
   * @param chunk New text chunk to process
   * @returns Array of matched results
   */
  update(chunk) {
    if (this.detectedFormat === "unknown") {
      this.detectedFormat = this.formatDetector.detectFormat(chunk);
    }
    let results = [];
    if (this.detectedFormat === "xml" || this.detectedFormat === "unknown") {
      const toolResultRegex = /<tool_result\s+tool_use_id="([^"]+)"(?:\s+status="([^"]+)")?>[\s\S]*?<\/tool_result>/g;
      let match;
      let lastIndex = 0;
      const matches = [];
      while ((match = toolResultRegex.exec(chunk)) !== null) {
        const toolUseId = match[1];
        matches.push({
          start: match.index,
          end: match.index + match[0].length,
          content: match[0],
          toolUseId
        });
      }
      for (let i = 0; i < matches.length; i++) {
        const match2 = matches[i];
        if (match2.start > lastIndex) {
          const textBefore = chunk.substring(lastIndex, match2.start);
          results.push(
            this.transformResult({
              matched: false,
              data: textBefore
            })
          );
        }
        try {
          const jsonStr = xmlToolResultToJson(match2.content);
          const obj = JSON.parse(jsonStr);
          results.push(
            this.transformResult({ matched: true, data: obj, type: "tool_result" })
          );
        } catch {
          results.push(
            this.transformResult({ matched: true, data: match2.content, type: "tool_result" })
          );
        }
        lastIndex = match2.end;
      }
      if (lastIndex < chunk.length) {
        results.push(
          this.transformResult({
            matched: false,
            data: chunk.substring(lastIndex)
          })
        );
      }
      return results;
    }
    if (this.detectedFormat === "json") {
      return this.jsonMatcher.update(chunk).map((r) => this.transformResult(r));
    } else {
      return this.xmlMatcher.update(chunk).map((r) => this.transformResult(r));
    }
  }
  /**
   * Process any remaining content and return final results
   *
   * @param chunk Optional final chunk to process
   * @returns Array of matched results
   */
  final(chunk) {
    if (chunk) {
      if (this.detectedFormat === "unknown") {
        this.detectedFormat = this.formatDetector.detectFormat(chunk);
      }
    }
    if (this.detectedFormat === "json") {
      return this.jsonMatcher.final(chunk).map((r) => this.transformResult(r));
    } else {
      return this.xmlMatcher.final(chunk).map((r) => this.transformResult(r));
    }
  }
  /**
   * Apply the transform function to a result
   *
   * @param result Result to transform
   * @returns Transformed result
   */
  transformResult(result) {
    if (!this.transform) {
      return result;
    }
    return this.transform(result);
  }
};
var HybridMatcher = class {
  /**
   * Create a new HybridMatcher
   *
   * @param tagName XML tag name to match
   * @param jsonType JSON type to match
   * @param transform Transform function for matched results
   * @param matchToolUse Whether to match tool use blocks (default: false)
   * @param matchToolResult Whether to match tool result blocks (default: false)
   */
  constructor(tagName, jsonType, transform, matchToolUse = false, matchToolResult = false) {
    this.tagName = tagName;
    this.jsonType = jsonType;
    this.transform = transform;
    this.matchToolUse = matchToolUse;
    this.matchToolResult = matchToolResult;
    this.xmlMatcher = new XmlMatcher(tagName);
    this.jsonMatcher = new JsonMatcher(jsonType);
    this.formatDetector = new FormatDetector();
    if (matchToolUse) {
      this.toolUseMatcher = new ToolUseMatcher(this.transform);
    }
    if (matchToolResult && this.toolUseMatcher) {
      this.toolResultMatcher = new ToolResultMatcher(this.toolUseMatcher.getToolUseIds(), this.transform);
    }
  }
  xmlMatcher;
  jsonMatcher;
  formatDetector;
  detectedFormat = "unknown";
  toolUseIds = /* @__PURE__ */ new Map();
  toolUseMatcher = null;
  toolResultMatcher = null;
  /**
   * Update the matcher with a new chunk of text
   *
   * @param chunk New text chunk to process
   * @returns Array of matched results
   */
  update(chunk) {
    if (this.detectedFormat === "unknown") {
      this.detectedFormat = this.formatDetector.detectFormat(chunk);
    }
    let results = [];
    if (this.tagName === "think" && this.jsonType === "thinking") {
      if (this.detectedFormat === "xml" || this.detectedFormat === "unknown") {
        const xmlResults = this.xmlMatcher.update(chunk);
        for (const result of xmlResults) {
          if (result.matched) {
            results.push(
              this.transformResult({
                matched: true,
                data: result.data,
                type: "thinking"
              })
            );
          } else {
            results.push(this.transformResult(result));
          }
        }
      } else if (this.detectedFormat === "json") {
        const jsonResults = this.jsonMatcher.update(chunk);
        for (const result of jsonResults) {
          if (result.matched && typeof result.data === "object" && result.data.type === "thinking") {
            results.push(
              this.transformResult({
                matched: true,
                data: result.data.content,
                type: "thinking"
              })
            );
          } else {
            results.push(this.transformResult(result));
          }
        }
      }
      if (results.some((r) => r.matched)) {
        return results;
      }
    }
    if (this.matchToolUse && this.toolUseMatcher) {
      const toolUseResults = this.toolUseMatcher.update(chunk);
      if (toolUseResults.length > 0) {
        results = [...results, ...toolUseResults];
      }
      this.toolUseIds = new Map(this.toolUseMatcher.getToolUseIds());
    }
    if (this.matchToolResult && this.toolResultMatcher) {
      const toolResultResults = this.toolResultMatcher.update(chunk);
      if (toolResultResults.length > 0) {
        results = [...results, ...toolResultResults];
      }
    }
    if (results.length === 0) {
      if (this.detectedFormat === "json") {
        results = this.jsonMatcher.update(chunk).map((r) => this.transformResult(r));
      } else {
        results = this.xmlMatcher.update(chunk).map((r) => this.transformResult(r));
      }
    }
    return results;
  }
  /**
   * Process any remaining content and return final results
   *
   * @param chunk Optional final chunk to process
   * @returns Array of matched results
   */
  final(chunk) {
    if (chunk) {
      if (this.detectedFormat === "unknown") {
        this.detectedFormat = this.formatDetector.detectFormat(chunk);
      }
    }
    let results = [];
    if (this.matchToolUse && this.toolUseMatcher) {
      const toolUseResults = this.toolUseMatcher.final(chunk);
      if (toolUseResults.length > 0) {
        results = [...results, ...toolUseResults];
      }
    }
    if (this.matchToolResult && this.toolResultMatcher) {
      const toolResultResults = this.toolResultMatcher.final(chunk);
      if (toolResultResults.length > 0) {
        results = [...results, ...toolResultResults];
      }
    }
    if (results.length === 0) {
      if (this.detectedFormat === "json") {
        results = this.jsonMatcher.final(chunk).map((r) => this.transformResult(r));
      } else {
        return this.xmlMatcher.final(chunk).map((r) => this.transformResult(r));
      }
    }
    return results;
  }
  /**
   * Apply the transform function to a result
   *
   * @param result Result to transform
   * @returns Transformed result
   */
  transformResult(result) {
    if (!this.transform) {
      return result;
    }
    return this.transform(result);
  }
  /**
   * Get the map of tool use IDs
   *
   * @returns Map of tool name to tool use ID
   */
  getToolUseIds() {
    return new Map(this.toolUseIds);
  }
};
// Annotate the CommonJS export names for ESM import in node:
0 && (module.exports = {
  FormatDetector,
  HybridMatcher,
  JsonMatcher,
  ToolResultMatcher,
  ToolUseMatcher,
  jsonThinkingToXml,
  jsonToolResultToXml,
  jsonToolUseToXml,
  neutralToolUseToOpenAiFunctionCall,
  openAiFunctionCallToNeutralToolUse,
  xmlThinkingToJson,
  xmlToolResultToJson,
  xmlToolUseToJson
});
