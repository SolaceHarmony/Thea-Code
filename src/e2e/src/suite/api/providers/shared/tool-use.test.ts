import { ToolCallAggregator, extractToolCallsFromDelta } from "../tool-use"
import type OpenAI from "openai"

import * as assert from 'assert'
function deltaToolCall(id: string, name: string, argsChunk: string, index?: number): OpenAI.Chat.Completions.ChatCompletionChunk.Choice.Delta {
  return {
    tool_calls: [{ id, index, type: "function", function: { name, arguments: argsChunk } }],
  } as OpenAI.Chat.Completions.ChatCompletionChunk.Choice.Delta

suite("ToolCallAggregator", () => {
  test("single complete tool call in one chunk", () => {
    const agg = new ToolCallAggregator()
    const out = agg.addFromDelta(deltaToolCall("a", "sum", `{"x":1,"y":2}`))
    assert.strictEqual(out.length, 1)
    assert.strictEqual(out[0].complete, true)
    assert.strictEqual(out[0].name, "sum")
    assert.deepStrictEqual(out[0].parsedArgs, { x: 1, y: 2 })

  test("arguments split across multiple chunks", () => {
    const agg = new ToolCallAggregator()
    expect(agg.addFromDelta(deltaToolCall("a", "sum", `{"x":1,`))).toHaveLength(0)
    const out = agg.addFromDelta(deltaToolCall("a", "sum", `"y":2}`))
    assert.strictEqual(out.length, 1)
    assert.strictEqual(out[0].complete, true)
    assert.strictEqual(out[0].argString, `{"x":1,"y":2}`)
    assert.deepStrictEqual(out[0].parsedArgs, { x: 1, y: 2 })

  test("multiple parallel tool calls with indices", () => {
    const agg = new ToolCallAggregator()
    const out1 = agg.addFromDelta(deltaToolCall("a", "one", `{"a":1}`, 0))
    const out2 = agg.addFromDelta(deltaToolCall("b", "two", `{"b":2}`, 1))
    
    assert.strictEqual(out1.length, 1)
    assert.strictEqual(out1[0].name, "one")
    assert.strictEqual(out1[0].index, 0)
    
    assert.strictEqual(out2.length, 1)
    assert.strictEqual(out2[0].name, "two")
    assert.strictEqual(out2[0].index, 1)

  test("malformed json never completes but finalizes", () => {
    const agg = new ToolCallAggregator()
    agg.addFromDelta(deltaToolCall("a", "bad", `{"x":1`))
    const flushed = agg.finalize()
    assert.strictEqual(flushed.length, 1)
    assert.strictEqual(flushed[0].complete, false) // still malformed; finalize doesn't force completion
    assert.strictEqual(flushed[0].argString, `{"x":1`)
    assert.strictEqual(flushed[0].parsedArgs, undefined)

  test("maxArgBytes safety cap", () => {
    const agg = new ToolCallAggregator({ maxArgBytes: 8 })
    const out = agg.addFromDelta(deltaToolCall("a", "sum", `{"x":123456}`))
    assert.strictEqual(out.length, 1)
    assert.strictEqual(out[0].limited, true)
    assert.strictEqual(out[0].complete, true)
    assert.strictEqual(out[0].parsedArgs, undefined) // too long, not parsed

  test("reset clears all accumulated state", () => {
    const agg = new ToolCallAggregator()
    agg.addFromDelta(deltaToolCall("a", "sum", `{"x":1`))
    agg.reset()
    const flushed = agg.finalize()
    assert.strictEqual(flushed.length, 0)

  test("parse attempts tracking", () => {
    const agg = new ToolCallAggregator()
    agg.addFromDelta(deltaToolCall("a", "sum", `{`))
    agg.addFromDelta(deltaToolCall("a", "sum", `"x"`))
    agg.addFromDelta(deltaToolCall("a", "sum", `:`))
    const out = agg.addFromDelta(deltaToolCall("a", "sum", `1}`))
    
    assert.strictEqual(out.length, 1)
    assert.strictEqual(out[0].parseAttempts, 4)

  test("handles tool calls without explicit id using generated ids", () => {
    const agg = new ToolCallAggregator()
    // Without explicit IDs, the system generates them based on name and timestamp
    const delta1: OpenAI.Chat.Completions.ChatCompletionChunk.Choice.Delta = { tool_calls: [{ type: "function", function: { name: "test1", arguments: `{"a":1}` }, index: 0 }] }
    const delta2: OpenAI.Chat.Completions.ChatCompletionChunk.Choice.Delta = { tool_calls: [{ type: "function", function: { name: "test2", arguments: `{"b":2}` }, index: 1 }] }
    
    const out1 = agg.addFromDelta(delta1)
    const out2 = agg.addFromDelta(delta2)
    
    assert.strictEqual(out1.length, 1)
    assert.strictEqual(out2.length, 1)
    assert.strictEqual(out1[0].name, "test1")
    assert.strictEqual(out2[0].name, "test2")
    assert.deepStrictEqual(out1[0].parsedArgs, { a: 1 })
    assert.deepStrictEqual(out2[0].parsedArgs, { b: 2 })

  test("accumulates chunks for same tool call without id", () => {
    const agg = new ToolCallAggregator()
    // Without ID, same name and index should accumulate
    const delta1: OpenAI.Chat.Completions.ChatCompletionChunk.Choice.Delta = { tool_calls: [{ type: "function", function: { name: "test", arguments: `{"a":` }, index: 0 }] }
    const delta2: OpenAI.Chat.Completions.ChatCompletionChunk.Choice.Delta = { tool_calls: [{ type: "function", function: { name: "test", arguments: `1}` }, index: 0 }] }
    
    const out1 = agg.addFromDelta(delta1)
    assert.strictEqual(out1.length, 0) // Not complete yet
    
    const out2 = agg.addFromDelta(delta2)
    assert.strictEqual(out2.length, 1) // Now complete
    assert.strictEqual(out2[0].argString, `{"a":1}`)
    assert.deepStrictEqual(out2[0].parsedArgs, { a: 1 })

suite("extractToolCallsFromDelta", () => {
  test("extracts standard OpenAI tool calls", () => {
    const delta = {
      tool_calls: [
        { id: "call_1", type: "function", function: { name: "test", arguments: `{"x":1}` } }

    const calls = extractToolCallsFromDelta(delta)
    assert.strictEqual(calls.length, 1)
    assert.strictEqual(calls[0].id, "call_1")
    assert.strictEqual(calls[0].function.name, "test")

  test("extracts XML tool calls from content", () => {
    const delta = {
      content: `<search><query>test query</query><limit>10</limit></search>`

    const calls = extractToolCallsFromDelta(delta)
    assert.strictEqual(calls.length, 1)
    assert.strictEqual(calls[0].function.name, "search")
    const args = JSON.parse(calls[0].function.arguments) as { query: string; limit: number }
    assert.strictEqual(args.query, "test query")
    assert.strictEqual(args.limit, 10)

  test("extracts JSON tool_use from content", () => {
    const delta = {
      content: `Here's the result: {"type":"tool_use","name":"calculate","id":"calc_1","input":{"x":5,"y":3}}`

    const calls = extractToolCallsFromDelta(delta)
    assert.strictEqual(calls.length, 1)
    assert.strictEqual(calls[0].id, "calc_1")
    assert.strictEqual(calls[0].function.name, "calculate")
    const args = JSON.parse(calls[0].function.arguments) as { x: number; y: number }
    assert.strictEqual(args.x, 5)
    assert.strictEqual(args.y, 3)

  test("handles multiple JSON tool_use blocks in content", () => {
    const delta = {
      content: `First: {"type":"tool_use","name":"tool1","input":{"a":1}} and second: {"type":"tool_use","name":"tool2","input":{"b":2}}`

    const calls = extractToolCallsFromDelta(delta)
    assert.strictEqual(calls.length, 2)
    assert.strictEqual(calls[0].function.name, "tool1")
    assert.strictEqual(calls[1].function.name, "tool2")

  test("ignores reserved XML tags", () => {
    const delta = {
      content: `<think>internal thought</think><tool_result>old result</tool_result><search><query>real tool</query></search>`

    const calls = extractToolCallsFromDelta(delta)
    assert.strictEqual(calls.length, 1)
    assert.strictEqual(calls[0].function.name, "search")
