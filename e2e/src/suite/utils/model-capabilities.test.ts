import * as assert from 'assert'
import * as sinon from 'sinon'
import {
  supportsComputerUse,
  supportsPromptCaching,
  supportsImages,
  supportsThinking,
  supportsTemperature,
  getMaxTokens,
  getReasoningEffort,
  hasCapability,
  getContextWindowSize
} from "../model-capabilities"
import { ModelInfo } from "../../schemas"

suite("Model Capability Detection", () => {
	let sandbox: sinon.SinonSandbox

	setup(() => {
		sandbox = sinon.createSandbox()
	})

	teardown(() => {
		sandbox.restore()
	})
  // Create mock model info objects for testing
  const fullFeaturedModel: ModelInfo = {
    maxTokens: 8192,
    contextWindow: 200_000,
    supportsPromptCache: true,
    supportsComputerUse: true,
    supportsImages: true,
    supportsTemperature: true,
    thinking: true,
    reasoningEffort: "high",
    inputPrice: 3.0,
    outputPrice: 15.0,
    description: "A full-featured model with all capabilities"
  }

  const limitedModel: ModelInfo = {
    maxTokens: 4096,
    contextWindow: 8192,
    supportsPromptCache: false,
    supportsComputerUse: false,
    supportsImages: false,
    supportsTemperature: false,
    thinking: false,
    reasoningEffort: "low",
    inputPrice: 1.0,
    outputPrice: 5.0,
    description: "A limited model with minimal capabilities"
  }

  const partiallyDefinedModel: ModelInfo = {
    contextWindow: 16384,
    supportsPromptCache: true,
    // Other properties are undefined
  }

  suite("supportsComputerUse", () => {
    test("should return true when supportsComputerUse is true", () => {
      expect(supportsComputerUse(fullFeaturedModel)).toBe(true)
    })

    test("should return false when supportsComputerUse is false", () => {
      expect(supportsComputerUse(limitedModel)).toBe(false)
    })

    test("should return false when supportsComputerUse is undefined", () => {
      expect(supportsComputerUse(partiallyDefinedModel)).toBe(false)
    })
  })

  suite("supportsPromptCaching", () => {
    test("should return true when supportsPromptCache is true", () => {
      expect(supportsPromptCaching(fullFeaturedModel)).toBe(true)
    })

    test("should return false when supportsPromptCache is false", () => {
      expect(supportsPromptCaching(limitedModel)).toBe(false)
    })

    test("should return false when supportsPromptCache is undefined", () => {
      // Create a model without the supportsPromptCache property
      const modelProps = { contextWindow: 16384 } as ModelInfo
      expect(supportsPromptCaching(modelProps)).toBe(false)
    })
  })

  suite("supportsImages", () => {
    test("should return true when supportsImages is true", () => {
      expect(supportsImages(fullFeaturedModel)).toBe(true)
    })

    test("should return false when supportsImages is false", () => {
      expect(supportsImages(limitedModel)).toBe(false)
    })

    test("should return false when supportsImages is undefined", () => {
      expect(supportsImages(partiallyDefinedModel)).toBe(false)
    })
  })

  suite("supportsThinking", () => {
    test("should return true when thinking is true", () => {
      expect(supportsThinking(fullFeaturedModel)).toBe(true)
    })

    test("should return false when thinking is false", () => {
      expect(supportsThinking(limitedModel)).toBe(false)
    })

    test("should return false when thinking is undefined", () => {
      expect(supportsThinking(partiallyDefinedModel)).toBe(false)
    })
  })

  suite("supportsTemperature", () => {
    test("should return true when supportsTemperature is true", () => {
      expect(supportsTemperature(fullFeaturedModel)).toBe(true)
    })

    test("should return false when supportsTemperature is false", () => {
      expect(supportsTemperature(limitedModel)).toBe(false)
    })

    test("should return true when supportsTemperature is undefined", () => {
      // The function defaults to true unless explicitly set to false
      expect(supportsTemperature(partiallyDefinedModel)).toBe(true)
    })
  })

  suite("getMaxTokens", () => {
    test("should return the maxTokens value when defined", () => {
      expect(getMaxTokens(fullFeaturedModel)).toBe(8192)
    })

    test("should return the default value when maxTokens is undefined", () => {
      expect(getMaxTokens(partiallyDefinedModel)).toBe(4096) // Default is 4096
    })

    test("should return the provided default value when maxTokens is undefined", () => {
      expect(getMaxTokens(partiallyDefinedModel, 2048)).toBe(2048)
    })
  })

  suite("getReasoningEffort", () => {
    test("should return the reasoningEffort value when defined", () => {
      expect(getReasoningEffort(fullFeaturedModel)).toBe("high")
    })

    test("should return undefined when reasoningEffort is undefined", () => {
      expect(getReasoningEffort(partiallyDefinedModel)).toBeUndefined()
    })
  })

  suite("hasCapability", () => {
    test("should correctly check computerUse capability", () => {
      expect(hasCapability(fullFeaturedModel, "computerUse")).toBe(true)
      expect(hasCapability(limitedModel, "computerUse")).toBe(false)
    })

    test("should correctly check promptCache capability", () => {
      expect(hasCapability(fullFeaturedModel, "promptCache")).toBe(true)
      expect(hasCapability(limitedModel, "promptCache")).toBe(false)
    })

    test("should correctly check images capability", () => {
      expect(hasCapability(fullFeaturedModel, "images")).toBe(true)
      expect(hasCapability(limitedModel, "images")).toBe(false)
    })

    test("should correctly check thinking capability", () => {
      expect(hasCapability(fullFeaturedModel, "thinking")).toBe(true)
      expect(hasCapability(limitedModel, "thinking")).toBe(false)
    })

    test("should correctly check temperature capability", () => {
      expect(hasCapability(fullFeaturedModel, "temperature")).toBe(true)
      expect(hasCapability(limitedModel, "temperature")).toBe(false)
    })
  })

  suite("getContextWindowSize", () => {
    test("should return the contextWindow value when defined", () => {
      expect(getContextWindowSize(fullFeaturedModel)).toBe(200_000)
    })

    test("should return the default value when contextWindow is undefined", () => {
      // Create a model without the contextWindow property
      const modelWithoutContextWindow = { 
        supportsPromptCache: true 
      } as ModelInfo
      
      expect(getContextWindowSize(modelWithoutContextWindow)).toBe(8192) // Default is 8192
    })

    test("should return the provided default value when contextWindow is undefined", () => {
      // Create a model without the contextWindow property
      const modelWithoutContextWindow = { 
        supportsPromptCache: true 
      } as ModelInfo
      
      expect(getContextWindowSize(modelWithoutContextWindow, 4096)).toBe(4096)
    })
  })
// Mock cleanup