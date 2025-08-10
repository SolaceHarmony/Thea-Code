// Example of updated webviewMessageHandler.ts with all dynamic providers

import { getBedrockModels, getGeminiModels, getVertexModels, getMistralModels, getDeepSeekModels } from "../../api/providers/provider-helpers"

// Add to the switch statement in webviewMessageHandler:

case "refreshBedrockModels": {
  const { apiConfiguration: configForRefresh } = await provider.getState()
  const bedrockModels = await getBedrockModels({
    awsRegion: configForRefresh.awsRegion,
    awsAccessKeyId: configForRefresh.awsAccessKeyId,
    awsSecretAccessKey: configForRefresh.awsSecretAccessKey,
    awsUseProfile: configForRefresh.awsUseProfile,
    awsProfile: configForRefresh.awsProfile,
  })
  
  if (Object.keys(bedrockModels).length > 0) {
    const cacheDir = await provider.ensureCacheDirectoryExists()
    await fs.writeFile(
      path.join(cacheDir, GlobalFileNames.bedrockModels),
      JSON.stringify(bedrockModels)
    )
    await provider.postMessageToWebview({ type: "bedrockModels", bedrockModels })
  }
  
  break
}

case "refreshGeminiModels": {
  const { apiConfiguration: configForRefresh } = await provider.getState()
  const geminiModels = await getGeminiModels({
    geminiApiKey: configForRefresh.geminiApiKey,
    googleGeminiBaseUrl: configForRefresh.googleGeminiBaseUrl,
  })
  
  if (Object.keys(geminiModels).length > 0) {
    const cacheDir = await provider.ensureCacheDirectoryExists()
    await fs.writeFile(
      path.join(cacheDir, GlobalFileNames.geminiModels),
      JSON.stringify(geminiModels)
    )
    await provider.postMessageToWebview({ type: "geminiModels", geminiModels })
  }
  
  break
}

case "refreshVertexModels": {
  const { apiConfiguration: configForRefresh } = await provider.getState()
  const vertexModels = await getVertexModels({
    vertexProjectId: configForRefresh.vertexProjectId,
    vertexRegion: configForRefresh.vertexRegion,
    vertexJsonCredentials: configForRefresh.vertexJsonCredentials,
    vertexKeyFile: configForRefresh.vertexKeyFile,
  })
  
  if (Object.keys(vertexModels).length > 0) {
    const cacheDir = await provider.ensureCacheDirectoryExists()
    await fs.writeFile(
      path.join(cacheDir, GlobalFileNames.vertexModels),
      JSON.stringify(vertexModels)
    )
    await provider.postMessageToWebview({ type: "vertexModels", vertexModels })
  }
  
  break
}

case "refreshMistralModels": {
  const { apiConfiguration: configForRefresh } = await provider.getState()
  const mistralModels = await getMistralModels({
    mistralApiKey: configForRefresh.mistralApiKey,
  })
  
  if (Object.keys(mistralModels).length > 0) {
    const cacheDir = await provider.ensureCacheDirectoryExists()
    await fs.writeFile(
      path.join(cacheDir, GlobalFileNames.mistralModels),
      JSON.stringify(mistralModels)
    )
    await provider.postMessageToWebview({ type: "mistralModels", mistralModels })
  }
  
  break
}

case "refreshDeepSeekModels": {
  const { apiConfiguration: configForRefresh } = await provider.getState()
  const deepSeekModels = await getDeepSeekModels({
    deepSeekApiKey: configForRefresh.deepSeekApiKey,
  })
  
  if (Object.keys(deepSeekModels).length > 0) {
    const cacheDir = await provider.ensureCacheDirectoryExists()
    await fs.writeFile(
      path.join(cacheDir, GlobalFileNames.deepSeekModels),
      JSON.stringify(deepSeekModels)
    )
    await provider.postMessageToWebview({ type: "deepSeekModels", deepSeekModels })
  }
  
  break
}

// Also add to the initialization section (webviewDidLaunch):

// Load cached models for all providers
const dynamicProviders = ['bedrock', 'gemini', 'vertex', 'mistral', 'deepseek']

for (const providerName of dynamicProviders) {
  // Load cached models
  void provider.readModelsFromCache(GlobalFileNames[`${providerName}Models`]).then((cachedModels) => {
    if (cachedModels) {
      void provider.postMessageToWebview({ 
        type: `${providerName}Models`, 
        [`${providerName}Models`]: cachedModels 
      })
    }
  })
  
  // Fetch fresh models in background
  void (async () => {
    try {
      const { apiConfiguration } = await provider.getState()
      let freshModels: Record<string, any> = {}
      
      switch (providerName) {
        case 'bedrock':
          freshModels = await getBedrockModels({
            awsRegion: apiConfiguration.awsRegion,
            awsAccessKeyId: apiConfiguration.awsAccessKeyId,
            awsSecretAccessKey: apiConfiguration.awsSecretAccessKey,
          })
          break
        case 'gemini':
          freshModels = await getGeminiModels({
            geminiApiKey: apiConfiguration.geminiApiKey,
          })
          break
        case 'vertex':
          freshModels = await getVertexModels({
            vertexProjectId: apiConfiguration.vertexProjectId,
            vertexRegion: apiConfiguration.vertexRegion,
          })
          break
        case 'mistral':
          freshModels = await getMistralModels({
            mistralApiKey: apiConfiguration.mistralApiKey,
          })
          break
        case 'deepseek':
          freshModels = await getDeepSeekModels({
            deepSeekApiKey: apiConfiguration.deepSeekApiKey,
          })
          break
      }
      
      if (Object.keys(freshModels).length > 0) {
        const cacheDir = await provider.ensureCacheDirectoryExists()
        await fs.writeFile(
          path.join(cacheDir, GlobalFileNames[`${providerName}Models`]),
          JSON.stringify(freshModels)
        )
        await provider.postMessageToWebview({ 
          type: `${providerName}Models`, 
          [`${providerName}Models`]: freshModels 
        })
      }
    } catch (error) {
      console.error(`Failed to fetch ${providerName} models:`, error)
      // Don't throw - just log and continue with other providers
    }
  })()
}