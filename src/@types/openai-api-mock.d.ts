declare module 'openai-api-mock' {
  type HttpMethod = 'GET' | 'POST' | 'PUT' | 'PATCH' | 'DELETE'

  interface MockRequestContext {
    req: {
      headers: Record<string, string | string[]>
    }
  }

  type MockEndpointHandler = (
    this: MockRequestContext,
    uri: string,
    requestBody: unknown,
  ) => unknown

  interface MockResponse {
    stopMocking(): void;

    // Used by our Mocha tests to capture request headers and return responses.
    addCustomEndpoint(method: HttpMethod, path: string, handler: MockEndpointHandler): void
  }
  export function mockOpenAIResponse(config: unknown): MockResponse;
}
