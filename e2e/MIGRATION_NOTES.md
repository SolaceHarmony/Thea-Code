Migration notes
- All tests should live under e2e/src/suite/** and run in a real VS Code host.
- Old tests in src/** and test/** will be removed once migrated.
- Prefer real classes and integration paths; only use mocks for hard external boundaries.
- Use the mock servers in test/ for OpenAI/MCP when real HTTP calls are required.
