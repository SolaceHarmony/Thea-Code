# Thea Code User Guide

Welcome to Thea Code! This guide will help you get started and make the most of this AI-powered coding assistant.

## Quick Start

1. [Install Thea Code](./getting-started.md#installation)
2. [Configure your API provider](./providers/README.md)
3. [Start your first task](./getting-started.md#first-task)

## What is Thea Code?

Thea Code is an autonomous AI coding agent that lives in your VSCode editor. It can:

- 💬 Communicate in natural language
- 📝 Read and write files directly in your workspace
- 🖥️ Run terminal commands
- 🌐 Automate browser actions
- 🔌 Integrate with any OpenAI-compatible API
- 🛠️ Use MCP tools with unified access
- 🎭 Adapt through Custom Modes

## Guide Contents

### Getting Started
- [Installation & Setup](./getting-started.md)
- [Configuration](./configuration.md)
- [Your First Task](./getting-started.md#first-task)

### Features
- [Custom Modes](./features/custom-modes.md) - Customize Thea's behavior
- [MCP Integration](./features/mcp-integration.md) - Model Context Protocol tools
- [Browser Automation](./features/browser-automation.md) - Web interactions
- [Terminal Commands](./features/terminal-commands.md) - Command execution

### Provider Configuration
- [Provider Overview](./providers/README.md)
- [Anthropic (Claude)](./providers/anthropic.md)
- [OpenAI](./providers/openai.md)
- [Ollama (Local)](./providers/ollama.md)
- [Google Vertex](./providers/vertex.md)
- [AWS Bedrock](./providers/bedrock.md)
- [More Providers](./providers/README.md)

### Help & Support
- [Troubleshooting](./troubleshooting.md)
- [FAQ](./faq.md)
- [Discord Community](https://discord.gg/EmberHarmony)

## Key Concepts

### Tasks
A task is a conversation with Thea where you work together to accomplish a goal. Tasks can be:
- Creating new features
- Fixing bugs
- Refactoring code
- Writing documentation
- Running tests

### Context
Thea understands your project through context:
- Files in your workspace
- Terminal output
- Browser content
- Previous conversations

### Modes
Modes customize how Thea behaves:
- **Code** - General software development
- **Architect** - System design and planning
- **Ask** - Quick questions without file edits
- **Custom** - Your own specialized modes

## Best Practices

### 1. Clear Instructions
Be specific about what you want:
- ✅ "Add error handling to the login function in auth.js"
- ❌ "Fix the login"

### 2. Provide Context
Include relevant information:
- Error messages
- Expected behavior
- Related files

### 3. Review Changes
Always review Thea's suggestions before accepting:
- Check the diff view
- Run tests
- Verify the logic

### 4. Use Modes Effectively
Choose the right mode for your task:
- **Code** for implementation
- **Architect** for planning
- **Ask** for questions

## Keyboard Shortcuts

| Action | Shortcut |
|--------|----------|
| New Task | `Ctrl/Cmd + Shift + T` |
| Open Thea | `Ctrl/Cmd + Shift + P` → "Thea: Open" |
| Switch Mode | Click mode button in chat |
| Cancel Task | `Esc` in chat |

## Tips & Tricks

### 1. Mention Files
Use `@filename` to reference specific files:
```
Fix the error in @auth.js on line 42
```

### 2. Use Screenshots
Paste images directly into chat for:
- UI designs
- Error screenshots
- Diagrams

### 3. Browser Automation
Ask Thea to interact with web pages:
```
Go to the documentation and find the API endpoint for user authentication
```

### 4. Terminal Integration
Thea can run commands and see output:
```
Run the tests and fix any failures
```

## Privacy & Security

- Thea only accesses files in your workspace
- API keys are stored securely in VSCode
- No data is sent without your consent
- Review the [Privacy Policy](../../PRIVACY.md)

## Getting Help

### Resources
1. **Documentation** - You're here!
2. **Discord** - [Join our community](https://discord.gg/EmberHarmony)
3. **GitHub Issues** - [Report bugs](https://github.com/SolaceHarmony/Thea-Code/issues)
4. **FAQ** - [Common questions](./faq.md)

### Common Issues
- [API Connection Problems](./troubleshooting.md#api-connection)
- [Performance Issues](./troubleshooting.md#performance)
- [File Access Errors](./troubleshooting.md#file-access)

## Next Steps

Ready to start? Here's what to do:

1. **[Install Thea Code](./getting-started.md)**
2. **[Configure your API provider](./providers/README.md)**
3. **[Try your first task](./getting-started.md#first-task)**
4. **[Explore Custom Modes](./features/custom-modes.md)**

---

**Need more help?** Join our [Discord community](https://discord.gg/EmberHarmony) for support and discussions!