<div align="center">
<sub>

English • [Català](locales/ca/README.md) • [Deutsch](locales/de/README.md) • [Español](locales/es/README.md) • [Français](locales/fr/README.md) • [हिन्दी](locales/hi/README.md) • [Italiano](locales/it/README.md)

</sub>
<sub>

[日本語](locales/ja/README.md) • [한국어](locales/ko/README.md) • [Polski](locales/pl/README.md) • [Português (BR)](locales/pt-BR/README.md) • [Türkçe](locales/tr/README.md) • [Tiếng Việt](locales/vi/README.md) • [简体中文](locales/zh-CN/README.md) • [繁體中文](locales/zh-TW/README.md)

</sub>
<h3>Welcome to Thea Code - A Community of Innovators</h3>
</div>
<p>Thea Code is not just an AI tool; it's a dynamic platform powered by the creativity and collaboration of its community. Whether you're a seasoned developer or just starting, your contributions help shape the future of AI-driven coding tools.</p>
<br>

<div align="center">
  <h2>Join the Thea Code Community</h2>
<p>By joining the Thea Code community, you become part of a global network of innovators. Share your ideas, contribute code, and explore the vast possibilities of AI-enhanced development.</p>
  <p>Connect with developers, contribute ideas, and stay ahead with the latest AI-powered coding tools.</p>
  
  <a href="https://discord.gg/EmberHarmony" target="_blank"><img src="https://img.shields.io/badge/Join%20Discord-5865F2?style=for-the-badge&logo=discord&logoColor=white" alt="Join Discord"></a>
  <a href="https://www.reddit.com/r/thea-placeholder/" target="_blank"><img src="https://img.shields.io/badge/Join%20Reddit-FF4500?style=for-the-badge&logo=reddit&logoColor=white" alt="Join Reddit"></a>
  
</div>
<br>
<br>

<div align="center">
<h1>Thea Code</h1>

<a href="SolaceHarmony/Thea-Code/discussions/categories/feature-requests?discussions_q=is%3Aopen+category%3A%22Feature+Requests%22+sort%3Atop" target="_blank"><img src="https://img.shields.io/badge/Feature%20Requests-yellow?style=for-the-badge" alt="Feature Requests"></a>
<a href="https://docs.thea-placeholder.com" target="_blank"><img src="https://img.shields.io/badge/Documentation-6B46C1?style=for-the-badge&logo=readthedocs&logoColor=white" alt="Documentation"></a>

</div>

**Thea Code** is an AI-powered **autonomous coding agent** that lives in your editor. It can:

- Communicate in natural language
- Read and write files directly in your workspace
- Run terminal commands
- Automate browser actions
- Integrate with any OpenAI-compatible or custom API/model
- Adapt its “personality” and capabilities through **Custom Modes**

Whether you’re seeking a flexible coding partner, a system architect, or specialized roles like a QA engineer or product manager, Thea Code can help you build software more efficiently.

Check out the [CHANGELOG](CHANGELOG.md) for detailed updates and fixes.

> **Note**: Thea Code is a fork of Roo Code. We'd like to express our special thanks to the Roo Code team for their excellent work which served as the foundation for this project.

---

## 🎉 Thea Code 0.0.5 Released

Thea Code 0.0.5 brings the initial release of this community fork!

- Initial fork from Roo Code with rebranding
- Updated documentation to reflect the new project identity
- Fixed references in contributing guidelines
- Maintained compatibility with existing features and functionality

---

## What Can Thea Code Do?

- 🚀 **Generate Code** from natural language descriptions
- 🔧 **Refactor & Debug** existing code
- 📝 **Write & Update** documentation
- 🤔 **Answer Questions** about your codebase
- 🔄 **Automate** repetitive tasks
- 🏗️ **Create** new files and projects

## Quick Start

1. [Install Thea Code](https://docs.thea-placeholder.com/getting-started/installing)
2. [Connect Your AI Provider](https://docs.thea-placeholder.com/getting-started/connecting-api-provider)
3. [Try Your First Task](https://docs.thea-placeholder.com/getting-started/your-first-task)

## Key Features

### Multiple Modes

Thea Code adapts to your needs with specialized [modes](https://docs.thea-placeholder.com/basic-usage/using-modes):

- **Code Mode:** For general-purpose coding tasks
- **Architect Mode:** For planning and technical leadership
- **Ask Mode:** For answering questions and providing information
- **Debug Mode:** For systematic problem diagnosis
- **[Custom Modes](https://docs.thea-placeholder.com/advanced-usage/custom-modes):** Create unlimited specialized personas for security auditing, performance optimization, documentation, or any other task

### Smart Tools

Thea Code comes with powerful [tools](https://docs.thea-placeholder.com/basic-usage/how-tools-work) that can:

- Read and write files in your project
- Execute commands in your VS Code terminal
- Control a web browser
- Use external tools via [MCP (Model Context Protocol)](https://docs.thea-placeholder.com/advanced-usage/mcp)

MCP extends Thea Code's capabilities by allowing you to add unlimited custom tools. Integrate with external APIs, connect to databases, or create specialized development tools - MCP provides the framework to expand Thea Code's functionality to meet your specific needs.

### Customization

Make Thea Code work your way with:

- [Custom Instructions](https://docs.thea-placeholder.com/advanced-usage/custom-instructions) for personalized behavior
- [Custom Modes](https://docs.thea-placeholder.com/advanced-usage/custom-modes) for specialized tasks
- [Local Models](https://docs.thea-placeholder.com/advanced-usage/local-models) for offline use
- [Auto-Approval Settings](https://docs.thea-placeholder.com/advanced-usage/auto-approving-actions) for faster workflows

## Resources

### Documentation

- [Basic Usage Guide](https://docs.thea-placeholder.com/basic-usage/the-chat-interface)
- [Advanced Features](https://docs.thea-placeholder.com/advanced-usage/auto-approving-actions)
- [Frequently Asked Questions](https://docs.thea-placeholder.com/faq)

### Community

- **Discord:** [Join our Discord server](https://discord.gg/EmberHarmony) for real-time help and discussions
- **Reddit:** [Visit our subreddit](https://www.reddit.com/r/thea-placeholder) to share experiences and tips
- **GitHub:** Report [issues](SolaceHarmony/Thea-Code/issues) or request [features](SolaceHarmony/Thea-Code/discussions/categories/feature-requests?discussions_q=is%3Aopen+category%3A%22Feature+Requests%22+sort%3Atop)

---

## Local Setup & Development

1. **Clone** the repo:

```sh
git clone https://github.com/SolaceHarmony/Thea-Code.git
```

1. **Install dependencies**:

```sh
npm run install:all
```

1. **Start the webview (Vite/React app with HMR)**:

```sh
npm run dev
```

1. **Debug**:
   Press `F5` (or **Run** → **Start Debugging**) in VSCode to open a new session with Thea Code loaded.

Changes to the webview will appear immediately. Changes to the core extension will require a restart of the extension host.

Alternatively you can build a .vsix and install it directly in VSCode:

```sh
npm run build
```

A `.vsix` file will appear in the `bin/` directory which can be installed with:

```sh
code --install-extension bin/thea-code-<version>.vsix
```

We use [changesets](https://github.com/changesets/changesets) for versioning and publishing. Check our `CHANGELOG.md` for release notes.

---

## Disclaimer

**Please note** that Solace Projectdoes **not** make any representations or warranties regarding any code, models, or other tools provided or made available in connection with Thea Code, any associated third-party tools, or any resulting outputs. You assume **all risks** associated with the use of any such tools or outputs; such tools are provided on an **"AS IS"** and **"AS AVAILABLE"** basis. Such risks may include, without limitation, intellectual property infringement, cyber vulnerabilities or attacks, bias, inaccuracies, errors, defects, viruses, downtime, property loss or damage, and/or personal injury. You are solely responsible for your use of any such tools or outputs (including, without limitation, the legality, appropriateness, and results thereof).

---

## Contributing

We love community contributions! Get started by reading our [CONTRIBUTING.md](CONTRIBUTING.md).

---

## Contributors

Thanks to all our contributors who have helped make Thea Code better!

<!-- START CONTRIBUTORS SECTION - AUTO-GENERATED, DO NOT EDIT MANUALLY -->

We welcome and appreciate contributions from the community. The Thea Code project is maintained by the Solace Harmony team and community contributors.

Special thanks to the original Roo Code contributors whose work provided the foundation for this project.

<!-- END CONTRIBUTORS SECTION -->

## License

[Apache 2.0 © 2025 Solace Project](./LICENSE)

---

**Enjoy Thea Code!** Whether you keep it on a short leash or let it roam autonomously, we can’t wait to see what you build. If you have questions or feature ideas, drop by our [Reddit community](https://www.reddit.com/r/thea-placeholder/) or [Discord](https://discord.gg/EmberHarmony). Happy coding!
