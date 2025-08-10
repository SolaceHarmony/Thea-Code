module.exports = {
  // ...existing code...
  overrides: [
    {
      files: ["e2e/**/*.ts"],
      parserOptions: {
        project: "./e2e/tsconfig.json"
      }
    }
  ]
  // ...existing code...
}