#!/bin/bash

# Migration script for Jest to Mocha tests
echo "Starting comprehensive test migration..."

# Create e2e test directories matching source structure
mkdir -p src/e2e/src/suite/shared
mkdir -p src/e2e/src/suite/utils
mkdir -p src/e2e/src/suite/core/tools
mkdir -p src/e2e/src/suite/core/sliding-window
mkdir -p src/e2e/src/suite/core/config
mkdir -p src/e2e/src/suite/core/ignore
mkdir -p src/e2e/src/suite/core/webview
mkdir -p src/e2e/src/suite/core/diff/strategies
mkdir -p src/e2e/src/suite/core/prompts
mkdir -p src/e2e/src/suite/core/mentions
mkdir -p src/e2e/src/suite/api/providers
mkdir -p src/e2e/src/suite/api/transform
mkdir -p src/e2e/src/suite/services
mkdir -p src/e2e/src/suite/integrations
mkdir -p src/e2e/src/suite/extension
mkdir -p src/e2e/src/suite/schemas

# Counter for progress
count=0
total=$(find src -name "*.test.ts" -type f | wc -l)

# Find all test files and migrate them
find src -name "*.test.ts" -type f | while read test_file; do
    count=$((count + 1))
    
    # Extract the relative path from src
    relative_path="${test_file#src/}"
    
    # Remove __tests__ from path and .test.ts extension
    clean_path="${relative_path//__tests__\//}"
    clean_path="${clean_path%.test.ts}"
    
    # Create the e2e test path
    e2e_path="src/e2e/src/suite/${clean_path}.e2e.test.ts"
    
    # Create directory if needed
    mkdir -p "$(dirname "$e2e_path")"
    
    echo "[$count/$total] Migrating: $test_file -> $e2e_path"
    
    # Run the migration using our node script
    if [ -f "migrate-jest-to-mocha.js" ]; then
        node migrate-jest-to-mocha.js "$test_file" "$e2e_path"
    else
        echo "Warning: migrate-jest-to-mocha.js not found, copying file as-is"
        cp "$test_file" "$e2e_path"
    fi
done

echo ""
echo "Migration complete!"
echo "Total files processed: $total"
echo ""
echo "Next steps:"
echo "1. Review complex mocking scenarios that may need manual adjustment"
echo "2. Update import paths for proxyquire-based mocking where needed"
echo "3. Run 'npm run test:e2e' to verify all tests pass"