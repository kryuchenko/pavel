#!/bin/bash
# Run tool with GitHub Secret for OpenAI API key

# Get the secret from GitHub
OPENAI_KEY=$(gh secret get OPENAI_API_KEY 2>/dev/null)

if [ -z "$OPENAI_KEY" ]; then
    echo "❌ Could not retrieve OPENAI_API_KEY from GitHub Secrets"
    echo "Make sure you're logged in with 'gh auth login'"
    exit 1
fi

echo "✅ Retrieved OpenAI API key from GitHub Secrets"

# Export and run the command
export OPENAI_API_KEY="$OPENAI_KEY"
PYTHONPATH=src python "$@"