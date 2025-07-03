# LangChain + Azure OpenAI Demo

This demo showcases how to use LangChain with Azure OpenAI for various prompt engineering tasks.

## Setup

1. **Install dependencies:**
   ```bash
   pip install -r ../../requirements.txt
   ```

2. **Set up environment variables:**
   Create a `.env` file in this directory with the following variables:
   ```
   AZURE_OPENAI_API_KEY=your_api_key_here
   AZURE_OPENAI_ENDPOINT=https://your-resource.openai.azure.com/
   AZURE_OPENAI_DEPLOYMENT_NAME=your_deployment_name
   AZURE_OPENAI_API_VERSION=2023-12-01-preview
   ```

3. **Run the demo:**
   ```bash
   python script01.py
   ```

## What the Demo Covers

### Demo 1: Simple Direct Prompt
- Basic usage of Azure OpenAI with system and human messages
- Direct invocation of the language model

### Demo 2: Using Prompt Templates
- Creating reusable prompt templates
- Variable substitution in prompts
- Formatted message generation

### Demo 3: Using LLM Chains
- Chaining prompts with LangChain
- Input/output handling
- Structured prompt execution

### Demo 4: Multi-turn Conversation
- Maintaining conversation context
- Building conversation history
- Stateful interactions

### Demo 5: Temperature Comparison
- Demonstrating different creativity levels
- Comparing outputs with different temperature settings
- Understanding model randomness

## Key Features Demonstrated

- **Azure OpenAI Integration**: How to properly configure and use Azure OpenAI with LangChain
- **Prompt Engineering**: Different approaches to crafting effective prompts
- **Template Usage**: Creating reusable and parameterized prompts
- **Chain Operations**: Combining multiple operations in a workflow
- **Conversation Management**: Handling multi-turn conversations
- **Parameter Tuning**: Understanding temperature and other model parameters

## Requirements

- Python 3.7+
- Azure OpenAI resource with a deployed model
- Valid Azure OpenAI API key and endpoint

## Troubleshooting

- Make sure your Azure OpenAI deployment is active
- Verify your API key and endpoint are correct
- Check that your deployment name matches the one in your Azure portal
- Ensure you have sufficient quota in your Azure OpenAI resource 