# Multi-Agent Financial and News Analysis System

This project implements a multi-agent system using the Phi framework for financial analysis, news gathering, and article summarization. The system consists of four specialized agents working together to provide comprehensive information services.

## Features

- **Finance Agent**: Retrieves financial data including stock prices, analyst recommendations, and fundamentals using YFinance
- **Blog Summarizer Agent**: Summarizes articles and blog posts using Newspaper4k
- **News Agent**: Gathers latest news and trends using Google Search
- **Main Agent**: Orchestrates the other agents and provides unified responses

## Prerequisites

- Python 3.x
- Required packages (install via pip):
  ```
  python-dotenv
  phi
  yfinance
  newspaper4k
  ```
- Groq API key (set in .env file)

## Installation

1. Clone the repository
2. Create a `.env` file and add your Groq API key:
   ```
   GROQ_API_KEY=your_api_key_here
   ```
3. Install the required packages:
   ```
   pip install -r requirements.txt
   ```

## Usage

### Running as a Web Application

The project can be run as a local web application using:

```python
python main.py
```

This will start a local server with a playground interface where you can interact with the agents.

### Using in CLI Mode

Alternatively, you can use the system in CLI mode by uncommenting and using the following code:

```python
is_true = True
while is_true:
    str = input("What would to like to know ?\n")
    if str == "Exit" or str == "exit":
        is_true = False
    agent_team.print_response(str, stream=True)
```

## Agent Descriptions

### Finance Agent
- Uses YFinance tools
- Displays financial data in table format
- Handles public company information
- Provides stock prices, analyst recommendations, and fundamentals

### Blog Summarize Agent
- Processes URLs using Newspaper4k
- Provides detailed summaries in bullet points and paragraphs
- Maintains context with history

### News Agent
- Uses Google Search for latest news
- Includes sources and dates in responses
- Focuses on current trends and news

### Main Agent (Team Leader)
- Coordinates all other agents
- Ensures proper formatting of responses
- Handles PDF limitations
- Maintains conversation history

## Storage

The system uses SQLite for storing agent conversations and history:
- Database file: `agents.db`
- Separate tables for each agent:
  - `finance_agent`
  - `summary_agent`
  - `news_agent`
  - `main_agent`

## Model Information

All agents use the Groq model `deepseek-r1-distill-llama-70b` for processing and generating responses.