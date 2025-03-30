# LinkedIn Content Publisher AI Agent (LangGraph)

###This system automates the process of monitoring a specific LinkedIn profile for new activity, classifying the content using AI, transforming it into formats suitable for a personal website backend, and publishing it via a REST API. It utilizes the LangGraph framework for orchestrating the AI agents.

# **WARNING:** This tool uses web scraping (`pyppeteer`) which relies on LinkedIn's website structure (HTML selectors). LinkedIn frequently updates its site, which **will break the scraping functionality**. You **must** be prepared to update the CSS selectors in `utils.py` regularly by inspecting the LinkedIn website with browser developer tools. Automated login can also trigger CAPTCHAs or account flags. Use responsibly and at your own risk.

## Features

###   **LinkedIn Monitoring:** Uses `pyppeteer` to log in, navigate to the specified profile's activity feed, and scrape new posts published since the last run (timestamp tracked in `last_timestamp.txt`).
###   **AI Classification:** Employs an LLM (via `langchain_openai`) to classify each new post into categories like `blog`, `work###experience`, `education`, `achievement`, or `skill`.
###   **AI Transformation:** Leverages the LLM to reformat and enhance the content from LinkedIn into structured JSON payloads matching predefined schemas required by the backend API.
###   **Data Validation:** Uses Pydantic (`schemas.py`) to validate the structure and data types of the transformed JSON *before* attempting to publish.
###   **Content Publishing:** Sends the validated JSON data via asynchronous HTTP POST requests (`httpx`) to the appropriate backend REST endpoints.
###   **State Management & Orchestration:** Uses LangGraph (`main.py`) to define the workflow (nodes and edges), manage the state (`AgentState` in `schemas.py`) between steps, and handle conditional logic based on classifications.
###   **Error Handling & Logging:** Includes error handling for scraping, LLM calls, validation, and publishing. Provides detailed logging via Python's `logging` module.
###   **Configuration:** Uses environment variables (`.env` file) for sensitive information and settings.

## Project Structure
