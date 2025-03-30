## Setup

1.  **Clone the Repository:**

    ```bash
    git clone <your-repo-url>
    cd <your-repo-directory>
    ```

2.  **Create Virtual Environment (Recommended):**

    ```bash
    python -m venv venv
    source venv/bin/activate  # Linux/macOS
    # venv\Scripts\activate    # Windows
    ```

3.  **Install Dependencies:**

    ```bash
    pip install -r requirements.txt
    ```

    _Note: `pyppeteer` will download Chromium the first time it runs if it's not found._

4.  **Configure Environment Variables:**

    - Copy `.env.example` to `.env`.
    - Edit `.env` and fill in **all** required values:
      - `OPENAI_API_KEY`: Your OpenAI API key.
      - `BACKEND_BASE_URL`: The base URL of your backend API (e.g., `http://localhost:3000/api`).
      - `LINKEDIN_EMAIL`: Your LinkedIn login email.
      - `LINKEDIN_PASSWORD`: Your LinkedIn login password.
      - `LINKEDIN_PROFILE_URL`: The URL of the LinkedIn profile you want to monitor (e.g., `https://www.linkedin.com/in/yourusername/`).
      - Optionally set `OPENAI_MODEL_NAME` and `LOG_LEVEL`.

5.  **Create State File:**
    - Create an empty file named `last_timestamp.txt` in the project root directory:
      ```bash
      touch last_timestamp.txt # Linux/macOS
      # type nul > last_timestamp.txt # Windows Command Prompt
      # '' > last_timestamp.txt      # Windows PowerShell
      ```
    - The script will use this file to track the timestamp of the last post processed to avoid duplicates.

## Running the System

Execute the main script from your terminal:

```bash
python main.py
```
