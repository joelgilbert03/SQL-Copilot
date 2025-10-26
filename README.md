# SQL Copilot: Your AI-Powered Database Assistant

SQL Copilot is an AI-powered assistant designed to help you interact with your SQL databases through a conversational chat interface. Configure your SQL database, and the AI assistant will help with everything from basic queries to complex data analysis.


## Key Features

*   **Conversational Database Interaction:** Chat with your database in natural language.
*   **Multi-LLM Support:** Powered by state-of-the-art language models from Mistral, OpenAI, and Anthropic.
*   **Text-to-SQL Conversion:** Automatically convert your questions into SQL queries.
*   **Vector-Powered Schema Retrieval:** Uses a Qdrant vector database to find the most relevant tables for your query.
*   **Multi-Step & Intermediate Queries:** Executes intermediate queries to validate data and construct more accurate final queries.
*   **Jupyter Notebook Integration:** Perform advanced data analysis and create visualizations in a Jupyter notebook environment.
*   **Vision-Enabled Insights:** Get automatic descriptions of your generated plots and charts.
*   **Containerized & Scalable:** Built with Docker for easy deployment and scalability.

## How It Works

1.  **User Asks a Question:** You ask a question in plain English in the chat interface.
2.  **Agent Understands:** The AI agent, powered by an LLM, analyzes your request.
3.  **Relevant Schema is Retrieved:** The agent queries the Qdrant vector database to find the most relevant table schemas for your question.
4.  **SQL is Generated:** The LLM generates an SQL query based on your question and the retrieved schemas.
5.  **Query is Executed:** The query is executed on your database.
6.  **Results are Displayed:** The results are displayed in the chat, and you can further analyze them using the integrated Jupyter notebook.

## Tech Stack

*   **Backend:** Python
*   **UI Framework:** [Chainlit](https://chainlit.io/)
*   **LLM Providers:** Mistral, OpenAI, Anthropic, Ollama
*   **Vector Database:** [Qdrant](https://qdrant.tech/)
*   **Data Science:** Pandas, NumPy, Scikit-learn, Plotly
*   **Database Connectors:** PyMySQL, Psycopg2, Snowflake, PyODBC
*   **Containerization:** Docker, Docker Compose

## Getting Started

### Prerequisites

*   [Docker](https://docs.docker.com/get-docker/) and [Docker Compose](https://docs.docker.com/compose/install/) installed on your machine.
*   API keys for your chosen LLM provider (e.g., Mistral, OpenAI).

### Installation

1.  **Clone the repository:**

    ```bash
    git clone https://github.com/your-username/sql-copilot.git
    cd sql-copilot
    ```

2.  **Configure your environment:**

    Create a `.env` file in the root of the project and add your database credentials and LLM API keys. You can use the `.env.example` file as a template:

    ```bash
    cp .env.example .env
    ```

    Then, edit the `.env` file with your credentials.

3.  **Run the application:**

    ```bash
    docker-compose up --build
    ```

4.  **Access the application:**

    Open your browser and navigate to `http://localhost:8080`.

## Configuration

You can configure the database connection and LLM provider by setting the following environment variables in your `.env` file:

```
# ----------------- Database Configuration -----------------
DB_TYPE=mysql
DB_HOST=your_database_host
DB_USER=your_database_user
DB_PASSWORD=your_database_password
DB_NAME=your_database_name

# ----------------- LLM Configuration -----------------
MISTRAL_API_KEY=your_mistral_api_key
# OPENAI_API_KEY=your_openai_api_key
# ANTHROPIC_API_KEY=your_anthropic_api_key
```
