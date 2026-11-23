# Financial Dashboard

```sh
pip install -r requirements.txt
streamlit run app.py
```

Open **AI Financial Copilot** in the sidebar. Suggested questions cover revenue,
unpaid invoices, forecasting, anomalies, KPIs, risk, and fraud review. The app
uses either the bundled ERP dataset or a sales CSV selected on **Dashboard**.
**New chat** starts a separate conversation, and **Download chat** exports the
selected conversation as JSON.

## How the Copilot answers

The built-in mode maps supported questions to parameterised SQL and Pandas
calculations. Set `OPENAI_API_KEY` in the server environment or Streamlit
Secrets to enable model-based question routing, SQL generation, and
evidence-based explanations. The number of available questions in built-in
mode depends on its prepared queries and calculations. The chat identifies
the active mode. [Runtime prompts](prompts.md)

## Chat storage

Conversations are stored in SQLite at `.copilot/chats.sqlite3` by default.
Set `COPILOT_DB_PATH` to place the database on a persistent volume. A browser
identifier and the active dataset determine which conversations are visible.
To reopen chats for an uploaded dataset, select the same CSV again in the same
browser. Streamlit Community Cloud does not guarantee persistence of files
created at runtime, so download chats you need to keep.
