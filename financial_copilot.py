"""Financial Copilot for the Lab 1 ERP dashboard (Week 10).

Run separately: streamlit run financial_copilot.py
Or import ``render_chat_panel`` into the existing Streamlit dashboard.

The default mode uses local analytics. Optional OpenAI mode explains financial
summaries and plans additional SQL queries for the read-only SQLite executor.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
import json
import os
from pathlib import Path
import re
import sqlite3
from typing import Any

import numpy as np
import pandas as pd

from data_processing import REQUIRED_COLUMNS


MAX_ROWS = 500
SALES_SQL_GENERATION_PROMPT = """You are a financial analytics SQL planner.
Generate ONE SQLite SELECT statement using ONLY the schema listed below.
Return strictly JSON: {"sql":"SELECT ...", "explanation":"..."}.
Never run commands or access external resources. Never invent columns, tables,
financial values, dates, data provenance, or causal explanations.
Sales revenue is SUM(sales_clean.revenue), excluding flagged invalid sales.
Use ISO dates and the historical snapshot. Prefer aggregates and limit listings to 50 rows.
Schema:
  sales_clean(sale_id, invoice_id, sale_date, customer_id, product_id,
              product_name, category, revenue, discount_rate)
"""
SQL_GENERATION_PROMPT = SALES_SQL_GENERATION_PROMPT + """\
  customers(customer_id, first_name, last_name, city, country, signup_date)
  invoices(invoice_id, customer_id, invoice_date, due_date, subtotal,
           tax_amount, invoice_value)
  payments(payment_id, invoice_id, payment_date, payment_method, amount_paid)
  products(product_id, product_name, category, unit_cost, unit_price, stock_quantity)
Conventions:
  For unpaid invoices, first aggregate payments BY invoice_id, then compare
  against invoices.invoice_value, using a tolerance of 0.01.
  Joining sales and payments directly may multiply rows: avoid this.
  Monetary currency is not specified in this dataset.
  For dates use YYYY-MM-DD; no CURRENT_DATE (historical snapshot).
  Do not select email or other private data; use customer IDs and names.
  Do not explain away anomalies or describe them as proven fraud.
"""


@dataclass
class QueryPlan:
    intent: str
    sql: str
    params: dict[str, Any] = field(default_factory=dict)


@dataclass
class CopilotAnswer:
    intent: str
    answer: str
    sql: str = ""
    data: pd.DataFrame = field(default_factory=pd.DataFrame)
    details: dict[str, Any] = field(default_factory=dict)


def find_project_root(explicit_root: str | Path | None = None) -> Path:
    """Locate original Lab 1 data from any project subdirectory."""
    seeds = [Path(explicit_root)] if explicit_root else [Path(__file__).resolve().parent, Path.cwd()]
    for seed in seeds:
        for ancestor in (seed.resolve(), *seed.resolve().parents):
            for candidate in (ancestor, ancestor / "lab1", ancestor / "lab1_work" / "lab1"):
                if ((candidate / "financial_analytics.db").is_file()
                        or (candidate / "week1-2" / "financial_analytics.db").is_file()):
                    return candidate
    raise FileNotFoundError(
        "Lab 1 ERP database not found. For standalone use, put "
        "financial_analytics.db next to financial_copilot.py."
    )


def _normalize_sales(source: Path | pd.DataFrame) -> pd.DataFrame:
    """Apply the dashboard's established validity rules to sales transactions."""
    data = pd.read_csv(source) if isinstance(source, Path) else source.copy()
    missing = set(REQUIRED_COLUMNS) - set(data.columns)
    if missing:
        raise ValueError(f"Missing required cleaned-sales columns: {sorted(missing)}")
    if "invoice_id" not in data.columns:
        data["invoice_id"] = pd.NA
    for col in ["sale_id", "invoice_id", "product_id", "customer_id", "product_name", "category"]:
        data[col] = data[col].astype("string").str.strip().replace("", pd.NA)
    data["sale_date"] = pd.to_datetime(data["sale_date"], errors="coerce", utc=True).dt.tz_convert(None).dt.normalize()
    data["revenue"] = pd.to_numeric(data["revenue"], errors="coerce")
    # Matches data_processing.load_data's treatment of known corrupted records.
    for flag in ("invalid_discount", "revenue_mismatch"):
        if flag in data.columns:
            allowed = data[flag].astype("string").str.lower().str.strip().isin(("false", "0", "0.0"))
            data = data[allowed]
    data = data.dropna(subset=["sale_id", "sale_date", "revenue"])
    data = data[np.isfinite(data["revenue"]) & (data["revenue"] >= 0)]
    data = data.drop_duplicates("sale_id").sort_values("sale_date")
    data["category"] = data["category"].fillna("Unknown category")
    if "discount_rate" not in data.columns:
        data["discount_rate"] = np.nan
    data["discount_rate"] = pd.to_numeric(data["discount_rate"], errors="coerce")
    return data[[
        "sale_id", "invoice_id", "sale_date", "customer_id", "product_id",
        "product_name", "category", "revenue", "discount_rate",
    ]].reset_index(drop=True)


def _date_value(value: str | date | pd.Timestamp | None) -> str | None:
    return pd.Timestamp(value).date().isoformat() if value is not None else None


def _date_where(column: str, start: str | None, end: str | None) -> str:
    where = " WHERE 1=1"
    if start:
        where += f" AND {column} >= :start_date"
    if end:
        where += f" AND {column} <= :end_date"
    return where


def _money(value: Any) -> str:
    return f"{float(value):,.2f}"


def _openai_completion(messages, api_key, model, *, json_mode=False):
    try:
        from openai import OpenAI, OpenAIError
    except ImportError as exc:
        raise RuntimeError("OpenAI mode requires the openai package.") from exc
    api_key = api_key or os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("Enter an OpenAI API key or set OPENAI_API_KEY.")
    options = {"response_format": {"type": "json_object"}} if json_mode else {}
    try:
        with OpenAI(api_key=api_key, timeout=20, max_retries=0) as client:
            response = client.chat.completions.create(
                model=model, temperature=0, messages=messages, **options,
            )
    except OpenAIError as exc:
        raise RuntimeError("OpenAI could not complete this request. Try again.") from exc
    text = response.choices[0].message.content if response.choices else None
    if not text or not text.strip():
        raise RuntimeError("OpenAI returned an empty response.")
    return text.strip()


def _financial_summary_messages(question, evidence):
    try:
        document = Path(__file__).with_name("prompts.md").read_text(encoding="utf-8")
    except OSError as exc:
        raise RuntimeError("The financial summary template in prompts.md is unavailable.") from exc
    prompts = []
    for heading in ("Shared system prompt", "1. Financial summary"):
        match = re.search(rf"^## {re.escape(heading)}\n\n```text\n(.*?)\n```", document, re.M | re.S)
        if not match:
            raise RuntimeError("The financial summary template in prompts.md is incomplete.")
        prompts.append(match[1])
    system, template = prompts
    content = template.replace("{{evidence_json}}", json.dumps(evidence, ensure_ascii=False))
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": content.replace("{{question}}", question)},
    ]


class FinancialCopilot:
    """Use SQLite for ERP retrieval, Pandas for forecasting and anomaly analysis."""

    def __init__(
        self,
        project_root: str | Path | None = None,
        sales_data: pd.DataFrame | None = None,
        *,
        use_demo_erp: bool = True,
    ) -> None:
        self.use_demo_erp = use_demo_erp
        self.root = None
        self.sales_source = None
        self.snapshot_date = None
        database = None
        if use_demo_erp:
            self.root = find_project_root(project_root)
            database = self.root / "financial_analytics.db"
            if not database.is_file():
                database = self.root / "week1-2" / "financial_analytics.db"
            self.sales_source = self.root / "clean_financial_data.csv"
            if not self.sales_source.is_file():
                self.sales_source = self.root / "week3-4" / "clean_financial_data.csv"
            if sales_data is None:
                if not self.sales_source.is_file():
                    raise FileNotFoundError(f"Cleaned sales file not found: {self.sales_source}")
                sales_data = self.sales_source
        if sales_data is None:
            raise ValueError("Upload a sales CSV to use the copilot in this mode.")

        self.sales = _normalize_sales(sales_data)
        self.connection = sqlite3.connect(":memory:", check_same_thread=False)
        try:
            if database is not None:
                source = sqlite3.connect(f"{database.as_uri()}?mode=ro", uri=True)
                try:
                    source.backup(self.connection)
                finally:
                    source.close()
                latest_payment = self.connection.execute("SELECT MAX(payment_date) FROM payments").fetchone()[0]
                self.snapshot_date = str(latest_payment) if latest_payment is not None else None
            staging = self.sales.copy()
            staging["sale_date"] = staging["sale_date"].dt.strftime("%Y-%m-%d")
            staging.to_sql("sales_clean", self.connection, index=False, if_exists="replace")
            self.connection.execute("CREATE INDEX IF NOT EXISTS idx_clean_date ON sales_clean(sale_date)")
            self.connection.execute("CREATE INDEX IF NOT EXISTS idx_clean_customer ON sales_clean(customer_id)")
        except Exception:
            self.connection.close()
            raise
        self.last_sales_date = self.sales["sale_date"].max().date() if not self.sales.empty else None

    def close(self) -> None:
        self.connection.close()

    def __enter__(self) -> "FinancialCopilot":
        return self

    def __exit__(self, *_: Any) -> None:
        self.close()

    def execute_readonly(self, sql: str, params: dict[str, Any] | None = None) -> pd.DataFrame:
        """Enforce a SELECT-only, allowlisted SQLite query, including LLM output."""
        statement = sql.strip()
        if statement.endswith(";"):
            statement = statement[:-1].strip()
        if (len(statement) > 6000 or ";" in statement or "--" in statement or "/*" in statement
                or not re.match(r"(?is)^\s*(SELECT|WITH)\b", statement)):
            raise ValueError("Only a single read-only SELECT query is allowed.")
        if re.search(r"(?i)\b(ATTACH|DETACH|PRAGMA|INSERT|UPDATE|DELETE|DROP|ALTER|CREATE|REPLACE|VACUUM|REINDEX|LOAD_EXTENSION)\b", statement):
            raise ValueError("Unsafe SQL keyword rejected.")
        allowed_tables = {"sales_clean"}
        if self.use_demo_erp:
            allowed_tables.update(("customers", "invoices", "payments", "products"))
        allowed_functions = {
            "sum", "count", "min", "max", "avg", "round", "coalesce", "nullif",
            "abs", "substr", "strftime", "date", "julianday", "trim", "length",
            "lower", "upper", "printf", "ifnull", "iif", "typeof",
        }
        allowed_actions = {sqlite3.SQLITE_SELECT, sqlite3.SQLITE_READ, sqlite3.SQLITE_FUNCTION}
        if hasattr(sqlite3, "SQLITE_RECURSIVE"):
            allowed_actions.add(sqlite3.SQLITE_RECURSIVE)

        def authorize(action: int, target: str | None, aux: str | None, *_: Any) -> int:
            if action not in allowed_actions:
                return sqlite3.SQLITE_DENY
            if action == sqlite3.SQLITE_READ:
                if target not in allowed_tables or (target == "customers" and aux == "email"):
                    return sqlite3.SQLITE_DENY
            if action == sqlite3.SQLITE_FUNCTION and (aux or "").lower() not in allowed_functions:
                return sqlite3.SQLITE_DENY
            return sqlite3.SQLITE_OK

        progress_calls = 0

        def progress() -> int:
            nonlocal progress_calls
            progress_calls += 1
            return int(progress_calls > 1000)  # query CPU budget

        self.connection.set_authorizer(authorize)
        self.connection.set_progress_handler(progress, 1000)
        try:
            cur = self.connection.execute(statement, params or {})
            rows = cur.fetchmany(MAX_ROWS + 1)
            columns = [c[0] for c in cur.description or []]
            result = pd.DataFrame(rows[:MAX_ROWS], columns=columns)
            result.attrs["truncated"] = len(rows) > MAX_ROWS
            return result
        except sqlite3.DatabaseError as exc:
            raise ValueError(f"SQL cannot be executed safely: {exc}") from exc
        finally:
            self.connection.set_progress_handler(None, 0)
            self.connection.set_authorizer(None)

    def plan_query(
        self, question: str, start_date: str | date | None = None,
        end_date: str | date | None = None,
    ) -> QueryPlan:
        """Explainable English/Russian NL-to-SQL for the standard ERP questions."""
        q = question.casefold().strip()
        start, end = _date_value(start_date), _date_value(end_date)
        if start and end and start > end:
            raise ValueError("The start date must be before the end date.")
        params = {"start_date": start, "end_date": end}
        sale_where = _date_where("s.sale_date", start, end)
        invoice_where = _date_where("i.invoice_date", start, end)
        nmatch = re.search(r"\b(?:top|first|первых|топ)\s*(\d{1,2})\b", q)
        limit = min(int(nmatch.group(1)), 50) if nmatch else 5

        if not self.use_demo_erp and re.search(r"\bpayments?\b|платеж|платёж|оплат", q):
            return QueryPlan("invoice_data_required", "")

        if re.search(r"\b(fraud|suspicious|fraudulent)\b|мошенн|подозр", q):
            if not self.use_demo_erp:
                return QueryPlan("invoice_data_required", "")
            return QueryPlan("fraud_review", """
                WITH payment_flags AS (
                  SELECT invoice_id,
                    SUM(CASE WHEN amount_paid < 0 THEN 1 ELSE 0 END) AS negative_payments,
                    SUM(amount_paid) AS total_paid
                  FROM payments GROUP BY invoice_id
                ), sales_flags AS (
                  SELECT invoice_id,
                    SUM(CASE WHEN discount_rate >= 0.40 THEN 1 ELSE 0 END) AS high_discount_lines
                  FROM sales_clean GROUP BY invoice_id
                )
                SELECT i.invoice_id, i.customer_id, i.invoice_value,
                  COALESCE(p.total_paid,0) AS total_paid,
                  COALESCE(p.negative_payments,0) AS negative_payments,
                  COALESCE(s.high_discount_lines,0) AS high_discount_lines,
                  CASE WHEN COALESCE(p.total_paid,0) > i.invoice_value+0.01 THEN 1 ELSE 0 END AS overpayment_flag
                FROM invoices i
                LEFT JOIN payment_flags p ON i.invoice_id = p.invoice_id
                LEFT JOIN sales_flags s ON i.invoice_id = s.invoice_id
                """ + invoice_where + """
                AND (COALESCE(p.negative_payments,0)>0 OR COALESCE(s.high_discount_lines,0)>0
                     OR COALESCE(p.total_paid,0) > i.invoice_value+0.01)
                ORDER BY overpayment_flag DESC, negative_payments DESC, high_discount_lines DESC
                LIMIT 50
                """, params)

        if re.search(r"anomal|outlier|аномал|выброс|необычн", q):
            return QueryPlan("anomalies", """
                SELECT s.sale_date,s.sale_id,s.invoice_id,s.customer_id,
                       s.product_name,s.category,s.revenue
                FROM sales_clean s
                """ + sale_where + " ORDER BY s.sale_date DESC, s.sale_id", params)

        if re.search(r"forecast|predict|projection|next month|прогноз|следующ.{0,8}месяц", q):
            return QueryPlan("forecast", """
                SELECT substr(s.sale_date,1,7) AS month,
                       ROUND(SUM(s.revenue),2) AS revenue
                FROM sales_clean s
                """ + sale_where + " GROUP BY substr(s.sale_date,1,7) ORDER BY month", params)

        if re.search(r"unpaid|outstanding|overdue|receivable|неоплач|просроч|задолж|долги|должник|invoice|счет[аов]?", q):
            if not self.use_demo_erp:
                return QueryPlan("invoice_data_required", "")
            invoice_id = re.search(r"\bINV\d{5}\b", question, re.I)
            customer_id = re.search(r"\bC\d{4}\b", question, re.I)
            if invoice_id:
                params["invoice_id"] = invoice_id.group(0).upper()
            if customer_id:
                params["customer_id"] = customer_id.group(0).upper()
            extra = ""
            if invoice_id:
                extra += " AND i.invoice_id = :invoice_id"
            if customer_id:
                extra += " AND i.customer_id = :customer_id"
            if not invoice_id and (not customer_id or re.search(r"unpaid|outstanding|overdue|receivable|неоплач|просроч|задолж", q)):
                extra += " AND ROUND(i.invoice_value - COALESCE(p.paid,0),2) > 0.01"
            if re.search(r"overdue|просроч", q):
                params["snapshot_date"] = self.snapshot_date
                extra += " AND i.due_date < :snapshot_date"
            return QueryPlan("unpaid_invoices", """
                WITH paid AS (
                  SELECT invoice_id, ROUND(SUM(amount_paid),2) AS paid
                  FROM payments GROUP BY invoice_id
                )
                SELECT i.invoice_id,i.customer_id,i.invoice_date,i.due_date,
                       ROUND(i.invoice_value,2) AS invoice_value,
                       ROUND(COALESCE(p.paid,0),2) AS amount_paid,
                       ROUND(i.invoice_value - COALESCE(p.paid,0),2) AS outstanding
                FROM invoices i LEFT JOIN paid p ON i.invoice_id=p.invoice_id
                """ + invoice_where + extra + " ORDER BY outstanding DESC, i.due_date ASC", params)

        if re.search(r"\b(country|countries|странам|страны|страна)\b", q) and re.search(r"revenue|sales|выручк|продаж|top|highest|больш", q):
            if not self.use_demo_erp:
                return QueryPlan("country_data_required", "")
            return QueryPlan("by_country", """
                SELECT COALESCE(c.country,'Unknown') AS country,
                       ROUND(SUM(s.revenue),2) AS revenue
                FROM sales_clean s LEFT JOIN customers c ON c.customer_id=s.customer_id
                """ + sale_where + " GROUP BY COALESCE(c.country,'Unknown') ORDER BY revenue DESC", params)

        if re.search(r"\b(products?|categories|category)\b|товар|продукт|категор", q) and re.search(r"revenue|sales|выручк|продаж|top|highest|больш", q):
            key = "category" if re.search(r"categor|категор", q) else "product_name"
            return QueryPlan("by_product", f"""
                SELECT s.{key}, ROUND(SUM(s.revenue),2) AS revenue
                FROM sales_clean s
                {sale_where} GROUP BY s.{key} ORDER BY revenue DESC LIMIT :result_limit
                """, {**params, "result_limit": limit})

        if re.search(r"customers?|client|клиент|покупател", q) and re.search(r"revenue|sales|выручк|продаж|top|highest|most|больш|лучш", q):
            customer_match = re.search(r"\bC\d{4}\b", question, re.I)
            extra_filter = ""
            if customer_match:
                params["customer_id"] = customer_match.group(0).upper()
                extra_filter = " AND s.customer_id = :customer_id "
                limit = 1
            if re.search(r"highest|most|maximum|biggest|largest|больш|максим|перв", q):
                limit = 1 if not nmatch else limit
            ranking = """
                SELECT s.customer_id,
                       ROUND(SUM(s.revenue),2) AS revenue,
                       COUNT(DISTINCT s.sale_id) AS transactions
                FROM sales_clean s
                """ + sale_where + " AND s.customer_id IS NOT NULL " + extra_filter + """
                GROUP BY s.customer_id
                ORDER BY revenue DESC, s.customer_id LIMIT :result_limit
                """
            if self.use_demo_erp:
                ranking = "WITH ranked AS (" + ranking + """
                ) SELECT r.customer_id,
                         TRIM(COALESCE(c.first_name,'') || ' ' || COALESCE(c.last_name,'')) AS customer_name,
                         r.revenue, r.transactions
                  FROM ranked r LEFT JOIN customers c ON c.customer_id = r.customer_id
                  ORDER BY r.revenue DESC, r.customer_id
                """
            return QueryPlan("top_customers", ranking, {**params, "result_limit": limit})

        if re.search(r"monthly|by month|month-by-month|по месяц|помесячн|динамик", q) and re.search(r"revenue|sales|выручк|продаж", q):
            return QueryPlan("monthly_revenue", """
                SELECT substr(s.sale_date,1,7) AS month,
                       ROUND(SUM(s.revenue),2) AS revenue
                FROM sales_clean s
                """ + sale_where + " GROUP BY substr(s.sale_date,1,7) ORDER BY month", params)

        if re.search(r"summary|summari|kpi|overview|financial|сводк|итог|показател|финансов", q):
            return QueryPlan("summary", """
                SELECT ROUND(SUM(s.revenue),2) AS total_revenue,
                       COUNT(DISTINCT s.sale_id) AS transactions,
                       COUNT(DISTINCT s.customer_id) AS customers,
                       ROUND(AVG(s.revenue),2) AS average_transaction
                FROM sales_clean s
                """ + sale_where, params)

        return QueryPlan("unknown", "")

    def _llm_sql(self, question: str, key: str | None, model: str) -> str:
        prompt = SQL_GENERATION_PROMPT if self.use_demo_erp else SALES_SQL_GENERATION_PROMPT
        raw = _openai_completion([
            {"role": "system", "content": prompt},
            {"role": "user", "content": question},
        ], key, model, json_mode=True)
        parsed = json.loads(raw)
        sql = parsed.get("sql") if isinstance(parsed, dict) else None
        if not isinstance(sql, str):
            raise ValueError("Model did not return a SQL string.")
        return sql

    def ask(
        self, question: str, *, start_date: str | date | None = None,
        end_date: str | date | None = None, use_llm: bool = False,
        api_key: str | None = None, model: str = "gpt-4.1-mini",
    ) -> CopilotAnswer:
        """Answer ERP questions with traceable SQL and measured calculations."""
        if not question.strip():
            return CopilotAnswer("help", "Please enter a financial question.")
        plan = self.plan_query(question, start_date, end_date)
        if plan.intent == "invoice_data_required":
            return CopilotAnswer(plan.intent, "This question requires invoice and payment records. "
                "Use Explore demo to try it with the sample ERP records.")
        if plan.intent == "country_data_required":
            return CopilotAnswer(plan.intent, "A country breakdown requires customer country records. "
                "You can rank customers from this CSV by customer ID.")
        if plan.intent == "unknown":
            if not use_llm:
                topics = "top customers, monthly revenue, forecasts, financial KPIs and anomalies"
                if self.use_demo_erp:
                    topics += ", plus unpaid invoices and fraud-risk flags"
                return CopilotAnswer("help", f"I can answer questions about {topics}. "
                    "Enable optional OpenAI mode for additional SQL questions.")
            sql = self._llm_sql(question, api_key, model)
            table = self.execute_readonly(sql)
            if table.empty:
                answer = "The read-only SQL query returned no matching records."
            elif table.shape == (1, 1):
                answer = f"SQL result — {table.columns[0]}: {table.iloc[0,0]:,}" if isinstance(table.iloc[0,0], (int,float)) else f"SQL result: {table.iloc[0,0]}"
            else:
                answer = f"The read-only query returned {len(table)} displayed rows. See the result table and SQL below."
            if table.attrs.get("truncated"):
                answer += f" Results are capped at {MAX_ROWS} rows."
            return CopilotAnswer("custom_sql", answer, sql, table)

        if plan.intent == "anomalies":
            # IQR methodology is imported from the existing Week 6 dashboard.
            from data_processing import detect_anomalies
            subset = self.sales.copy()
            if plan.params["start_date"]:
                subset = subset[subset.sale_date >= pd.Timestamp(plan.params["start_date"])]
            if plan.params["end_date"]:
                subset = subset[subset.sale_date <= pd.Timestamp(plan.params["end_date"])]
            if len(subset) < 20:
                return CopilotAnswer("anomalies", "At least 20 transactions are needed for IQR anomaly detection.", plan.sql)
            flagged = detect_anomalies(subset)
            if flagged is None:
                return CopilotAnswer("anomalies", "Insufficient transactions for anomaly detection.", plan.sql)
            display = flagged[["sale_date", "sale_id", "invoice_id", "customer_id", "product_name",
                "category", "revenue", "threshold", "reason", "comparison"]].head(30).copy()
            if not display.empty:
                display["sale_date"] = display["sale_date"].dt.strftime("%Y-%m-%d")
                example = flagged.sort_values("revenue", ascending=False).iloc[0]
                example_text = (f" Example: sale {example['sale_id']} ({example['category']}) "
                    f"has revenue {_money(example['revenue'])} against an IQR threshold "
                    f"of {_money(example['threshold'])}.")
            else:
                example_text = ""
            answer = (f"Found {len(flagged)} IQR outliers among {len(subset)} valid sales. "
                "Following the Week 6 dashboard, values outside 1.5 × IQR of their product "
                "category are flagged; small categories use all selected sales."
                + example_text + " Potential explanations include order size, discounts, or data errors; "
                "the data alone cannot establish the cause or fraud.")
            return CopilotAnswer("anomalies", answer, plan.sql,
                display, {"anomaly_count": len(flagged), "transaction_count": len(subset)})

        if plan.intent == "forecast":
            from data_processing import revenue_forecast
            monthly = self.execute_readonly(plan.sql, plan.params)
            if monthly.empty:
                return CopilotAnswer("forecast", "No sales data available for the chosen period.", plan.sql)
            requested_start = pd.Timestamp(plan.params["start_date"]) if plan.params["start_date"] else self.sales.sale_date.min()
            requested_end = pd.Timestamp(plan.params["end_date"]) if plan.params["end_date"] else self.sales.sale_date.max()
            first = requested_start.replace(day=1)
            last = requested_end.replace(day=1)
            if requested_start.day != 1:
                first += pd.offsets.MonthBegin(1)
            if not requested_end.is_month_end:
                last -= pd.offsets.MonthBegin(1)
            if first > last:
                return CopilotAnswer("forecast", "No complete calendar months in the selected period.", plan.sql)
            month_series = pd.to_datetime(monthly["month"] + "-01")
            monthly_series = pd.Series(monthly["revenue"].values, index=month_series)
            months = pd.date_range(first, last, freq="MS")
            series = pd.DataFrame({"sale_date": months,
                "revenue": monthly_series.reindex(months, fill_value=0).astype(float).values})
            if len(series) < 12:
                return CopilotAnswer("forecast", "A forecast needs at least 12 complete months of sales history. "
                    f"The selected period contains {len(series)}.", plan.sql, series)
            results = revenue_forecast(series, horizon=1)
            future = results["future"].copy()
            prediction = float(future["revenue"].iloc[0])
            target_month = future["sale_date"].iloc[0].strftime("%B %Y")
            quality = f"Backtest MAE: {_money(results['mae'])}; previous-month baseline MAE: {_money(results['baseline_mae'])}."
            answer = (f"Forecast for {target_month}: {_money(prediction)} (currency not specified). "
                f"Based on {len(series)} complete months through {last:%B %Y}, using the "
                "dashboard's lagged-revenue ridge regression. " + quality
                + " This is a statistical estimate, not a guarantee or a live current-month forecast.")
            future["sale_date"] = future["sale_date"].dt.strftime("%Y-%m-%d")
            return CopilotAnswer("forecast", answer, plan.sql, future,
                {"forecast": prediction, "forecast_month": target_month,
                 "mae": float(results["mae"]), "baseline_mae": float(results["baseline_mae"]),
                 "historical_months": len(series)})

        table = self.execute_readonly(plan.sql, plan.params)
        if plan.intent == "top_customers":
            if table.empty:
                msg = "No customers with valid sales match the selected period."
            elif len(table) == 1:
                top = table.iloc[0]
                name = top.get("customer_name")
                customer = f"{name} ({top['customer_id']})" if name else top["customer_id"]
                descriptor = ("Revenue for customer" if "customer_id" in plan.params
                              else "Highest-revenue customer")
                msg = (f"{descriptor}: {customer}, "
                    f"with {_money(top['revenue'])} in valid sales revenue across "
                    f"{int(top['transactions'])} transactions.")
            else:
                msg = f"Top {len(table)} customers ranked by valid sales revenue."
            return CopilotAnswer(plan.intent, msg, plan.sql, table)
        if plan.intent == "unpaid_invoices":
            if table.empty:
                msg = "No matching outstanding invoices were found."
            else:
                outstanding = table["outstanding"].clip(lower=0)
                paid0 = int((table["amount_paid"].abs() < 0.01).sum())
                msg = (f"{len(table)} matching invoices; total positive outstanding balance "
                    f"{_money(outstanding.sum())}. {paid0} have no recorded payment. "
                    f"Payments are available through {self.snapshot_date}; this is a historical "
                    "ERP snapshot, not a live banking balance.")
                if table.attrs.get("truncated"):
                    msg = (f"Showing only the first {MAX_ROWS} matches; displayed outstanding subtotal "
                        f"is {_money(outstanding.sum())}. Data snapshot through {self.snapshot_date}.")
            return CopilotAnswer(plan.intent, msg, plan.sql, table)
        if plan.intent == "summary":
            row = table.iloc[0]
            if not row["transactions"]:
                return CopilotAnswer(plan.intent, "No valid sales match the selected period.", plan.sql, table)
            first = plan.params["start_date"] or _date_value(self.sales.sale_date.min())
            last = plan.params["end_date"] or _date_value(self.last_sales_date)
            source = "demo ERP" if self.use_demo_erp else "uploaded sales CSV"
            evidence = {
                "source": f"sales_clean ({source})",
                "period": f"{first} to {last}",
                "total_revenue": float(row["total_revenue"]),
                "transactions": int(row["transactions"]),
                "customers": int(row["customers"]),
                "average_transaction_value": float(row["average_transaction"]),
                "currency": None,
            }
            summary = CopilotAnswer(plan.intent,
                f"Total valid sales revenue: {_money(evidence['total_revenue'])}; "
                f"transactions: {evidence['transactions']}; customers: {evidence['customers']}; "
                f"average sales transaction: {_money(evidence['average_transaction_value'])}. "
                "Amounts are in unspecified source currency and exclude invalid sales.", plan.sql, table)
            if use_llm:
                try:
                    messages = _financial_summary_messages(question, evidence)
                    summary.answer = _openai_completion(messages, api_key, model)
                    summary.details["note"] = "OpenAI summary based on the displayed KPI totals."
                except RuntimeError as exc:
                    summary.details["note"] = f"{exc} Showing the local KPI summary."
            return summary
        if plan.intent == "monthly_revenue":
            msg = f"Monthly valid-sales revenue across {len(table)} months."
            return CopilotAnswer(plan.intent, msg, plan.sql, table)
        if plan.intent in ("by_product", "by_country"):
            if plan.intent == "by_country" and not table.empty:
                msg = f"Highest-revenue country: {table.iloc[0]['country']} ({_money(table.iloc[0]['revenue'])}). "
            else:
                msg = ""
            msg += f"Revenue breakdown across {len(table)} displayed groups, using cleaned sales."
            return CopilotAnswer(plan.intent, msg, plan.sql, table)
        if plan.intent == "fraud_review":
            if not table.empty:
                msg = (f"{len(table)} invoice(s) flagged by payment/discount review rules "
                    "(negative payments, discounts ≥40%, or overpayments). "
                    "These are leads for checking source documents, not proof of fraud.")
                return CopilotAnswer(plan.intent, msg, plan.sql, table)
            # Supplement payment rules with the existing dashboard's independent
            # statistical outlier evidence, without asserting fraud.
            anomaly = self.ask("Explain anomalies", start_date=start_date, end_date=end_date)
            msg = ("No invoices matched the negative-payment, high-discount or overpayment rules. "
                + f"Separately, statistical sales screening found {anomaly.details.get('anomaly_count', 0)} "
                "IQR outliers; a sample is shown for manual review. "
                "Statistical irregularity does not establish intent or fraud.")
            return CopilotAnswer(plan.intent, msg,
                plan.sql + "\n\n-- IQR sales screening performed in Pandas (see Week 6 dashboard)",
                anomaly.data, {"anomaly_count": anomaly.details.get("anomaly_count", 0)})
        return CopilotAnswer("help", "No handler for this question.")


def render_chat_panel(
    copilot: FinancialCopilot | None = None,
    *, start_date: date | None = None, end_date: date | None = None,
) -> None:
    """Reusable Streamlit chat UI, suitable as a fifth dashboard tab."""
    import streamlit as st

    st.header("AI Financial Copilot")
    if copilot is None:
        try:
            copilot = FinancialCopilot()
        except (FileNotFoundError, ValueError) as exc:
            st.error(str(exc))
            return
    if copilot.use_demo_erp:
        st.caption("Demo ERP data · Ask about customers, unpaid invoices, KPIs, forecasts and anomalies.")
    else:
        st.caption("Uploaded sales CSV · Ask about customer IDs, KPIs, forecasts and anomalies. "
            "Invoice questions require invoice and payment records.")
    if start_date is not None and end_date is not None:
        st.caption(f"Report period: {start_date} to {end_date}.")

    with st.expander("Optional AI mode & data provenance"):
        use_llm = st.checkbox("Use OpenAI for financial summaries and extra SQL questions", key="copilot_use_llm")
        key = st.text_input("OpenAI API key (optional; also accepts OPENAI_API_KEY)",
            type="password", key="copilot_api_key") if use_llm else None
        st.caption("OpenAI receives your question and the aggregate KPIs for a financial summary. "
            "SQL planning uses your question and the table schema.")
        if copilot.use_demo_erp:
            st.caption("Sales: bundled CSV. Invoices, customers and payments: bundled SQLite database. "
                f"Latest recorded payment: {copilot.snapshot_date}.")

    examples = [
        "Which customer generated the highest revenue?",
        "Forecast next month's revenue.",
        "Explain anomalies.",
        "Give me a financial KPI summary",
    ]
    if copilot.use_demo_erp:
        examples.insert(1, "Show unpaid invoices.")
    st.markdown("**Try a question**")
    columns = st.columns(2)
    clicked = None
    for i, example in enumerate(examples):
        if columns[i % 2].button(example, key=f"copilot_example_{i}", width="stretch"):
            clicked = example

    if "copilot_messages" not in st.session_state:
        st.session_state["copilot_messages"] = []
    for item in st.session_state["copilot_messages"]:
        with st.chat_message(item["role"]):
            st.markdown(item["text"])
            if item["role"] == "assistant":
                if item.get("note"):
                    st.caption(item["note"])
                if item.get("table") is not None and not item["table"].empty:
                    st.dataframe(item["table"].head(50), width="stretch", hide_index=True)
                if item.get("sql"):
                    with st.expander("View generated SQL"):
                        st.code(item["sql"], language="sql")

    question = st.chat_input("Ask about your ERP finances...", key="copilot_chat_input")
    request = clicked or question
    if request:
        st.session_state["copilot_messages"].append({"role": "user", "text": request})
        try:
            response = copilot.ask(request, start_date=start_date, end_date=end_date,
                use_llm=use_llm, api_key=key)
            st.session_state["copilot_messages"].append({
                "role": "assistant", "text": response.answer, "sql": response.sql,
                "table": response.data, "note": response.details.get("note"),
            })
        except (ValueError, RuntimeError, sqlite3.Error) as exc:
            st.session_state["copilot_messages"].append({
                "role": "assistant", "text": f"Could not answer safely: {exc}",
            })
        st.rerun()


def main() -> None:
    import streamlit as st
    st.set_page_config(page_title="Lab 1 | AI Financial Copilot", layout="wide")
    st.title("Lab 1 — AI Financial Copilot")
    try:
        copilot = FinancialCopilot()
    except (FileNotFoundError, ValueError) as exc:
        st.error(str(exc))
        st.stop()
    render_chat_panel(copilot)


if __name__ == "__main__":
    main()
