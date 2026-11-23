"""Financial Copilot for the Lab 1 ERP dashboard (Week 10).

Use the bundled analytics locally, or enable model-based planning and
explanations with a server-side OpenAI credential.
"""

from __future__ import annotations

from contextlib import closing
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

from copilot_questions import (
    COUNT, CUSTOMERS, FORECAST, GREETINGS, INVOICES, MONTHLY, MOST, ORDERS, RECENT,
    REVENUE, SALES, THANKS, UNPAID, YEAR_OR_DATE, follow_up, is_russian, normalize_question,
)
from data_processing import REQUIRED_COLUMNS, calculate_var, monthly_growth


MAX_ROWS = 500
SALES_SCHEMA = """  sales_clean(sale_id, invoice_id, sale_date, customer_id, product_id,
              product_name, category, revenue, discount_rate)
"""
ERP_SCHEMA = SALES_SCHEMA + """\
  customers(customer_id, first_name, last_name, city, country, signup_date)
  invoices(invoice_id, customer_id, invoice_date, due_date, subtotal,
           tax_amount, invoice_value)
  payments(payment_id, invoice_id, payment_date, payment_method, amount_paid)
  products(product_id, product_name, category, unit_cost, unit_price, stock_quantity)
"""
ANALYSIS_TOOLS = {
    "summary": ("Give me a financial KPI summary", "1. Financial summary"),
    "kpi": ("Explain average transaction value", "2. KPI explanation"),
    "risk": ("Explain revenue risk", "3. Risk explanation"),
    "fraud": ("Investigate fraud risk", "4. Fraud investigation"),
    "forecast": ("Forecast next month's revenue", None),
    "anomalies": ("Explain anomalies", None),
}


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
        raise RuntimeError("OpenAI is unavailable on the server.")
    options = {"response_format": {"type": "json_object"}} if json_mode else {}
    try:
        with OpenAI(api_key=api_key, timeout=20, max_retries=0) as client:
            response = client.chat.completions.create(
                model=model, temperature=0, max_completion_tokens=700, messages=messages, **options,
            )
    except OpenAIError as exc:
        raise RuntimeError("OpenAI could not complete this request. Try again.") from exc
    text = response.choices[0].message.content if response.choices else None
    if not text or not text.strip():
        raise RuntimeError("OpenAI returned an empty response.")
    return text.strip()


def _prompt(heading):
    try:
        document = Path(__file__).with_name("prompts.md").read_text(encoding="utf-8")
    except OSError as exc:
        raise RuntimeError("The Copilot prompts are unavailable.") from exc
    match = re.search(rf"^## {re.escape(heading)}\n\n```text\n(.*?)\n```", document, re.M | re.S)
    if not match:
        raise RuntimeError("The Copilot prompt is missing.")
    return match[1]


def _explanation_messages(heading, question, evidence):
    content = _prompt(heading).replace("{{evidence_json}}", json.dumps(evidence, ensure_ascii=False))
    return [
        {"role": "system", "content": _prompt("Shared system prompt")},
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

    def execute_readonly(self, sql: str, params: dict[str, Any] | None = None,
                         *, connection: sqlite3.Connection | None = None) -> pd.DataFrame:
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

        db = connection if connection is not None else self.connection
        db.set_authorizer(authorize)
        db.set_progress_handler(progress, 1000)
        try:
            cur = db.execute(statement, params or {})
            rows = cur.fetchmany(MAX_ROWS + 1)
            columns = [c[0] for c in cur.description or []]
            if len(columns) != len(set(columns)):
                raise ValueError("Query result columns need distinct names.")
            result = pd.DataFrame(rows[:MAX_ROWS], columns=columns)
            result.attrs["truncated"] = len(rows) > MAX_ROWS
            return result
        except sqlite3.DatabaseError as exc:
            raise ValueError(f"SQL cannot be executed safely: {exc}") from exc
        finally:
            db.set_progress_handler(None, 0)
            db.set_authorizer(None)

    def plan_query(
        self, question: str, start_date: str | date | None = None,
        end_date: str | date | None = None,
    ) -> QueryPlan:
        """Explainable English/Russian NL-to-SQL for the standard ERP questions."""
        q = normalize_question(question)
        start, end = _date_value(start_date), _date_value(end_date)
        if start and end and start > end:
            raise ValueError("The start date must be before the end date.")
        params = {"start_date": start, "end_date": end}
        sale_where = _date_where("s.sale_date", start, end)
        invoice_where = _date_where("i.invoice_date", start, end)
        nmatch = re.search(r"\b(?:top|first|первых|топ)\s*(\d{1,2})\b", q)
        limit = max(1, min(int(nmatch.group(1)), 50)) if nmatch else 5

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

        if re.search(r"\brisk\b|\bvar\b|риск", q):
            return QueryPlan("risk", "", params)

        if FORECAST.search(q):
            return QueryPlan("forecast", """
                SELECT substr(s.sale_date,1,7) AS month,
                       ROUND(SUM(s.revenue),2) AS revenue
                FROM sales_clean s
                """ + sale_where + " GROUP BY substr(s.sale_date,1,7) ORDER BY month", params)

        unpaid = bool(UNPAID.search(q))
        if unpaid or INVOICES.search(q):
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
            if unpaid and not invoice_id:
                extra += " AND ROUND(i.invoice_value - COALESCE(p.paid,0),2) > 0.01"
            paid = bool(re.search(r"\bpaid\b|\bоплачен\w*\b", q))
            if paid and not unpaid:
                extra += " AND ROUND(i.invoice_value - COALESCE(p.paid,0),2) <= 0.01"
            overdue = bool(re.search(r"overdue|просроч", q))
            if overdue:
                params["snapshot_date"] = self.snapshot_date
                extra += " AND i.due_date < :snapshot_date"
            invoices = """
                WITH paid AS (
                  SELECT invoice_id, ROUND(SUM(amount_paid),2) AS paid
                  FROM payments GROUP BY invoice_id
                )
            """
            status = "overdue" if overdue else "outstanding" if unpaid else "paid" if paid else "any"
            if MONTHLY.search(q):
                return QueryPlan("monthly_invoices", invoices + """
                    SELECT substr(i.invoice_date,1,7) AS month, COUNT(*) AS invoices
                    FROM invoices i LEFT JOIN paid p ON i.invoice_id=p.invoice_id
                    """ + invoice_where + extra + """
                    GROUP BY substr(i.invoice_date,1,7) ORDER BY month
                    """, {**params, "invoice_status": status})
            if COUNT.search(q):
                return QueryPlan("invoice_count", invoices + """
                    SELECT COUNT(*) AS invoices
                    FROM invoices i LEFT JOIN paid p ON i.invoice_id=p.invoice_id
                    """ + invoice_where + extra, {**params, "invoice_status": status})
            invoices += """
                SELECT i.invoice_id,i.customer_id,i.invoice_date,i.due_date,
                       ROUND(i.invoice_value,2) AS invoice_value,
                       ROUND(COALESCE(p.paid,0),2) AS amount_paid,
                       ROUND(i.invoice_value - COALESCE(p.paid,0),2) AS outstanding
                FROM invoices i LEFT JOIN paid p ON i.invoice_id=p.invoice_id
                """ + invoice_where + extra
            if unpaid and not RECENT.search(q):
                return QueryPlan("unpaid_invoices", invoices + " ORDER BY outstanding DESC, i.due_date ASC", params)
            requested_limit = re.search(r"\b(\d{1,2})\s+(?:invoices?|bills?|счет\w*)\b", q)
            params["result_limit"] = max(1, min(int(requested_limit[1]), 50)) if requested_limit else 10
            return QueryPlan("recent_invoices", invoices + """
                ORDER BY i.invoice_date DESC, i.invoice_id DESC LIMIT :result_limit
                """, params)

        has_sales_metric = REVENUE.search(q) or SALES.search(q)
        rank_words = MOST.search(q)
        if re.search(r"\b(country|countries|странам|страны|страна)\b", q) and (has_sales_metric or rank_words):
            if not self.use_demo_erp:
                return QueryPlan("country_data_required", "")
            return QueryPlan("by_country", """
                SELECT COALESCE(c.country,'Unknown') AS country,
                       ROUND(SUM(s.revenue),2) AS revenue
                FROM sales_clean s LEFT JOIN customers c ON c.customer_id=s.customer_id
                """ + sale_where + " GROUP BY COALESCE(c.country,'Unknown') ORDER BY revenue DESC", params)

        if re.search(r"\b(products?|categories|category)\b|товар|продукт|категор", q) and (has_sales_metric or rank_words):
            key = "category" if re.search(r"categor|категор", q) else "product_name"
            return QueryPlan("by_product", f"""
                SELECT s.{key}, ROUND(SUM(s.revenue),2) AS revenue
                FROM sales_clean s
                {sale_where} GROUP BY s.{key} ORDER BY revenue DESC LIMIT :result_limit
                """, {**params, "result_limit": limit})

        spent_most = re.search(r"\b(?:spent|paid)\s+(?:the\s+)?most\b|кто\s+(?:потратил|купил)\s+больше", q)
        if spent_most or (CUSTOMERS.search(q) and not COUNT.search(q) and (has_sales_metric or rank_words)):
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

        if MONTHLY.search(q) and has_sales_metric:
            if COUNT.search(q) and SALES.search(q) and not REVENUE.search(q):
                intent, metric = "monthly_sales_count", "COUNT(*) AS transactions"
            else:
                intent, metric = "monthly_revenue", "ROUND(SUM(s.revenue),2) AS revenue"
            return QueryPlan(intent, f"""
                SELECT substr(s.sale_date,1,7) AS month, {metric}
                FROM sales_clean s {sale_where}
                GROUP BY substr(s.sale_date,1,7) ORDER BY month
                """, params)

        if ORDERS.search(q):
            return QueryPlan("clarify_orders", "")

        if COUNT.search(q) and CUSTOMERS.search(q):
            return QueryPlan("customer_count", "SELECT COUNT(DISTINCT s.customer_id) AS customers FROM sales_clean s" + sale_where, params)
        if COUNT.search(q) and SALES.search(q) and not REVENUE.search(q):
            return QueryPlan("sales_count", "SELECT COUNT(*) AS transactions FROM sales_clean s" + sale_where, params)
        explain_kpi = re.search(r"average transaction|average sale|средн.{0,15}(чек|транзакц|продаж)", q)
        if not explain_kpi and SALES.search(q) and (RECENT.search(q) or re.search(r"\b(?:transactions?|транзакци\w*)\b", q)):
            size = re.search(r"\b(\d{1,2})\s+(?:sales?|transactions?|продаж\w*|транзакци\w*)\b", q)
            params["result_limit"] = max(1, min(int(size[1]), 50)) if size else 10
            return QueryPlan("recent_sales", """
                SELECT s.sale_date, s.sale_id, s.customer_id, s.product_name, ROUND(s.revenue, 2) AS revenue
                FROM sales_clean s
                """ + sale_where + " ORDER BY s.sale_date DESC, s.sale_id DESC LIMIT :result_limit", params)

        if explain_kpi or re.search(r"summary|summari|kpi|overview|financial|сводк|итог|показател|финансов", q):
            return QueryPlan("kpi" if explain_kpi else "summary", """
                SELECT ROUND(SUM(s.revenue),2) AS total_revenue,
                       COUNT(DISTINCT s.sale_id) AS transactions,
                       COUNT(DISTINCT s.customer_id) AS customers,
                       ROUND(AVG(s.revenue),2) AS average_transaction
                FROM sales_clean s
                """ + sale_where, params)

        if has_sales_metric:
            return QueryPlan("total_revenue", "SELECT ROUND(SUM(s.revenue), 2) AS total_revenue FROM sales_clean s" + sale_where, params)

        return QueryPlan("unknown", "")

    def _model_sql(self, sql, start, end):
        # Give the model only the selected report, regardless of filters in its SQL.
        with closing(sqlite3.connect(":memory:")) as scoped:
            self.connection.backup(scoped)
            params = {"start_date": start, "end_date": end}
            sales_filter = _date_where("sale_date", start, end).removeprefix(" WHERE ")
            scoped.execute(f"DELETE FROM sales_clean WHERE NOT COALESCE(({sales_filter}), 0)", params)
            if self.use_demo_erp:
                invoice_filter = _date_where("invoice_date", start, end).removeprefix(" WHERE ")
                scoped.execute(f"DELETE FROM invoices WHERE NOT COALESCE(({invoice_filter}), 0)", params)
                scoped.execute("DELETE FROM payments WHERE invoice_id NOT IN (SELECT invoice_id FROM invoices)")
            data = self.execute_readonly(sql, connection=scoped)
        if data.empty:
            answer = "No records matched the query in the selected report."
        elif data.shape == (1, 1):
            answer = f"**{data.columns[0].replace('_', ' ')}** — {data.iloc[0, 0]}"
        else:
            answer = f"The query returned {len(data):,} rows. The table shows up to 50 rows."
        if data.attrs.get("truncated"):
            answer += f" Results reached the {MAX_ROWS}-row query limit."
        return CopilotAnswer("custom_sql", answer, sql, data, {"note": "SQL planned by OpenAI."})

    def _ask_model(self, question, start, end, history, api_key, model):
        schema = ERP_SCHEMA if self.use_demo_erp else SALES_SCHEMA
        prompt = _prompt("5. Question routing and SQL").replace("{{schema}}", schema)
        context = {
            "report_start": start, "report_end": end, "latest_payment": self.snapshot_date,
            "recent_questions": list(history)[-4:], "question": question,
        }
        raw = _openai_completion([
            {"role": "system", "content": prompt},
            {"role": "user", "content": json.dumps(context, ensure_ascii=False)},
        ], api_key, model, json_mode=True)
        decision = json.loads(raw)
        if not isinstance(decision, dict) or not isinstance(action := decision.get("action"), str):
            raise ValueError("The model did not choose a data tool.")
        if action == "sql":
            sql = decision.get("sql")
            if not isinstance(sql, str):
                raise ValueError("The model did not return SQL.")
            return self._model_sql(sql, start, end)
        if action == "clarify":
            message = decision.get("message")
            if not isinstance(message, str) or not message.strip() or len(message) > 400:
                raise ValueError("The model did not return a short clarification.")
            return CopilotAnswer("clarify", message.strip())
        if action == "unsupported":
            return None
        if action not in ANALYSIS_TOOLS:
            raise ValueError("The model chose an unavailable data tool.")
        tool_question, heading = ANALYSIS_TOOLS[action]
        result = self._ask_local(tool_question, start, end)
        evidence = result.details.get("evidence")
        if heading and evidence:
            try:
                result.answer = _openai_completion(
                    _explanation_messages(heading, question, evidence), api_key, model,
                )
            except RuntimeError:
                result.details["note"] = "Showing the calculated result. The model explanation was unavailable."
                return result
            result.details["note"] = "OpenAI explained the calculated evidence."
        else:
            result.details["note"] = "OpenAI selected the analysis. Calculations ran locally."
        return result

    def ask(
        self, question: str, *, start_date: str | date | None = None,
        end_date: str | date | None = None, use_llm: bool = False,
        api_key: str | None = None, model: str = "gpt-4.1-mini", history=(),
    ) -> CopilotAnswer:
        """Answer ERP questions with traceable SQL and measured calculations."""
        if not question.strip():
            return CopilotAnswer("help", "Please enter a financial question.")
        start, end = _date_value(start_date), _date_value(end_date)
        if start and end and start > end:
            raise ValueError("The start date must be before the end date.")
        history = list(history)
        interpreted, requested = follow_up(question, history)
        russian = is_russian(question) or (not re.search(r"[a-zа-я]", question, re.I) and is_russian(interpreted))
        if requested and not FORECAST.search(interpreted):
            first, last = requested
            if (end and first > end) or (start and last < start):
                message = (
                    "Эти даты выходят за пределы выбранного отчёта. Выбери другие даты в Reporting period."
                    if russian else "Those dates are outside the selected report. Choose different dates in Reporting period."
                )
                return CopilotAnswer("outside_report", message)
            start = max(start, first) if start else first
            end = min(end, last) if end else last
        if use_llm:
            try:
                answer = self._ask_model(question, start, end, history, api_key, model)
                if answer is not None:
                    return answer
            except (ValueError, RuntimeError, sqlite3.Error):
                result = self._ask_local(interpreted, start, end, russian=russian)
                result.details["note"] = (
                    "Модель недоступна для этого вопроса. Ответил встроенный режим."
                    if russian else "The model could not complete this request. Showing the local result."
                )
                return result
        return self._ask_local(interpreted, start, end, russian=russian)

    def _explain_risk(self, start_date, end_date):
        if self.sales.empty:
            return CopilotAnswer("risk", "No valid sales are available for revenue risk analysis.")
        start = pd.Timestamp(start_date) if start_date else self.sales.sale_date.min()
        end = pd.Timestamp(end_date) if end_date else self.sales.sale_date.max()
        sales = self.sales[self.sales.sale_date.between(start, end)]
        growth = monthly_growth(sales, start, end)
        risk = calculate_var(growth)
        evidence = {
            "source": "Monthly growth of valid sales revenue",
            "period": f"{start:%Y-%m-%d} to {end:%Y-%m-%d}",
            "growth_observations": int(growth["growth_pct"].count()),
            "minimum_observations": 12,
            "currency": None,
        }
        if risk is None:
            return CopilotAnswer("risk", "Revenue VaR needs at least 12 monthly growth observations. "
                f"This period has {evidence['growth_observations']}.", details={"evidence": evidence})
        evidence.update({
            "fifth_percentile_growth_pct": float(growth["growth_pct"].quantile(0.05)),
            "var_pct": risk["var_pct"],
            "baseline_month": risk["baseline_month"].strftime("%Y-%m"),
            "baseline_revenue": risk["baseline_revenue"],
            "estimated_shortfall": risk["shortfall"],
        })
        return CopilotAnswer("risk",
            f"The historical 95% revenue VaR is {risk['var_pct']:.1f}%, or "
            f"{_money(risk['shortfall'])} relative to {evidence['baseline_month']} revenue. "
            f"It uses {risk['observations']} monthly growth observations. Future declines can exceed it.",
            details={"evidence": evidence})

    def _ask_local(self, question, start_date, end_date, *, russian=False):
        q = normalize_question(question).strip(" ?!.")
        if q in GREETINGS:
            return CopilotAnswer("greeting", "Привет! Что посмотрим? Например, «выручка по месяцам»."
                                if russian else "Hi! What would you like to see? Try “revenue by month”.")
        if q in THANKS:
            return CopilotAnswer("acknowledgement", "Пожалуйста." if russian else "You're welcome.")
        if len(YEAR_OR_DATE.findall(q)) > 1:
            return CopilotAnswer("clarify_period",
                "Какой период показать первым? Например, «выручка за 2025»." if russian
                else "Which period should I show first? For example, “revenue in 2025”.")
        if re.search(r"\b(?:last|previous) (?:month|quarter|year)\b|прошл\w* (?:месяц\w*|квартал\w*|год\w*)", q):
            return CopilotAnswer("clarify_period",
                "Укажи месяц и год или только год. Например, «выручка за март 2025»." if russian
                else "Which month and year should I use? For example, “revenue in March 2025”.")
        if re.search(r"\b(?:profits?|margins?|прибыл\w*|марж\w*)\b", q):
            return CopilotAnswer("unsupported_metric",
                "Для прибыли нужны себестоимость и расходы. По этому отчёту могу посчитать выручку."
                if russian else "Profit requires costs and expenses. I can calculate sales revenue from this report.")
        plan = self.plan_query(question, start_date, end_date)
        if plan.intent == "clarify_orders":
            if self.use_demo_erp:
                message = ("Под заказами ты имеешь в виду **счета** или **строки продаж**? "
                           "Могу показать «последние счета», «неоплаченные счета» или «последние продажи».") if russian else (
                    "By orders, do you mean **invoices** or **sales transactions**? "
                    "I can show “latest invoices”, “unpaid invoices”, or “recent sales”.")
            else:
                message = ("По этому CSV могу показать **последние продажи** или **количество продаж**. Что нужно?"
                           if russian else "From this CSV I can show **recent sales** or **the sales count**. Which one?")
            return CopilotAnswer(plan.intent, message)
        if plan.intent == "invoice_data_required":
            return CopilotAnswer(plan.intent, ("Нужны данные о счетах и платежах. Их можно посмотреть в режиме **Demo data**."
                if russian else "This requires invoice and payment records. You can try it using **Demo data**."))
        if plan.intent == "country_data_required":
            return CopilotAnswer(plan.intent, ("Для разбивки по странам нужны страны клиентов. По этому файлу доступен рейтинг по ID."
                if russian else "A country breakdown requires customer country records. This CSV supports rankings by customer ID."))
        if plan.intent == "unknown":
            if MONTHLY.search(q):
                message = ("Что разбить по месяцам — **выручку** или **число продаж**?" if russian
                           else "What should I break down by month — **revenue** or **sales count**?")
            elif CUSTOMERS.search(q):
                message = ("Показать **топ покупателей по выручке** или **количество покупателей**?"
                           if russian else "Would you like the **top buyers by revenue** or the **number of customers**?")
            elif re.search(r"\bowners?\b|владелец|владельц", q):
                message = ("Владелец чего? В данных можно найти клиента по счёту, если ты про это."
                           if russian else "Which owner do you mean? The demo can look up the customer on an invoice.")
            else:
                message = ("Что нужно посчитать или показать? Например, «выручка за 2025» или «последние продажи»."
                           if russian else "What would you like to calculate or see? Try “revenue in 2025” or “recent sales”.")
            return CopilotAnswer("clarify", message)

        if plan.intent == "risk":
            return self._explain_risk(start_date, end_date)

        if plan.intent == "anomalies":
            # IQR methodology is imported from the existing Week 6 dashboard.
            from data_processing import detect_anomalies
            subset = self.sales.copy()
            if plan.params["start_date"]:
                subset = subset[subset.sale_date >= pd.Timestamp(plan.params["start_date"])]
            if plan.params["end_date"]:
                subset = subset[subset.sale_date <= pd.Timestamp(plan.params["end_date"])]
            if len(subset) < 20:
                return CopilotAnswer("anomalies", "At least 20 transactions are needed for IQR anomaly detection.")
            flagged = detect_anomalies(subset)
            if flagged is None:
                return CopilotAnswer("anomalies", "Insufficient transactions for anomaly detection.")
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
            return CopilotAnswer("anomalies", answer, data=display,
                details={"anomaly_count": len(flagged), "transaction_count": len(subset)})

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
        invoice_labels = {
            "overdue": ("Overdue invoices", "Просроченные счета"),
            "outstanding": ("Invoices with a balance", "Счета с остатком"),
            "paid": ("Paid invoices", "Оплаченные счета"),
            "any": ("Invoices", "Счета"),
        }
        if plan.intent in ("recent_sales", "recent_invoices"):
            if plan.intent == "recent_sales":
                label, label_ru = "sales transactions", "строки продаж"
            else:
                label, label_ru = "matching invoices", "подходящие счета"
            if table.empty:
                msg = "За этот период подходящих записей нет." if russian else "No matching records for this period."
            else:
                msg = (f"Показываю {len(table):,} последних записей за выбранный период ({label_ru})." if russian
                       else f"Here are the {len(table):,} most recent {label} in the selected period.")
            return CopilotAnswer(plan.intent, msg, plan.sql, table)
        if plan.intent in ("sales_count", "customer_count", "invoice_count", "total_revenue"):
            label, label_ru = {
                "sales_count": ("Sales transactions", "Строки продаж"),
                "customer_count": ("Customers with sales", "Клиенты с покупками"),
                "invoice_count": invoice_labels[plan.params.get("invoice_status", "any")],
                "total_revenue": ("Valid sales revenue", "Выручка по продажам"),
            }[plan.intent]
            value = table.iloc[0, 0]
            if pd.isna(value):
                msg = "За этот период нет продаж." if russian else "No sales match this period."
            else:
                number = _money(value) if plan.intent == "total_revenue" else f"{int(value):,}"
                period = f" ({start_date} — {end_date})" if start_date and end_date else ""
                msg = f"{label_ru if russian else label}{period} — **{number}**."
            return CopilotAnswer(plan.intent, msg, plan.sql, table)
        if plan.intent == "top_customers":
            if table.empty:
                msg = ("За этот период нет покупателей с продажами."
                       if russian else "No customers with valid sales match the selected period.")
            elif len(table) == 1:
                top = table.iloc[0]
                name = top.get("customer_name")
                customer = f"{name} ({top['customer_id']})" if name else top["customer_id"]
                descriptor = ("Revenue for customer" if "customer_id" in plan.params
                              else "Highest-revenue customer")
                msg = (f"{descriptor}: {customer}, "
                    f"with {_money(top['revenue'])} in valid sales revenue across "
                    f"{int(top['transactions'])} transactions.")
                if russian:
                    lead = "Выручка по покупателю" if "customer_id" in plan.params else "Лидер по выручке"
                    msg = f"{lead} — **{customer}**, {_money(top['revenue'])} по {int(top['transactions'])} продажам."
            else:
                msg = (f"Топ-{len(table)} покупателей по выручке." if russian
                       else f"Top {len(table)} customers ranked by valid sales revenue.")
            return CopilotAnswer(plan.intent, msg, plan.sql, table)
        if plan.intent == "unpaid_invoices":
            if table.empty:
                msg = "Неоплаченных счетов за этот период нет." if russian else "No matching outstanding invoices were found."
            else:
                outstanding = table["outstanding"].clip(lower=0)
                paid0 = int((table["amount_paid"].abs() < 0.01).sum())
                msg = (f"{len(table)} matching invoices; total positive outstanding balance "
                    f"{_money(outstanding.sum())}. {paid0} have no recorded payment. "
                    f"Payments are available through {self.snapshot_date}; this is a historical "
                    "ERP snapshot, not a live banking balance.")
                if russian:
                    msg = (f"Найдено {len(table):,} счетов с остатком на сумму {_money(outstanding.sum())}. "
                           f"По {paid0} из них нет записей об оплате. Данные о платежах по {self.snapshot_date}.")
                if table.attrs.get("truncated"):
                    msg = (f"Первые {MAX_ROWS} счетов. Остаток по ним — {_money(outstanding.sum())}. "
                           f"Данные по {self.snapshot_date}." if russian else
                           f"Showing only the first {MAX_ROWS} matches; displayed outstanding subtotal "
                           f"is {_money(outstanding.sum())}. Data snapshot through {self.snapshot_date}.")
            return CopilotAnswer(plan.intent, msg, plan.sql, table)
        if plan.intent in ("summary", "kpi"):
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
                "distinct_sale_ids": int(row["transactions"]),
                "customers": int(row["customers"]),
                "average_transaction_value": float(row["average_transaction"]),
                "currency": None,
            }
            if plan.intent == "kpi":
                return CopilotAnswer(plan.intent,
                    "Average transaction value is valid sales revenue divided by distinct sale IDs. "
                    f"For this period, {_money(evidence['total_revenue'])} across "
                    f"{evidence['transactions']:,} sales gives {_money(evidence['average_transaction_value'])} per sale.",
                    plan.sql, table, {"evidence": evidence})
            return CopilotAnswer(plan.intent,
                f"Total valid sales revenue: {_money(evidence['total_revenue'])}; "
                f"transactions: {evidence['transactions']}; customers: {evidence['customers']}; "
                f"average sales transaction: {_money(evidence['average_transaction_value'])}. "
                "Amounts are in unspecified source currency and exclude invalid sales.",
                plan.sql, table, {"evidence": evidence})
        if plan.intent in ("monthly_revenue", "monthly_sales_count", "monthly_invoices"):
            if table.empty:
                msg = "За этот период нет записей." if russian else "No records match this period."
            else:
                first_month, last_month = table["month"].iloc[0], table["month"].iloc[-1]
                label, label_ru = {
                    "monthly_revenue": ("Sales revenue", "Выручка"),
                    "monthly_sales_count": ("Sales transactions", "Число продаж"),
                    "monthly_invoices": invoice_labels[plan.params.get("invoice_status", "any")],
                }[plan.intent]
                msg = (f"{label_ru} по месяцам за {first_month} — {last_month}." if russian
                       else f"{label} by month from {first_month} to {last_month}.")
            return CopilotAnswer(plan.intent, msg, plan.sql, table)
        if plan.intent in ("by_product", "by_country"):
            if russian:
                group = "категориям" if "category" in table else "странам" if "country" in table else "товарам"
                message = f"Выручка по {group} за выбранный период." if not table.empty else "За этот период нет продаж."
                return CopilotAnswer(plan.intent, message, plan.sql, table)
            if plan.intent == "by_country" and not table.empty:
                msg = f"Highest-revenue country: {table.iloc[0]['country']} ({_money(table.iloc[0]['revenue'])}). "
            else:
                msg = ""
            msg += f"Revenue breakdown across {len(table)} displayed groups, using cleaned sales."
            return CopilotAnswer(plan.intent, msg, plan.sql, table)
        if plan.intent == "fraud_review":
            evidence = {
                "source": "Demo ERP invoices, payments and valid sales",
                "snapshot_date": self.snapshot_date,
                "period": f"{start_date or _date_value(self.sales.sale_date.min())} to {end_date or _date_value(self.last_sales_date)}",
                "currency": None,
                "flagged_invoice_rows_returned": len(table),
                "invoice_query_limit": 50,
                "flags": json.loads(table.head(5).to_json(orient="records")),
            }
            if not table.empty:
                msg = (f"{len(table)} invoice(s) flagged by payment/discount review rules "
                    "(negative payments, discounts ≥40%, or overpayments). "
                    "These are leads for checking source documents, not proof of fraud.")
                return CopilotAnswer(plan.intent, msg, plan.sql, table, {"evidence": evidence})
            # Supplement payment rules with the existing dashboard's independent
            # statistical outlier evidence, without asserting fraud.
            anomaly = self._ask_local("Explain anomalies", start_date, end_date)
            evidence["sales_iqr_outliers"] = anomaly.details.get("anomaly_count")
            msg = ("No invoices matched the negative-payment, high-discount or overpayment rules. "
                + f"Separately, statistical sales screening found {anomaly.details.get('anomaly_count', 0)} "
                "IQR outliers; a sample is shown for manual review. "
                "Statistical irregularity does not establish intent or fraud.")
            return CopilotAnswer(plan.intent, msg,
                plan.sql + "\n\n-- IQR sales screening performed in Pandas (see Week 6 dashboard)",
                anomaly.data, {"evidence": evidence})
        return CopilotAnswer("help", "No handler for this question.")


def render_chat_panel(copilot=None, *, start_date=None, end_date=None):
    """Open the conversation workspace when running the Copilot by itself."""
    import streamlit as st
    from chat_ui import render_chat_panel as render
    if copilot is not None:
        render(copilot, start_date=start_date, end_date=end_date)
        return
    try:
        with FinancialCopilot() as standalone:
            render(standalone, start_date=start_date, end_date=end_date)
    except (FileNotFoundError, ValueError, sqlite3.Error):
        st.error("The bundled ERP data could not be loaded.")


def main():
    import streamlit as st
    st.set_page_config(page_title="AI Financial Copilot", layout="wide")
    render_chat_panel()


if __name__ == "__main__":
    main()
