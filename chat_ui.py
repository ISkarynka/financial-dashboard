"""Streamlit conversation workspace for the ERP Copilot."""

from hashlib import sha256
import json
import os
from pathlib import Path
import re
import sqlite3

import pandas as pd
import streamlit as st
from streamlit.errors import StreamlitSecretNotFoundError

from chat_store import ChatStore


BROWSER_IDENTITY = st.components.v2.component(
    "copilot_browser_identity",
    js="""
        export default function ({setStateValue}) {
            const key = "financial-dashboard.copilot-owner";
            let token;
            try { token = localStorage.getItem(key); } catch {}
            if (!/^[a-f0-9]{64}$/.test(token || "")) {
                token = Array.from(crypto.getRandomValues(new Uint8Array(32)),
                    byte => byte.toString(16).padStart(2, "0")).join("");
                try { localStorage.setItem(key, token); } catch {}
            }
            setStateValue("token", token);
        }
    """,
)


def _browser_owner():
    token = st.session_state.get("_chat_browser_token")
    if not token:
        result = BROWSER_IDENTITY(key="copilot_identity", on_token_change=lambda: None)
        token = result.token
        if not isinstance(token, str) or not re.fullmatch(r"[a-f0-9]{64}", token):
            return None
        st.session_state["_chat_browser_token"] = token
    return sha256(token.encode()).hexdigest()


def _server_key():
    key = os.environ.get("OPENAI_API_KEY")
    if not key:
        try:
            key = st.secrets.get("OPENAI_API_KEY")
        except StreamlitSecretNotFoundError:
            return None
    return key.strip() if isinstance(key, str) else None


def _select_chat(chat_id):
    st.session_state["_active_chat"] = chat_id


def _export(title, source, turns):
    messages = []
    for turn in turns:
        message = {key: value for key, value in turn.items() if key != "table_json"}
        if turn["table_json"]:
            message["table"] = json.loads(turn["table_json"])
        messages.append(message)
    return json.dumps({"title": title, "source": source, "turns": messages}, ensure_ascii=False, indent=2)


def _show_turn(turn, period):
    with st.chat_message("user"):
        st.markdown(turn["question"])
    with st.chat_message("assistant", avatar=":material/insights:"):
        if turn["period"] != period:
            st.caption(turn["period"])
        st.markdown(turn["answer"])
        if turn["note"]:
            st.caption(turn["note"])
        if turn["table_json"]:
            table = json.loads(turn["table_json"])
            st.dataframe(pd.DataFrame(table["data"], columns=table["columns"]), width="stretch", hide_index=True)
        if turn["sql"]:
            with st.expander("View SQL"):
                st.code(turn["sql"], language="sql")


def render_chat_panel(copilot, *, start_date=None, end_date=None, source="demo"):
    """Open saved conversations for the browser and selected dataset."""
    st.title("AI Financial Copilot", anchor=False)
    owner = _browser_owner()
    if owner is None:
        st.caption("Opening your chats…")
        return

    location = Path(os.environ.get("COPILOT_DB_PATH", Path(__file__).with_name(".copilot") / "chats.sqlite3"))
    try:
        store = ChatStore(location, owner, source)
        chats = store.chats()
    except (OSError, sqlite3.Error):
        st.error("Chat storage is unavailable. Try opening Copilot again.")
        return

    if st.session_state.get("_chat_source") != source:
        st.session_state["_chat_source"] = source
        _select_chat(chats[0]["id"] if chats else None)
    active_chat = st.session_state.get("_active_chat")
    active_title = next((chat["title"] for chat in chats if chat["id"] == active_chat), None)
    if active_chat and active_title is None:
        _select_chat(None)
        active_chat = None
    turns = store.turns(active_chat) if active_chat else []
    source_label = "Demo ERP data" if copilot.use_demo_erp else "Uploaded sales CSV"
    with st.sidebar, st.container(key="copilot_chat_list"):
        st.subheader("Chats")
        st.button("New chat", icon=":material/add:", width="stretch", on_click=_select_chat, args=(None,))
        with st.container(height=260, border=False) if len(chats) > 5 else st.container():
            for chat in chats:
                st.button(chat["title"], key=f"thread_{chat['id']}", width="stretch",
                          type="primary" if chat["id"] == active_chat else "tertiary",
                          on_click=_select_chat, args=(chat["id"],))
        if active_chat and turns:
            st.download_button(
                "Download chat", _export(active_title, source_label, turns),
                file_name="financial-copilot-chat.json", mime="application/json", on_click="ignore",
                icon=":material/download:",
            )

    api_key = _server_key()
    period = f"{start_date or copilot.sales.sale_date.min().date()} to {end_date or copilot.last_sales_date}"
    mode = "OpenAI" if api_key else "Built-in analytics"
    st.caption(f"{source_label} · {period} · {mode}")
    if api_key:
        st.caption("OpenAI processes your question and selected evidence for explanations.")

    prompt_buttons = [
        ("Which customer generated the highest revenue?", ":material/leaderboard:"),
        ("Forecast next month's revenue.", ":material/trending_up:"),
        ("Explain anomalies.", ":material/troubleshoot:"),
        ("Give me a financial KPI summary", ":material/summarize:"),
        ("Explain average transaction value", ":material/calculate:"),
        ("Explain revenue risk", ":material/monitoring:"),
    ]
    if copilot.use_demo_erp:
        prompt_buttons.insert(1, ("Show unpaid invoices.", ":material/receipt_long:"))
        prompt_buttons.append(("Investigate fraud risk", ":material/policy:"))
    request = None
    with st.expander("Suggested questions", expanded=not turns):
        columns = st.columns(2)
        for index, (question, icon) in enumerate(prompt_buttons):
            if columns[index % 2].button(question, icon=icon, width="stretch", key=f"example_{index}"):
                request = question

    for turn in turns:
        _show_turn(turn, period)
    with st.bottom:
        typed = st.chat_input("Ask about your financial data", max_chars=2000)
    question = (request or typed or "").strip()
    if not question:
        return
    with st.chat_message("user"):
        st.markdown(question)
    with st.spinner("Checking the report…"):
        try:
            result = copilot.ask(question, start_date=start_date, end_date=end_date,
                                 use_llm=bool(api_key), api_key=api_key,
                                 history=[turn["question"] for turn in turns[-4:]])
            _select_chat(store.save_turn(active_chat, question, result, period))
        except (OSError, ValueError, RuntimeError, sqlite3.Error):
            st.error("The Copilot could not complete and save this answer. Please try again.")
            return
    st.rerun()
