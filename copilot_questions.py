"""Recognise common English and Russian questions in the built-in Copilot."""

import re
from calendar import monthrange
from datetime import date


REVENUE = re.compile(r"\b(?:revenue|turnover|выручк\w*|оборот\w*)\b")
SALES = re.compile(r"\b(?:sales?|transactions?|продаж\w*|транзакци\w*)\b")
ORDERS = re.compile(r"\b(?:orders?|ordrs?|заказ\w*|ордер\w*|орнер)\b")
INVOICES = re.compile(r"\b(?:invoices?|bills?|инвойс\w*|счет(?:а|ов|у|ом|е|ам|ами|ах)?)\b")
UNPAID = re.compile(r"\b(?:unpaid|outstanding|overdue|receivables?|неоплач\w*|просроч\w*|задолж\w*|долг\w*|должник\w*)\b")
CUSTOMERS = re.compile(r"\b(?:customers?|clients?|buyers?|клиент\w*|покупател\w*)\b")
COUNT = re.compile(r"\b(?:how many|number of|count|сколько|количество|число)\b")
RECENT = re.compile(r"\b(?:recent|latest|last|newest|последн\w*|свеж\w*)\b")
MOST = re.compile(r"\b(?:top|highest|most|best|biggest|largest|maximum|топ|больш\w*|лучш\w*|максим\w*|перв\w*)\b")
MONTHLY = re.compile(r"\b(?:monthly|by month|month-by-month|по месяцам|помесячно|динамик\w*)\b")
CATEGORY = re.compile(r"\b(?:by category|by product|по категори\w*|по товар\w*)\b")
FORECAST = re.compile(r"forecast|predict|projection|next month|прогноз|следующ.{0,8}месяц")
GREETINGS = {"hi", "hey", "hello", "привет", "здравствуйте", "добрый день"}
THANKS = {"thanks", "thank you", "спасибо", "благодарю"}

YEAR_OR_DATE = re.compile(r"\b(20\d{2})(?:-(\d{2})(?:-(\d{2}))?)?\b")
MONTH_PREFIXES = (
    ("jan", "january", "январ"), ("feb", "february", "феврал"),
    ("mar", "march", "март"), ("apr", "april", "апрел"),
    ("may", "май", "мая", "мае"), ("jun", "june", "июн"),
    ("jul", "july", "июл"), ("aug", "august", "август"),
    ("sep", "sept", "september", "сентябр"), ("oct", "october", "октябр"),
    ("nov", "november", "ноябр"), ("dec", "december", "декабр"),
)
FOLLOW_UP_WORDS = {"and", "for", "in", "during", "what", "about", "how",
                   "а", "и", "что", "как", "насчет", "за", "в", "г", "год", "года", "году"}
PERIOD_PREPOSITIONS = {"in", "for", "during", "from", "of", "about", "and", "за", "в", "с", "по", "на", "насчет", "а", "и"}


def normalize_question(question):
    return " ".join(question.casefold().replace("ё", "е").split())


def is_russian(question):
    return bool(re.search(r"[а-яё]", question, re.I))


def _month_word(word):
    for month, names in enumerate(MONTH_PREFIXES, 1):
        if any(word == name if name.isascii() else word.startswith(name) for name in names):
            return month
    return None


def question_period(text):
    """Return one explicit day, month or year as inclusive ISO dates."""
    matches = list(YEAR_OR_DATE.finditer(text))
    if len(matches) != 1:
        return None
    match = matches[0]
    year, month, day = match.groups()
    year = int(year)
    preceding = re.search(r"\b([a-zа-я]+)\.?\s+$", text[:match.start()])
    month = int(month) if month else _month_word(preceding[1]) if preceding else None
    if month is None and match.start() > 0 and not re.search(r"^\s*(?:г\b|год\w*\b)", text[match.end():]):
        word = preceding[1] if preceding else ""
        if word not in PERIOD_PREPOSITIONS and not any(
            pattern.fullmatch(word) for pattern in (REVENUE, SALES, ORDERS, INVOICES, CUSTOMERS)
        ):
            return None
    try:
        if day is not None:
            day = date(year, month, int(day)).isoformat()
            return day, day
        if month is not None:
            return date(year, month, 1).isoformat(), date(year, month, monthrange(year, month)[1]).isoformat()
        return date(year, 1, 1).isoformat(), date(year, 12, 31).isoformat()
    except ValueError:
        return None


def follow_up(question, history):
    """Carry a report topic into short date or revenue-breakdown follow-ups."""
    question_text = normalize_question(question)
    short_question = question_text.strip(" ?!.")
    if short_question in GREETINGS or short_question in THANKS:
        return question_text, None
    topic, period, breakdown = "", None, None
    for text in [*(normalize_question(turn) for turn in history[-4:]), question_text]:
        short = text.strip(" ?!.")
        if short in GREETINGS or short in THANKS:
            continue
        requested = question_period(text)
        words = re.findall(r"[a-zа-я]+", text)
        period_only = requested and all(word in FOLLOW_UP_WORDS or _month_word(word) for word in words)
        short = re.sub(r"^(?:and|а|и|what about)\s+", "", short)
        grouping = MONTHLY.fullmatch(short) or CATEGORY.fullmatch(short)
        if period_only:
            period = requested
        elif grouping:
            breakdown = short
        else:
            order_choice = (
                len(words) <= 5 and ORDERS.search(topic)
                and any(pattern.search(text) for pattern in (INVOICES, UNPAID, SALES))
            )
            if requested or not order_choice:
                period = requested
            if order_choice and COUNT.search(topic) and not (COUNT.search(text) or RECENT.search(text)):
                text = f"count {text}"
            topic, breakdown = text, None
    if breakdown:
        invoice_topic = INVOICES.search(topic) or UNPAID.search(topic)
        count_topic = COUNT.search(topic) and SALES.search(topic) and not REVENUE.search(topic)
        if MONTHLY.fullmatch(breakdown) and (invoice_topic or count_topic):
            topic = f"{topic} {breakdown}"
        elif not (invoice_topic or count_topic or ORDERS.search(topic)) and (REVENUE.search(topic) or SALES.search(topic)):
            topic = f"revenue {breakdown}"
        else:
            topic = breakdown
    return topic or question_text, period
