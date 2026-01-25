import time
from datetime import datetime
from typing import Optional, Tuple

import markdown
from markupsafe import Markup


def render_md(text: str) -> Markup:
    html = markdown.markdown(text, extensions=["extra", "sane_lists", "nl2br"])
    return Markup(html)


def plural_ru(value: int, form1: str, form2: str, form5: str) -> str:
    n = abs(value)
    if 11 <= n % 100 <= 14:
        form = form5
    else:
        last = n % 10
        if last == 1:
            form = form1
        elif 2 <= last <= 4:
            form = form2
        else:
            form = form5
    return f"{value} {form}"


def humanize_timestamp(ts: int | str, tz_offset: int = 0, now: Optional[int] = None) -> str:
    # если ts — строка ISO 8601, преобразуем в timestamp
    if isinstance(ts, str):
        try:
            dt = datetime.fromisoformat(ts.replace("Z", "+00:00"))
            ts_int = int(dt.timestamp())
        except ValueError:
            return "неверный формат даты"
    else:
        ts_int = ts

    offset = tz_offset * 3600
    if now is None:
        now_int = int(time.time())
    else:
        now_int = now

    delta = (now_int - ts_int) - offset

    if delta < 0:
        return "в будущем"
    if delta < 5:
        return "только что"

    units: Tuple[Tuple[float, str, str, str], ...] = (
        (60.0, "секунду", "секунды", "секунд"),
        (60.0, "минуту", "минуты", "минут"),
        (24.0, "час", "часа", "часов"),
        (7.0, "день", "дня", "дней"),
        (4.34524, "неделю", "недели", "недель"),
        (12.0, "месяц", "месяца", "месяцев"),
        (float("inf"), "год", "года", "лет"),
    )

    value: float = float(delta)
    prev_value: Optional[float] = None
    prev_forms: Optional[Tuple[str, str, str]] = None

    for limit, f1, f2, f5 in units:
        if value < limit:
            main = int(value)
            result = plural_ru(main, f1, f2, f5)
            # добавляем предыдущий разряд, если он есть и ненулевой
            if prev_value is not None and prev_forms is not None:
                extra = int((value - main) * prev_value)
                if extra > 0:
                    result += " " + plural_ru(extra, *prev_forms)
            return result + " назад"
        prev_value = limit
        prev_forms = (f1, f2, f5)
        value /= limit

    # Этот код никогда не выполнится из-за float("inf") в последнем элементе,
    # но для полноты возвращаем fallback значение
    return "давно (UNKNOWN ERROR)"
