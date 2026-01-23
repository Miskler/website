import time
from markupsafe import Markup
import markdown

def render_md(text):
    html = markdown.markdown(
        text,
        extensions=[
            "extra",
            "sane_lists",
            "nl2br"
        ]
    )
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


def humanize_timestamp(ts: int, tz_offset: int = 0, now: int | None = None) -> str:
    offset = tz_offset * 3600
    if now is None:
        now = int(time.time())

    delta = (now - ts) - offset
    if delta < 0:
        return "в будущем"
    if delta < 5:
        return "только что"

    units = (
        (60, "секунду", "секунды", "секунд"),
        (60, "минуту", "минуты", "минут"),
        (24, "час", "часа", "часов"),
        (7, "день", "дня", "дней"),
        (4.34524, "неделю", "недели", "недель"),
        (12, "месяц", "месяца", "месяцев"),
        (float("inf"), "год", "года", "лет"),
    )

    value = delta
    prev_value = None
    prev_forms = None

    for limit, f1, f2, f5 in units:
        if value < limit:
            main = int(value)

            result = plural_ru(main, f1, f2, f5)

            # добавляем предыдущий разряд, если он есть и ненулевой
            if prev_value is not None:
                extra = int((value - main) * prev_value)
                if extra > 0:
                    result += " " + plural_ru(extra, *prev_forms)

            return result + " назад"

        prev_value = limit
        prev_forms = (f1, f2, f5)
        value /= limit
