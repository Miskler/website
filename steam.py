import json
import aiohttp
import asyncio
import time


with open("configs/secrets.json", encoding="utf-8") as f:
    SECRETS = json.load(f)

STEAM_KEY = SECRETS["steam"]
STEAM_ID = SECRETS["steam_id"]
STEAM_API = "https://api.steampowered.com"


# ------------------------
# Steam API helpers
# ------------------------

async def steam_get(session, interface, method, version="v1", **params):
    url = f"{STEAM_API}/{interface}/{method}/{version}"
    params["key"] = STEAM_KEY

    async with session.get(url, params=params) as resp:
        resp.raise_for_status()
        return await resp.json()

def country_code_to_flag(code: str) -> str:
    """
    Преобразует ISO 3166-1 alpha-2 код страны в emoji-флаг.
    Пример: 'KZ' -> 🇰🇿
    """
    if not code or len(code) != 2:
        return ""

    code = code.upper()
    return chr(ord(code[0]) + 127397) + chr(ord(code[1]) + 127397)

def plural_ru(value: int, form1: str, form2: str, form5: str) -> str:
    """
    value  — число
    form1  — 1 год/день/час/минута
    form2  — 2 года/дня/часа/минуты
    form5  — 5 лет/дней/часов/минут

    Возвращает: "<value> <правильная форма>"
    """
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

def humanize_timestamp(
    ts: int,
    tz_offset: int = 0,
    now: int | None = None
) -> str:
    """
    ts         — Unix timestamp (UTC)
    tz_offset  — смещение часового пояса в часах (например: +3)
    now        — текущий Unix timestamp (UTC), опционально
    """
    offset = tz_offset * 3600

    if now is None:
        now = int(time.time())

    # Сдвигаем оба времени в одну зону

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
    for limit, f1, f2, f5 in units:
        if value < limit:
            return plural_ru(int(value), f1, f2, f5) + " назад"
        value /= limit


async def get_user_data():
    async with aiohttp.ClientSession() as session:
        user, badges, games = await asyncio.gather(
            steam_get(
                session,
                "ISteamUser",
                "GetPlayerSummaries",
                steamids=STEAM_ID
            ),
            steam_get(
                session,
                "IPlayerService",
                "GetBadges",
                steamid=STEAM_ID
            ),
            steam_get(
                session,
                "IPlayerService",
                "GetOwnedGames",
                steamid=STEAM_ID,
                include_appinfo=1,
                include_played_free_games=1
            )
        )
        user = user["response"]["players"]["player"][0]
        badges = badges["response"]
        games = games["response"]

        user["loccountyflag"] = country_code_to_flag(user["loccountrycode"])

        real_state = ["offline", "online", "занят", "отошел", "спит", "торгует", "ищет игру", "играет в {game}"]
        online_state = ["offline", "online", "busy", "away", "away", "online", "online", "busy"]
        user["onlineState"] = online_state[user["personastate"]].replace(" ", "")
        if "gameextrainfo" in user:
            user["lastlog"] = real_state[-1].format(game=user["gameextrainfo"])
        else:
            if user["personastate"] == 0:
                user["lastlog"] = humanize_timestamp(user["lastlogoff"], tz_offset=0)
            else:
                user["lastlog"] = real_state[user["personastate"]]

        badges["player_level_word"] = f"{badges['player_level']} уровень"
        badges["percent"] = round(100 / (badges['player_xp']+badges['player_xp_needed_to_level_up']) * badges['player_xp'], 1)

        return {
            "user": user,
            "badges": badges,
            "games": games
        }
