import json
import aiohttp
import asyncio
from tools import humanize_timestamp, plural_ru

with open("configs/secrets.json", encoding="utf-8") as f:
    SECRETS = json.load(f)

STEAM_KEY = SECRETS["steam"]
STEAM_ID = SECRETS["steam_id"]
STEAM_API = "https://api.steampowered.com"


async def steam_get(session, interface, method, version="v1", **params):
    url = f"{STEAM_API}/{interface}/{method}/{version}"
    params["key"] = STEAM_KEY
    async with session.get(url, params=params) as resp:
        resp.raise_for_status()
        return await resp.json()

def humanize_playtime(ts: int) -> str:
    value = ts * 60
    if value < 0:
        return "не играл"
    if value < 5:
        return "пару минут"
    units = (
        (60, "секунду", "секунды", "секунд"),
        (60, "минуту", "минуты", "минут"),
        (float("inf"), "час", "часа", "часов")
    )
    for limit, f1, f2, f5 in units:
        if value < limit:
            return plural_ru(int(value), f1, f2, f5)
        value /= limit


async def get_user_data():
    async with aiohttp.ClientSession() as session:
        # Получаем пользователя, бейджи и игры
        user, badges, games = await asyncio.gather(
            steam_get(session, "ISteamUser", "GetPlayerSummaries", steamids=STEAM_ID),
            steam_get(session, "IPlayerService", "GetBadges", steamid=STEAM_ID),
            steam_get(session, "IPlayerService", "GetOwnedGames",
                      steamid=STEAM_ID, include_appinfo=1, include_played_free_games=1)
        )

        user = user["response"]["players"]["player"][0]
        badges = badges["response"]
        games = games["response"]["games"]

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
        
        user["timecreated_word"] = humanize_timestamp(user["timecreated"])

        badges["player_level_word"] = f"{badges['player_level']} уровень"
        badges["percent"] = round(100 / (badges['player_xp']+badges['player_xp_needed_to_level_up']) * badges['player_xp'], 1)

        # Сортируем игры по playtime_forever и берём первые 30
        top_games = sorted(games, key=lambda x: x.get("playtime_forever", 0), reverse=True)[:20]

        for g in top_games:
            g["vlogo"] = f"https://steamcdn-a.akamaihd.net/steam/apps/{g['appid']}/library_600x900.jpg"
            g["playtime_word"] = humanize_playtime(int(g["playtime_forever"]))
            g["gamelink"] = f"https://store.steampowered.com/app/{g['appid']}/{g['name'].replace(' ', '_')}"

        return {
            "user": user,
            "badges": badges,
            "games": top_games
        }


# Пример запуска
if __name__ == "__main__":
    data = asyncio.run(get_user_data())
    print(json.dumps(data, ensure_ascii=False, indent=2))
