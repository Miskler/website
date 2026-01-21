import json
import aiohttp
import asyncio


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

        return {
            "user": user["response"],
            "badges": badges["response"],
            "games": games["response"]
        }
