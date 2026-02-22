from datetime import datetime
from typing import Any, Dict

import aiohttp
from async_lru import alru_cache

from tools import plural_ru

OPENWORKSHOP_STATISTICS_API = "https://api.openworkshop.miskler.ru/catalog/statistics"
OPENWORKSHOP_MODS_API = "https://api.openworkshop.miskler.ru/mods"
OPENWORKSHOP_MODS_PARAMS: Dict[str, Any] = {
    "page_size": 10,
    "page": 0,
    "sort": "DOWNLOADS",
    "show_not_public": "false",
    "short_description": "true",
    "description": "false",
    "dates": "true",
    "general": "true",
}


def _safe_int(value: Any) -> int:
    try:
        return max(0, int(value))
    except (TypeError, ValueError):
        return 0


def _format_number(value: int) -> str:
    return f"{value:,}".replace(",", " ")


def _plural_form(value: int, form1: str, form2: str, form5: str) -> str:
    return plural_ru(value, form1, form2, form5).split(" ", maxsplit=1)[1]


def _format_count(value: int, form1: str, form2: str, form5: str) -> str:
    return f"{_format_number(value)} {_plural_form(value, form1, form2, form5)}"


def _format_float(value: float, digits: int = 1) -> str:
    formatted = f"{value:.{digits}f}".rstrip("0").rstrip(".")
    return formatted.replace(".", ",")


def _format_bytes(size_bytes: int) -> str:
    units = ("Б", "КиБ", "МиБ", "ГиБ", "ТиБ", "ПиБ")
    value = float(max(size_bytes, 0))
    unit_index = 0

    while value >= 1024 and unit_index < len(units) - 1:
        value /= 1024
        unit_index += 1

    if unit_index == 0:
        return f"{_format_number(int(value))} {units[unit_index]}"
    return f"{_format_float(value, 2)} {units[unit_index]}"


def _safe_text(value: Any, fallback: str = "—") -> str:
    if value is None:
        return fallback
    if isinstance(value, str):
        value = value.strip()
        return value if value else fallback

    value_str = str(value).strip()
    return value_str if value_str else fallback


def _format_date(value: Any) -> str:
    if not isinstance(value, str):
        return "—"

    try:
        dt = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return "—"

    return dt.strftime("%d.%m.%Y")


def _format_source(source: Any) -> str:
    if not isinstance(source, str):
        return "UNKNOWN"

    labels = {
        "steam": "Steam",
        "moddb": "ModDB",
        "github": "GitHub",
        "nexusmods": "NexusMods",
    }
    return labels.get(source.lower(), source.upper())


def _build_mod_item(payload: Dict[str, Any]) -> Dict[str, Any]:
    mod_id = _safe_int(payload.get("id"))
    downloads = _safe_int(payload.get("downloads"))

    return {
        "id": mod_id,
        "name": _safe_text(payload.get("name"), fallback=f"Mod #{mod_id}" if mod_id else "Unknown mod"),
        "short_description": _safe_text(payload.get("short_description"), fallback="Описание отсутствует"),
        "source_label": _format_source(payload.get("source")),
        "size_word": _format_bytes(_safe_int(payload.get("size"))),
        "downloads_word": _format_count(downloads, "загрузка", "загрузки", "загрузок"),
        "date_creation_word": _format_date(payload.get("date_creation")),
        "date_edit_word": _format_date(payload.get("date_edit")),
        "date_update_word": _format_date(payload.get("date_update_file")),
        "source_id": _safe_text(payload.get("source_id"), fallback="—"),
    }


def _build_stats(payload: Dict[str, Any], api_error: str | None = None) -> Dict[str, Any]:
    mods_count = _safe_int(payload.get("mods_count"))
    users_count = _safe_int(payload.get("users_count"))
    database_size_bytes = _safe_int(payload.get("database_size_bytes"))
    database_size_unpacked_bytes = _safe_int(payload.get("database_size_unpacked_bytes"))
    resources_size_bytes = _safe_int(payload.get("resources_size_bytes"))

    raw_db_fill_percent = (
        (database_size_bytes / database_size_unpacked_bytes * 100)
        if database_size_unpacked_bytes
        else 0.0
    )
    db_fill_percent = round(raw_db_fill_percent, 1)
    db_fill_percent_bar = round(min(max(raw_db_fill_percent, 0.0), 100.0), 1)
    db_saving_percent = round(max(0.0, 100.0 - raw_db_fill_percent), 1)
    db_compression_ratio = (
        round(database_size_unpacked_bytes / database_size_bytes, 2) if database_size_bytes else 0.0
    )
    database_saved_bytes = max(database_size_unpacked_bytes - database_size_bytes, 0)
    total_storage_bytes = database_size_bytes + resources_size_bytes
    mods_per_user = (mods_count / users_count) if users_count else 0.0

    return {
        "mods_count": mods_count,
        "users_count": users_count,
        "database_size_bytes": database_size_bytes,
        "database_size_unpacked_bytes": database_size_unpacked_bytes,
        "resources_size_bytes": resources_size_bytes,
        "mods_count_word": _format_count(mods_count, "мод", "мода", "модов"),
        "users_count_word": _format_count(users_count, "пользователь", "пользователя", "пользователей"),
        "database_size_word": _format_bytes(database_size_bytes),
        "database_size_unpacked_word": _format_bytes(database_size_unpacked_bytes),
        "resources_size_word": _format_bytes(resources_size_bytes),
        "database_fill_percent": db_fill_percent_bar,
        "database_fill_percent_word": f"{_format_float(db_fill_percent)}%",
        "database_saving_percent_word": f"{_format_float(db_saving_percent)}%",
        "database_compression_ratio_word": f"{_format_float(db_compression_ratio, 2)}x",
        "database_saved_word": _format_bytes(database_saved_bytes),
        "total_storage_word": _format_bytes(total_storage_bytes),
        "mods_per_user_word": f"{_format_float(mods_per_user, 1)} модов/польз.",
        "api_error": api_error,
    }


@alru_cache(ttl=240)
async def fetch_openworkshop_stats() -> Dict[str, Any]:
    payload: Any
    timeout = aiohttp.ClientTimeout(total=8)

    try:
        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.get(OPENWORKSHOP_STATISTICS_API) as response:
                response.raise_for_status()
                payload = await response.json()
    except Exception as exc:
        return _build_stats({}, api_error=f"{type(exc).__name__}: {exc}")

    if not isinstance(payload, dict):
        return _build_stats({}, api_error="Invalid API response shape")

    return _build_stats(payload)


@alru_cache(ttl=180)
async def fetch_openworkshop_mods() -> Dict[str, Any]:
    payload: Any
    timeout = aiohttp.ClientTimeout(total=8)

    try:
        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.get(OPENWORKSHOP_MODS_API, params=OPENWORKSHOP_MODS_PARAMS) as response:
                response.raise_for_status()
                payload = await response.json()
    except Exception as exc:
        return {
            "mods": [],
            "mods_total_word": _format_count(0, "мод", "мода", "модов"),
            "offset": 0,
            "api_error": f"{type(exc).__name__}: {exc}",
        }

    if not isinstance(payload, dict):
        return {
            "mods": [],
            "mods_total_word": _format_count(0, "мод", "мода", "модов"),
            "offset": 0,
            "api_error": "Invalid API response shape",
        }

    raw_results = payload.get("results")
    if not isinstance(raw_results, list):
        raw_results = []

    mods = [_build_mod_item(mod) for mod in raw_results if isinstance(mod, dict)]
    mods_total = _safe_int(payload.get("database_size"))

    return {
        "mods": mods,
        "mods_total_word": _format_count(mods_total, "мод", "мода", "модов"),
        "offset": _safe_int(payload.get("offset")),
        "api_error": None,
    }
