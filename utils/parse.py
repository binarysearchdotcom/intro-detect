import re


def parse_show_and_season(name: str):
    show_match = re.split(r"\b\d+\s*сезон", name, maxsplit=1, flags=re.IGNORECASE)
    show_name = show_match[0].strip(" .,") if len(show_match) > 1 else "unknown"

    season_match = re.search(r"(\d+)\s*сезон", name, flags=re.IGNORECASE)
    series_match = re.search(r"(\d+)\s*серия", name, flags=re.IGNORECASE)

    season_num = season_match.group(1) if season_match else "0"
    series_num = series_match.group(1) if series_match else "0"

    return show_name, season_num, series_num


def hms_to_sec(hms: str) -> int:
    h, m, s = map(int, hms.strip().split(":"))
    return h * 3600 + m * 60 + s


def frames_to_hms(fr: int, FPS) -> str:
    t = fr / FPS
    h, m, s = int(t // 3600), int((t % 3600) // 60), int(t % 60)
    return f"{h:02d}:{m:02d}:{s:02d}"
