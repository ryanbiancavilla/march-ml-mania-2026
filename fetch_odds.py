"""Fetch NCAAB odds and ESPN scores, save to cached JSON files.
Run by GitHub Actions every 2 hours, or manually."""

import json
import os
import sys
from datetime import datetime, timezone

import requests


def fetch_odds(api_key):
    """Fetch NCAAB odds from The Odds API."""
    url = "https://api.the-odds-api.com/v4/sports/basketball_ncaab/odds"
    resp = requests.get(url, params={
        "apiKey": api_key,
        "regions": "us",
        "markets": "h2h,spreads,totals",
        "oddsFormat": "american",
    }, timeout=15)

    if resp.status_code == 401:
        print("ERROR: Invalid API key")
        return None
    if resp.status_code == 429:
        print("ERROR: Rate limited (monthly credits exhausted)")
        return None
    if resp.status_code != 200:
        print(f"ERROR: API returned {resp.status_code}")
        return None

    remaining = resp.headers.get("x-requests-remaining", "?")
    print(f"Odds API: {len(resp.json())} games fetched, {remaining} credits remaining")
    return {
        "games": resp.json(),
        "remaining": remaining,
        "fetched_at": datetime.now(timezone.utc).isoformat(),
    }


def fetch_espn_scores():
    """Fetch live NCAA tournament scores from ESPN's free API."""
    url = "https://site.api.espn.com/apis/site/v2/sports/basketball/mens-college-basketball/scoreboard"
    resp = requests.get(url, params={"groups": 100, "limit": 200}, timeout=15)

    if resp.status_code != 200:
        print(f"ESPN API returned {resp.status_code}")
        return None

    data = resp.json()
    games = []
    for ev in data.get("events", []):
        status = ev.get("status", {}).get("type", {}).get("name", "")
        comps = ev.get("competitions", [{}])
        if not comps:
            continue
        comp = comps[0]
        competitors = comp.get("competitors", [])
        if len(competitors) < 2:
            continue

        home = competitors[0]
        away = competitors[1]
        # Extract broadcast info
        broadcasts = comp.get("broadcasts", [])
        broadcast_names = []
        for b in broadcasts:
            for name_entry in b.get("names", []):
                broadcast_names.append(name_entry)
        broadcast = ", ".join(broadcast_names) if broadcast_names else ""

        # Extract game start time
        start_time = comp.get("date", ev.get("date", ""))

        games.append({
            "game_id": ev.get("id"),
            "name": ev.get("name", ""),
            "status": status,
            "status_detail": ev.get("status", {}).get("type", {}).get("shortDetail", ""),
            "home_team": home.get("team", {}).get("displayName", ""),
            "home_score": int(home.get("score", 0)),
            "home_id": home.get("team", {}).get("id", ""),
            "away_team": away.get("team", {}).get("displayName", ""),
            "away_score": int(away.get("score", 0)),
            "away_id": away.get("team", {}).get("id", ""),
            "broadcast": broadcast,
            "start_time": start_time,
        })

    print(f"ESPN: {len(games)} games fetched")
    return {
        "games": games,
        "fetched_at": datetime.now(timezone.utc).isoformat(),
    }


def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))

    # Fetch ESPN scores (free, no key needed)
    espn_data = fetch_espn_scores()
    if espn_data:
        espn_path = os.path.join(script_dir, "cached_espn.json")
        with open(espn_path, "w") as f:
            json.dump(espn_data, f, indent=2)
        print(f"Saved ESPN scores to cached_espn.json")

    # Fetch odds (requires API key)
    api_key = os.environ.get("ODDS_API_KEY", "")
    if not api_key:
        print("ODDS_API_KEY not set, skipping odds fetch")
        return

    odds_data = fetch_odds(api_key)
    if odds_data:
        odds_path = os.path.join(script_dir, "cached_odds.json")
        with open(odds_path, "w") as f:
            json.dump(odds_data, f, indent=2)
        print(f"Saved odds to cached_odds.json")
    else:
        print("Failed to fetch odds")
        sys.exit(1)


if __name__ == "__main__":
    main()
