"""Fetch NCAAB odds and ESPN scores, save to cached JSON files.
Run by GitHub Actions on schedule, or manually.

Key design: MERGE new data into existing cache so past finals persist.
ESPN only returns today's games, so we archive completed games to avoid losing them."""

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
    return resp.json(), remaining


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
    return games


def load_json(path):
    """Load existing JSON cache file, return empty dict on error."""
    if not os.path.exists(path):
        return {}
    try:
        with open(path) as f:
            return json.load(f)
    except Exception:
        return {}


def merge_espn(existing, new_games):
    """Merge new ESPN games into existing cache. Keep past finals, update live games."""
    # Index existing games by game_id for fast lookup
    by_id = {}
    for g in existing.get("games", []):
        gid = g.get("game_id")
        if gid:
            by_id[gid] = g

    # Update/add new games
    for g in new_games:
        gid = g.get("game_id")
        if gid:
            old = by_id.get(gid)
            if old and old.get("status") == "STATUS_FINAL" and g.get("status") != "STATUS_FINAL":
                # Don't overwrite a final game with a non-final status (ESPN quirk)
                continue
            by_id[gid] = g

    return list(by_id.values())


def merge_odds(existing, new_games):
    """Merge new odds games into existing cache. Update current games, keep old ones with odds."""
    by_id = {}
    for g in existing.get("games", []):
        gid = g.get("id")
        if gid:
            by_id[gid] = g

    for g in new_games:
        gid = g.get("id")
        if gid:
            by_id[gid] = g  # Always update with latest odds

    return list(by_id.values())


def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))

    # ── ESPN scores (free, no key needed) ──
    espn_path = os.path.join(script_dir, "cached_espn.json")
    existing_espn = load_json(espn_path)
    new_espn_games = fetch_espn_scores()

    if new_espn_games is not None:
        merged_games = merge_espn(existing_espn, new_espn_games)
        espn_out = {
            "games": merged_games,
            "fetched_at": datetime.now(timezone.utc).isoformat(),
        }
        with open(espn_path, "w") as f:
            json.dump(espn_out, f, indent=2)
        finals = sum(1 for g in merged_games if g.get("status") == "STATUS_FINAL")
        print(f"Saved ESPN: {len(merged_games)} total games ({finals} finals archived)")

    # ── Odds (requires API key) ──
    api_key = os.environ.get("ODDS_API_KEY", "")
    if not api_key:
        print("ODDS_API_KEY not set, skipping odds fetch")
        return

    result = fetch_odds(api_key)
    if result:
        new_odds_games, remaining = result
        odds_path = os.path.join(script_dir, "cached_odds.json")
        existing_odds = load_json(odds_path)
        merged_odds = merge_odds(existing_odds, new_odds_games)
        odds_out = {
            "games": merged_odds,
            "remaining": remaining,
            "fetched_at": datetime.now(timezone.utc).isoformat(),
        }
        with open(odds_path, "w") as f:
            json.dump(odds_out, f, indent=2)
        print(f"Saved odds: {len(merged_odds)} total games (was {len(existing_odds.get('games', []))})")
    else:
        print("Failed to fetch odds")
        sys.exit(1)


if __name__ == "__main__":
    main()
