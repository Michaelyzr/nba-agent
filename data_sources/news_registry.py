"""Reviewed account identities and per-game source plans (no network required)."""
import argparse
import json
import re
from pathlib import Path
from types import SimpleNamespace

DEFAULT_REGISTRY = Path(__file__).with_name("news_sources.json")
TEAM_CODES = set("ATL BOS BKN CHA CHI CLE DAL DEN DET GSW HOU IND LAC LAL MEM MIA MIL MIN NOP NYK OKC ORL PHI PHX POR SAC SAS TOR UTA WAS".split())
TEAM_ALIASES = {"GS": "GSW", "NY": "NYK", "NO": "NOP", "SA": "SAS", "PHO": "PHX", "WSH": "WAS", "BRK": "BKN"}
PRIORITIES = {"official_report": 50, "team_official": 40, "authority_media": 30, "insider": 20, "team_reporter": 10, "unclassified": 0}


class NewsRegistry:
    def __init__(self, data):
        self.data = data
        if set(data["teams"]) != TEAM_CODES:
            raise ValueError("news registry must cover exactly the 30 NBA teams")
        self.accounts = {}
        self._add(data["insider"], "insider", None)
        for team, entry in data["teams"].items():
            self._add({"handle": entry["official"], "name": entry["name"],
                       "evidence_urls": [data["official_accounts_evidence"]]}, "team_official", team)
            for handle in entry.get("pr", []):
                self._add({"handle": handle, "name": entry["name"] + " PR"}, "team_official", team)
            self._add(entry["reporter"], "team_reporter", team)
        self.media = {m["source"]: m for m in data["media"]}
        if len(self.media) != len(data["media"]):
            raise ValueError("duplicate media source id")

    def _add(self, account, role, team):
        handle = account["handle"]
        if not re.fullmatch(r"[A-Za-z0-9_]{1,15}", handle):
            raise ValueError(f"invalid X handle: {handle}")
        key = handle.lower()
        if key in self.accounts:
            raise ValueError(f"duplicate X account: {handle}")
        self.accounts[key] = {**account, "source": f"x:{key}", "role": role, "team": team,
                              "url": f"https://x.com/{handle}", "reviewed_at": self.data["reviewed_at"]}

    @classmethod
    def load(cls, path=DEFAULT_REGISTRY):
        return cls(json.loads(Path(path).read_text()))

    def for_game(self, game):
        teams = {TEAM_ALIASES.get(t, t) for t in (game.home_team, game.away_team)}
        missing = teams - TEAM_CODES
        return {"teams": sorted(teams), "media": list(self.media.values()),
                "x_accounts": [a for a in self.accounts.values() if a["team"] in teams or a["role"] == "insider"],
                "missing_teams": sorted(missing), "reviewed_at": self.data["reviewed_at"]}

    def role(self, source):
        if source == "nba_injury_report":
            return "official_report"
        if source in self.media:
            return "authority_media"
        if source.startswith("x:"):
            return self.accounts.get(source[2:].lower(), {}).get("role", "unclassified")
        return "unclassified"

    def priority(self, source):
        return PRIORITIES[self.role(source)]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--registry", type=Path, default=DEFAULT_REGISTRY)
    ap.add_argument("--teams", nargs=2, metavar=("HOME", "AWAY"))
    args = ap.parse_args()
    registry = NewsRegistry.load(args.registry)
    value = registry.for_game(SimpleNamespace(home_team=args.teams[0], away_team=args.teams[1])) if args.teams else registry.data
    print(json.dumps(value, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
