# Skill: trade split

Question shape: did a player's true shooting change after he changed teams?

Tables:
- `trades`: PLAYER_ID, TRADE_DATE, FROM_TEAM_ID, TO_TEAM_ID
- `player_regular`: one row per player-game (GAME_ID, GAME_DATE, PLAYER_ID, TEAM_ID, PTS, FGA, FTA)

Steps:
1. Look up the player's row in `trades`. Use its TRADE_DATE as the split.
2. Before window: regular-season games with GAME_DATE < TRADE_DATE and TEAM_ID == FROM_TEAM_ID.
3. After window: regular-season games with GAME_DATE >= TRADE_DATE and TEAM_ID == TO_TEAM_ID.
4. True shooting for a window = sum(PTS) / (2 * (sum(FGA) + 0.44 * sum(FTA))). Compute it from totals, not as an average of per-game values.
5. Report both windows with their game count and game ids.

Never:
- read `player_playoffs` for a regular-season question
- charge a game to a team the player was not on that night
