# Skill: rate basis

Question shape: a player's scoring rate per game and per 100 possessions over the same games.

Tables:
- `player_regular`: PTS per player-game
- `advanced_regular`: POSS per player-game (the player's possessions while on the floor)

Steps:
1. Select the player's regular-season games in the window (whole season if no window is given).
2. Points per game = sum(PTS) / number of games.
3. Points per 100 possessions = sum(PTS) / sum(POSS) * 100, over the same games.
4. Report the two numbers as two different quantities, with the game count and game ids.

Never:
- compare per-game with per-100 as if they were the same unit
- report a rate without the number of games it covers
