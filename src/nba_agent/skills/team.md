# Skill: team window

Question shape: a team's net rating before and after a stated date.

Tables:
- `team_regular`: one row per team-game (GAME_ID, GAME_DATE, TEAM_ID, PTS, OPP_PTS, POSS)

Steps:
1. Before window: the team's regular-season games with GAME_DATE < the stated date.
2. After window: GAME_DATE >= the stated date.
3. Net rating for a window = (sum(PTS) - sum(OPP_PTS)) / sum(POSS) * 100. Compute from totals.
4. Report both windows with game count and game ids.

Never:
- include playoff games
- average per-game net ratings
