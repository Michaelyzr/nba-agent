# Skill: signing pay

Question shape: what annual salary would a team have to offer to sign this player?

Feature row (regular season only):
- MPG: minutes per game
- PPG: points per game
- TS: true shooting from season totals
- USG: mean usage from `advanced_regular`
- GP: games played
- AGE: from `players`
- PRIOR_PAY: last annual pay from `contracts`, if present. If missing, use 0 and set HAS_PRIOR = 0.

Steps:
1. Build the feature row. Reject if the player has no regular-season games.
2. Run the signing network. The output is predicted annual pay in dollars.
3. Attach the held-out mean absolute error from `models/signing/metrics.json`.

The note must:
- call the figure a prediction
- state the held-out error
- list the features that went in

Never:
- say the team has cap room, or quote a cap number
- recommend signing the player
