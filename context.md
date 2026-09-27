# Skill: context

Question shape: what did a saved report say about this player in a date window?

Tables:
- `paragraphs`: PARA_ID, DATE, PLAYER_ID, KIND (injury, trade, role), SOURCE, URL, TEXT
- `player_regular`: for any number the note states (for example games played in the window)

Steps:
1. Filter `paragraphs` to the player and the date window (whole season if none).
2. Rank the remaining paragraphs against the question and keep the top five.
3. Quote only text that appears word for word in a kept paragraph. Cite SOURCE, DATE, URL.
4. If the note states a number outside a quote, it must come from the box score (games in the window).

Never:
- invent or paraphrase a quote inside quotation marks
- use a paragraph about another player or outside the window
- turn a number in an article into a statistic
- forecast an injury
