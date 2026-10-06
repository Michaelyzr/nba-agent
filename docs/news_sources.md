# Pregame news sources

The machine-readable configuration is [`backend/data_sources/news_sources.json`](../backend/data_sources/news_sources.json).
Account identities and affiliations were reviewed on 2026-10-06. This is a
maintained source list, not a guarantee that every account will post every game.
Recheck the linked bios/publication pages when reporters change assignments.

Each game polls ESPN and CBS Sports NBA RSS, the NBA injury report, Shams, and
only the two playing teams' official/PR and reporter accounts. All 30 official
handles are linked in [NBA.com's schedule-release roundup](https://www.nba.com/news/nba-teams-release-schedules-on-social-media).
[Shams Charania](https://x.com/ShamsCharania) is tracked as an insider; his
[ESPN author page](https://www.espn.com/contributor/shams-charania/) records his affiliation.

| Source | Retrieval | Identity / feed reference |
| --- | --- | --- |
| ESPN | NBA RSS | [ESPN RSS information](https://www.espn.com/espn/news/story?page=rssinfo) |
| CBS Sports | NBA RSS | [CBS feed directory](https://www.cbssports.com/xml/rss) |
| NBA | Latest matching injury-report PDF | [NBA official injury reporting](https://official.nba.com/nba-injury-report-2025-26-season/) |
| Selected X accounts | Official recent-search API | [X API documentation](https://docs.x.com/x-api/posts/search/quickstart/recent-search) |

## Team accounts

The reference column links one review source. The JSON retains additional
verification links and assignment notes. Sam Perley and Jim Eichenhofer are
team-employed writers, explicitly marked `team_affiliated`; their personal
accounts remain in the reporter tier and are not counted as independent
confirmation of their teams' official announcements.

| Team | Official X | Reporter / team writer | Outlet | Review reference |
| --- | --- | --- | --- | --- |
| ATL — Atlanta Hawks | [@ATLHawks](https://x.com/ATLHawks) | Lauren Williams · [@WilliamsLaurenL](https://x.com/WilliamsLaurenL) | Atlanta Journal-Constitution | [Reference](https://www.ajc.com/staff/lauren-williams/) |
| BOS — Boston Celtics | [@celtics](https://x.com/celtics) | Souichi Terada · [@SouichiTerada](https://x.com/SouichiTerada) | MassLive | [Reference](https://muckrack.com/souichi-terada) |
| BKN — Brooklyn Nets | [@BrooklynNets](https://x.com/BrooklynNets) | Brian Lewis · [@NYPost_Lewis](https://x.com/NYPost_Lewis) | New York Post | [Reference](https://x.com/NYPost_Lewis/with_replies) |
| CHA — Charlotte Hornets | [@hornets](https://x.com/hornets) | Sam Perley · [@sam_perley](https://x.com/sam_perley) | Hornets.com | [Reference](https://cdn.nba.com/teams/uploads/sites/1610612766/2026/04/2025-26_CHA_MediaGuide_NEW.pdf) |
| CHI — Chicago Bulls | [@chicagobulls](https://x.com/chicagobulls) | K.C. Johnson · [@KCJHoop](https://x.com/KCJHoop) | Chicago Sports Network | [Reference](https://www.chsn.com/videos/bulls-mailbag-with-k-c-johnson-1736973789275) |
| CLE — Cleveland Cavaliers | [@cavs](https://x.com/cavs) | Chris Fedor · [@ChrisFedor](https://x.com/ChrisFedor) | cleveland.com / The Plain Dealer | [Reference](https://muckrack.com/chris-fedor) |
| DAL — Dallas Mavericks | [@dallasmavs](https://x.com/dallasmavs) | Mike Curtis · [@MikeACurtis2](https://x.com/MikeACurtis2) | Dallas Morning News | [Reference](https://www.reddit.com/r/Mavericks/comments/1ccqpno/) |
| DEN — Denver Nuggets | [@nuggets](https://x.com/nuggets) | Vinny Benedetto · [@VBenedetto](https://x.com/VBenedetto) | Denver Gazette | [Reference](https://gazette.com/author/vinny-benedetto-vinny-benedettogazette-com/) |
| DET — Detroit Pistons | [@DetroitPistons](https://x.com/DetroitPistons) | Omari Sankofa II · [@omarisankofa](https://x.com/omarisankofa) | Detroit Free Press | [Reference](https://omny.fm/shows/the-pistons-pulse/pistons-pulse-offseason-special-sankofa-and-son-and-bryce) |
| GSW — Golden State Warriors | [@warriors](https://x.com/warriors) | Dalton Johnson · [@DaltonJ_Johnson](https://x.com/DaltonJ_Johnson) | NBC Sports Bay Area | [Reference](https://x.com/DaltonJ_Johnson/status/2040981959598559718) |
| HOU — Houston Rockets | [@HoustonRockets](https://x.com/HoustonRockets) | Will Guillory · [@WillGuillory](https://x.com/WillGuillory) | The Athletic | [Reference](https://www.chron.com/sports/rockets/article/will-guillory-pelicans-rockets-22352753.php) |
| IND — Indiana Pacers | [@Pacers](https://x.com/Pacers) | Tony East · [@TonyREast](https://x.com/TonyREast) | Circle City Spin / Forbes | [Reference](https://x.com/TonyREast/with_replies) |
| LAC — LA Clippers | [@LAClippers](https://x.com/LAClippers) | Law Murray · [@LawMurrayTheNU](https://x.com/LawMurrayTheNU) | The Athletic | [Reference](https://x.com/LawMurrayTheNU/status/2043118150146023757) |
| LAL — Los Angeles Lakers | [@Lakers](https://x.com/Lakers) | Dan Woike · [@DanWoikeSports](https://x.com/DanWoikeSports) | The Athletic | [Reference](https://x.com/DanWoikeSports/with_replies) |
| MEM — Memphis Grizzlies | [@memgrizz](https://x.com/memgrizz) | Damichael Cole · [@DamichaelC](https://x.com/DamichaelC) | The Commercial Appeal | [Reference](https://x.com/DamichaelC/with_replies) |
| MIA — Miami Heat | [@MiamiHEAT](https://x.com/MiamiHEAT) | Anthony Chiang · [@Anthony_Chiang](https://x.com/Anthony_Chiang) | Miami Herald | [Reference](https://www.miamiherald.com/sports/nba/miami-heat/article239370443.html) |
| MIL — Milwaukee Bucks | [@Bucks](https://x.com/Bucks) | Eric Nehm · [@eric_nehm](https://x.com/eric_nehm) | The Athletic | [Reference](https://muckrack.com/eric-nehm) |
| MIN — Minnesota Timberwolves | [@Timberwolves](https://x.com/Timberwolves) | Jon Krawczynski · [@JonKrawczynski](https://x.com/JonKrawczynski) | The Athletic | [Reference](https://bsky.app/profile/jonkrawczynski.bsky.social) |
| NOP — New Orleans Pelicans | [@PelicansNBA](https://x.com/PelicansNBA) | Jim Eichenhofer · [@Jim_Eichenhofer](https://x.com/Jim_Eichenhofer) | Pelicans.com | [Reference](https://www.nba.com/pelicans/staff) |
| NYK — New York Knicks | [@nyknicks](https://x.com/nyknicks) | Ian Begley · [@IanBegley](https://x.com/IanBegley) | SNY | [Reference](https://sny.tv/authors/ian-begley) |
| OKC — Oklahoma City Thunder | [@okcthunder](https://x.com/okcthunder) | Clemente Almanza · [@CAlmanza1007](https://x.com/CAlmanza1007) | Thunder Wire | [Reference](https://x.com/CAlmanza1007/with_replies) |
| ORL — Orlando Magic | [@OrlandoMagic](https://x.com/OrlandoMagic) | Jason Beede · [@therealBeede](https://x.com/therealBeede) | Orlando Sentinel | [Reference](https://www.si.com/nba/magic/onsi/instant-grade-orlando-magic-sign-malaki-branham) |
| PHI — Philadelphia 76ers | [@sixers](https://x.com/sixers) | Adam Aaronson · [@SixersAdam](https://x.com/SixersAdam) | PhillyVoice | [Reference](https://www.phillyvoice.com/sixers-news-analysis-lebron-james-reaction-nba-press-conference-release-mike-gansey-josh-harris-bob-myers/) |
| PHX — Phoenix Suns | [@Suns](https://x.com/Suns) | Duane Rankin · [@DuaneRankin](https://x.com/DuaneRankin) | Arizona Republic / azcentral | [Reference](https://x.com/DuaneRankin/status/2013829795696476301) |
| POR — Portland Trail Blazers | [@trailblazers](https://x.com/trailblazers) | Sean Highkin · [@highkin](https://x.com/highkin) | Rose Garden Report | [Reference](https://www.rosegardenreport.com/) |
| SAC — Sacramento Kings | [@SacramentoKings](https://x.com/SacramentoKings) | Jason Anderson · [@JandersonSacBee](https://x.com/JandersonSacBee) | Sacramento Bee | [Reference](https://www.sacbee.com/profile/221350135/) |
| SAS — San Antonio Spurs | [@spurs](https://x.com/spurs) | Tom Orsborn · [@tom_orsborn](https://x.com/tom_orsborn) | San Antonio Express-News | [Reference](https://www.expressnews.com/author/tom-orsborn/) |
| TOR — Toronto Raptors | [@Raptors](https://x.com/Raptors) | Josh Lewenberg · [@JLew1050](https://x.com/JLew1050) | TSN | [Reference](https://x.com/JLew1050/with_replies) |
| UTA — Utah Jazz | [@utahjazz](https://x.com/utahjazz) | Ryan Miller · [@millerjryan](https://x.com/millerjryan) | KSL | [Reference](https://www.ksl.com/author/Ryan_Miller) |
| WAS — Washington Wizards | [@WashWizards](https://x.com/WashWizards) | Josh Robbins · [@JoshuaBRobbins](https://x.com/JoshuaBRobbins) | The Athletic | [Reference](https://www.reddit.com/r/OrlandoMagic/comments/1o8e6ks/) |

Additional verified PR accounts: [@HawksPR](https://x.com/HawksPR),
[@HornetsPR](https://x.com/HornetsPR), [@WarriorsPR](https://x.com/WarriorsPR).
Will Guillory is assigned to HOU in this list after his July 2026 move from NOP.

## Conflicts and coverage

Priority for an explicit supported player factor is:
NBA injury report → team official/PR → ESPN/CBS → Shams → team reporter → other.
The latest publication wins within the same tier. A newer lower-tier conflict
is saved and flagged pending; the selected authoritative status remains until
that tier publishes a correction. Without a primary report, personal reports
can update the model but are marked `secondary_only`.

A successful request is not proof that all important news was published or
retrieved. Snapshots record per-source success, missing credentials, disabled
sources, failures, rate-limit cooldowns and unfinished pagination. ESPN/CBS
adapters read RSS titles and summaries, not full article bodies. Reports not
in those feeds may be missed. Original relevant posts/articles and images are
stored in `evidence.jsonl`; ambiguous statements and image-only lineups do not
create numerical model adjustments. Personal X posts are filtered by the
configured author identity returned by X, not an author name appearing in text.

Set `X_BEARER_TOKEN` locally with recent-search access before live use. The
reader does not follow accounts, post to X or scrape logged-in pages. Recent
search covers at most seven days. Polling defaults to 60 seconds, allows 30
seconds for indexing and overlaps five minutes; final updates can still be
unavailable before the pregame loop stops at tip-off. Tokens are never persisted.

Inspect the plan with `python -m data_sources.news_registry --teams BOS LAL`.
Update the JSON (or use `--news-registry`) when accounts or assignments change.
See the [pregame runner instructions](project-plan.md#pregame-news-polling-and-fair-odds)
for live/replay commands and output files.
