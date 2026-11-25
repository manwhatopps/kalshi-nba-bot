# QuantSport resource evaluation

Checked 2026-11-24 from the build sandbox. Every row was inspected directly (GitHub search API, raw READMEs,
PyPI JSON, vendor docs, test downloads). Anything that could not be fetched says **not verified**.

**Verdicts.** INTEGRATE = use as a dependency or data source as is. ADAPT = use part of it, or use it behind a
wrapper or restriction. BENCHMARK = do not depend on it; compare our output against it. REJECT = do not use.

**Profitability claims.** None of the projects below has a profitability claim that we verified. Treat every
accuracy, ROI or "edge" figure in a README as unverified until it is reproduced in `python -m bot walk`.

**Sandbox reachability (tested with curl).** Reachable: `docs.kalshi.com`, `help.kalshi.com`,
`api.elections.kalshi.com`, `demo-api.kalshi.co`, `huggingface.co` (+ `us.aws.cdn.hf.co`), `pypi.org`,
`raw.githubusercontent.com`, `github.com/.../releases/download/...`, `api.the-odds-api.com` (401 without key).
Not reachable from the build sandbox: `stats.nba.com`, `cdn.nba.com` (403), `collegefootballdata.com`,
`api.collegefootballdata.com`, `the-odds-api.com` (website), `sportsgameodds.com`, `oddsjam.com`,
`opticodds.com`, `developer.sportradar.com`, `site.api.espn.com`, `kaggle.com`, `external-api.kalshi.com`,
`nflreadpy.nflverse.com`. `kalshi.com/docs/kalshi-fee-schedule.pdf` returned 429. Prices marked "search
snippet" come from a web-search summary of the vendor page, not from the page itself.

---

## A. Kalshi

| Item | What it does / what we reuse | Key deps | Licence | Maintenance | Data, keys, cost | Difficulty | Known limitations | Verdict |
|---|---|---|---|---|---|---|---|---|
| [docs.kalshi.com](https://docs.kalshi.com) + [openapi.yaml](https://docs.kalshi.com/openapi.yaml) + [asyncapi.yaml](https://docs.kalshi.com/asyncapi.yaml) | REST spec v3.37.0 (13,190 lines), WebSocket spec v2.0.0, guides as `.md` (index: `/llms.txt`). Reuse: the spec as the contract for `bot/kalshi.py`; a test that diffs our paths and required fields against it. | none | Kalshi developer terms (`/getting_started/terms`, not read) | Changelog page; SDKs "published Tuesday-Wednesday each week" | Free. Public market data needs no key; trading needs an RSA or Ed25519 key | low | Spec title says "manually defined ... being migrated"; some operations are `x-excluded` (TP/SL orders, RFQ batches) and not live | **INTEGRATE** - it is the only source of truth; Kalshi itself says SDKs may lag it |
| [Kalshi/kalshi-starter-code-python](https://github.com/Kalshi/kalshi-starter-code-python) | Example of RSA-PSS request signing, a REST wrapper (balance, status, trades) and a WebSocket ticker subscription. "This is not an SDK." Reuse: nothing new; our client already signs the same way. | requests, cryptography, websockets | none (no LICENSE file) | 102 stars, last push 2025-03-07 | Free; key for auth calls | low | No orders, no RFQ, no combos; pinned 2025 versions | **BENCHMARK** - signing and WS-handshake reference only |
| [PyPI kalshi-python](https://pypi.org/project/kalshi-python/) | Old official generated client. | urllib3, pydantic, cryptography, python-dateutil | LicenseRef-Proprietary | Published by "Kalshi Support <support@kalshi.com>"; last release 2.1.4 on 2025-09-06; docs call it deprecated | Free | n/a | Wheel inspected: **no `/portfolio/events/orders`**; it only has the legacy `/portfolio/orders` write paths, which now return `410 Gone`. RFQ: create/list/delete and the old `/communications/quotes/{id}/accept` only; still calls the removed `.../lookup` path | **REJECT** - deprecated and cannot place an order today |
| [PyPI kalshi_python_sync](https://pypi.org/project/kalshi_python_sync/) / kalshi_python_async | Current official generated clients. Wheel 3.36.0 inspected: has `/portfolio/events/orders`, RFQ-scoped accept/confirm, block trades, multivariate create. | urllib3, pydantic; **Python >= 3.13** (our .venv is 3.13.18) | LicenseRef-Proprietary | 3.36.0 on 2026-11-11, 37 releases since 2025-12-02; one minor behind spec 3.37.0 | Free | med | Generated code, proprietary licence, lags the spec by design; Kalshi recommends generating your own client or integrating directly for production | **BENCHMARK** - cross-check our request shapes against it; keep `bot/kalshi.py` |
| [TexasCoding/kalshi-python-sdk](https://github.com/TexasCoding/kalshi-python-sdk) (PyPI name is **`kalshi-sdk`**) | Third-party typed SDK: sync + async REST, 13 typed WebSocket channels, FIX, perps. README claims 144 of 145 operations of OpenAPI 3.37.0 and V2 orders as "the only order-write surface" (claim not tested by us). | httpx, pydantic v2, cryptography, websockets; Python >= 3.12 | MIT | 10 stars, last push 2026-11-24; `kalshi-sdk` 22.16.0 on 2026-11-24, 61 releases since 2026-05-10 | Free | med | One maintainer, very fast version churn, few users. PyPI **`kalshi-python-sdk`** is a different publisher (NaxosData, one release 2026-11-01) - do not install by repo name | **BENCHMARK** - candidate only if we ever need WebSocket streaming; pre-game-only does not |
| Help: [Combos](https://help.kalshi.com/en/articles/13823820-combos), [Market FAQs](https://help.kalshi.com/en/articles/13823821-market-faqs) | The retail rules for combos: payout, RFQ pricing, settlement, DNP. Reuse: encode them in `bot/sim_env.py` settlement and in slip explanations. | none | n/a | Both fetched 2026-11-24 (HTTP 200) | Free | low | Retail wording; "Protected Player Markets" page referenced but not fetched; contract terms per market govern | **INTEGRATE** - the settlement rules below must be in the simulator |

### A.1 API facts read from the spec and guides (v3.37.0)

- **Hosts.** Spec lists `external-api.kalshi.com` as production and `api.elections.kalshi.com` as "shared, also
  supported"; demo is `external-api.demo.kalshi.co` / `demo-api.kalshi.co`.
- **Order writes are V2 only.** `POST /portfolio/events/orders` (create), `DELETE /portfolio/events/orders`
  (cancel all), `POST|DELETE /portfolio/events/orders/batched`, `DELETE /portfolio/events/orders/{order_id}`,
  `POST .../{order_id}/amend`, `POST .../{order_id}/decrease`. The legacy `/portfolio/orders` create, cancel,
  amend, decrease and batch forms "have been removed and now return `410 Gone`"; only the `GET` reads remain
  there (`/portfolio/orders`, `/{order_id}`, `/queue_positions`).
- **V2 request shape.** Required: `ticker`, `side` (`bid` or `ask`, where `bid` = yes and `ask` = no),
  `count` (string, 2 decimals, fractional contracts allowed down to 0.01), `price` (fixed-point dollar string,
  up to 4 decimals), `time_in_force` (`fill_or_kill`, `good_till_canceled`, `immediate_or_cancel`),
  `self_trade_prevention_type`. A `201` means accepted, not resting: check `remaining_count`. Valid prices come
  from the market's `price_ranges`, not from an assumed 1-cent grid.
- **Combos (multivariate).** `GET /multivariate_event_collections`, `GET /multivariate_event_collections/{t}`,
  `POST /multivariate_event_collections/{t}` (create the combo market; required before it can be quoted or
  traded; same legs return the existing market), `GET /events/multivariate`; `GET /markets` takes `mve_filter`.
  The old `.../lookup` path is gone. Limit: 5,000 new combo markets per user per rolling 7 days. Rejections
  come back as `400` with `conflicting_leg_outcomes`, `duplicated_legs`, `invalid_market_combination` or
  `invalid_parameters`; legs are re-validated on every call. Live check: the open sports collection
  `KXMVESPORTSMULTIGAMEEXTENDED-R` has `size_min` 2 and `size_max` 0.
- **RFQ.** `POST|GET /communications/rfqs`, `GET|DELETE /communications/rfqs/{rfq_id}`,
  `PUT /communications/rfqs/{rfq_id}/quotes/{quote_id}/accept` (body `accepted_side`: `yes` or `no`),
  `.../confirm` (maker), plus `/communications/quotes...` for makers. `rest_remainder` is required on create;
  size is exactly one of `contracts_fp` or `target_cost_dollars` (which by default caps principal plus fees).
  About 100 open RFQs per subaccount; an open RFQ closes after about 3 hours. All combo markets and all
  `KXNBA`/`KXNFL`/`KXMLB` events are "high volatility": maker confirmation window 3 s, execution timer 1 s.
  The fill can come from a different maker than the quote you named, at the same price. Creating many RFQs
  that are never accepted raises the token cost of creating RFQs. RFQ batches exist in the spec but are hidden.
- **WebSocket.** `wss://external-api-ws.kalshi.com/trade-api/ws/v2` (shared host also works). **The handshake
  itself must be signed**, even for public channels (`ticker`, `trade`, `market_lifecycle_v2`,
  `multivariate_market_lifecycle`). Quotes arrive on the `communications` channel. 200 concurrent connections
  on Basic.
- **Rate limits.** Token buckets, not requests per second. Default cost 10 tokens; cancel and single-order read
  2; batch create 10 per order (no saving). Budgets in tokens per second, read / write: Basic 200 / 100,
  Advanced 300 / 300, Expert 600 / 600, Premier 1,200 / 1,200, Paragon 2,400 / 2,400, Prime 4,800 / 4,800,
  Prestige 12,000 / 9,600. Basic is therefore about 20 reads and 10 order writes per second. Advanced is a free
  API call once one of your last 100 orders came through the API. `429` carries no `Retry-After`.
  `GET /portfolio/balance` has its own bucket: once per second on Basic and Advanced.
- **Fees.** Series carry `fee_type` (`quadratic`, `quadratic_with_maker_fees`,
  `quadratic_with_combo_maker_fees`, `flat`) and `fee_multiplier`; events can override both
  (`fee_type_override`, `fee_multiplier_override`); scheduled changes are at `/series/fee_changes` and
  `/events/fee_changes`. Fills report `fee_cost`. Live check 2026-11-24: `KXNBAGAME` and `KXNCAAFGAME` are
  `quadratic_with_maker_fees`, the combo series is `quadratic_with_combo_maker_fees` (spec: maker multiplier
  0.5 instead of 0.25), the 12 other `KXNCAAF*` series sampled are `quadratic`. Trade fee is rounded up to $0.000001,
  then a rounding fee aligns the balance to $0.0001 (direct members) or $0.01 (FCM-cleared), with a per-order
  accumulator and rebate. The 0.07 coefficient
  lives in the fee-schedule PDF, which we could not fetch: **not verified**.
- **History.** Old markets, trades, orders and fills move to `/historical/*`; cutoffs at `GET /historical/cutoff`.

### A.2 Combo rules (from the two help pages)

- A combo is its own market with its own order book. Payout per contract = **product of the settlement values
  of the legs**, maximum $1.00. All legs at $1.00 pays $1.00; any leg at $0 pays $0.
- Price comes only from the RFQ system. No maker quote means no fill ("payout unavailable"). Quotes can move
  between viewing and execution.
- Eligibility: events are added on a rolling basis and usually appear near start time; the FAQ says combo
  buttons typically appear when lineups are finalised. The combo builder (or the collections endpoint) is the
  list; there is no fixed public list.
- Exit: "Once a combo is placed and filled, it cannot be canceled or reversed by the user." Plan every combo as
  held to settlement.
- **DNP does not void a leg and does not refund the combo.** A player who never enters the game makes that leg
  settle under the underlying market's rules, generally at "its last fair price as determined by the Exchange",
  a value between $0 and $1. Example given: legs at $0.70, $1.00, $1.00 pay $0.70. Once the player has entered
  the game (NFL: one play), the leg settles on actual statistics regardless of injury or ejection.
- Protected NFL and NBA player markets differ: played the first half, not after halftime, target not reached ->
  settles at the pre-game fair price.
- Settlement: after all legs are determined, or immediately at $0 once any leg is $0; typically 1 to 12 hours
  after the last leg, through a separate review.

## B. NBA

| Item | What it does / what we reuse | Key deps | Licence | Maintenance | Data, keys, cost | Difficulty | Known limitations | Verdict |
|---|---|---|---|---|---|---|---|---|
| [swar/nba_api](https://github.com/swar/nba_api) | Client for 130+ `stats.nba.com` endpoints (139 documented) and the `cdn.nba.com` live feeds; returns DataFrames. Reuse: the endpoint classes, already in `requirements.txt` as an optional path. | pandas, requests | MIT | 3,902 stars, last push 2026-11-11, PyPI 1.12.0 on 2026-10-24 | Free, no key | low (home connection) / not possible (cloud) | README: stats.nba.com "throttles rapid requests and does not respond to many cloud-hosting IP ranges"; advice is `time.sleep(1)`, longer timeout, or a proxy. No published rate limit. Both NBA hosts are not reachable from the build sandbox. Unofficial API; endpoints get replaced (v2 -> v3) | **ADAPT** - home-connection fetcher only, cached to parquet; the Hugging Face game logs stay the daily source |

Endpoints that matter (all present in the docs index): player logs `playergamelogs`, `playergamelog`,
`leaguegamelog`; box scores `boxscoretraditionalv3`, `boxscoreadvancedv3`, `boxscoreusagev3`; tracking
`leaguedashptstats`, `boxscoreplayertrackv3`, `playerdashptpass`, `playerdashptreb`; matchups and defence
`leagueseasonmatchups`, `matchupsrollup`, `boxscorematchupsv3`, `leaguedashptdefend`, `playerdashptshotdefend`;
shot profiles `shotchartdetail`, `leaguedashplayershotlocations`, `playerdashptshots`, `synergyplaytypes`;
context `leaguehustlestatsplayer`, `leaguedashlineups`, `commonteamroster`, `scoreboardv3`, `playbyplayv3`.

## C. NFL

| Item | What it does / what we reuse | Key deps | Licence | Maintenance | Data, keys, cost | Difficulty | Known limitations | Verdict |
|---|---|---|---|---|---|---|---|---|
| [nflverse/nflverse-data](https://github.com/nflverse/nflverse-data) | Release assets (parquet/csv/rds) per data type. Reuse: direct `pd.read_parquet(url)`, as `bot/nfl_scenarios.py` already does. | none (files) | CC-BY-4.0 | 448 stars, last push 2026-11-01; assets rebuilt daily (see dates below) | Free, no key | low | Tag and file names change (see `player_stats`); participation arrives after the season | **INTEGRATE** - already the NFL source; extend the tag list |
| [nflverse/nflreadpy](https://github.com/nflverse/nflreadpy) | Python loaders (`load_pbp`, `load_player_stats`, `load_snap_counts`, `load_injuries`, `load_depth_charts`, `load_nextgen_stats`, `load_participation`, `load_schedules`, ...) with caching. Reuse: its URL map as the catalogue of names. | **polars**, pydantic, pydantic-settings, requests, tqdm | MIT | 253 stars, last push 2026-10-30; PyPI 0.1.5 on 2025-11-19 (a year old) | Free | low | Returns Polars frames; adds five dependencies to replace a one-line parquet read; version 0.1.x | **ADAPT** - copy the tag/file naming, keep direct parquet reads, no Polars |
| [nflverse/nflverse-pbp](https://github.com/nflverse/nflverse-pbp) | The R build jobs that produce pbp and player stats for nflverse-data. | R | CC-BY-4.0 | 360 stars, last push 2026-09-11 | n/a | high | A pipeline, not a consumer interface | **REJECT** - we consume its output through nflverse-data |
| [nflverse/nfl_data_py](https://github.com/nflverse/nfl_data_py) | Old Python loader. | pandas | MIT | **Archived** (GitHub flag true); README: "deprecated in favour of nflreadpy"; last release 0.3.3 on 2024-09-20 | n/a | n/a | No further updates planned | **REJECT** - confirmed deprecated and archived |

Release tags tested 2026-11-24 (`https://github.com/nflverse/nflverse-data/releases/download/<tag>/<file>`):

| Tag | File tested | Result | Asset last modified |
|---|---|---|---|
| `pbp` | `play_by_play_2026.parquet` | OK | 2026-11-24 |
| `stats_player` | `stats_player_week_2026.parquet` | OK | 2026-11-24 |
| `player_stats` (legacy) | `player_stats_2024.parquet` OK; `player_stats_2025` and `_2026` **404** | stops at 2024 | - |
| `stats_team` | `stats_team_week_2026.parquet` | OK | not checked |
| `snap_counts` | `snap_counts_2026.parquet` | OK | 2026-11-24 |
| `injuries` | `injuries_2026.parquet` | OK | 2026-11-24 |
| `depth_charts` | `depth_charts_2026.parquet` | OK | 2026-11-24 |
| `schedules` | `games.parquet` | OK | 2026-11-24 |
| `pbp_participation` | `pbp_participation_2025.parquet` OK (2026-02-10); `_2026` **404** | prior season only | 2026-02-10 |
| `nextgen_stats` | `ngs_receiving.parquet` (also `ngs_passing`, `ngs_rushing` by naming; not tested) | OK | 2026-11-24 |
| `rosters`, `weekly_rosters`, `players`, `ftn_charting` | `roster_2026`, `roster_weekly_2026`, `players`, `ftn_charting_2026` | OK | not checked |

## D. College football

| Item | What it does / what we reuse | Key deps | Licence | Maintenance | Data, keys, cost | Difficulty | Known limitations | Verdict |
|---|---|---|---|---|---|---|---|---|
| [CollegeFootballData API](https://collegefootballdata.com) | Games, lines, plays, drives, player and team stats, ratings, recruiting. | none | Terms page not read. Search snippet: stored raw responses may not be shared with third parties | API v2 repo `CFBD/cfb-api-v2` pushed 2026-11-18 | **Key required** (Bearer token; README: "Request an API key at CollegeFootballData.com"). Free tier 1,000 calls/month; Patreon tiers $1 = 5,000, $5 = 30,000, $10 = 75,000 calls/month - search snippets, the $ figures from an older blog post: **not verified** | med | Site and API not reachable from the build sandbox; monthly call caps | **ADAPT** - optional keyed enrichment (lines history, player usage); not on the v1 critical path |
| [github.com/CFBD](https://github.com/CFBD): [cfbd-python](https://github.com/CFBD/cfbd-python) (PyPI `cfbd`) | Generated Python client for the API above. Other org repos: `cfb-api` (64 stars, MIT, 2025-02-21), `cfb-api-v2`, `cfbd-typescript`, `cfbd-net`, `cbbd-python` (basketball). | urllib3, pydantic | PyPI metadata MIT; no LICENSE detected on GitHub | 222 stars, last push 2026-11-14; `cfbd` 5.33.3 on 2026-11-14 (196 releases) | Needs the CFBD key | low once a key exists | Useless without key and network access | **ADAPT** - the client to use if the owner supplies a key; do not hand-write CFBD calls |
| [sportsdataverse/cfbfastR-data](https://github.com/sportsdataverse/cfbfastR-data) | Committed files over raw.githubusercontent.com: CFBD-shaped schedules with results and Elo, team info, play-level player stats, betting lines. Reuse: schedules + results + pregame Elo for NCAAF models. | none (files) | **none stated** (data originates from CFBD and ESPN; their terms apply) | 20 stars, last commit 2026-11-23 ("Updated 2026-11-23 ... cfbfastR 3.1.0") | Free, no key | low | README: `pbp/` (2002-2021), `teams/`, `data/parquet/` are frozen. 2026 schedule file is FBS home games only (895 rows; 2024 and 2025 files hold all divisions, about 3,800). `cfb/schedules/.../cfb_schedule_2026.parquet` is empty. Betting file ends with season 2025 | **INTEGRATE** - keyless results and Elo, tested; internal research use only until the licence is clear |
| [sportsdataverse-data](https://github.com/sportsdataverse/sportsdataverse-data) `espn_cfb_*` release tags | ESPN-derived parquet per season: pbp with EPA/WP (509 columns), schedules, team box, player box, closing spread and total, injuries. Reuse: features and a reference spread per game. | none (files) | repo MIT; ESPN-sourced data | 15 stars, last push 2026-11-21; `timestamp.json` per tag shows updates 2026-11-22 to 2026-11-24 | Free, no key | low | ESPN IDs and names (need a crosswalk to Kalshi team names); 2026 schedule file holds completed games only (822); no moneyline in `betting_2026`; young project, small user base | **INTEGRATE** - current-season pbp, box scores and a spread/total reference with no key |
| HF [24601p/cfb-data](https://huggingface.co/datasets/24601p/cfb-data) | `games-2025.parquet` (3,831 rows), `games-2026.parquet` (3,758 rows, all divisions), `drives-*`, `teams-2026.csv`; CFBD column set (points, line scores, pregame Elo). | none | none stated, no dataset card | last modified 2026-11-23; 1,030 downloads | Free, no key | low | Anonymous publisher, no documentation, could vanish | **BENCHMARK** - cross-check results and fill the non-FBS gap; not a primary source |
| HF [SmartStake/cfb-mainlines](https://huggingface.co/datasets/SmartStake/cfb-mainlines) | Every pregame price change on NCAAF moneyline, spread, total at 12 venues **including Kalshi and Polymarket** (minute-level best bettable price) for 1,067 games, 2026-08-27 to 2026-10-31; 172.7M rows, about 1 GB. | none | CC-BY-4.0 | last modified 2026-11-17; one-off study dataset | Free, no key | med (size; per-day files) | No results; ends 2026-10-31; raw odds, not de-vigged; Kalshi offers under $100 ignored | **ADAPT** - reference-price history for NCAAF backtests, joined to results from the files above |

No-key downloads that returned HTTP 200 on 2026-11-24 (full GET; columns read with pyarrow):

| URL | Rows | Columns (abridged) |
|---|---|---|
| `https://raw.githubusercontent.com/sportsdataverse/cfbfastR-data/main/schedules/parquet/cfb_schedules_2026.parquet` (also `_2025`, `_2024`; csv at `schedules/csv/cfb_schedules_2026.csv`) | 895 (822 completed, last 2026-11-22); 3,831; 3,801 | `game_id, season, week, season_type, start_date, completed, neutral_site, conference_game, home_id, home_team, home_division, home_conference, home_points, home_pregame_elo, home_postgame_elo, home_post_win_prob, away_*, excitement_index` |
| `https://raw.githubusercontent.com/sportsdataverse/cfbfastR-data/main/team_info/parquet/cfb_team_info_2026.parquet` | 684 | `team_id, school, mascot, abbreviation, alt_name1..3, conference, division, classification, venue_id, venue_name, city, state, timezone, latitude, longitude, elevation, capacity, grass, dome` |
| `https://raw.githubusercontent.com/sportsdataverse/cfbfastR-data/main/betting/parquet/cfb_line_odds.parquet` | 1,183,529 (seasons 2006-2025) | `game_id, season, week, season_type, date_time, market_type (spread/total/money_line), abbr, lines, odds, opening_lines, opening_odds, book, home_team_id, away_team_id` |
| `https://raw.githubusercontent.com/sportsdataverse/cfbfastR-data/main/player_stats/parquet/player_stats_2026.parquet` | 211,156 (weeks 1-12) | play-level: `game_id, week, team, opponent, play_id, down, distance, rush_player(_id), rush_yds, reception_player(_id), reception_yds, completion_player(_id), completion_yds, target_player, touchdown_player, sack_player, ...` |
| `https://raw.githubusercontent.com/sportsdataverse/cfbfastR-data/main/pbp/parquet/play_by_play_2021.parquet` | not read (range request OK) | frozen legacy pbp, 2002-2021 only |
| `https://github.com/sportsdataverse/sportsdataverse-data/releases/download/espn_cfb_schedules/cfb_schedule_2026.parquet` (also `_2025`, `_2024`) | 822; 958; 966 | `game_id, season, week, season_type, game_date, neutral_site, home_id, away_id, home_team, away_team, home_score, away_score, home_winner, venue, attendance, status` |
| `https://github.com/sportsdataverse/sportsdataverse-data/releases/download/espn_cfb_pbp/play_by_play_2026.parquet` (also `_2025`) | 142,832 (53.5 MB) | 509 columns incl. `game_id, pos_team, def_pos_team, down, distance, EPA, wpa, wp_before, wp_after, home_wp_before, rusher_player_name, passer_player_name, receiver_player_name, yds_rushed, yds_receiving` |
| `https://github.com/sportsdataverse/sportsdataverse-data/releases/download/espn_cfb_team_box/team_box_2026.parquet` | 1,644 | `game_id, team_id, team_name, home_away, firstDowns, totalYards, netPassingYards, rushingYards, rushingAttempts, turnovers, possessionTime, thirdDownEff` |
| `https://github.com/sportsdataverse/sportsdataverse-data/releases/download/espn_cfb_player_box/player_box_2026.parquet` | 70,234 | `game_id, athlete_id, athlete_name, team_id, category, passingYards, passingTouchdowns, rushingAttempts, rushingYards, receptions, receivingYards, totalTackles, sacks, adjQBR` |
| `https://github.com/sportsdataverse/sportsdataverse-data/releases/download/espn_cfb_betting/betting_2026.parquet` | 822 | `game_id, season, week, game_spread, over_under, home_favorite, home_team_spread, odds_source` |
| `https://github.com/sportsdataverse/sportsdataverse-data/releases/download/espn_cfb_injuries/injuries_2026.parquet` | 243 | `as_of_date, team_id, athlete_id, athlete_display_name, athlete_position, status, injury_date, type_name, detail_type, detail_return_date` |
| `https://huggingface.co/datasets/24601p/cfb-data/resolve/main/games-2026.parquet` (also `games-2025.parquet`) | 3,758; 3,831 | `Id, Season, Week, SeasonType, StartDate, Completed, HomeTeam, HomeClassification, HomePoints, HomeLineScores, HomePregameElo, AwayTeam, AwayPoints, ...` |
| `https://huggingface.co/datasets/SmartStake/cfb-mainlines/resolve/main/kick_day=2026-08-27/data.parquet` (one file per kickoff day) | 554,260 in this file | `matchKey, market, book, contract, side, line, ts, ts_last, start_ms, home, away, eff, del, mx` |

`.../espn_cfb_<tag>/timestamp.json` returns the last build time of each tag (for example `espn_cfb_pbp`:
2026-11-24 16:01 EST). 404s seen: `espn_cfb_schedules/schedules_2026.parquet`, `espn_cfb_betting/cfb_betting_2026.parquet`.

## E. Odds and reference prices

Execution is Kalshi-only. Outside prices are a reference to measure our probabilities and Kalshi's prices
against; they never trigger or size a trade by themselves.

| Item | What it does / what we reuse | Key deps | Licence | Maintenance | Data, keys, cost | Difficulty | Known limitations | Verdict |
|---|---|---|---|---|---|---|---|---|
| [The Odds API](https://the-odds-api.com) | Live and historical sportsbook odds as JSON; de-vig with `penaltyblog` (already a dependency). | requests | Terms on storing or redistributing data: **not verified** (site not reachable; search found no text) | `api.the-odds-api.com/v4` answers (401 `MISSING_KEY`) | Key required. Search snippets, **not verified**: free 500 credits/month; paid $30 = 20,000, $59 = 100,000, $119 = 5M, $249 = 15M credits/month; historical calls cost 10x; historical data "only available on paid usage plans"; player props and other extra markets have history from 2023-05-03, one event per call | low | Credit cost grows with markets x regions; sources disagree on whether the free tier includes props | **BENCHMARK** - cheapest reference feed if the owner wants one; not needed for pre-game Kalshi-only v1 |
| [SportsGameOdds](https://sportsgameodds.com) | Odds API priced per event object. | requests | not verified | not verified | Search snippets, **not verified**: free 2,500 objects/month at 10 requests/minute; Rookie $149/month (100,000 objects); Pro $499/month | low | Site not reachable; numbers come from the vendor's own comparison pages | **REJECT** - five times the entry price of The Odds API for a reference-only use |
| [OddsJam](https://oddsjam.com) API (sold as OpticOdds) | Enterprise odds feed. | - | not verified | not verified | Price not found (demo form; an unverified community figure of about $5,000/month per sport) | high | Sales-gated; not reachable | **REJECT** - enterprise pricing for a reference input |
| [Sportradar](https://developer.sportradar.com) | Official league data and odds. | - | commercial contract | not verified | Price not found (sales-gated) | high | Not reachable; contract required | **REJECT** - no self-serve tier found |
| HF [minmaxxxer/nba-closing-lines-ats-2025-26](https://huggingface.co/datasets/minmaxxxer/nba-closing-lines-ats-2025-26) (+ `-2024-25`, `nfl-closing-lines-ats-2025`, `nba-odds-2026-27`, `nfl-odds-2026`) | One row per game: consensus opening and closing moneyline probability (de-vigged), spread, total, scores, ATS and total results. Downloaded: 1,315 NBA games, 34 columns. | none | CC-BY-4.0 | last modified 2026-10-18 (season files), 2026-11-23 (current-season files, "updated every Monday") | Free, no key | low | Vendor's free sample for a paid scraper; "consensus" of about 21 books with undocumented method; own team IDs; no props | **BENCHMARK** - a free closing-line yardstick for game models; spot-check before trusting |
| HF [SmartStake/nfl-mainlines](https://huggingface.co/datasets/SmartStake/nfl-mainlines) | Same format as the CFB set: pregame price runs at 12 venues including Kalshi, 120 NFL games from 2026-09-09. | none | CC-BY-4.0 | last modified 2026-11-23 | Free, no key | med | Mainlines only (no props); no results; study snapshot | **ADAPT** - Kalshi versus sharp-book price paths for NFL game lines |
| HF [oliviersportsdata/NBA-Opening-Closing-Odds-US-Books-Pinnacle](https://huggingface.co/datasets/oliviersportsdata/NBA-Opening-Closing-Odds-US-Books-Pinnacle) (+ NFL) | Per-book open/close odds 2018-2026. | none | "single-commercial-user" (paid via Gumroad) | last modified 2026-11-17 | Free part is a **sample of 20 NBA / 26 NFL games**; full price not found | low | Sample too small to use | **REJECT** - sample only, restrictive licence |
| `cfbfastR-data` `betting/parquet/cfb_line_odds.parquet` (section D) | NCAAF spread, total, moneyline by book, 2006-2025. | none | none stated | file ends at 2025 season | Free, no key | low | 2020-2025 coverage is thin (about 15-18k rows per season: ESPN Bet, Bovada, DraftKings) | **ADAPT** - prior-season NCAAF reference lines only |

Also seen, not evaluated: HF `kennyhyder/sportsbookish-daily-odds` (daily Kalshi versus de-vigged book median,
CC-BY-4.0, latest snapshot only), HF `minmaxxxer/pinnacle-player-props-results-nba-nhl` (one week of graded
Pinnacle props, CC-BY-4.0).

## F. Quant libraries

"In .venv" is the project's virtualenv on 2026-11-24. Latest release from PyPI.

| Library | Licence | Latest release | In .venv | Role in a same-day binary-contract sports portfolio | Verdict |
|---|---|---|---|---|---|
| numpy | BSD-3-Clause (bundled parts 0BSD, MIT, Zlib) | 2.5.5, 2026-11-15 | yes | Joint-outcome simulation: draw correlated player stat lines, turn them into a scenarios x contracts 0/1 payoff matrix; combo payoff = product of leg columns. | **INTEGRATE** - in use |
| scipy | BSD-3-Clause | 1.18.1, 2026-08-21 | yes | Distributions for counts (Poisson, negative binomial), Gaussian copula for leg correlation, binomial and Wilson intervals for rule grading, `optimize` for one-parameter calibration. | **INTEGRATE** - in use |
| pandas | BSD-3-Clause | 3.1.1, 2026-11-04 | yes | Parquet I/O, the ledger, walk-forward splits by night. | **INTEGRATE** - in use |
| scikit-learn | BSD-3-Clause | 1.9.1, 2026-09-10 | yes | Logistic baselines, isotonic and Platt calibration, `TimeSeriesSplit`, Brier score and log-loss. Calibration matters more than accuracy for pricing a binary contract. | **INTEGRATE** - in use |
| lightgbm | MIT | 4.7.0, 2026-07-18 | yes | The one gradient-boosting library: player-stat mean and quantile models, P(over line) classifiers. | **INTEGRATE** - in use; the only GBM we keep |
| xgboost | Apache-2.0 | 3.4.1, 2026-08-15 | no | Same job as lightgbm. A second GBM adds a dependency without a new capability. | **BENCHMARK** - only if lightgbm fails a calibration check and we want a second opinion |
| catboost | Apache-2.0 | 1.2.10, 2026-02-18 | no | Same job; strength is high-cardinality categoricals (player, team IDs). Large wheel. | **REJECT** - duplicate of lightgbm for our feature set |
| optuna | MIT | 5.1.0, 2026-11-16 | no | Hyper-parameter search. Every extra trial is another chance to overfit a few hundred game-nights; the walk-forward test would have to wrap the whole search. | **REJECT** for now - fixed, documented parameters; revisit when a model clears `walk --pre` |
| pymc | Apache-2.0 | 6.4.0, 2026-11-01 | no | Hierarchical partial pooling: player rates shrunk toward position/team means, minutes uncertainty, posterior predictive P(stat >= line) with honest intervals for small samples (rookies, first game back, NCAAF players). Heavy (pytensor, arviz); slow in a 20-minute daily run. | **BENCHMARK** - offline study against the lightgbm/empirical-Bayes path; not in the daily run |
| cvxpy | Apache-2.0 | 1.9.3, 2026-09-19 | **yes** | Growth-optimal (Kelly) allocation over simulated joint outcomes: maximise mean log(1 + payoff_matrix @ stakes) subject to bankroll, per-market, per-game, per-sleeve and daily-loss caps; fractional Kelly as a scaling or a risk constraint. Handles correlated legs and combos, which per-bet Kelly cannot. Solvers bundled (Clarabel, SCS, OSQP, HiGHS). | **INTEGRATE** - the portfolio/risk layer; already installed |
| Riskfolio-Lib | BSD-3-Clause | 7.4.1, 2026-11-04 | no | Mean-risk, risk-parity and HRP optimisation over a matrix of **continuous asset-return time series**. Binary contracts that settle the same day have no return series, no stationary covariance and a two-point payoff; its estimators do not apply. It also pulls in arch, astropy, networkx, xlsxwriter and vectorbt. | **REJECT** - wrong data model; cvxpy on our own scenario matrix does the job |
| vectorbt | Apache-2.0 **with Commons Clause** ("fair-code") | 1.2.1, 2026-11-11 | no | Vectorised backtests of signals on **price bars** with entries and exits. Our bets are placed once pre-game and held to settlement at a known fill price and fee; there is no bar series to vectorise. Pulls in numba, plotly, ipywidgets. | **REJECT** - wrong data model and a restricted licence; `bot/walkforward.py` and `bot/sim_env.py` already replay real nights |

Already in use and kept: statsmodels (BSD-3-Clause), gymnasium (MIT), penaltyblog (MIT, 1.13.2 on 2026-11-23;
power and Shin de-vig). Net change to dependencies from this section: **none added**; cvxpy gets declared in
`requirements.txt` because it is installed but not listed.

## G. Existing projects

Picked by stars and recent activity from GitHub searches on 2026-11-24 (NBA/NFL prediction, props, parlay/EV,
Kelly, backtesters, Kalshi/Polymarket tools). "Claim verified" refers to profitability or accuracy.

| Item | What it does / what we reuse | Key deps | Licence | Maintenance | Data, keys, cost | Difficulty | Known limitations | Claim verified? | Verdict |
|---|---|---|---|---|---|---|---|---|---|
| [kyleskom/NBA-Machine-Learning-Sports-Betting](https://github.com/kyleskom/NBA-Machine-Learning-Sports-Betting) | NBA moneyline and totals models (XGBoost, neural net, logistic) on team stats 2007-08 onward, EV and Kelly output, Flask app. Reuse: feature list and its sigmoid-calibrated training script as a baseline. | TensorFlow, XGBoost, pandas | none (no LICENSE) | 1,795 stars, last push 2026-11-11 | stats.nba.com + SBR odds scrape; no key | med | Team-level only, no props; scrapes blocked from cloud; no licence means no code reuse | No. Current README states no accuracy or ROI and shows no out-of-sample betting record | **BENCHMARK** - compare our game-line model to its published approach |
| [georgedouzas/sports-betting](https://github.com/georgedouzas/sports-betting) (`sportsbet`) | Dataloaders + "bettors" that wrap any scikit-learn estimator; `backtest(bettor, X, Y, O, cv=TimeSeriesSplit(5))`. Reuse: the time-ordered backtest interface design. | scikit-learn, pandas | MIT | 849 stars, last push 2026-11-15 | Football-Data sources (soccer); paid feeds optional | med | Soccer data sources; fixed-odds bookmaker model, no exchange fees or RFQ fills | No profitability claim made | **BENCHMARK** - check our walk-forward against its backtest conventions |
| [Jon-Becker/prediction-market-analysis](https://github.com/Jon-Becker/prediction-market-analysis) | Indexers and analysis scripts for Kalshi and Polymarket markets and trades; pre-collected dataset (36 GiB compressed). README lists academic papers built on it. Reuse: analysis recipes for calibration and maker/taker splits. | duckdb, pandas, pyarrow, httpx, scipy; also pins the deprecated `kalshi-python>=2.1.4` | MIT | 3,959 stars, last push 2026-10-29 | Free; dataset on the author's Cloudflare R2 (host not tested) | med | 36 GiB download; overlaps `TimeSeventeen/Kalshi`, which we already use | Research framework; cited papers not reviewed by us | **BENCHMARK** - reproduce one of its calibration tables on our fills as a sanity check |
| [pmxt-dev/pmxt](https://github.com/pmxt-dev/pmxt) | "CCXT for prediction markets": one API over Polymarket, Kalshi and others. | **Node.js >= 18** sidecar + Python SDK | MIT | 2,218 stars, last push 2026-09-30 | Venue keys | high | Needs a Node server process; README: Kalshi writes only when self-hosted "where supported" | No claim | **REJECT** - we trade one venue and already have a signed client |
| [nautechsystems/nautilus_trader](https://github.com/nautechsystems/nautilus_trader) | Event-driven backtest and live engine; adapters: Betfair (stable), Polymarket (stable), **Kalshi (beta)**. | Rust core, Python | LGPL-3.0 | 31,788 stars, last push 2026-11-24 | Venue keys | high | Built for order-book replay and intraday execution; far more machinery than pre-game, hold-to-settlement bets need | No claim | **REJECT** - wrong scale for this project; revisit only if live quoting returns |
| [ryanfrigo/kalshi-ai-trading-bot](https://github.com/ryanfrigo/kalshi-ai-trading-bot) | Kalshi toolkit whose main feature is measuring edge: Brier, log-loss, a "forward-only guard" that counts only trades resolved after entry. Reuse: its scorecard layout for our audit tier. | Python 3.12+; LLM provider (not verified which) | MIT | 662 stars, last push 2026-11-09 | Kalshi key, LLM keys | med | LLM-driven picks; no offline backtest ("placeholder") | **Reports a negative result**: "NO MEASURED EDGE - won 58% vs. 60% implied", Brier 0.2356, n=43, one live account. Not independently verified | **BENCHMARK** - the forward-only audit idea and metrics; not its strategy |
| [rodlaf/KalshiMarketMaker](https://github.com/rodlaf/KalshiMarketMaker) | Avellaneda-Stoikov market maker per Kalshi market with inventory caps. | requests, PyYAML, cryptography | MIT | 470 stars, last push 2026-04-14 | Kalshi key | med | Live quoting, which the owner has switched off; may predate the V2 order endpoints (not checked) | No claim | **REJECT** - out of scope (pre-game only, no live watcher) |
| [OctagonAI/kalshi-trading-bot-cli](https://github.com/OctagonAI/kalshi-trading-bot-cli) | LLM research CLI that prints model probability versus market price, Kelly sizes, `calibration` command with Brier by category. | Node.js (`package.json` 2.3.3), vendor research API | MIT | 418 stars, last push 2026-10-01 | Kalshi key + paid research API | med | Edge numbers in the README are illustrative output, not a track record | No | **REJECT** - vendor-tied LLM picks; nothing reusable beyond the idea of per-category calibration |
| [braedonsaunders/homerun](https://github.com/braedonsaunders/homerun) | Full-stack platform for Polymarket and Kalshi strategies: L2 book replay, fill-probability model, walk-forward, parameter sweeps. | Python 3.10+, web stack, DB | **AGPL-3.0** | 231 stars, last push 2026-10-27 | Venue keys | high | AGPL; platform-sized; order-book focus | No | **REJECT** - licence and scope |
| [NBA-Betting/NBA_Betting](https://github.com/NBA-Betting/NBA_Betting) | End-to-end NBA spread prediction pipeline; documents Vegas spread error by era (about 23,000 games, 2006-2026). | not verified | MIT | 209 stars; **archived**, "on hiatus" | NBA stats + odds scrapes | med | Archived; goal stated as "profitable" without a verified record | No | **BENCHMARK** - its spread-error-by-era table is a useful baseline for game-line models |
| [greerreNFL/nfelo](https://github.com/greerreNFL/nfelo) | NFL Elo-style ratings and predictions; publishes model outputs as CSV in `output_data/`. | requests | no standard licence detected | 66 stars, last push 2026-11-24 | Free CSVs | low | README: "Model code has moved private" as of 2026-11-11; outputs only, method no longer auditable | No | **BENCHMARK** - an independent NFL win-probability series to score ours against |
| [sedemmler/WagerBrain](https://github.com/sedemmler/WagerBrain) | Odds conversion, parlay payout, Kelly, EV helpers. | not verified | MIT | 322 stars, last push 2020-05-02 | none | low | Stale; sportsbook parlay maths (independent legs), not Kalshi's product-of-legs with scalar DNP values | n/a | **REJECT** - a few formulas we already have, six years unmaintained |
| [parlayparlor/nba-prop-prediction-model](https://github.com/parlayparlor/nba-prop-prediction-model) | NBA prop tool using nba_api averages; the most-starred open prop projector found. | nba_api | MIT | 33 stars, last push 2024-12-20 | stats.nba.com | low | Simple averages; stale | No | **REJECT** - weaker than `bot/legs.py` |

No repository found had a pre-registered evaluation. The closest to an honest out-of-sample record is
`ryanfrigo/kalshi-ai-trading-bot` (forward-only, negative result). Searches for parlay/EV calculators,
Kelly/bankroll simulators and sports-betting backtesters returned nothing above about 120 stars that is
maintained (`deltaray-io/kelly-criterion` 116 stars, last push 2022-12-08;
`xandao-dev/monte-carlo-betting-simulations` 26 stars, 2022). Those pieces stay in-house on numpy + cvxpy.

---

## 1. Data-source matrix

| Domain | Source | Free / paid | Key needed | Reachable from build sandbox (tested 2026-11-24) | History depth | Update lag | Licence |
|---|---|---|---|---|---|---|---|
| NBA | HF `cbratkovics/nba-game-logs` | free | no | yes | not checked here (522 files) | last modified 2026-11-24 | CC0-1.0 |
| NBA | `gabriel1200/player_sheets` (GitHub) | free | no | host reachable; files not re-fetched here | not checked | last push 2026-11-24 | none stated |
| NBA | stats.nba.com via `nba_api` | free | no | **no** (also `cdn.nba.com` 403) | 1996-97 onward for most endpoints (not verified) | minutes | NBA terms; client MIT |
| NFL | nflverse-data releases | free | no | yes | pbp 1999 onward (not verified); participation to 2025 | rebuilt daily, files dated 2026-11-24 | CC-BY-4.0 |
| NCAAF | cfbfastR-data (raw GitHub) | free | no | yes | schedules 2001-2026 (file listing; 2024-2026 opened); lines 2006-2025; pbp frozen at 2021 | last commit 2026-11-23 | none stated |
| NCAAF | sportsdataverse-data `espn_cfb_*` | free | no | yes | 2024-2026 tested; earlier not checked | 0-2 days (`timestamp.json`) | repo MIT; ESPN-sourced |
| NCAAF | HF `24601p/cfb-data` | free | no | yes | 2025-2026 | last modified 2026-11-23 | none stated |
| NCAAF | CollegeFootballData API | free tier + paid | **yes** | **no** | not verified | not verified | terms not read |
| Kalshi prices | Kalshi REST (`api.elections.kalshi.com`) | free | no for market data; yes for portfolio and RFQ | yes (prod shared host and demo) | live + `/historical/*` | real time | Kalshi developer terms |
| Kalshi prices | HF `TimeSeventeen/Kalshi` (every fill) | free | no | yes | not checked here (2,153 files) | last modified 2026-11-24 | none stated |
| Kalshi prices | HF `SmartStake/cfb-mainlines`, `nfl-mainlines` | free | no | yes | NCAAF 2026-08-27 to 2026-10-31; NFL from 2026-09-09 | study snapshots | CC-BY-4.0 |
| Reference odds | HF `minmaxxxer/*` closing lines | free | no | yes | NBA 2024-25 onward, NFL 2025 onward | weekly (Mondays) | CC-BY-4.0 |
| Reference odds | sportsdataverse `espn_cfb_betting`, cfbfastR `cfb_line_odds` | free | no | yes | 2026 closing spread/total; 2006-2025 by book | 0-2 days; season-end | MIT repo / none stated |
| Reference odds | The Odds API | free tier + paid | **yes** | API host yes (401 without key); website no | props history from 2023-05-03 (search snippet) | live | terms not verified |
| Reference odds | SportsGameOdds, OddsJam/OpticOdds, Sportradar | paid | yes | **no** | not verified | not verified | commercial |

## 2. Cost analysis

Free today, and enough for the pre-game NBA + NFL + NCAAF build: Kalshi API (Basic tier; Advanced is a free
upgrade call), Hugging Face datasets, nflverse-data, cfbfastR-data, sportsdataverse-data, every library in
section F, GitHub. **Current data cost: $0 per month.**

| Paid option | Smallest useful tier | Cost per month | Source of the price |
|---|---|---|---|
| The Odds API | 20,000 credits | $30 | search snippet, not verified on the vendor page |
| The Odds API with regular historical or prop pulls | 100,000 credits (historical costs 10x) | $59 | search snippet, not verified |
| CollegeFootballData | free key, 1,000 calls | $0 | search snippet, not verified |
| CollegeFootballData Patreon | Tier 1 (5,000 calls) / Tier 2 (30,000) / Tier 3 (75,000) | $1 / $5 / $10 | older CFBD blog post via search snippet, not verified |
| SportsGameOdds | Rookie, 100,000 objects | $149 ($99 if billed annually) | search snippet of vendor pages, not verified |
| OddsJam / OpticOdds | - | price not found | sales-gated |
| Sportradar | - | price not found | sales-gated |
| oliviersportsdata full odds files | one-off purchase | price not found | Gumroad link on the dataset card |
| Kalshi trading fees | per fill | variable: `fee_type` x `fee_multiplier` per series | spec; coefficient PDF not fetched |

## 3. Reuse vs build decisions

- Kalshi client: **keep** `bot/kalshi.py`; add a spec-drift test against `openapi.yaml`; do not add an SDK.
- Kalshi fees: **build** a small fee function driven by the series/event `fee_type` and `fee_multiplier`
  (taker and maker, combo maker multiplier, rounding per the fee-rounding guide); the current flat `0.07 * P * (1-P)` taker
  formula ignores maker fees and overrides.
- Combo settlement: **build** in the simulator as product of leg values with scalar (DNP / protected) legs;
  no library models this.
- NBA stats: **reuse** the Hugging Face game logs daily; `nba_api` only from a home connection.
- NFL data: **reuse** nflverse-data parquet directly; take names from nflreadpy; no Polars dependency.
- NCAAF data: **reuse** cfbfastR-data (results, Elo) + sportsdataverse `espn_cfb_*` (pbp, box, spread); build
  only a team-name crosswalk to Kalshi tickers. CFBD key optional later, via `cfbd`.
- Reference prices: **reuse** free Hugging Face closing-line and SmartStake sets for research; buy The Odds
  API only if a live reference is needed; de-vig with `penaltyblog`.
- Probability models: **reuse** lightgbm + scikit-learn calibration; no xgboost, catboost or optuna.
- Bayesian player models: **defer**; pymc as an offline comparison only.
- Portfolio and risk: **reuse** cvxpy for Kelly over the simulated payoff matrix; build the scenario
  generator (numpy/scipy) and the caps ourselves. No Riskfolio-Lib, no vectorbt.
- Backtesting: **keep** `bot/walkforward.py` and `bot/sim_env.py`; borrow the forward-only audit idea, not code.
- Audit and dashboard: **build** on the existing ledger; no project above fits a hold-to-settlement book.
- Third-party bots and "profitable" models: **no code reuse**; BENCHMARK rows are comparisons only.

## Not verified (carry forward)

- Kalshi fee coefficient and maker-fee table (fee-schedule PDF returned 429); Kalshi developer terms;
  the "Protected Player Markets" help page.
- All vendor prices in sections D, E and 2 (vendor sites not reachable; figures are search snippets).
- The Odds API terms on storing data; whether its free tier includes props.
- CollegeFootballData terms, rate limits and current tier prices.
- TexasCoding SDK's coverage claim (README only; not run). `rodlaf/KalshiMarketMaker` V2 compatibility.
- History depth of the NBA and Kalshi Hugging Face datasets (already in use; not re-measured here).
- `ngs_passing` / `ngs_rushing` file names; nflverse pbp first season.
