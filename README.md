# draft-assistant-data

Weekly NFL fantasy projections for the Draft Assistant app, rebuilt every morning (and before Thursday and Sunday kickoffs) from public nflverse data and Vegas game lines.

- `projections.csv`: one row per active player for the coming week: our PPR projection, expected stats, and what moved it.
- `updated.txt`: when it last ran.
- `pipeline/`: the code. Nothing paid or private goes in this repo.

## Rookie board (dynasty)

`rookies_2027.csv` is a consensus superflex rookie board: each player's average rank across several
public rankings sites (listed in `rookies_updated.txt`), with his best and worst rank. Only the
average is published here, not any site's own list. It is rebuilt weekly by `rookies/consensus.py`.
