# diprally-engine

Multi-ticker swing decision engine. Daily dip-and-rally round-trip analysis
across a 17-ticker volatile-name universe.

Status: **W0** (scaffolding + v2 migration). Real README arrives at W2 when
the multi-ticker generalisation lands. See `CLAUDE.md` for session contract
and constraints.

## Quick start (single ticker, W0 surface)

```bash
pip install -r requirements.txt
export FMP_API_KEY=...
export ANTHROPIC_API_KEY=...
python tools/run.py SNDK --no-ai --peers MU WDC
```

Output:
- `output/round_trip_history_<TICKER>.csv` (appended per run)
- `output/<ticker>_dipnrally_dashboard.html` (regenerated per run)
