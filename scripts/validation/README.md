# UX affect validation study (Phase 4)

Pilot protocol for measuring decoder transfer from movie-trained weights to website walkthroughs.

## Run a validation session

```powershell
python scripts/validation/record_validation_session.py --session-id <id> --url https://example.com
# After walkthrough, export participant ratings JSON and merge:
python scripts/validation/record_validation_session.py --session-id <id> --ratings path/to/ratings.json
```

## Aggregate transfer metrics

```powershell
python scripts/validation/aggregate_validation_metrics.py
```

Output: `scout_data/validation/reports/transfer_report.json`

See `docs/validation/ux-affect-study-protocol.md` for full protocol.
