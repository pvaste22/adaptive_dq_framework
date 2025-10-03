# Phase 1 Validation Checklist

## Pre-flight Checks
- [ ] All config files present (config.yaml, band_configs.json, drift_thresholds.yaml)
- [ ] Raw data files present (CellReports.csv, UEReports.csv)
- [ ] Directory structure created (data/raw, data/processed, data/artifacts, data/logs)

## Data Loading
- [ ] Cell data loads without errors
- [ ] UE data loads without errors
- [ ] Entity counts match config (52 cells, 48 UEs)
- [ ] Timestamps parse correctly
- [ ] Band extraction works (S1/B13/C1 → B13)

## Unit Conversions
- [ ] PrbTot converts from percentage to absolute
- [ ] Energy converts from cumulative to interval
- [ ] Energy validation against Power×Time formula
- [ ] QosFlow 1-second semantics flagged
- [ ] TB counters marked as unreliable

## Window Generation
- [ ] 5-minute windows created
- [ ] 80% overlap implemented
- [ ] Completeness validation works
- [ ] Windows with <95% completeness dropped
- [ ] Window metadata saved

## Quality Validation
- [ ] PRB limit violations detected
- [ ] CQI range validation (0-15)
- [ ] TB reliability flagging
- [ ] Energy formula deviations calculated

## Artifacts
- [ ] Window metadata saved
- [ ] Artifacts versioned correctly
- [ ] Can load saved artifacts

## Logging
- [ ] Phase-specific logs created
- [ ] No errors in logs
- [ ] Warnings documented

## Performance
- [ ] Processes 7 days of data in reasonable time
- [ ] Memory usage acceptable
- [ ] No data loss during processing