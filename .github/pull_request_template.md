## Summary
Describe the change, motivation, and related issue.

## Validation
- [ ] Python compile check
- [ ] `python scripts/validate_engine.py --skip-load`
- [ ] `python -m unittest discover -s tests -v`
- [ ] GPU runtime validation (if applicable)
- [ ] GPU performance regression comparison (if applicable)

**Not run / not applicable:** State which checks you could not run and why. Do not check a box for unexecuted checks.

## Evidence and compatibility
For changes affecting kernels, scheduling, or inference, specify tested hardware, model weights, baseline, accuracy, throughput and known limitations. Note any third-party code or data and its license/provenance.

## Review notes
Potential risks, compatibility concerns, or follow-up tasks.
