Using the existing CLSmith test harness, run CLgen programs and look for errors.

1. Generate programs using CLgen, matching the CLsmith prototype.
2. Run these programs using cl_launcher.
    * TODO: either modify cl_launcher to accept 'A' as kernel name, or rewrite synthesized kernels to have name 'entry'.
3. Difftest.
4. If diffs found, hand-pick bugs and report.
