# Sample Import Files

Synthetic sample datasets that match the Long Health Coach ingestion formats. Values are realistic but not derived from any individual.

Files
- Apple Health Export Sample.zip (contains apple_health_export/export.xml)
- labs_wide_sample.csv (wide-format labs)
- labs_long_sample.csv (long-format labs)
- cronometer_servings_sample.csv (Cronometer serving export format)
- vcf_sample.vcf (VCF sample for genomic summary ingestion)

Example usage
```
PYTHONPATH=src python -m long_health_coach.main --subject-id sample --run-id sample-run --artifact-root artifacts/sample-run --timeline artifacts/sample-run/timeline.parquet --source "sample-import-files/Apple Health Export Sample.zip" --source "sample-import-files/labs_wide_sample.csv" --source "sample-import-files/labs_long_sample.csv" --source "sample-import-files/cronometer_servings_sample.csv" --source "sample-import-files/vcf_sample.vcf"
```
