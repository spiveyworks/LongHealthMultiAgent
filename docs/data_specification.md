# Long Health Coach Data Specification

## Overview
- Audience: data engineers and analysts preparing longitudinal health data for the Long Health Coach multi-agent workflow.
- Goal: standardize heterogeneous exports (labs, wearables, diet logs, supplement trackers) into the canonical timeline consumed by `TimelineStore` (`src/long_health_coach/data/timeline.py`).
- Scope: single-subject datasets covering serum labs, vitals, wearable metrics, nutrition macros/micros, supplements, medications, and procedural context.
- Output: a normalized long-format table saved as Parquet at `artifacts/ingest/timeline.parquet` (or an equivalent path passed via `RunConfig.ingestion.timeline_table`).

## Deliverables For The Data Engineer
- Raw source exports (CSV, XLSX, or Parquet) organized under `data/` with descriptive filenames.
- This specification and any supplementary mapping sheets used during standardization.
- The canonical timeline dataset (`timeline.parquet`) plus optional CSV preview for manual review.
- A brief README or change log (optional) that notes preprocessing steps or anomalies encountered.

## Canonical Timeline Table
| column | type | required | description | example |
| --- | --- | --- | --- | --- |
| subject_id | string | yes | Unique identifier for the subject; must match `RunConfig.subject_id`. | johndoe |
| measurement_date | datetime (ISO 8601, UTC) | yes | Observation date. Store day-level timestamps at midnight UTC (`YYYY-MM-DDT00:00:00Z`). | 2024-03-12T00:00:00Z |
| variable_name | string | yes | Canonical measurement name from the dictionaries below. | HDL Cholesterol |
| value | float64 | yes | Numeric value expressed in canonical units. | 43.0 |
| unit | string | conditional | Canonical unit label. Leave empty only when the quantity is dimensionless. | mg/dL |
| source | string | yes | Provenance label (usually the original filename or system name). | Bob Blood Tests.xlsx |
| metadata | JSON string | optional | Minified JSON capturing auxiliary columns (e.g., `{"panel":"Lipid Panel","raw_unit":"ng/mL"}`). | {"panel":"Lipid Panel"} |

Persist the table as Parquet (preferred) or CSV with UTF-8 encoding. Downstream components expect timezone-aware timestamps and column order as listed above.

## Accepted Source Formats

### Long-form input (preferred)
- Required columns: `measurement_date`, `variable_name`, `value`.
- Optional columns: `unit`, `subject_id`, `source`, any additional descriptors.
- Normalize column names with lowercase and underscores before mapping.
- Dates may arrive with or without times; floor to date, convert to UTC, and drop time-of-day unless the metric is inherently intraday.

### Wide-form lab workbook
Many lab exports list analytes across columns. Reshape them using:
1. Treat column 1 as the analyte label (`variable_name`).
2. If column 2 contains units, map it to `unit`.
3. Remaining columns are date headers; melt them into long rows with `measurement_date`.
4. Coerce `value` to numeric; cast non-numeric entries such as `"<1.0"` using domain logic (e.g., replace with `0.5` and flag in `metadata`).

**Example:**

Wide sheet excerpt:

| Test | Unit | 2024-03-01 | 2024-06-01 |
| --- | --- | --- | --- |
| HDL Cholesterol | mg/dL | 41 | 45 |
| Triglycerides | mg/dL | 86 | 74 |

Canonical rows:

| subject_id | measurement_date | variable_name | value | unit | source |
| --- | --- | --- | --- | --- | --- |
| johndoe | 2024-03-01T00:00:00Z | HDL Cholesterol | 41 | mg/dL | Bob Blood Tests.xlsx |
| johndoe | 2024-06-01T00:00:00Z | HDL Cholesterol | 45 | mg/dL | Bob Blood Tests.xlsx |
| johndoe | 2024-03-01T00:00:00Z | Triglycerides | 86 | mg/dL | Bob Blood Tests.xlsx |
| johndoe | 2024-06-01T00:00:00Z | Triglycerides | 74 | mg/dL | Bob Blood Tests.xlsx |

## Transformation Checklist
1. Inventory each source export and record its provenance (`source` value).
2. Trim and normalize column headers; ensure dates parse with `pandas.to_datetime(..., utc=True)`.
3. Reshape wide sheets, then append long-form sheets directly.
4. Map raw metric names to the canonical `variable_name` strings in the dictionary below. Log any unmapped names.
5. Convert units into the canonical units; store original units in `metadata.raw_unit` when conversions are applied.
6. Populate `subject_id` on every row (default `johndoe` for the current dataset).
7. Drop duplicates on (`subject_id`, `measurement_date`, `variable_name`) or average them after confirming they represent the same reading.
8. Remove placeholder zeros that represent missing measurements (e.g., `Body Temperature == 0`) by setting them to null and flagging in `metadata`.
9. Validate using the checklist below, then export the clean timeline.

## Data Quality Checks
- No nulls in `subject_id`, `measurement_date`, or `variable_name`.
- `measurement_date` is monotonic within each `variable_name`.
- Each metric uses the canonical unit and plausible range (see dictionary).
- Dimensionless measures (`ratio` values, binary flags) use an empty unit string.
- Supplements and medications are recorded as daily average dosage/servings; 0 denotes no intake.
- Run `python -m long_health_coach.main --subject-id johndoe --run-id dryrun --source data/<file>` to smoke test ingestion after exporting.

## Canonical Variable Dictionary

### Serum and Laboratory Biomarkers
| variable_name | description | canonical_unit | cadence | notes |
| --- | --- | --- | --- | --- |
| 25-(OH) Vitamin D | Serum 25-hydroxy vitamin D | ng/mL | Lab draw | Convert from nmol/L by dividing by 2.5 when necessary. |
| ApoB:ApoA1 Ratio | Apolipoprotein B to A1 ratio | ratio | Lab draw | Compute as ApoB divided by ApoA1; leave unit empty. |
| Apolipoprotein A1 (APOA1) | Serum ApoA1 | mg/dL | Lab draw | Convert from g/L by multiplying by 100. |
| Apolipoprotein B (APOB) | Serum ApoB | mg/dL | Lab draw | Convert from g/L by multiplying by 100. |
| Cystatin C | Serum cystatin C | mg/L | Lab draw | Round to three decimals. |
| Dehydroepiandrosterone Sulfate (DHEA-S) | Serum DHEA-S | ug/dL | Lab draw | Convert from umol/L using lab reference factors when present. |
| Ferritin | Serum ferritin | ng/mL | Lab draw | For "<" results, substitute half the detection limit and capture the raw string in metadata. |
| HDL Cholesterol | High-density lipoprotein cholesterol | mg/dL | Lab draw | |
| LDL Cholesterol | Low-density lipoprotein cholesterol | mg/dL | Lab draw | |
| Total Cholesterol | Total cholesterol | mg/dL | Lab draw | |
| Triglycerides | Serum triglycerides | mg/dL | Lab draw | |
| Total Cholesterol:HDL Ratio | Total cholesterol to HDL ratio | ratio | Lab draw | Derive or ingest; unit remains empty. |
| Triglycerides:HDL Ratio | Triglyceride to HDL ratio | ratio | Lab draw | |
| Hemoglobin A1C | Glycated hemoglobin | percent | Lab draw | Convert from fraction (0-1) by multiplying by 100. |
| High-Sensitivity CRP | High-sensitivity C-reactive protein | mg/L | Lab draw | Use detection limit / 2 for below-range values. |
| Homocysteine (HCY) | Plasma homocysteine | umol/L | Lab draw | |
| Insulin | Fasting insulin | uIU/mL | Lab draw | Convert from pmol/L by dividing by 6.0. |
| Morning Cortisol | Morning serum cortisol | ug/dL | Lab draw | Record draw time in metadata if available. |
| Testosterone, Total (Males) | Total testosterone | ng/mL | Lab draw | Divide ng/dL results by 100 to convert. |
| Testosterone:Cortisol Ratio | Testosterone to cortisol ratio | ratio | Lab draw | Derived from canonical testosterone and cortisol values. |
| Thyroid Stimulating Hormone (TSH) | Thyroid stimulating hormone | uIU/mL | Lab draw | |

### Cardiometabolic Vitals & Fitness
| variable_name | description | canonical_unit | cadence | notes |
| --- | --- | --- | --- | --- |
| Blood Pressure Diastolic - Morning | Morning seated diastolic blood pressure | mmHg | Daily | Average multiple readings; convert from kPa by multiplying by 7.5 when needed. |
| Blood Pressure Diastolic - Afternoon | Afternoon diastolic blood pressure | mmHg | Daily | Capture measurement window in metadata (`{"daypart":"afternoon"}`). |
| Blood Pressure Diastolic - Night | Evening/night diastolic blood pressure | mmHg | Daily | |
| Blood Pressure Diastolic - Any Time | Unspecified-time diastolic blood pressure | mmHg | Daily | Use when device does not tag daypart. |
| Blood Pressure Systolic - Morning | Morning seated systolic blood pressure | mmHg | Daily | |
| Blood Pressure Systolic - Afternoon | Afternoon systolic blood pressure | mmHg | Daily | |
| Blood Pressure Systolic - Night | Evening/night systolic blood pressure | mmHg | Daily | |
| Blood Pressure Systolic - Any Time | Unspecified-time systolic blood pressure | mmHg | Daily | |
| Body Fat Percentage | Body fat proportion | fraction | Weekly | Store as 0-1 fraction (e.g., 0.162 = 16.2%). |
| Body Mass | Body weight | lb | Daily | Convert from kg by multiplying by 2.20462. |
| Body Mass Index | BMI | kg/m^2 | Daily | Compute from mass and height if not supplied. |
| Pulse Wave Velocity | Pulse wave velocity | m/s | Weekly | Replace zeros with null unless a true zero is confirmed. |
| VO2 Max | Estimated VO2 max | mL/kg/min | Weekly | Typically sourced from Apple Watch cardio fitness. |

### Wearable Activity & Recovery
| variable_name | description | canonical_unit | cadence | notes |
| --- | --- | --- | --- | --- |
| Active Energy Burned | Active energy expenditure | kcal | Daily | Sum of exercise and activity calories. |
| Basal Energy Burned | Basal metabolic energy | kcal | Daily | Exported BMR; usually near 2200 kcal/day. |
| Apple Exercise Time | Minutes in Apple Fitness "exercise" zone | minutes | Daily | Ensure fractional minutes are preserved. |
| Body Temperature | Peripheral body temperature | degF | Daily | Replace zero readings with null and flag `{"raw_value":"0"}`. |
| Heart Rate Variability SDNN (intraday median, across day mean) | Daily HRV SDNN | ms | Daily | Derived from wearable intraday HRV; ensure smoothing matches exporter. |
| Resting Heart Rate | Resting heart rate | bpm | Daily | Nightly resting rate from wearable. |
| Respiratory Rate | Respiratory rate | breaths/min | Daily | Derived from wearable overnight respiration. |

### Nutrition Intake - Energy & Macros
| variable_name | description | canonical_unit | cadence | notes |
| --- | --- | --- | --- | --- |
| Daily Average Dietary Energy Consumed | Total caloric intake | kcal | Daily | Average across logging window preceding the measurement date. |
| Daily Average Dietary Carbohydrates | Total carbohydrates | g | Daily | Include net + fiber as exported. |
| Daily Average Dietary Protein | Protein intake | g | Daily | |
| Daily Average Dietary Fat Total | Total fat intake | g | Daily | |
| Daily Average Dietary Fat Saturated | Saturated fat | g | Daily | |
| Daily Average Dietary Fat Monounsaturated | Monounsaturated fat | g | Daily | |
| Daily Average Dietary Fat Polyunsaturated | Polyunsaturated fat | g | Daily | |
| Daily Average Dietary Fiber | Dietary fiber | g | Daily | |
| Daily Average Dietary Sugar | Total sugars | g | Daily | |
| Daily Average Dietary Water | Water intake | mL | Daily | Convert from grams; 1 g = 1 mL. |

### Nutrition Intake - Micronutrients
| variable_name | description | canonical_unit | cadence | notes |
| --- | --- | --- | --- | --- |
| Daily Average Dietary Biotin | Biotin | mcg | Daily | |
| Daily Average Dietary Caffeine | Caffeine | mg | Daily | |
| Daily Average Dietary Calcium | Calcium | mg | Daily | |
| Daily Average Dietary Cholesterol | Cholesterol | mg | Daily | |
| Daily Average Dietary Chromium | Chromium | mcg | Daily | |
| Daily Average Dietary Copper | Copper | mg | Daily | |
| Daily Average Dietary Folate | Folate (DFE) | mcg | Daily | Ensure exporter uses DFE; log alias if different. |
| Daily Average Dietary Iodine | Iodine | mcg | Daily | |
| Daily Average Dietary Iron | Iron | mg | Daily | |
| Daily Average Dietary Magnesium | Magnesium | mg | Daily | |
| Daily Average Dietary Manganese | Manganese | mg | Daily | |
| Daily Average Dietary Molybdenum | Molybdenum | mcg | Daily | Treat zeros as missing if no supplement logged. |
| Daily Average Dietary Niacin | Niacin | mg | Daily | |
| Daily Average Dietary Pantothenic Acid | Pantothenic acid | mg | Daily | |
| Daily Average Dietary Phosphorus | Phosphorus | mg | Daily | |
| Daily Average Dietary Potassium | Potassium | mg | Daily | |
| Daily Average Dietary Riboflavin | Riboflavin (B2) | mg | Daily | |
| Daily Average Dietary Selenium | Selenium | mcg | Daily | |
| Daily Average Dietary Sodium | Sodium | mg | Daily | |
| Daily Average Dietary Thiamin | Thiamin (B1) | mg | Daily | |
| Daily Average Dietary Vitamin A | Vitamin A (RAE) | mcg | Daily | Convert IU to mcg (IU / 3.33) when needed. |
| Daily Average Dietary Vitamin B12 | Vitamin B12 | mcg | Daily | |
| Daily Average Dietary Vitamin B6 | Vitamin B6 | mg | Daily | |
| Daily Average Dietary Vitamin C | Vitamin C | mg | Daily | Large spikes often indicate supplementation. |
| Daily Average Dietary Vitamin D | Vitamin D | mcg | Daily | Convert IU to mcg (IU / 40). |
| Daily Average Dietary Vitamin E | Vitamin E | mg | Daily | |
| Daily Average Dietary Vitamin K | Vitamin K | mcg | Daily | |
| Daily Average Dietary Zinc | Zinc | mg | Daily | |

### Supplements & Specific Items
| variable_name | description | canonical_unit | cadence | notes |
| --- | --- | --- | --- | --- |
| Daily Average BroccoMax | BroccoMax capsules | capsules/day | Daily | Record number of capsules consumed. |
| Daily Average Cvs Health, 100% Pure Omega-3 Krill Oil, 500 mg | CVS Health Krill Oil servings | servings/day | Daily | One serving corresponds to the label dose; capture capsule count in metadata if available. |
| Daily Average Force Factor, Total Beets Original Drink Powder, Pomegranate Berry | Total Beets powder scoops | servings/day | Daily | |
| Daily Average Nature Made, Fish Oil 1200 mg 720 mg Omega 3 | Nature Made Fish Oil | servings/day | Daily | |
| Daily Average Nature's Bounty, Krill Oil | Nature's Bounty Krill Oil | servings/day | Daily | |
| Daily Average Whole Eggs | Whole eggs consumed | count/day | Daily | Average count across logged days. |

### Medications & Context Flags
| variable_name | description | canonical_unit | cadence | notes |
| --- | --- | --- | --- | --- |
| Statin | Statin daily dose | mg/day | Daily | Normalize spelling to "Statin" during transformation; legacy exports include the typo. |
| Donated Blood | Blood donation volume | pints/day | Event-driven | Use 1 for a full pint donation; values <1 represent fractional units. |
| Nutrition Number of Days Before Blood Test | Diet logging coverage window | days | Per lab panel | Count of tracked nutrition days in the look-back window preceding lab draw. |

## Recommended Metadata Keys
Capture informative context when available:
- `panel`: lab panel or wearable export batch (e.g., "Lipid Panel", "Apple Health Summary").
- `raw_unit`: original unit string before conversion.
- `raw_variable`: original column/header name.
- `daypart`: measurement window for vitals (e.g., `morning`, `afternoon`).
- `notes`: free-text flags such as "duplicate removed" or "converted from IU".

## Verification Workflow
1. Regenerate the timeline and save to `artifacts/ingest/timeline.parquet`.
2. Inspect a sample with `python -c "import pandas as pd; df=pd.read_parquet('artifacts/ingest/timeline.parquet'); print(df.sample(10))"`.
3. Run the orchestrator dry run (`python -m long_health_coach.main --subject-id johndoe --run-id spec_check --source data/<file>`).
4. Share the refreshed timeline, this spec, and notes with downstream developers or operators.
