# Image Resolution Benchmark Report

Measured performance and extraction accuracy across image resolutions (`max_side`):

| max_side | average seconds per file | files flagged "Needs review" | total-field accuracy |
|:---:|:---:|:---:|:---:|
| 1400 | 36.50s | 0 / 5 | 100.0% (5/5) |
| 1000 | 8.57s | 0 / 5 | 100.0% (5/5) |
| 800 | 9.28s | 0 / 5 | 100.0% (5/5) |

## Analysis & Recommendation

**Recommended `max_side`: 1000**

- **Accuracy**: Achieved 100.0% total-field accuracy with 0 files flagged.
- **Speed**: Averaged 8.57 seconds per file.
- **Rationale**: Balances visual clarity required by Ollama's vision model with minimal latency and memory footprint. Downscaling prevents out-of-memory errors on large images while preserving fine receipt details.

## Detailed Measurements by File

### max_side = 1400
| File | Total Elapsed | Prep Time | Ollama Time | Status | Expected Total | Extracted Total |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| receipt_01_amazon.jpg | 7.28s | 0.019s | 7.257s | OK | 1299.00 | 1299.00 |
| receipt_02_swiggy.jpg | 8.22s | 0.006s | 8.213s | OK | 540.00 | 540.00 |
| receipt_03_invoice_consulting.jpg | 56.21s | 0.014s | 56.191s | OK | 35400.00 | 35400.00 |
| receipt_04_aws.jpg | 57.37s | 0.011s | 57.362s | OK | 7823.45 | 7823.45 |
| receipt_05_uber.jpg | 53.43s | 0.016s | 53.413s | OK | 285.00 | 285.00 |

### max_side = 1000
| File | Total Elapsed | Prep Time | Ollama Time | Status | Expected Total | Extracted Total |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| receipt_01_amazon.jpg | 8.47s | 0.016s | 8.452s | OK | 1299.00 | 1299.00 |
| receipt_02_swiggy.jpg | 8.18s | 0.013s | 8.167s | OK | 540.00 | 540.00 |
| receipt_03_invoice_consulting.jpg | 8.56s | 0.015s | 8.542s | OK | 35400.00 | 35400.00 |
| receipt_04_aws.jpg | 8.66s | 0.014s | 8.645s | OK | 7823.45 | 7823.45 |
| receipt_05_uber.jpg | 9.00s | 0.016s | 8.981s | OK | 285.00 | 285.00 |

### max_side = 800
| File | Total Elapsed | Prep Time | Ollama Time | Status | Expected Total | Extracted Total |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| receipt_01_amazon.jpg | 8.59s | 0.018s | 8.567s | OK | 1299.00 | 1299.00 |
| receipt_02_swiggy.jpg | 9.74s | 0.019s | 9.719s | OK | 540.00 | 540.00 |
| receipt_03_invoice_consulting.jpg | 10.77s | 0.016s | 10.756s | OK | 35400.00 | 35400.00 |
| receipt_04_aws.jpg | 9.16s | 0.022s | 9.135s | OK | 7823.45 | 7823.45 |
| receipt_05_uber.jpg | 8.16s | 0.017s | 8.139s | OK | 285.00 | 285.00 |
