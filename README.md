# Data-to-Insight Agent

An AI-powered data analysis prototype that reads a
structured dataset (CSV or Excel) and automatically
generates business insights, identifies trends, and
recommends actions — built for a mid-size retail
client reviewing weekly sales data to replace manual
analysis processes.

---

## What It Does

The agent uses an iterative back-and-forth architecture
where Claude (Anthropic's AI) and a pandas computation
engine work together across multiple rounds:

1. The app sends Claude the dataset schema and
   pre-computed statistics as opening context
2. Claude responds with a REASONING (why it chose
   this computation) and a REQUEST (what to compute)
3. The app runs that exact pandas computation and
   returns the real numeric result
4. Claude reads the result and either requests
   another computation or produces final insights

This means Claude never generates numbers from
memory — every figure in the insights came from
a real pandas computation run against the actual
dataset. Hallucination of metrics is architecturally
prevented.

---

## Key Features

- Agentic analysis loop — up to 7 rounds of
  iterative computation and reasoning, with Round 6
  receiving a wrap-up warning and Round 7 reserved
  exclusively for structured JSON output
- Automatic data cleaning — detects and converts
  datetime formats (DD/MM/YYYY, MM/DD/YYYY, YYYY-MM-DD
  and more), currency strings, and numeric columns
  automatically for any uploaded dataset
- Statistical anomaly detection — Z-score
  (threshold 3.5 standard deviations) and
  IsolationForest methods combined, capped at 10
  results sorted by severity
- Time trend analysis — week-over-week,
  month-over-month, and year-over-year percentage
  changes detected and computed automatically
- Natural language queries — ask plain English
  questions about your data and receive grounded
  answers backed by real pandas computations
- Transparent reasoning — every insight card
  shows which computation round produced it and
  Claude's reasoning for choosing that computation
- Domain agnostic — works on retail, HR,
  e-commerce, and any structured dataset without
  code changes

---

## Architecture Decision

Three approaches were considered before choosing
the final architecture:

Option 1 — Precompute Stats (rejected):
Hardcodes which statistics to compute before calling
the LLM. Misses patterns like "Sunday is peak sales
day" or "March consistently outperforms" that were
not anticipated at build time. Fast but shallow.

Option 2 — Code Generation (partially used):
Claude writes pandas code directly. Risk of buggy
code on edge cases and unusual column formats.
Used only for the computation execution layer.

Option 3 — Back-and-forth Loop (chosen):
Claude requests computations in plain English, the
app runs them via pandas, Claude reads real results.
Each round builds on the previous finding — genuinely
exploratory analysis that adapts to what the data
actually contains.

The back-and-forth approach was chosen because:
- Claude decides what to investigate based on actual
  findings, not a predetermined script
- Numbers come from pandas not Claude's training data
- Progressive deepening produces insights that require
  real computation to discover
- Works on any dataset regardless of domain or
  column naming conventions

---

## Setup Instructions

Prerequisites:
- Python 3.8 or higher
- Anthropic API key (get one at console.anthropic.com)

Installation:

1. Clone the repository:
   git clone https://github.com/[your-username]/data-insight-agent
   cd data-insight-agent

2. Install dependencies:
   pip install -r requirements.txt

3. Create a .env file in the root folder:
   ANTHROPIC_API_KEY=your_api_key_here

4. Run the app:
   streamlit run app.py

5. Open your browser at:
   http://localhost:8501

Supported File Formats:
- CSV (.csv)
- Excel (.xlsx)
- Maximum recommended size: 50,000 rows
  (larger datasets are automatically sampled)

---

## Project Structure

data_agent/
├── app.py                  # Entry point — page config
│                           # and tab routing only
├── core/
│   ├── loader.py           # File upload and cleaning
│   │                       # pipeline (runs once on upload)
│   ├── schema.py           # Column type classification
│   │                       # (numeric/categorical/datetime)
│   ├── anomalies.py        # Z-score + IsolationForest
│   │                       # anomaly detection
│   ├── analysis.py         # Back-and-forth AI loop,
│   │                       # computation engine, JSON parser
│   └── trends.py           # Time trend precomputation
│                           # passed to Claude as context
├── ui/
│   ├── sidebar.py          # File upload, Run Analysis
│   │                       # button, round progress display
│   ├── insights_tab.py     # Insight, recommendation,
│   │                       # and anomaly flag cards
│   └── query_tab.py        # Natural language query
│                           # interface and history
└── requirements.txt

---

## How the Analysis Loop Works

Upload CSV/Excel
      ↓
Clean data once → store in session_state
      ↓
Extract schema → numeric / categorical / datetime
      ↓
Detect anomalies → Z-score + IsolationForest
      ↓
Precompute trends → period-over-period changes
      ↓
Send Claude: schema + stats + trends + anomalies
      ↓
Claude: REASONING + REQUEST
      ↓
Pandas: runs exact computation
      ↓
Result returned to Claude
      ↓
Repeat rounds 1-5 (full analysis)
Round 6: final computation + wrap-up warning
Round 7: ANALYSIS_COMPLETE + JSON output only
      ↓
5 insights + 2 recommendations + anomaly commentary

---

## Assumptions

- Dataset is a single CSV or Excel file
  (multi-file datasets not supported in this version)
- At least one numeric column exists as the primary
  metric to analyze
- Approximately 7 Claude API calls are made per
  analysis run (cost borne by the user)
- For best results, datasets should have 100+ rows

---

## Sample Output

Dataset: Superstore Sales (train.csv)
9,800 rows, 18 columns, US retail orders 2015-2018
Source: kaggle.com/datasets/vivek468/superstore-dataset-final

Insight 1 — Annual Growth Recovery:
Sales revenue grew 30.6% year-over-year in 2017
and 20.3% in 2018, following a 4.3% decline in 2016,
indicating strong recovery and sustained growth
momentum after an initial contraction.

Insight 2 — Category Performance:
Office Supplies experienced explosive growth of
37.0% YoY in 2017 and 31.8% in 2018, significantly
outpacing Technology (36.8% and 21.4%) and Furniture
(19.4% and 8.4%) in the same periods.

Insight 3 — Regional Divergence:
West region sales grew 36.5% YoY in 2017 and 36.0%
in 2018, while Central region grew 42.2% in 2017 but
declined 2.8% in 2018 — the only region to turn
negative while all others sustained 17-36% growth.

Insight 4 — Revenue Concentration Risk:
The top 10% of transactions generate 60.4% of total
sales — 980 orders out of 9,800 drive the majority
of revenue, creating significant dependency on
high-value orders.

Insight 5 — Seasonal Volatility Pattern:
Month-over-month volatility ranges from -75% to
+200%, with March consistently peaking (1,121% MoM
in 2015, 171% in 2016, 123% in 2017, 195% in 2018)
and January consistently troughing (-73% to -75%
across all years).

Recommendation 1:
Implement revenue diversification strategy to reduce
dependency on high-value transactions — 60% of revenue
from 10% of transactions creates substantial business
risk if large customers reduce spending.

Recommendation 2:
Investigate Central region's shift from 42% growth
to -2.8% decline in 2018 while all other regions
sustained 17-36% growth — reallocate resources from
underperforming to high-growth West and South regions.

Anomaly Finding:
8 of the 10 highest-value transactions are Technology
purchases (Copiers and Machines), distributed across
all 4 regions and all 4 years with no temporal
clustering — consistent enterprise purchasing behavior
rather than one-time events.

Natural Language Query Example:
Question: What was the worst performing week?
Answer: The worst performing week was February 16-22,
2015, with total sales of only $224.91 — significantly
below typical weekly performance.
Computation used:
df.groupby(df['Order Date'].dt.to_period('W'))['Sales'].sum().sort_values().head(1)

---

## Optional Enhancements Implemented

From the assessment brief's optional features:

Anomaly detection logic
  Z-score threshold 3.5 standard deviations
  combined with IsolationForest

Natural language query capability
  Example: Which region performed best last quarter?

Transparent reasoning chain
  Every insight shows which round produced it
  and Claude's reasoning for choosing that computation

---

## Known Limitations

- Visualization layer not included in this version
  (removed to focus on analytical depth and
  reliability of the insight generation pipeline)
- Single file uploads only — relational datasets
  requiring joins across multiple files are not
  supported
- Analysis quality depends on dataset having
  meaningful datetime and categorical columns
- Browser refresh requires clicking Run Analysis
  twice on first run after refresh — known
  Streamlit session_state behaviour

---

## Potential Next Steps

Given more time, the following would strengthen
the prototype:

- Visualization layer — dynamic Plotly charts
  driven by Claude's chart specifications
- Streaming output — show insights as they generate
  rather than waiting for all rounds to complete
- Multi-file support — automatic join detection
  for relational datasets
- Caching — store analysis results by file hash
  so re-running on the same file is instant
- Export — PDF report generation from insights
  and recommendations

---

## Dependencies

streamlit          — web application framework
anthropic          — Claude API client
pandas             — data manipulation and computation
numpy              — numerical operations
scikit-learn       — IsolationForest anomaly detection
python-dotenv      — environment variable management
openpyxl           — Excel file reading

---

## How Numbers Are Verified

Every number in the generated insights is traceable
to a specific pandas computation. The sidebar shows
Claude's reasoning for each round — what it
investigated and why — proving the analysis is
grounded in real data.

To verify any specific figure:

1. Find the insight and open its
   Why this insight? expander
2. Note which round produced it
3. That round ran a specific pandas computation
   against the uploaded dataset
4. Claude read the numeric result — it did not
   calculate it from memory

The How was this computed? expander in the
Ask a Question tab shows the exact pandas code
and raw output for every query answer.
