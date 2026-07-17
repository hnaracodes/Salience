# Human intervention: Clarity data for Attention T2

## What Clarity is (open source check)

- **Client instrumentation** (`clarity-js`) is open source under the MIT license:
  https://github.com/microsoft/clarity
- **Hosted project data** on clarity.microsoft.com is **private** to the project owner.
- You **cannot** download click heatmaps for Stripe, Wikipedia, etc. Public sites in
  `configs/validation_site_corpus.yaml` are for TRIBE/norm engineering only.

## What we need from you

1. List **owned or authorized** properties (domains) where you can install Clarity.
2. Confirm whether a Clarity project already exists (yes/no). Do **not** paste API tokens in chat.
3. For each page in the attention corpus (target ≥12 pages / ≥4 properties):
   - Install Clarity tracking code (or confirm it is live).
   - Wait until the page has **≥200 unique sessions** and **≥50 mapped clicks** on actionable elements.
   - Export **Click heatmap → Download CSV** from the Clarity UI for that URL/device window.
   - Drop the CSV at:
     `scout_data/validation/attention_v1/exports/<property>__<page>.csv`
   - Tell the agent the `property_id`, page URL, session_id of the matching TRIBE capture,
     unique session count, and mapped click count.

## Export path (UI)

1. Open https://clarity.microsoft.com → your project
2. Heatmaps → select the page URL
3. Download → **CSV** (not only PNG)
4. Preferred columns include Selector/Element and Clicks (adapter also accepts common aliases)

## What the agent will do after you provide CSVs

```bash
# After TRIBE analyze (without --clarity-csv on the claim bundles) and CSV placement:
python scripts/validate_attention_proxy.py \
  --fit-weights \
  --evaluate-holdout \
  --write-config configs/attribution_calibrated.yaml \
  --write-memo docs/validation/attention_validation_memo.md
```

Holdout properties stay sealed until weights are frozen.
