# 0.1,0.2 temparature


import vertexai
# Import GenerationConfig
from vertexai.generative_models import GenerativeModel, GenerationConfig
from google.cloud import bigquery
from google.cloud.exceptions import NotFound
import os
import json
import traceback
import re  # Import re module for regular expressions

# --- FastAPI Imports ---
from fastapi import FastAPI, HTTPException, status
from pydantic import BaseModel
import uvicorn

# ==============================================================================
# Configuration
# ==============================================================================
PROJECT_ID = os.getenv("VERTEX_PROJECT_ID", "california-desig-1536902647897")
LOCATION = os.getenv("LOCATION", "us-central1")
# Uses a capable model
# Updated to a generally available Pro model
MODEL_NAME = os.getenv("MODEL_NAME", "gemini-2.5-pro-preview-05-06")

BQ_PROJECT_ID = os.getenv("BQ_PROJECT_ID", "california-desig-1536902647897")
BQ_DATASET_ID = os.getenv("BQ_DATASET_ID", "cdd_pluto7_dataset")
BQ_TABLE_ID = os.getenv("BQ_TABLE_ID", "master_ledger_US")

# --- Temperature Settings for LLM ---
SQL_GENERATION_TEMPERATURE = float(
    os.getenv("SQL_GENERATION_TEMPERATURE", 0.1))
SUMMARY_GENERATION_TEMPERATURE = float(
    os.getenv("SUMMARY_GENERATION_TEMPERATURE", 0.2))


# ==============================================================================
# Detailed Schema Description & SQL Generation Guidelines
# ==============================================================================
MASTER_LEDGER_US_SCHEMA_DESCRIPTION = f"""
Table: `{BQ_PROJECT_ID}.{BQ_DATASET_ID}.{BQ_TABLE_ID}`
This table contains a daily summary of e-commerce operations, including sales, advertising, inventory, and product performance for the US marketplace.

Key Columns and Their Common Interpretations (Data Types are from your provided list):
*   **Identifiers, Attributes & Internal Planning:**
    *   `parent` (STRING): General product category or parent grouping. **Primary entity for "program" or "product line" queries.**
    *   `sku` (STRING): Stock Keeping Unit.
    *   `colour` (STRING): Color variation.
    *   `size` (STRING): Size variation.
    *   `child_asin` (STRING): Amazon Standard Identification Number for the specific product variation.
    *   `geo` (STRING): Geographic region (e.g., 'US').
    *   `marketplace` (STRING): Sales channel (e.g., 'Amazon').
    *   `f1_analysis` (STRING): Internal sales performance/replenishment strategy code.
    *   `classification` (STRING): Internal product classification.
    *   `threshold_NOD` (NUMERIC/INTEGER): Target Number of Days of stock.
    *   `final_weighted_average` (NUMERIC/FLOAT): Adjusted average daily sales quantity target.
*   **Time Dimension:**
    *   `date_ordered` (DATE): Primary date (YYYY-MM-DD) for daily data.
        *   For weekly trends/aggregations starting Sunday: Use `DATE_TRUNC(date_ordered, WEEK(SUNDAY))` and alias it as `week_start_date`.
        *   **"Current Week" or "Most Recently Completed Week (CW)" (Assumed Sunday-Saturday):**
            `date_ordered >= DATE_SUB(DATE_TRUNC(CURRENT_DATE(), WEEK(SUNDAY)), INTERVAL 1 WEEK)` AND
            `date_ordered < DATE_TRUNC(CURRENT_DATE(), WEEK(SUNDAY))`
        *   **"Previous N Weeks Baseline" (e.g., the 6 weeks immediately before CW):**
            For a baseline of 6 weeks prior to CW (P6WB):
            `date_ordered >= DATE_SUB(DATE_TRUNC(CURRENT_DATE(), WEEK(SUNDAY)), INTERVAL (1+6) WEEK)` (i.e., `INTERVAL 7 WEEK`) AND
            `date_ordered < DATE_SUB(DATE_TRUNC(CURRENT_DATE(), WEEK(SUNDAY)), INTERVAL 1 WEEK)`
        *   Adapt N accordingly for other N-week comparisons (e.g., for "last N full weeks").
*   **Sales & Units Metrics:**
    *   `product_sales` (FLOAT): Revenue specifically attributed to the product itself for shipped items (from payment report). This is a primary revenue metric.
    *   `bussiness_report_units` (FLOAT): Total units ordered (B2C & B2B) for this ASIN as per Amazon Business Report. Often used as the primary metric for "units sold". (Note: original field name has typo 'bussiness').
    *   `profitability_units` (FLOAT): Actual net number of units sold after deducting cancellations and refunds.
    *   `quantity_purchased` (INTEGER): Gross total number of units of this ASIN purchased/ordered on this date, sourced from Order report (transaction level, pre-cancellation).
    *   `sales` (FLOAT): Gross sales revenue generated from 'quantity_purchased' on this date, sourced from Order report (includes cancelled and returned orders).
*   **Advertising Metrics:**
    *   `clicks` (INTEGER): Total clicks received on associated Sponsored Products (SP) and/or Sponsored Display (SD) ads for this ASIN on this date. This is a COMBINED total.
    *   `sp_spends` (FLOAT): Amount spent on Sponsored Products (SP) ads for this ASIN on this date.
    *   `sd_spends` (FLOAT): Amount spent on Sponsored Display (SD) ads for this ASIN on this date.
    *   `sp_impressions` (INTEGER): Number of times Sponsored Products (SP) ads were displayed (impressions) for this ASIN on this date.
    *   `sd_impressions` (INTEGER): Number of times Sponsored Display (SD) ads were displayed (impressions) for this ASIN on this date.
    *   `sp_ads_attributed_sales` (FLOAT): Sales revenue attributed to Sponsored Products (SP) ad clicks within a 7-day attribution window.
    *   `sd_ads_attributed_sales` (FLOAT): Sales revenue attributed to Sponsored Display (SD) ad clicks/views within a 14-day attribution window.
    *   `sp_ads_attributed_units` (INTEGER): Number of units sold attributed to Sponsored Products (SP) ad clicks within a 7-day attribution window.
    *   `sd_ads_attributed_units` (FLOAT): Number of units sold attributed to Sponsored Display (SD) ad clicks/views within a 14-day attribution window.
*   **Inventory Metrics:**
    *   `fba_cost_inventory` (FLOAT): Total cost of inventory held within FBA. This column represents the *cost value* of FBA inventory.
    *   **!!! IMPORTANT NOTE ON FBA INVENTORY !!!** The column `total_fba` (INTEGER) is assumed here to be the primary FBA on-hand *quantity* (units). PLEASE VERIFY THE ACTUAL COLUMN NAME FOR 'Total FBA on-hand quantity/units' in your complete schema and REPLACE `total_fba` in this description and in derived NOD/WOC calculations if it's different (e.g., if it's `fba_onhand_units` or similar from a more complete schema list).
    *   `total_fba` (INTEGER): Assumed column for total quantity of available units in Amazon FBA. **(VERIFY THIS COLUMN NAME)**.
    *   Other on-hand columns for specific warehouses exist (e.g., `onhand_victorville`, `onhand_bristol`) but the FBA-specific on-hand quantity is typically used for Amazon channel analysis.
*   **Session & Page View Metrics (Traffic):**
    *   `sessions_total` (FLOAT): Total unique user sessions (within 24 hours) across all platforms (Browser + Mobile App) for this ASIN's page. Sourced from Amazon Business Reports. This is a key "traffic" metric.
    *   `page_views_total` (FLOAT): Total page views across all platforms (Browser + Mobile App) for this ASIN's page. Sourced from Amazon Business Reports.
    *   Breakdowns also available: `browser_sessions` (FLOAT), `browser_page_views` (FLOAT), `sessions_mobile_app` (FLOAT), `page_view_mobile_app` (FLOAT).
*   **Other Key Financial & Operational Metrics:**
    *   `promotional_rebates` (FLOAT): Cost of promotional rebates or discounts (e.g., Deal, Coupon, PED). Typically a negative value.
    *   `selling_fees` (FLOAT): Marketplace selling fees (e.g., referral fees). Typically negative.
    *   `fba_fees` (FLOAT): Fees for FBA services. Typically negative.
    *   `net_sales` (FLOAT): Calculated as: `product_sales + shipping_credits + gift_wrap_credits + promotional_rebates + other`.
    *   `landed_cost` (FLOAT): Total landed cost (COGS) for `profitability_units` sold.
    *   `gross_profit_before_ads` (FLOAT): Calculated as: `(net_sales + selling_fees + fba_fees) - (landed_cost + inventory_storage_fee)`.
*   **Derived Metrics (Important for Analysis - Calculate these in SQL):**
    *   **Total Ad Spend**: `COALESCE(sp_spends, 0) + COALESCE(sd_spends, 0)`.
    *   **Total Ad Clicks**: `COALESCE(clicks, 0)`. (Uses the combined `clicks` column).
    *   **Total Ad Impressions**: `COALESCE(sp_impressions, 0) + COALESCE(sd_impressions, 0)`.
    *   **Total Ad Attributed Sales**: `COALESCE(sp_ads_attributed_sales, 0) + COALESCE(sd_ads_attributed_sales, 0)`.
    *   **Total Ad Attributed Units**: `COALESCE(sp_ads_attributed_units, 0) + COALESCE(sd_ads_attributed_units, 0)`.
    *   **Average Selling Price (ASP)**: `SAFE_DIVIDE(SUM(product_sales), SUM(bussiness_report_units))`.
    *   **Overall Conversion Rate (Units per Session)**: `SAFE_DIVIDE(SUM(bussiness_report_units), SUM(sessions_total))`.
    *   **Ad Conversion Rate (Ad Units per Ad Click)**: `SAFE_DIVIDE(SUM(Total Ad Attributed Units), SUM(Total Ad Clicks))`.
    *   **Overall Ad Cost Per Click (CPC)**: `SAFE_DIVIDE(SUM(Total Ad Spend), SUM(Total Ad Clicks))`.
        *   **Note:** SP-specific or SD-specific CPC *cannot* be calculated directly as the `clicks` column is a combined total. The query should calculate Overall Ad CPC.
    *   **Overall Ad Click-Through Rate (CTR)**: `SAFE_DIVIDE(SUM(Total Ad Clicks), SUM(Total Ad Impressions))`.
        *   **Note:** SP-specific or SD-specific CTR *cannot* be reliably calculated without separate SP/SD click counts. This overall CTR uses total clicks and total (SP+SD) impressions.
    *   **SP Impression Share, SD Impression Share, Total Impression Share:** These *cannot be calculated* from this table as it does not contain data on total available impressions in the market (Impression Rank data is needed from advertising platforms).
    *   **ACOS (Advertising Cost of Sales)**: `SAFE_DIVIDE(SUM(Total Ad Spend), SUM(Total Ad Attributed Sales))`.
    *   **TACOS (Total Advertising Cost of Sales)**: `SAFE_DIVIDE(SUM(Total Ad Spend), SUM(product_sales))`. (Uses total product sales as denominator).
    *   **Organic Sales**: `GREATEST(0, SUM(product_sales) - SUM(Total Ad Attributed Sales))`. (Ensure non-negative).
    *   **Ad Sales to Organic Sales Ratio**: `SAFE_DIVIDE(SUM(Total Ad Attributed Sales), SUM(Organic Sales))`.
    *   **Number of Days (NOD) of Stock / Weeks of Cover (WOC)**: Calculate using `AVG(total_fba)` (VERIFY `total_fba` column name for FBA *quantity*) for a period and average daily/weekly sales of `bussiness_report_units` over a recent representative period (e.g., last 30 days).
        *   NOD = `SAFE_DIVIDE(AVG(total_fba), AVG_Daily_Units_Sold_Recent_Period)`.
        *   WOC = `SAFE_DIVIDE(AVG(total_fba), AVG_Weekly_Units_Sold_Recent_Period)`.
*   **Important Considerations for Certain Questions:**
    *   **Market Share / Overall Amazon Trends:** This table contains *your* performance data. Direct "market share" or "overall Amazon traffic trends" (not tied to your ASINs) are not columns here. Queries should focus on your metrics.
    *   **Competitor Data:** Direct data on competitor actions (deals, stock, pricing) is not in this table.
    *   **Content Changes (Titles, Images, A+):** This table tracks sales/traffic, *not* logs of when product content was modified. If a user provides an approximate date of change, query for performance metrics before and after that date.
    *   **Deals/Promotions (Coupons, PEDs):** Use `promotional_rebates` trends. A significant negative value (cost) in `promotional_rebates` concurrent with sales/unit spikes might *suggest* a promotion.
    *   **Impression Share:** This table lacks data for calculating true Impression Share. Queries should focus on your own `sp_impressions` and `sd_impressions` trends. State this limitation.
    *   Whenever the user asks a question involving a week, consider the week starting from the most recent Sunday before the question date, not from the date the question is asked.
**Core Table for Analysis:** `master_ledger_US`

(Many other columns exist, e.g., `onhand_mitou_cost`, `target_units`, `deliver_cost`, etc. Focus on the ones most relevant to sales, ads, traffic, and inventory for general e-commerce questions unless a very specific question targets those other columns.)
"""

SQL_GENERATION_GUIDELINES = f"""
**System Persona & Objective:**
You are an expert BigQuery SQL query generator. Your primary goal is to answer questions about sales, advertising, inventory, and product performance by querying the `{BQ_PROJECT_ID}.{BQ_DATASET_ID}.{BQ_TABLE_ID}` table in BigQuery, based on the detailed schema and calculation methods provided.

**General Querying and Analysis Guidelines:**
1.  **Understand the Question:**
    *   Identify primary entities (e.g., `parent` for "program," `sku`, `child_asin`, `parent, size`, `parent, colour`), key metrics, specific timeframes (daily, weekly, WoW, vs. N-week average), and the analytical intent.
    *   If a question uses terms like "program" or "product line," assume it refers to the `parent` column unless specified otherwise (e.g., "Parent-Size level").
    *   Break down complex, multi-part questions. Aim for a single SQL query (using CTEs) that gathers all necessary data.

2.  **Data Retrieval Strategy:**
    *   **Aggregation:** Use `SUM()` for transactional data (sales, units, spend, clicks, impressions, sessions) and `AVG()` for metrics like inventory (e.g., `AVG(total_fba)` - using the verified FBA quantity column) or ASP, or for averaging weekly totals over a period.
    *   **Time Granularity:** Adapt to the question (daily, weekly using `DATE_TRUNC(date_ordered, WEEK(SUNDAY))`, monthly, period-over-period).
    *   **Filtering:** Apply `WHERE` clauses accurately based on the question's criteria.

3.  **Specific Metric Interpretation & Calculation (Refer to Schema Description):**
    *   Use the derived metric definitions provided in the schema (ACOS, TACOS, Overall Ad CPC, Overall Ad CTR, Organic Sales, NOD, WOC, etc.). Ensure `COALESCE(column, 0)` for numerics in calculations.
    *   Use `SAFE_DIVIDE(numerator, denominator)` for all divisions.
    *   **Pay close attention to notes in the schema about what CANNOT be calculated (e.g., SP-specific CPC, SD-specific CPC, Impression Share). Do not attempt to generate SQL to calculate these specific breakdowns if the base data isn't there as specified in the schema.**
    *   For questions about metrics noted as "not calculable" in the schema, the SQL should instead retrieve trends of *related available data*. For example, if asked about "Impression Share," provide trends for `sp_impressions` and `sd_impressions`. The business summary will then explain the limitation.

4.  **Handling Time-Based Analysis & Comparisons:**
    *   Use `DATE_TRUNC` for weekly/monthly aggregations as specified in the schema.
    *   **For "current week vs. past N-week average" (e.g., "sales drop vs past 6 weeks"):**
        1.  Define "Current Week (CW)" period (typically last completed week, e.g., Sunday to Saturday).
        2.  Define "Past N-Weeks Baseline (PNWB)" period (e.g., the N weeks immediately prior to CW).
        3.  CTE 1: Calculate `SUM(metric)` or `AVG(metric)` for CW, grouped by `parent` (or other relevant dimensions like `parent, size`).
        4.  CTE 2: Calculate weekly `SUM(metric)` or `AVG(metric)` for PNWB, grouped by `week_start_date` and `parent` (or other dimensions).
        5.  CTE 3: From CTE 2, calculate `AVG(weekly_sum_metric)` or overall `SUM(metric)/N` for the PNWB, grouped by `parent` (or other dimensions).
        6.  `JOIN` CTE 1 and CTE 3 on the grouping dimensions to compare CW metric with PNWB average. Calculate differences and percentage changes.
    *   **For "Week-over-Week (WoW)" or "last X weeks vs. previous X weeks":** Use the date range patterns from the schema. Employ CTEs for each distinct period (e.g., `CurrentPeriodData`, `PreviousPeriodData`), then `JOIN` to compare.

5.  **Addressing "Why," "Reason For," "Contributing Factors," or "Impact Of" Questions:**
    *   Identify potential influencing factors from the schema (e.g., ad spend, CPCs, impressions, ASP, sessions, inventory levels (`total_fba`), `promotional_rebates`, organic sales proportion).
    *   The query should retrieve data for the primary metric of interest AND these potential influencing factors for the relevant time periods and entities.
    *   Example: If `sessions_total` dropped for a `parent`, also show changes in its `Total Ad Spend`, `Overall Ad CPC`, `Total Ad Impressions`, `ASP`, and average `total_fba` (FBA quantity) for the same periods.
    *   The goal is to provide data for *correlational analysis*. The summary will interpret these correlations.

6.  **Quantifying Impact (e.g., "How many units reduced because of X"):**
    *   Direct attribution is hard. The query should show the `change in units` (e.g., `bussiness_report_units` or `profitability_units`) alongside the `change in the potential influencing factor(s)` (e.g., ad spend, ASP) for the same items and periods.
    *   Example: For "how many units were reduced because of lower Ad Spends," list programs, their change in units, and their change in `Total Ad Spend`.

7.  **"Driver" Analysis (e.g., "is sales or traffic the primary driver of WoW change?"):**
    *   Calculate the WoW (or other relevant period) percentage change for sales (e.g., `product_sales` or `bussiness_report_units`) and for traffic (`sessions_total`).
    *   The query should output both percentage changes for the relevant entities. The summary will then highlight the metric with the larger absolute percentage change as a potential primary driver.

8.  **Handling Data Not Directly in Schema or Explicitly Unavailable:**
    *   **Competitor/Platform Data:** If questions involve competitor actions or broad Amazon platform trends not in `master_ledger_US`, generate SQL to retrieve *your company's* relevant metrics (sales, traffic, ad performance, ASP, inventory) for the specified period. The summary will then discuss these internal metrics in the context of the (unqueriable) external factors.
    *   **Content Change Dates:** If a question asks about impact of content changes (titles, images) and *does not* provide a date, state that the change date is not in the dataset. If a date *is* provided, query for performance metrics (conversion rate, sessions, units) before and after that date.
    *   **Deal/Promotion Flags:** Since explicit deal flags may be missing, use `promotional_rebates` as an indicator. If a question refers to deals, query for sales, units, ASP, and `promotional_rebates` trends. A significant negative `promotional_rebates` value (cost) concurrent with a sales spike suggests a promotion.
    *   **Impression Share:** If the schema explicitly states this cannot be calculated, focus the query on trends of `sp_impressions` and `sd_impressions`. The summary must clearly state the limitation.
    *   **SP/SD Specific CPC/CTR:** If the schema indicates only combined `clicks` is available, the query should calculate `Overall Ad CPC` and `Overall Ad CTR`. The summary must state that campaign-type specific CPC/CTR is not available from the provided `clicks` column.

9.  **Output Structure for Complex Questions:**
    *   The final `SELECT` statement should provide a comprehensive table. Each row might represent a `parent` (or `parent,size`, etc.), with columns showing the primary metric of interest (e.g., session change), changes in related factors (ad spend change, ASP change, inventory change), and unit changes.
    *   Use `ORDER BY` and potentially `LIMIT` if ranking or "top N" is implied.

10. **SQL Syntax and Best Practices:**
    *   Ensure the query is syntactically correct for BigQuery Standard SQL.
    *   Use `COALESCE(column, 0)` for numeric columns involved in calculations if they can be NULL, to avoid calculations resulting in NULL.
    *   Alias calculated fields clearly (e.g., `avg_asp`, `session_change_percentage`, `overall_ad_cpc`).
    *   Use `GREATEST(0, ...)` for calculations like Organic Sales to prevent negative values where illogical.

Only output the SQL query. Do not include any other text, comments, or explanations before or after the SQL query.
"""

# --- FastAPI App Initialization ---
app = FastAPI(
    title="NLP to BigQuery SQL API with Business Summary",
    description="Takes an NLP question, generates SQL, executes on BigQuery, returns results and a business summary.",
    version="1.4.0"  # Version bump for temperature feature
)

# --- Global Client Variables ---
bq_client: bigquery.Client | None = None
vertex_ai_initialized: bool = False


@app.on_event("startup")
async def startup_event():
    global bq_client, vertex_ai_initialized
    try:
        print(
            f"Initializing Vertex AI for Project: {PROJECT_ID}, Location: {LOCATION}")
        vertexai.init(project=PROJECT_ID, location=LOCATION)
        vertex_ai_initialized = True
        print("Vertex AI initialized successfully.")
    except Exception as e:
        print(f"FATAL ERROR during Vertex AI initialization: {e}")
        traceback.print_exc()
        vertex_ai_initialized = False  # Ensure this is set on failure
    try:
        print(f"Initializing BigQuery Client for Project: {BQ_PROJECT_ID}")
        bq_client = bigquery.Client(project=BQ_PROJECT_ID)
        print("BigQuery Client initialized successfully.")
    except Exception as e:
        print(f"FATAL ERROR during BigQuery Client initialization: {e}")
        traceback.print_exc()
        bq_client = None  # Ensure this is set on failure

# ==============================================================================
# Helper Functions
# ==============================================================================


def get_bigquery_table_schema(project_id: str, dataset_id: str, table_id: str) -> str:
    if bq_client is None:  # Check moved here as it's the first use
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                            detail="BigQuery client not initialized or unavailable.")
    if project_id == BQ_PROJECT_ID and dataset_id == BQ_DATASET_ID and table_id == BQ_TABLE_ID:
        return MASTER_LEDGER_US_SCHEMA_DESCRIPTION
    # Fallback for other tables (less likely to be used with current hardcoding but good practice)
    try:
        table_ref = bq_client.dataset(
            dataset_id, project=project_id).table(table_id)
        table = bq_client.get_table(table_ref)
        schema_parts = [
            f"- {field.name} ({field.field_type})" for field in table.schema]
        return f"Table: `{project_id}.{dataset_id}.{table_id}`\nColumns:\n" + "\n".join(schema_parts)
    except NotFound:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND,
                            detail=f"Table {project_id}.{dataset_id}.{table_id} not found.")
    except Exception as e:  # Catch other potential BQ client errors
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                            detail=f"Error fetching schema for {project_id}.{dataset_id}.{table_id}: {str(e)}")


def generate_sql_from_nlp(table_schema_info: str, nlp_question: str) -> str:
    if not vertex_ai_initialized:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                            detail="Vertex AI service not initialized or unavailable.")
    try:
        model = GenerativeModel(MODEL_NAME)
    except Exception as model_init_err:  # Catch specific model init errors
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                            detail=f"Could not initialize LLM model for SQL generation: {str(model_init_err)}")

    prompt_content = f"""
    {SQL_GENERATION_GUIDELINES}

    Full Schema Details of `{BQ_PROJECT_ID}.{BQ_DATASET_ID}.{BQ_TABLE_ID}`:
    {table_schema_info}

    Natural Language Question:
    {nlp_question}

    SQL Query:
    """
    print(
        f"\n--- Sending Prompt for SQL Generation (LLM: {MODEL_NAME}, Temp: {SQL_GENERATION_TEMPERATURE}) ---")
    # Limit logging of potentially large prompts
    print("SQL Gen Prompt (first 500 chars):",
          prompt_content[:500].replace('\n', ' ') + "...")
    try:
        generation_config = GenerationConfig(
            temperature=SQL_GENERATION_TEMPERATURE
        )
        response = model.generate_content(
            prompt_content,
            generation_config=generation_config
        )
        # Accessing the text content safely, considering potential API response structures
        generated_sql = ""
        if response and response.candidates and response.candidates[0].content and response.candidates[0].content.parts:
            generated_sql = response.candidates[0].content.parts[0].text
        else:  # Fallback for simpler response structure if any, or if error in response
            generated_sql = response.text if hasattr(response, 'text') else ""

        generated_sql = generated_sql.strip()

        # Clean up common LLM-added markdown for SQL blocks
        if generated_sql.startswith("```sql"):
            generated_sql = generated_sql[6:]
        elif generated_sql.startswith("```"):
            generated_sql = generated_sql[3:]
        if generated_sql.endswith("```"):
            generated_sql = generated_sql[:-3]
        generated_sql = generated_sql.strip()

        print(f"--- Generated SQL ---")
        print(generated_sql)
        print("---------------------\n")
        if not generated_sql:
            print("Warning: LLM returned an empty SQL query.")
        return generated_sql
    except ValueError as ve:  # Specific error for LLM content issues
        # Log the problematic response if possible, for debugging
        error_response_text = str(getattr(ve, 'args', 'No response details'))
        print(
            f"LLM content generation error. Response: {error_response_text[:500]}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"LLM failed to generate valid SQL content: {str(ve)}")
    except Exception as e:
        traceback.print_exc()
        error_detail = f"Error generating SQL from LLM: {str(e)}"
        # Safely access candidate and parts details if they exist in the exception
        candidates_info = getattr(e, 'candidates', None)
        parts_info = getattr(e, 'parts', None)
        if candidates_info:
            error_detail += f" Candidates: {candidates_info}"
        if parts_info:
            error_detail += f" Parts: {parts_info}"
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                            detail=error_detail)


def execute_bigquery_query(sql_query: str) -> list:
    if bq_client is None:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                            detail="BigQuery client not initialized or unavailable.")
    if not sql_query:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST,
                            detail="No SQL query provided for execution.")
    print(
        f"--- Executing BigQuery Query ---\n{sql_query}\n------------------------------")
    try:
        query_job = bq_client.query(sql_query)
        results = query_job.result(timeout=120)  # Set a timeout
        records = []
        for row in results:
            record = {}
            for key, value in row.items():
                if hasattr(value, 'isoformat'):  # Handles DATE, DATETIME, TIMESTAMP
                    record[key] = value.isoformat()
                elif isinstance(value, (float, int, str, bool)) or value is None:
                    record[key] = value
                # Attempt to convert other types (e.g., decimal.Decimal from NUMERIC)
                else:
                    try:
                        # Common conversion for NUMERIC to float
                        record[key] = float(value)
                    except (TypeError, ValueError):
                        # Fallback to string if float conversion fails
                        record[key] = str(value)
            records.append(record)
        print(
            f"--- Query Results ---\nRetrieved {len(records)} records.\n---------------------")
        return records
    except Exception as e:
        traceback.print_exc()
        # Provide more specific error message if it's a BQ API error
        error_message = f"Error executing BigQuery query: {str(e)}"
        if hasattr(e, 'errors') and e.errors:
            error_message += f"; Details: {e.errors[0]['message'] if e.errors[0] else 'Unknown BQ error'}"
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                            detail=error_message)


def generate_business_summary(
    nlp_question: str,
    sql_query: str,
    query_results: list,
    # Max rows to show in prompt for summary
    max_rows_for_summary_context: int = 10
) -> str:
    if not vertex_ai_initialized:
        print("Vertex AI service not available for summarization.")
        # Return a more informative message if summarization cannot proceed
        return f"Retrieved {len(query_results) if query_results else 0} records. Summary generation failed: Vertex AI unavailable."

    if query_results is None:  # Should ideally not happen if called correctly
        return "Internal error: Query results were unexpectedly None before summarization."

    try:
        model = GenerativeModel(MODEL_NAME)
    except Exception as model_init_err:
        print(
            f"Error initializing GenerativeModel for Summary: {model_init_err}")
        return f"Retrieved {len(query_results)} records. Error initializing summary model: {str(model_init_err)}"

    num_total_rows = len(query_results)
    context_data_str = ""
    if num_total_rows > 0:
        rows_to_show = query_results[:max_rows_for_summary_context]
        if rows_to_show:  # Ensure there are rows to get keys from
            column_names = list(rows_to_show[0].keys())
            context_data_str += f"Column Names: {', '.join(column_names)}\n"
            context_data_str += "Sample Data (first {} of {} total rows):\n".format(
                len(rows_to_show), num_total_rows)
            for row_idx, row_data in enumerate(rows_to_show):
                # Use json.dumps for better representation of complex row data
                context_data_str += f"Row {row_idx+1}: {json.dumps(row_data)}\n"
        if num_total_rows > max_rows_for_summary_context:
            context_data_str += f"... and {num_total_rows - max_rows_for_summary_context} more rows.\n"
        context_data_str += f"Total rows retrieved: {num_total_rows}\n"
    else:
        context_data_str = "The query returned no data (0 rows).\n"

    summary_prompt = f"""
    You are a business analyst. Your task is to provide a concise, insightful summary based on the results of a database query that was run to answer a user's question.

    Original Natural Language Question from User:
    "{nlp_question}"

    The SQL Query that was executed to get the data:
    ```sql
    {sql_query}
    ```

    Data Results from the Query:
    {context_data_str}

    Instructions for your summary:
    1. Directly and clearly answer the user's original question using the insights from the provided data.
    2. If the data shows specific items (like products, parents/programs, SKUs), mention the top few relevant ones if appropriate for the question (e.g., "The top programs with declining sessions are A, B, and C.").
    3. If the query returned numerical results (e.g., totals, averages, changes, percentages), state the key figures.
    4. If the query returned no data, clearly state that "The query returned no data that matched your criteria."
    5. Keep the summary brief (typically 2-5 sentences) and easy for a non-technical business user to understand.
    6. Do NOT just list the raw data. Provide an interpretation or summary of what the data means in context of the question.
    7. Do not make up information not present in the data. Stick to what the data implies.
    8. **Interpreting "Reasons," "Causes," or "Correlations":**
        *   If the question asked for "reasons," "causes," or "why," and the data provides metrics for potential influencing factors (e.g., ad spend changes, inventory level changes, ASP changes alongside changes in sales or sessions), discuss these as *correlations* or *potential contributing factors*.
        *   Example: "Sessions for Program X decreased by 15%. During the same period, ad spend for this program also fell by 20%, and average inventory was 30% lower. These factors *may have contributed* to the decline in sessions."
        *   Avoid stating definitive causation (e.g., "sessions dropped *because* ad spend decreased") unless the data explicitly supports direct attribution (like `sp_ads_attributed_sales`).
    9. **Addressing "How many units reduced/increased due to X":**
        *   If the question asks to quantify unit changes due to specific broad factors (e.g., "how many units were reduced because of Ads Spends"), and the query provides changes in units alongside changes in those factors:
            *   State the overall unit change for the relevant items.
            *   Then, mention the concurrent changes in the correlated factors. Example: "For programs with declining sessions, total units decreased by 500. This coincided with a $2000 aggregate reduction in ad spend and a 10% average increase in ASP for these programs."
            *   Do *not* attempt to assign a specific number of units lost *solely* to ad spend, ASP, or inventory from this general correlational data. Focus on presenting the concurrent changes as potential influences.
    10. **If the SQL query could not be fully answered due to data limitations mentioned in the schema (e.g., Impression Share not calculable, SP-specific CPC not calculable), acknowledge this limitation clearly in your summary.** Example: "Impression Share cannot be calculated from the available data. However, your total SP impressions were X and SD impressions were Y..."

    Business Summary:
    """
    print(
        f"\n--- Sending Prompt for Business Summary (LLM: {MODEL_NAME}, Temp: {SUMMARY_GENERATION_TEMPERATURE}) ---")
    print("Summary Prompt (first 500 chars):",
          summary_prompt[:500].replace('\n', ' ') + "...")
    try:
        generation_config = GenerationConfig(
            temperature=SUMMARY_GENERATION_TEMPERATURE
        )
        response = model.generate_content(
            summary_prompt,
            generation_config=generation_config
        )
        # Safely access text from response
        summary_text = ""
        if response and response.candidates and response.candidates[0].content and response.candidates[0].content.parts:
            summary_text = response.candidates[0].content.parts[0].text
        else:
            summary_text = response.text if hasattr(response, 'text') else ""

        summary_text = summary_text.strip()
        print(
            f"--- Generated Business Summary ---\n{summary_text}\n----------------------------------")
        return summary_text if summary_text else f"Retrieved {num_total_rows} records, but could not generate a specific summary."
    except Exception as e:
        traceback.print_exc()
        error_detail = f"Successfully retrieved {num_total_rows} records. An error occurred while generating a business summary: {str(e)}"
        # Safely access candidate and parts details if they exist in the exception
        candidates_info = getattr(e, 'candidates', None)
        parts_info = getattr(e, 'parts', None)
        if candidates_info:
            error_detail += f" Candidates: {candidates_info}"
        if parts_info:
            error_detail += f" Parts: {parts_info}"
        return error_detail

# --- Pydantic Models for Request and Response ---


class NLPQueryRequest(BaseModel):
    question: str


class DataTableContent(BaseModel):
    data: list
    columns: list


class QueryResponse(BaseModel):
    query: str
    sql_query: str | None  # SQL might not be generated if initial error
    dataframe_content: DataTableContent | None  # Data might not be fetched
    answer: str  # Business summary or error message

# --- FastAPI Endpoint ---


@app.post("/ask", response_model=QueryResponse)
async def ask_nlp_query(request: NLPQueryRequest):
    nlp_question = request.question
    print(f"\nReceived API Request for Question: \"{nlp_question}\"")

    if not vertex_ai_initialized:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                            detail="Vertex AI service failed to initialize on startup.")
    if bq_client is None:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                            detail="BigQuery client failed to initialize on startup.")

    generated_sql: str | None = None
    query_results: list | None = None
    df_content_for_response: DataTableContent | None = None
    # business_summary: str # Will be assigned or will be an error message

    try:
        table_schema = get_bigquery_table_schema(
            BQ_PROJECT_ID, BQ_DATASET_ID, BQ_TABLE_ID)
        generated_sql = generate_sql_from_nlp(table_schema, nlp_question)

        # --- Improved SQL Validation ---
        is_valid_query_structure = False
        is_safe_for_readonly = True  # Assume safe until proven otherwise
        validation_error_message = ""

        if not generated_sql or not generated_sql.strip():
            validation_error_message = "LLM returned an empty SQL query."
            print(
                f"VALIDATION FAILED: {validation_error_message} SQL: '{generated_sql}'")
        else:
            normalized_sql_for_matching = generated_sql.strip()
            match = re.match(r"^\s*(SELECT|WITH)\b",
                             normalized_sql_for_matching, re.IGNORECASE)

            if match:
                first_keyword = match.group(1).upper()
                if first_keyword == "WITH":
                    if re.search(r"\bSELECT\b", normalized_sql_for_matching, re.IGNORECASE):
                        is_valid_query_structure = True
                    else:
                        validation_error_message = "SQL starts with WITH but does not contain a SELECT statement."
                else:  # Starts with SELECT
                    is_valid_query_structure = True
            else:
                validation_error_message = "Generated SQL must start with SELECT or WITH."

            if not is_valid_query_structure:
                print(
                    f"VALIDATION FAILED (Structure): {validation_error_message} SQL: '{generated_sql}'")

            # Check for forbidden DML/DDL keywords only if structure is potentially valid
            if is_valid_query_structure:
                forbidden_keywords = [
                    "INSERT", "UPDATE", "DELETE", "DROP", "CREATE",
                    "ALTER", "TRUNCATE", "GRANT", "REVOKE", "MERGE"
                ]
                # Check for whole word, case-insensitive matches
                for keyword in forbidden_keywords:
                    if re.search(r'\b' + keyword + r'\b', normalized_sql_for_matching, re.IGNORECASE):
                        is_safe_for_readonly = False
                        validation_error_message = f"Generated SQL contains a disallowed keyword: {keyword}. Only SELECT queries are permitted."
                        print(
                            f"VALIDATION FAILED (Safety): {validation_error_message} SQL: '{generated_sql}'")
                        break  # Stop checking on first forbidden keyword

        if not is_valid_query_structure or not is_safe_for_readonly:
            if not validation_error_message:  # Generic message if a specific one wasn't set
                validation_error_message = "Generated SQL failed validation (structure or safety)."
            # For validation failures, we return the generated SQL (if any) for debugging, but no data and error as answer.
            return QueryResponse(
                query=nlp_question,
                sql_query=generated_sql,  # Show the problematic SQL
                dataframe_content=None,
                answer=validation_error_message
            )
        # --- End of Improved SQL Validation ---

        print("Generated SQL passed validation. Proceeding with execution.")
        query_results = execute_bigquery_query(generated_sql)

        if query_results is not None:  # query_results could be an empty list if query ran but returned 0 rows
            df_content_for_response = DataTableContent(
                data=query_results,
                # Get columns from first row if data exists
                columns=list(query_results[0].keys()) if query_results else []
            )

        # Generate summary even if query_results is an empty list (0 rows)
        business_summary = generate_business_summary(
            nlp_question, generated_sql, query_results if query_results is not None else [])

        return QueryResponse(
            query=nlp_question,
            sql_query=generated_sql,
            dataframe_content=df_content_for_response,
            answer=business_summary
        )

    except HTTPException as http_exc:  # Catch HTTPExceptions raised by our helper functions
        print(
            f"HTTPException in /ask: {http_exc.status_code} - {http_exc.detail}")
        # Return the generated SQL if available, for context
        return QueryResponse(
            query=nlp_question,
            sql_query=generated_sql,
            # Might be None if error occurred before BQ execution
            dataframe_content=df_content_for_response,
            answer=http_exc.detail
        )
    except Exception as e:  # Catch any other unexpected errors
        print(f"Unhandled error in /ask endpoint: {e}")
        traceback.print_exc()
        err_answer = f"An unexpected server error occurred: {str(e)}."

        # Attempt to build df_content if results were fetched before an error (e.g., in summarization)
        if query_results is not None and df_content_for_response is None:
            df_content_for_response = DataTableContent(
                data=query_results,
                columns=list(query_results[0].keys()) if query_results else []
            )

        return QueryResponse(
            query=nlp_question,
            sql_query=generated_sql,  # Include SQL if generated
            dataframe_content=df_content_for_response,  # Include data if fetched
            answer=err_answer
        )

# --- Health Check Endpoint ---


@app.get("/health", status_code=status.HTTP_200_OK)
async def health_check():
    bq_ok = bq_client is not None
    vertex_ok = vertex_ai_initialized
    overall_status = "ok" if bq_ok and vertex_ok else "error"
    # More detailed status
    services_status = {
        "bigquery_client": "ok" if bq_ok else "unavailable",
        "vertex_ai_initialized": "ok" if vertex_ok else "unavailable",
        "sql_generation_temperature": SQL_GENERATION_TEMPERATURE,
        "summary_generation_temperature": SUMMARY_GENERATION_TEMPERATURE
    }
    # Try a simple query to BQ as a deeper health check if client exists
    if bq_ok:
        try:
            bq_client.query("SELECT 1").result(timeout=5)  # Short timeout
            services_status["bigquery_connectivity"] = "ok"
        except Exception as bq_conn_err:
            print(f"Health check BQ connectivity error: {bq_conn_err}")
            services_status["bigquery_connectivity"] = "error"
            overall_status = "error"  # Degrade overall status if BQ connectivity fails
    else:
        services_status["bigquery_connectivity"] = "not_tested (client unavailable)"

    return {"status": overall_status, "services": services_status}

# --- Main execution block for running Uvicorn ---
if __name__ == "__main__":
    # Basic check for essential environment variables
    essential_vars = {
        "VERTEX_PROJECT_ID": PROJECT_ID,
        "LOCATION": LOCATION,
        "MODEL_NAME": MODEL_NAME,
        "BQ_PROJECT_ID": BQ_PROJECT_ID,
        "BQ_DATASET_ID": BQ_DATASET_ID,
        "BQ_TABLE_ID": BQ_TABLE_ID
    }
    missing_vars = [k for k, v in essential_vars.items(
    ) if not v or "your-" in str(v).lower() or "placeholder" in str(v).lower()]
    if missing_vars:
        print(
            f"CRITICAL ERROR: Essential configuration environment variable(s) are missing or using default placeholders: {', '.join(missing_vars)}")
        for var_name in missing_vars:
            print(f"{var_name}: {essential_vars[var_name]}")
        print("Please set these environment variables or update the script.")
    else:
        print("Configuration seems okay. Starting FastAPI server with Uvicorn...")
        print(f"SQL Generation Temperature: {SQL_GENERATION_TEMPERATURE}")
        print(
            f"Summary Generation Temperature: {SUMMARY_GENERATION_TEMPERATURE}")
        # Ensure the script filename matches "piagent:app" or adjust accordingly
        # The port 8091 was specified, you can change it as needed.
        # Assuming your script is named piagent.py
        uvicorn.run("piagent:app", host="0.0.0.0", port=8092, reload=True)
