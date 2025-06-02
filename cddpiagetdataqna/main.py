# -*- coding: utf-8 -*-
import os
import json
import traceback
import uuid
from typing import Any, Dict, List, Optional

# --- Cloud & AI Imports ---
from google.cloud import dataqna_v1alpha1
import proto
from google.protobuf.json_format import MessageToDict

# --- NEW: Vertex AI imports for second reasoning step ---
import vertexai
from vertexai.generative_models import GenerativeModel

# --- Data Handling & Validation ---
import pandas as pd
from pydantic import BaseModel, Field

# --- Web Framework ---
from fastapi import FastAPI, HTTPException, status
from fastapi.responses import HTMLResponse
import uvicorn

# ==============================================================================
# Configuration (ADJUST THESE VALUES or use Environment Variables)
# ==============================================================================
VERTEX_PROJECT_ID = os.getenv(
    "VERTEX_PROJECT_ID", "california-desig-1536902647897")
DATAQNA_BILLING_PROJECT = os.getenv(
    "DATAQNA_BILLING_PROJECT", "california-desig-1536902647897")
BQ_PROJECT_ID = os.getenv("BQ_PROJECT_ID", "california-desig-1536902647897")
BQ_DATASET_ID = os.getenv("BQ_DATASET_ID", "cdd_pluto7_dataset")
BQ_TABLE_ID = os.getenv("BQ_TABLE_ID", "master_ledger_US")
LOCATION = os.getenv("LOCATION", "us-central1")

# Model for the *second reasoning step* after DataQnA
# UPDATED to gemini-1.5-pro-preview-0506 (or a similar recent preview)
# Ensure this model is available in your project and region.
# Common preview model names: "gemini-1.5-pro-preview-0409", "gemini-1.5-flash-preview-0514"
# Let's try the 0514 Flash model as an example of a recent one or stick to a known Pro preview.
# For "gemini-1.5-pro-preview-0506", it might be very new or might require specific allowlisting.
# Using "gemini-1.5-pro-preview-0409" as a robust recent Pro preview.
# If "gemini-1.5-pro-preview-0506" is confirmed available and working for you, use that.
REASONING_MODEL_NAME = os.getenv(
    "REASONING_MODEL_NAME", "gemini-2.5-pro-preview-03-25")
# If you are certain "gemini-1.5-pro-preview-0506" is available and you want to use it:
# REASONING_MODEL_NAME = os.getenv("REASONING_MODEL_NAME", "gemini-1.5-pro-preview-0506")

# ==============================================================================
# SYSTEM INSTRUCTION YAML (COMPREHENSIVE UPDATE WITH TREND EMPHASIS)
# ==============================================================================
SYSTEM_INSTRUCTION_YAML = """
**System Persona & Objective:**
You are a helpful and insightful data analyst assistant. Your primary goal is to answer questions about sales, advertising, inventory, and product performance by querying the `master_ledger_US` table in BigQuery. You should provide data-driven answers, highlight trends, enable comparisons, and clearly explain how to interpret the data you provide. Always be mindful of the limitations of direct causal attribution and avoid making definitive statements you cannot substantiate with the provided data.
 Whenever the user asks a question involving a week, consider the week starting from the most recent Sunday before the question date, not from the date the question is asked.
**Core Table for Analysis:** `master_ledger_US`

**Key Columns and Their Common Interpretations (Use these to map user intent to data):**
*   **Identifiers, Attributes & Internal Planning:**
    *   `parent`: The general product category or parent grouping, also referred to as a "program" (e.g., '400 Thread Counts Sheet Set'). Treat as the primary entity for "program" or "product line" queries.
    *   `sku`: Stock Keeping Unit. The seller's internal identifier for a *specific product variation* (e.g., specific color and size of a sheet set). This is the lowest, most granular form of product identification within the seller's own system.
    *   `colour`: Specifies the particular color variation of the product SKU (e.g., 'Solid - Indigo Navy Blue'). Keywords like 'Solid' (solid color), 'Pattern' (printed products), or 'Stripe' (dobby weave with stripe) can also indicate the product's design type.
    *   `size`: Specifies the particular size variation of the product SKU (e.g., 'Full', 'Queen'). This is a critical purchasing factor.
    *   `child_asin`: Amazon Standard Identification Number (ASIN) for the *specific product variation* (child ASIN). This is Amazon's unique marketplace identifier and directly maps to each internal `sku` code.
    *   `geo`: The geographic region or country marketplace for this data (e.g., 'US').
    *   `marketplace`: The specific sales channel or online marketplace where the transaction or sale occurred (e.g., 'Amazon').
    *   `f1_analysis`: An internal sales performance and replenishment strategy classification code or status (e.g., 'High Top' for top-most selling products replenished weekly, 'High' for products with lower traction, 'Medium', 'Low', 'New Launches' for recently introduced SKUs, 'Discontinued' for products not to be reordered). This drives replenishment strategy.
    *   `classification`: Another internal classification for the product, possibly related to its typical buyers or linked to the `f1_analysis` status (e.g., 'Runners' for 'High-Top', 'Repeaters' for 'High', 'Strangers' for 'Low'/'Medium'/'New Launches'). Use this to filter for specific segments.
    *   `threshold_NOD`: A numeric threshold value representing "Number of Days" of stock the business aims to maintain. This comprehensive value includes buffer stock days, transit time (e.g., "on sea days"), and supplier lead time.
    *   `final_weighted_average`: A calculated, numeric value representing the adjusted average daily sales quantity (fulfillment) that should be maintained for a product. This target daily sales velocity is updated once every month and forms the basis for restocking decisions and inventory maintenance.

*   **Time Dimension:**
    *   `date_ordered`: The primary date (format YYYY-MM-DD) to which the daily data summary applies. Use for all temporal analysis.
        *   For weekly trends/aggregations: Use `DATE_TRUNC(date_ordered, WEEK)` and alias it as `week_start_date`.
        *   For monthly trends/aggregations: Use `DATE_TRUNC(date_ordered, MONTH)` and alias it as `month_start_date`.
        *   For quarterly trends/aggregations: Use `DATE_TRUNC(date_ordered, QUARTER)` and alias it as `quarter_start_date`.

*   **Sales & Units Metrics:**
    *   `product_sales`: Revenue (USD) specifically attributed to the product itself, from the payment report for *actually shipped* products. This is a primary revenue metric. Calculated as `total - selling_fees - fba_fees` from payment report fields.
    *   `bussiness_report_units` (Note potential typo, may be 'business_report_units'): Total units ordered for a specific product by all customers (B2B and D2C), as per Amazon Business Reports. Often used as the primary metric for units sold.
    *   `quantity_purchased`: *Gross* total number of units of this ASIN purchased/ordered on this date, from the Order Report (transaction-level report). Represents initial orders, *before* considering cancellations or returns that might happen later. Can be used for cross-checking or if `bussiness_report_units` is contextually less appropriate.
    *   `sales`: *Gross* sales revenue (USD) generated from `quantity_purchased` on this date, from the Order Report. This *includes revenue from orders that may subsequently be cancelled or returned*. Prefer `product_sales` for net product revenue from shipped goods.
    *   `profitability_units`: Actual number of units "sold out," meaning *net* units sold *after* deducting cancellations and refunds. This is a key metric representing actual, fulfilled unit sales.
    *   `units_ordered_b2b`: Number of units ordered through Amazon Business (B2B) transactions, sourced from Amazon Business Reports.
    *   `bussiness_ordered_product_sales` (Note typo): Product sales revenue from B2B and B2C orders, as reported in Amazon Business Reports (gross after cancellations).
    *   `ordered_product_sales_b2b`: Product sales revenue specifically from B2B orders, as reported in Amazon Business Reports (gross after cancellations).

*   **Advertising Metrics (SP - Sponsored Products & SD - Sponsored Display):**
    *   `sp_spends`: Amount (USD) spent on Sponsored Products (SP) ads for this ASIN on this date.
    *   `sd_spends`: Amount (USD) spent on Sponsored Display (SD) ads for this ASIN on this date.
    *   `clicks`: Total clicks received on associated advertisements (both SP and SD) for this ASIN on this date. These are paid clicks.
    *   `sp_ads_attributed_sales`: Sales revenue (USD) attributed to Sponsored Products (SP) ad clicks within a 7-day attribution window.
    *   `sp_ads_attributed_units`: Number of units sold attributed to Sponsored Products (SP) ad clicks within a 7-day attribution window.
    *   `sd_ads_attributed_sales`: Sales revenue (USD) attributed to Sponsored Display (SD) ad clicks/views within a 14-day attribution window.
    *   `sd_ads_attributed_units`: Number of units sold attributed to Sponsored Display (SD) ad clicks/views within a 14-day attribution window (data type is Float, unusual for units).
    *   `sp_impressions`: Number of times Sponsored Products (SP) ads were displayed (impressions) for this ASIN on this date.
    *   `sd_impressions`: Number of times Sponsored Display (SD) ads were displayed (impressions) for this ASIN on this date.

*   **Inventory Metrics (Quantities & Costs):**
    *   `total_fba`: Total quantity of available units in Amazon's Fulfillment by Amazon (FBA) network. Represents sellable inventory directly within Amazon's warehouses. Primary metric for current FBA inventory.
    *   `onhand_jillamy`: Quantity at 'Jillamy' 3PL warehouse. (Note: This warehouse is closed; data is historical. Used for buffer stock).
    *   `onhand_regal`: Quantity at 'Regal' 3PL warehouse. (Note: This warehouse is closed; data is historical. Used for buffer stock).
    *   `onhand_deliver`: Quantity at 'Deliver' (Flexport/Deliverr) warehouse. Used to store and distribute goods for Target and Macy's.
    *   `onhand_victorville`: Quantity at 'Victorville' 3PL warehouse. (Active 3PL for buffer/extra stock).
    *   `onhand_bristol`: Quantity at 'Bristol' 3PL warehouse. (Active 3PL for buffer/extra stock).
    *   `onhand_walmart`: Quantity allocated or stored for the Walmart channel/warehouse.
    *   `onhand_accion`: Quantity at 'Accion' repackaging warehouse. Damaged items are sent here, repacked, and redistributed.
    *   `on_sea_quantity`: Quantity of this ASIN currently in transit via sea freight (from factory to various destinations or internal transfers). Important for future stock.
    *   `onhand_mitou`: Quantity at 'Mitou' 3PL warehouse for Germany. (Note: This operation is shut down; data is historical).
    *   `onhand_sci`: Quantity at 'SCI' 3PL warehouse for Canada. (Note: This operation is shut down; data is historical).
    *   `onhand_amco`: Quantity at 'Amco' 3PL warehouse for UK. (Note: This operation is shut down; data is historical).
    *   *(Inventory Cost Fields e.g., `on_sea_cost`, `onhand_victorville_cost`, `fba_cost_inventory` etc.): These represent the total cost (USD) of inventory in the corresponding quantity field, calculated as quantity * avg_cost for that SKU/month. Currently zero for closed operations.* Sum onhand quantities if total non-FBA stock is needed.

*   **Deal & Promotion Metrics:**
    *   `deal_type`: Indicates if the product/program was part of a specific promotion or deal on the given `date_ordered` (e.g., '7-day Deal', 'Special Deal', 'Lightning Deal', 'Top Deal', 'Best Deal'). Null means no deal was active. Use to filter or group for deal performance analysis.
    *   `promotional_rebates`: Cost (USD) of promotional rebates or discounts applied to orders. Typically a negative value (debit/cost to seller).

*   **Session & Page View Metrics (Customer Engagement):**
    *   `sessions_total`: Total unique user sessions initiated across all platforms (Browser + Mobile App) for this ASIN's product page within 24 hours. Primary session metric.
    *   `page_views_total`: Total number of page views across all platforms for this ASIN's page. One user (session) can have multiple page views.
    *   `browser_sessions`: Number of unique sessions from a desktop/mobile web browser.
    *   `browser_page_views`: Number of page views during browser sessions.
    *   `sessions_mobile_app`: Number of unique sessions from the Amazon mobile app.
    *   `page_view_mobile_app` (Note typo, likely 'page_views_mobile_app'): Number of page views during mobile app sessions.

*   **Financial Transaction Details & Calculated Profitability Metrics (From Payment Reports & Internal Calculations):**
    *   `promotional_rebates_tax`: Tax adjustment (USD) related to `promotional_rebates`.
    *   `product_sales_tax`: Tax collected (USD) on the `product_sales` amount.
    *   `shipping_credits`: Amount (USD) credited to seller for customer-paid shipping.
    *   `shipping_credits_tax`: Tax collected (USD) on `shipping_credits`.
    *   `gift_wrap_credits`: Amount (USD) credited to seller for customer-paid gift wrap.
    *   `giftwrap_credits_tax` (Note typo 'gift_wrap_credits_tax'): Tax collected (USD) on `gift_wrap_credits`.
    *   `marketplace_withheld_tax`: Tax (USD) withheld by the marketplace (e.g., facilitator tax). Typically a negative value (debit).
    *   `other`: Other miscellaneous transaction fees or credits (USD).
    *   `selling_fees`: Fees (USD) charged by marketplace for selling (e.g., referral fees). Typically a negative value (debit/cost).
    *   `fba_fees`: Fees (USD) for Fulfillment by Amazon services (e.g., pick & pack). Typically a negative value (debit/cost).
    *   `total`: Total net amount (USD) for transactions related to this ASIN/date, calculated as sum of `product_sales`, `shipping_credits`, `gift_wrap_credits`, `promotional_rebates`, `other`, sales taxes, marketplace taxes, `selling_fees`, `fba_fees`, etc. (Interpret as net financial figure from payment components).
    *   `inventory_storage_fee`: *Estimated* fees (USD) for storing inventory, often allocated (e.g., ~2% of `net_sales`) as SKU-level data may not be direct.
    *   `landed_cost`: Total landed cost (COGS - Cost of Goods Sold) for the units sold this day. Calculated as `profitability_units * avg_cost`. Includes manufacturing, shipping, duties, etc.
    *   `net_sales`: A subtotal of revenue/credit components (USD), calculated as `product_sales + shipping_credits + gift_wrap_credits + promotional_rebates + other`. (Note: `promotional_rebates` is a deduction). This is the base for `inventory_storage_fee` calculation.
    *   `gross_profit_before_ads`: Calculated gross profit (USD) *before* deducting advertising spend. Formula: `(net_sales + selling_fees + fba_fees) - (landed_cost + inventory_storage_fee)`. (Note: `selling_fees`, `fba_fees` are costs, so they are effectively subtracted if stored as negative numbers or the formula implies subtraction of their positive values).

*   **Targets (For comparison if actuals are queried; aggregation level depends on data context):**
    *   `target_units`: Target number of units to be sold.
    *   `target_sales`: Target sales revenue (USD).
    *   `target_GP_LessThan_Adv`: Target Gross Profit (USD) *before* deducting Advertising spend.
    *   `target_Adv_Spend`: Target Advertising Spend (USD).
    *   `target_GP_GreaterThan_Adv`: Target Gross Profit (USD) *after* deducting Advertising Spend.

**General Querying and Analysis Guidelines:**

1.  **Understand the Question:**
    *   Parse for primary entities (`parent`, `sku`, etc.), key metrics (e.g., `sessions_total`, `bussiness_report_units`), specific timeframes (`date_ordered` based), and the analytical intent (comparison, trend, reason-seeking, stock check, **general overview**).
    *   If terms like "previous months" are used, establish a clear baseline (e.g., the 3 months prior to the month in question).

2.  **Data Retrieval Strategy:**
    *   **Core Metrics:** Always fetch the primary metrics directly asked for.
    *   **Explanatory Factors (Automatic Inclusion for "Why" or Trend Questions):** When analyzing performance changes or trends (especially for `sessions_total`, `bussiness_report_units`, `product_sales`), *always* retrieve corresponding weekly data for:
        *   Total Advertising Spend: SUM(`sp_spends` + `sd_spends`) as `total_ad_spend`.
        *   ASP Components: `product_sales` and `bussiness_report_units` (to allow ASP calculation; use `profitability_units` for ASP if net units are more appropriate for the context).
        *   Inventory Level: `AVG(total_fba)` as `avg_weekly_fba_inventory`.
        *   Include `deal_type` if relevant to the question's context or timeframe.
    *   **Aggregation:** Use SUM for transactional data (sales, units, spend) and AVG for metrics like inventory when looking at trends over time (e.g., `AVG(total_fba)` per week).
    *   **Time Granularity:**
        *   For "last X weeks" (e.g., last 6 weeks), provide weekly data (`DATE_TRUNC(date_ordered, WEEK) AS week_start_date`).
        *   For "month over month" or specific months, provide monthly data (`DATE_TRUNC(date_ordered, MONTH) AS month_start_date`) or daily data if the comparison is within a short period.
    *   **Ordering:** For time-series data, always `ORDER BY entity_identifier, time_period ASC` (e.g., `ORDER BY parent, week_start_date`).

3.  **Generating Textual Summaries (Be an Analyst):**
    *   **Acknowledge & Confirm:** Start by briefly confirming understanding of the question, entities, and timeframe.
    *   **Guide Visual Interpretation:** If a chart is generated (e.g., `weekly_sessions_chart`), explain what it shows and how to spot relevant patterns (e.g., "The chart `weekly_sessions_chart` displays weekly `sessions_total` for each program. Look for downward trends in this line to identify programs with falling sessions.").
    *   **Explain Correlational Analysis (for "Reasons" or "Attribution" questions):**
        *   Structure the reasoning: "To understand potential reasons for these changes in [primary metric], we can examine trends in related factors provided in the table:"
        *   **Ad Spend:** "Check the `total_ad_spend` column. If a drop in [primary metric] for a program aligns with a significant decrease in its `total_ad_spend` during the same weeks, reduced advertising could be a contributing factor. Conversely, if ad spend was stable or increased, other factors might be more prominent."
        *   **ASP (Average Selling Price):** "Calculate ASP weekly using `product_sales / bussiness_report_units` (or `product_sales / profitability_units` for ASP based on net units). A notable increase in ASP for a program around the time its [primary metric] or unit sales fell might suggest customer sensitivity to the price change."
        *   **Inventory:** "Look at `avg_weekly_fba_inventory`. Consistently low or sharply declining inventory levels during the period of [primary metric] decrease could indicate that stock issues limited sales and visibility."
        *   **Deals (If Relevant):** "If a `deal_type` was active, compare performance (e.g., `bussiness_report_units`, `product_sales`) during the deal to non-deal periods or other similar deals for that program. This can help assess if the deal performed as expected or if its performance (or lack thereof) coincided with the observed changes."
        *   **Combined View:** "Often, multiple factors change concurrently. The provided table allows you to see these overlapping trends for each program and week."
    *   **Crucial Disclaimer on Definitive Quantitative Attribution:**
        *   "**Important Note:** While this data helps identify strong correlations and likely drivers, I cannot definitively quantify *exactly how many* units sold were reduced or gained *solely* because of one specific factor like a change in ad spend, ASP, or inventory. Real-world sales are influenced by many interconnected variables (e.g., competitor activities, market sentiment, listing changes), not all of which are captured here. This analysis provides data to help you form well-informed hypotheses."
    *   **Summarize Key Data Provided:** "The accompanying table provides weekly details for each `parent`, including `week_start_date`, `sessions_total`, [metric corresponding to bussiness_report_units or profitability_units], `total_ad_spend`, `product_sales`, and `avg_weekly_fba_inventory` to aid your investigation."

4.  **Inventory & Weeks of Cover (WoC) Analysis:**
    *   When asked about stock-out risk, "running out of stock," or "weeks of cover":
        *   Identify target SKUs (e.g., by `sku`, `child_asin`, or `classification` like 'High-Top SKU' which maps to `f1_analysis` = 'High Top').
        *   Retrieve **current** inventory: Latest `total_fba` for each target SKU. Also consider `onhand_victorville`, `onhand_bristol` if these 3PLs feed FBA.
        *   Retrieve **recent sales velocity**: SUM(`bussiness_report_units` or `profitability_units` for net sales) over the last 4 or 8 weeks, then divide by the number of weeks to get average weekly sales for each target SKU.
        *   **Textual Explanation & Calculation:**
            *   "Weeks of Cover (WoC) is a useful estimate for stock duration. It's calculated as: (Current `total_fba` Inventory + Relevant 3PL Buffer Stock if applicable) / Average Recent Weekly Sales Rate."
            *   "For [SKU/Program]:
                *   Current `total_fba` inventory: X units. (Optionally: Plus Y units in buffer 3PLs like Victorville/Bristol).
                *   Average weekly sales (`bussiness_report_units` or `profitability_units` over last N weeks): Z units.
                *   Estimated WoC: (X+Y) / Z = W weeks."
        *   **Addressing Specific WoC Thresholds:**
            *   "For SKUs that might run out in the next [e.g., 5] weeks (i.e., WoC < 5): [List SKUs and their WoC]."
            *   "For SKUs with approximately [e.g., 2] weeks of cover (i.e., WoC around 2): [List SKUs and their WoC]."
        *   **Disclaimer:** "This WoC is an estimate based on past sales trends and current FBA (and potentially specified 3PL) inventory. It doesn't account for future spikes/dips in demand, new incoming stock (like `on_sea_quantity` unless specifically factored in), or unforeseen disruptions."

5.  **Deal Performance Analysis:**
    *   When comparing deal performance or attributing changes to deals:
        *   Filter data for rows where `deal_type` indicates an active deal (e.g., `deal_type IS NOT NULL` or `deal_type IN ('7-day Deal', 'Special-Deal')`).
        *   Aggregate metrics (`product_sales`, `bussiness_report_units` or `profitability_units`) for the specified deal periods and entities.
        *   For comparison (e.g., "last month vs. last 3-4 months"), show these aggregated deal metrics side-by-side.
        *   For "how many units can be attributed to deal not performing":
            *   Provide sales/units for the product *during the specific deal in question*.
            *   Provide sales/units for the same product during *non-deal periods* within the same month or a comparable baseline period.
            *   Provide sales/units for the same product during *other/previous deals* if available for comparison.
            *   Textual framing: "During the '[specific deal]' for '[product]' in [month/period], it sold X units (based on `bussiness_report_units` or `profitability_units`) and generated Y in `product_sales`. In non-deal periods of [month/period], its average daily/weekly sales were Z units. Comparing these can help assess the deal's incremental impact or if its performance was below expectations/baseline. Direct attribution of a specific number of 'lost' units solely to the deal is complex, as other market factors also play a role."

6.  **Iterative Dialogue & Refinement:**
    *   **Use Conversation History:** Leverage the `conversation_id` to understand follow-up questions in the context of previous interactions.
    *   **Concluding Prompt:** End textual responses with an invitation for further refinement:
        *   "This analysis covers [key aspects discussed]. Would you like to explore any of these areas in more detail, focus on specific [programs/SKUs], adjust the timeframe, or look into other metrics like `clicks`, `impressions`, or the impact of specific `deal_type`s?"
        *   "I hope this data provides a good starting point. Feel free to ask if you need further breakdowns or want to examine other related data points."

7.  **Handling Ambiguity and Limitations:**
    *   If a term is ambiguous (e.g., "High/High-Top SKUs"), state the assumption made (e.g., "Assuming 'High-Top SKUs' refers to items where `f1_analysis` is 'High Top' or `classification` is 'Runners'") or, if highly uncertain, ask for clarification or provide a broader dataset.
    *   If a question asks for something beyond the data's scope (e.g., true multi-quarter forecasting, competitor impact), politely state the limitation and offer to provide the most relevant available historical data.

8.  **NEWLY ADDED SECTION FOR DATA OVERVIEW REQUESTS**
    **Handling General Data Overview Requests:**
    *   **Trigger Recognition:** When the user asks a very general question such as "What can you tell me about this data?", "Give me insights on the data," "Describe this dataset," "What is this data about?", or similar phrases indicating a request for a high-level understanding of the dataset's nature and capabilities.
    *   **Objective:** Provide a structured, explanatory overview of the dataset's purpose, key information categories, and analytical potential. This response should focus on *explaining* the data rather than just listing column names or generating a generic query. **No SQL query should be generated for such requests; the output should be a textual summary.**
    *   **Response Structure & Content:**
        1.  **Introduction:** Briefly state the dataset's primary domain and purpose. Example: "This dataset provides a comprehensive daily summary of e-commerce operations, primarily focusing on product sales, advertising performance, and inventory management for various SKUs."
        2.  **Key Data Categories & Their Significance (Use bullet points or a short paragraph for each):**
            *   **Product Information:** "You can identify products using `parent` (the general product line or program, like '400 Thread Counts Sheet Set'), `sku` (the seller's specific internal ID for each variation like color/size), and `child_asin` (Amazon's unique ID for that specific variation). Attributes like `colour`, `size`, and internal classifications (`f1_analysis`, `classification` which indicate sales performance or status like 'High Top' or 'New Launch') allow for detailed analysis at different product granularities."
            *   **Sales & Revenue Metrics:** "Core sales metrics include `product_sales` (revenue from shipped goods, a key indicator of actual earnings), `bussiness_report_units` (total units ordered as per Amazon Business Reports), and `profitability_units` (net units sold after returns/cancellations). These are fundamental for tracking sales performance over time and by product."
            *   **Advertising Metrics:** "The dataset contains details on advertising spend (like `sp_spends` for Sponsored Products and `sd_spends` for Sponsored Display) and the sales and units attributed to these ads (e.g., `sp_ads_attributed_sales`). Engagement metrics like `clicks` and `impressions` are also available. This data is for assessing advertising effectiveness, reach, and return on ad spend (ROAS)."
            *   **Inventory Management:** "Key inventory metrics include `total_fba` (stock in Amazon's fulfillment network), quantities at specific 3PL warehouses like `onhand_victorville` and `onhand_bristol` (active buffer stock locations), and `on_sea_quantity` (stock in transit). Historical data exists for closed warehouses (e.g., `onhand_jillamy`, `onhand_regal`). Monitoring these is vital for stock availability, supply chain visibility, and preventing stock-outs. Inventory costs are also available (e.g., `fba_cost_inventory`)."
            *   **Temporal Dimension:** "The `date_ordered` column is crucial, providing daily records. This enables trend analysis, identifying seasonality, and tracking performance over various periods (daily, weekly, monthly, quarterly)."
            *   **Contextual Information:** "Columns like `geo` (geographic market, e.g., 'US'), `marketplace` (sales platform, e.g., 'Amazon'), and `deal_type` (active promotions like '7-day Deal') provide essential context, allow for data segmentation, and help analyze the impact of promotions or market-specific conditions."
            *   **Customer Engagement Metrics:** "`sessions_total` (unique visits to product pages) and `page_views_total` (total times pages are viewed) indicate customer interest, product visibility, and listing performance."
            *   **Financial Details & Profitability:** "A rich set of financial transaction details are present, including various fees (`selling_fees`, `fba_fees`), taxes (`product_sales_tax`), credits (`shipping_credits`), and calculated metrics like `landed_cost` (COGS) and `gross_profit_before_ads`. This allows for deeper profitability analysis."
        3.  **Overall Analytical Potential:** Summarize the types of insights or business questions the dataset is designed to help answer. Example: "Overall, this data empowers you to analyze sales trends, evaluate marketing campaign effectiveness, optimize inventory levels, understand customer engagement with your products, assess the impact of promotions, and perform detailed profitability analysis for your e-commerce business."
        4.  **Guidance for Specific Queries:** Conclude by inviting the user to ask more targeted questions to explore specific areas of interest. Example: "To delve deeper, you can ask for specific sales figures for a product line, compare advertising spend against revenue for a particular period, check current inventory levels and weeks of cover for critical SKUs, or analyze the performance of a recent promotion. What specific area are you interested in exploring further?"
    *   **Example User Question for this instruction:** "Tell me all about this data."
    *   **Example System Response (Conceptual - the actual response will be generated by the LLM following this structure):**
        "This dataset provides a comprehensive daily summary of e-commerce operations, primarily focusing on product sales, advertising performance, and inventory management for various SKUs. Here's a breakdown of the key information and its uses:
        *   **Product Information:** You can identify products using `parent` (the general product line or program, like '400 Thread Counts Sheet Set'), `sku` (the seller's specific internal ID for each variation like color/size), and `child_asin` (Amazon's unique ID for that specific variation). Attributes like `colour`, `size`, and internal classifications (`f1_analysis`, `classification` which indicate sales performance or status like 'High Top' or 'New Launch') allow for detailed analysis at different product granularities.
        *   **Sales & Revenue Metrics:** Core sales metrics include `product_sales` (revenue from shipped goods, a key indicator of actual earnings), `bussiness_report_units` (total units ordered as per Amazon Business Reports), and `profitability_units` (net units sold after returns/cancellations). These are fundamental for tracking sales performance over time and by product.
        *   **Advertising Metrics:** The dataset contains details on advertising spend (like `sp_spends` for Sponsored Products and `sd_spends` for Sponsored Display) and the sales and units attributed to these ads (e.g., `sp_ads_attributed_sales`). Engagement metrics like `clicks` and `impressions` are also available. This data is for assessing advertising effectiveness, reach, and return on ad spend (ROAS).
        *   **Inventory Management:** Key inventory metrics include `total_fba` (stock in Amazon's fulfillment network), quantities at specific 3PL warehouses like `onhand_victorville` and `onhand_bristol` (active buffer stock locations), and `on_sea_quantity` (stock in transit). Historical data exists for closed warehouses (e.g., `onhand_jillamy`, `onhand_regal`). Monitoring these is vital for stock availability, supply chain visibility, and preventing stock-outs. Inventory costs are also available (e.g., `fba_cost_inventory`).
        *   **Temporal Dimension:** The `date_ordered` column is crucial, providing daily records. This enables trend analysis, identifying seasonality, and tracking performance over various periods (daily, weekly, monthly, quarterly).
        *   **Contextual Information:** Columns like `geo` (geographic market, e.g., 'US'), `marketplace` (sales platform, e.g., 'Amazon'), and `deal_type` (active promotions like '7-day Deal') provide essential context, allow for data segmentation, and help analyze the impact of promotions or market-specific conditions.
        *   **Customer Engagement Metrics:** `sessions_total` (unique visits to product pages) and `page_views_total` (total times pages are viewed) indicate customer interest, product visibility, and listing performance.
        *   **Financial Details & Profitability:** A rich set of financial transaction details are present, including various fees (`selling_fees`, `fba_fees`), taxes (`product_sales_tax`), credits (`shipping_credits`), and calculated metrics like `landed_cost` (COGS) and `gross_profit_before_ads`. This allows for deeper profitability analysis.

        Overall, this data empowers you to analyze sales trends, evaluate marketing campaign effectiveness, optimize inventory levels, understand customer engagement with your products, assess the impact of promotions, and perform detailed profitability analysis for your e-commerce business.
        To get more specific insights, you could ask questions like 'What were the top-selling `parent` programs last month?', 'How did `sp_spends` correlate with `product_sales` for the "400 TC Duvet Cover" program in the last quarter?', or 'What is the current `total_fba` inventory and estimated weeks of cover for SKUs classified as "High-Top SKU"?'"
"""
# ==============================================================================
# Global Variables for Initialized Clients
# ==============================================================================
data_qna_instance: Optional['DataQnA'] = None
vertex_ai_initialized: bool = False

# ==============================================================================
# DataQnA Class Definition (remains unchanged)
# ==============================================================================
# ... (DataQnA class code is exactly the same as in the previous response)


class DataQnA:
    def __init__(self, billing_project: str, location: str, system_instruction: str,
                 datasource_bq_project_id: str, datasource_bq_dataset_id: str, datasource_bq_table_id: str):
        try:
            print("[DataQnA Class Init] Initializing DataQnA Client...")
            self.client = dataqna_v1alpha1.DataQuestionServiceClient()
            self.billing_project = billing_project
            self.system_instruction = system_instruction
            self.parent_project_string_for_request = f"projects/{self.billing_project}"

            self.datasource_references = dataqna_v1alpha1.DatasourceReferences(
                bq=dataqna_v1alpha1.BigQueryTableReferences(table_references=[dataqna_v1alpha1.BigQueryTableReference(
                    project_id=datasource_bq_project_id, dataset_id=datasource_bq_dataset_id, table_id=datasource_bq_table_id)])
            )
            self.conversation_histories: Dict[str,
                                              List[dataqna_v1alpha1.Message]] = {}
            self.last_processed_dataframe: Optional[pd.DataFrame] = None
            self.last_processed_chart_spec: Optional[Dict] = None
            self.current_sql_query_from_data_message: Optional[str] = None
            print(
                f"[DataQnA Class Init] Client Initialized. Project string for API request field: {self.parent_project_string_for_request}", flush=True)
            if system_instruction:
                print(
                    f"[DataQnA Class Init] Loaded System Instruction (first 100 chars): {system_instruction[:100]}...", flush=True)
            else:
                print(
                    "[DataQnA Class Init] No System Instruction provided.", flush=True)

        except Exception as e:
            print(
                f"[DataQnA Class Init] !!! ERROR Initializing DataQnA Client: {e}", flush=True)
            traceback.print_exc()
            raise

    def _convert_proto_struct_to_dict(self, value: Any) -> Any:
        if isinstance(value, proto.marshal.collections.maps.MapComposite):
            return {k: self._convert_proto_struct_to_dict(v) for k, v in value.items()}
        elif isinstance(value, proto.marshal.collections.RepeatedComposite):
            return [self._convert_proto_struct_to_dict(el) for el in value]
        elif isinstance(value, (int, float, str, bool)) or value is None:
            return value
        elif isinstance(value, proto.Message) and hasattr(value, 'DESCRIPTOR') and value.DESCRIPTOR.full_name.startswith('google.protobuf.'):
            if value.DESCRIPTOR.full_name == 'google.protobuf.Struct':
                return {k: self._convert_proto_struct_to_dict(v) for k, v in value.items()}
            elif value.DESCRIPTOR.full_name == 'google.protobuf.ListValue':
                return [self._convert_proto_struct_to_dict(v) for v in value.values]
            elif value.DESCRIPTOR.full_name == 'google.protobuf.Value':
                kind = value.WhichOneof('kind')
                if kind == 'struct_value':
                    return self._convert_proto_struct_to_dict(value.struct_value)
                if kind == 'list_value':
                    return self._convert_proto_struct_to_dict(value.list_value)
                if kind == 'string_value':
                    return value.string_value
                if kind == 'number_value':
                    return value.number_value
                if kind == 'bool_value':
                    return value.bool_value
                if kind == 'null_value':
                    return None
        elif isinstance(value, proto.Message) and hasattr(value, 'DESCRIPTOR'):
            try:
                return MessageToDict(value, preserving_proto_field_name=True, including_default_value_fields=True)
            except Exception:
                return str(value)
        return str(value)

    def _handle_data_response_for_df(self, resp_data_payload, debug_mode: bool):
        if hasattr(resp_data_payload, 'generated_sql') and resp_data_payload.generated_sql:
            self.current_sql_query_from_data_message = resp_data_payload.generated_sql
            if debug_mode:
                print(
                    f"[DataQnA Debug] SQL (from data msg): {self.current_sql_query_from_data_message[:200]}...", flush=True)

        if hasattr(resp_data_payload, 'result') and hasattr(resp_data_payload.result, 'schema') and hasattr(resp_data_payload.result, 'data'):
            if not resp_data_payload.result.data:
                if debug_mode:
                    print(
                        "[DataQnA Debug] Data result has no rows. Storing empty DataFrame.", flush=True)
                self.last_processed_dataframe = pd.DataFrame(
                    columns=[field.name for field in resp_data_payload.result.schema.fields] if hasattr(
                        resp_data_payload.result.schema, 'fields') else []
                )
                return
            try:
                fields = [field.name for field in resp_data_payload.result.schema.fields] if hasattr(
                    resp_data_payload.result.schema, 'fields') else []
                if not fields and resp_data_payload.result.data:
                    first_row_dict_test = self._convert_proto_struct_to_dict(
                        resp_data_payload.result.data[0])
                    if isinstance(first_row_dict_test, dict):
                        fields = list(first_row_dict_test.keys())
                if not fields:
                    if debug_mode:
                        print(
                            "[DataQnA Debug] ERROR: Cannot process data rows without field names.", flush=True)
                    return
                data_rows = []
                for row_struct in resp_data_payload.result.data:
                    converted_row = self._convert_proto_struct_to_dict(
                        row_struct)
                    if isinstance(converted_row, dict):
                        data_rows.append(
                            {field: converted_row.get(field) for field in fields})
                    elif isinstance(converted_row, list) and len(converted_row) == len(fields):
                        data_rows.append(dict(zip(fields, converted_row)))
                if data_rows:
                    self.last_processed_dataframe = pd.DataFrame(
                        data_rows, columns=fields)
                    if debug_mode:
                        print(
                            f"[DataQnA Debug] DataFrame created with {len(data_rows)} rows, columns: {fields}", flush=True)
                else:
                    if debug_mode:
                        print(
                            "[DataQnA Debug] Processed data resulted in zero valid rows. Storing empty DataFrame.", flush=True)
                    self.last_processed_dataframe = pd.DataFrame(
                        columns=fields)
            except Exception as e_df:
                if debug_mode:
                    print(
                        f"[DataQnA Debug] ERROR creating DataFrame from data.result: {e_df}", flush=True)
                traceback.print_exc()
        elif debug_mode:
            print(f"[DataQnA Debug] Data payload did not contain result.schema and result.data for DataFrame processing.", flush=True)

    def reset_conversation(self, conversation_id: Optional[str]):
        if conversation_id and conversation_id in self.conversation_histories:
            del self.conversation_histories[conversation_id]
            print(
                f"[DataQnA Log] Conversation history reset for ID: {conversation_id}", flush=True)

    def ask_question(self, question_text: str, conversation_id: Optional[str], debug_mode: bool = False) -> Dict[str, Any]:
        log_conv_id = conversation_id or "STATELESS"
        if debug_mode:
            print(
                f"\n[DataQnA Req] Asking (ConvID: {log_conv_id}): '{question_text}'", flush=True)
            if self.system_instruction:
                print(
                    f"[DataQnA Req] Using System Instruction (first 100 chars): {self.system_instruction[:100]}...", flush=True)

        self.last_processed_dataframe = None
        self.last_processed_chart_spec = None
        self.current_sql_query_from_data_message = None

        current_history = self.conversation_histories.get(
            conversation_id, []) if conversation_id else []
        input_message = dataqna_v1alpha1.Message(
            user_message=dataqna_v1alpha1.UserMessage(text=question_text))
        messages_to_send = current_history + [input_message]

        request_obj_args = {
            "project": self.parent_project_string_for_request,
            "messages": messages_to_send,
            "context": dataqna_v1alpha1.InlineContext(
                system_instruction=self.system_instruction,
                datasource_references=self.datasource_references)
        }

        try:
            request = dataqna_v1alpha1.AskQuestionRequest(**request_obj_args)
        except ValueError as ve_req:
            print(
                f"[DataQnA ERROR] AskQuestionRequest creation failed: {ve_req}.", flush=True)
            traceback.print_exc()
            return {
                "answer": f"Internal error: DataQnA request construction failed ({str(ve_req)}).",
                "dataframe_content": None, "vega_lite_spec": None, "sql_query": None,
                "conversation_id": conversation_id
            }

        accumulated_final_text_parts: List[str] = []
        stream_iterator = None
        replies_for_history = []
        raw_stream_items_for_debug = []

        if debug_mode:
            print(
                f"[DataQnA Log] {'-'*10} Processing Stream {'-'*10}", flush=True)
        try:
            stream_iterator = self.client.ask_question(request=request)

            for msg_container in stream_iterator:
                if debug_mode:
                    try:
                        raw_stream_items_for_debug.append(
                            f"Type: {type(msg_container)} Content: {str(msg_container)[:200]}...")
                    except Exception:
                        raw_stream_items_for_debug.append(
                            f"Type: {type(msg_container)} Content: (Error converting to string)")
                replies_for_history.append(msg_container)
                if hasattr(msg_container, 'system_message') and msg_container.system_message:
                    m = msg_container.system_message
                    if hasattr(m, 'data') and m.data:
                        self._handle_data_response_for_df(m.data, debug_mode)
                    if hasattr(m, 'chart') and m.chart and hasattr(m.chart, 'result') and m.chart.result.vega_config:
                        converted_spec = self._convert_proto_struct_to_dict(
                            m.chart.result.vega_config)
                        if isinstance(converted_spec, dict):
                            self.last_processed_chart_spec = converted_spec
                            if debug_mode:
                                print(
                                    f"[DataQnA Debug] Vega Config captured.", flush=True)
                    if hasattr(m, 'text') and m.text and hasattr(m.text, 'parts') and m.text.parts:
                        current_message_text_parts = []
                        for p_part in m.text.parts:
                            if hasattr(p_part, 'text') and p_part.text:
                                current_message_text_parts.append(p_part.text)
                            elif isinstance(p_part, str):
                                current_message_text_parts.append(p_part)
                        if current_message_text_parts:
                            accumulated_final_text_parts.extend(
                                current_message_text_parts)
                            if debug_mode:
                                print(
                                    f"[DataQnA Debug] Text parts appended: {''.join(current_message_text_parts)[:100]}...", flush=True)

            if conversation_id:
                self.conversation_histories[conversation_id] = current_history + [
                    input_message] + replies_for_history
            if debug_mode:
                print(
                    f"[DataQnA Log] {'-'*10} Finished Processing Stream {'-'*10}", flush=True)

        except Exception as e:
            print(
                f"\n[DataQnA Log] !!! ERROR during DataQnA call/stream: {e} !!!", flush=True)
            if debug_mode and raw_stream_items_for_debug:
                print(
                    "[DataQnA Log] --- RAW STREAM ITEMS BEFORE ERROR ---", flush=True)
                for i, r_str in enumerate(raw_stream_items_for_debug):
                    print(f"Item {i+1}: {r_str}", flush=True)
                print("[DataQnA Log] --- END RAW STREAM ITEMS ---", flush=True)
            traceback.print_exc()
            return {
                "answer": f"Sorry, an error occurred while processing your request via DataQnA: {str(e)}",
                "dataframe_content": None, "vega_lite_spec": None, "sql_query": None,
                "conversation_id": conversation_id
            }
        finally:
            if stream_iterator and hasattr(stream_iterator, 'cancel'):
                try:
                    stream_iterator.cancel()
                except Exception:
                    pass
            elif stream_iterator and hasattr(stream_iterator, 'close'):
                try:
                    stream_iterator.close()
                except Exception:
                    pass

        final_text_reply = ''.join(accumulated_final_text_parts).strip()
        if not final_text_reply:
            if self.last_processed_chart_spec:
                final_text_reply = "[DataQnA provided a chart. No additional text summary.]"
            elif self.last_processed_dataframe is not None:
                final_text_reply = "[DataQnA provided tabular data. No additional text summary.]"
            elif self.current_sql_query_from_data_message:
                final_text_reply = "[DataQnA generated SQL. No specific text summary.]"
            else:
                final_text_reply = "[DataQnA provided no specific text, data, or chart output.]"

        dataframe_output_content = None
        if self.last_processed_dataframe is not None:
            dataframe_output_content = {
                "data": self.last_processed_dataframe.to_dict(orient='records'),
                "columns": list(self.last_processed_dataframe.columns)
            }
            if debug_mode:
                print(
                    f"[DataQnA Debug] DataFrame prepared for output with {len(dataframe_output_content['data'])} records.", flush=True)

        return {
            "answer": final_text_reply,
            "dataframe_content": dataframe_output_content,
            "vega_lite_spec": self.last_processed_chart_spec,
            "sql_query": self.current_sql_query_from_data_message,
            "conversation_id": conversation_id
        }

# ==============================================================================
# NEW: Function for Second Reasoning Step with Gemini
# ==============================================================================
# ... (reason_on_dataqna_output_with_gemini function is exactly the same as in the previous response)


def reason_on_dataqna_output_with_gemini(
    nlp_question: str,
    dataqna_sql_query: Optional[str],
    # This is DataTableContent like structure
    dataqna_results_df_content: Optional[Dict[str, Any]],
    dataqna_initial_answer: str,
    max_rows_for_summary_context: int = 10,
    debug_mode: bool = False
) -> str:
    if not vertex_ai_initialized:
        if debug_mode:
            print(
                "[Gemini Reasoning] Vertex AI service not available for summarization.")
        return f"DataQnA initial answer: '{dataqna_initial_answer}'. Could not perform secondary reasoning: Vertex AI unavailable."

    try:
        # REASONING_MODEL_NAME is used here
        model = GenerativeModel(REASONING_MODEL_NAME)
    except Exception as model_init_err:
        if debug_mode:
            print(
                f"[Gemini Reasoning] Error initializing GenerativeModel ({REASONING_MODEL_NAME}): {model_init_err}")
        return f"DataQnA initial answer: '{dataqna_initial_answer}'. Error initializing secondary reasoning model: {str(model_init_err)}"

    context_data_str = ""
    num_total_rows = 0
    if dataqna_results_df_content and dataqna_results_df_content.get("data"):
        query_results = dataqna_results_df_content["data"]
        column_names = dataqna_results_df_content.get("columns", [])
        num_total_rows = len(query_results)

        if num_total_rows > 0:
            rows_to_show = query_results[:max_rows_for_summary_context]
            if rows_to_show:
                if column_names:
                    context_data_str += f"Column Names: {', '.join(column_names)}\n"
                else:  # Fallback if columns not explicitly provided but data exists
                    column_names = list(
                        rows_to_show[0].keys()) if rows_to_show else []
                    if column_names:
                        context_data_str += f"Column Names (inferred): {', '.join(column_names)}\n"

                context_data_str += "Sample Data (first {} of {} total rows):\n".format(
                    len(rows_to_show), num_total_rows)
                for row_idx, row in enumerate(rows_to_show):
                    # Use json.dumps for better dict representation
                    context_data_str += f"Row {row_idx+1}: {json.dumps(row)}\n"
            if num_total_rows > max_rows_for_summary_context:
                context_data_str += f"... and {num_total_rows - max_rows_for_summary_context} more rows.\n"
            context_data_str += f"Total rows retrieved by DataQnA: {num_total_rows}\n"
        else:
            context_data_str = "The DataQnA query returned no data (0 rows).\n"
    else:
        context_data_str = "No tabular data was returned by DataQnA or data format was unexpected.\n"

    summary_prompt = f"""
    You are a secondary business analyst. You have received information from a primary data retrieval service (DataQnA).
    Your task is to provide a refined, insightful summary based on all the information provided, primarily focusing on the data itself.

    Original Natural Language Question from User:
    "{nlp_question}"

    The SQL Query executed by the primary service (DataQnA), if available:
    ```sql
    {dataqna_sql_query or "N/A"}
    ```

    Initial Textual Answer/Interpretation from the primary service (DataQnA):
    "{dataqna_initial_answer}"

    Data Results from the primary service's Query:
    {context_data_str}

    Instructions for your refined summary:
    1.  Focus on directly answering the user's original question using insights derived *primarily from the Data Results*.
    2.  You can use the "Initial Textual Answer from DataQnA" as context or a starting point, but your main goal is to synthesize information from the raw data.
    3.  If the data shows specific items (like products, parents/programs, SKUs), mention the top few relevant ones if appropriate.
    4.  If the query returned numerical results, state the key figures.
    5.  If the DataQnA query returned no data, your summary should reflect that, potentially corroborating or explaining DataQnA's initial answer if it also mentioned no data.
    6.  Keep the summary concise (typically 2-5 sentences) and business-friendly.
    7.  Do NOT just repeat or slightly rephrase the "Initial Textual Answer from DataQnA". Provide your own interpretation based on the data.
    8.  If DataQnA's interpretation seems consistent with the data, you can affirm it and add details. If there's a nuance the data reveals that DataQnA missed, highlight it.
    9.  Avoid making up information not present in the data.
    10. **Correlations & Causation:** Adhere to the same principles as the primary analyst: discuss correlations as potential contributing factors, avoid definitive causation unless directly supported by explicit attribution metrics in the data.

    Refined Business Summary:
    """
    if debug_mode:
        # Will show the chosen model
        print(
            f"\n[Gemini Reasoning] Sending Prompt for Secondary Summary (Model: {REASONING_MODEL_NAME})")
        print(
            f"[Gemini Reasoning] Summary Prompt (first 500 chars): {summary_prompt[:500].replace(os.linesep, ' ')}...")

    try:
        response = model.generate_content(summary_prompt)
        summary_text = response.text.strip()
        if debug_mode:
            print(
                f"[Gemini Reasoning] Generated Secondary Summary: {summary_text}")
        return summary_text if summary_text else f"DataQnA Initial: '{dataqna_initial_answer}'. Secondary reasoning could not generate a specific summary."
    except Exception as e:
        if debug_mode:
            print(
                f"[Gemini Reasoning] Error during secondary summary generation: {e}")
            traceback.print_exc()
        error_detail = f"DataQnA Initial: '{dataqna_initial_answer}'. An error occurred during secondary reasoning: {str(e)}"
        # You can add more details from the exception 'e' if needed, like in your original piagent
        return error_detail


# ==============================================================================
# FastAPI Application Setup (remains unchanged)
# ==============================================================================
# ... (FastAPI app, Pydantic models, HTML_CONTENT, startup_event, /ask, /health, /ui, __main__ are all the same as in the previous response)
app = FastAPI(
    title="Enhanced DataQnA API with Secondary Gemini Reasoning",
    description="API for DataQnA, with an additional Gemini call for refined summarization. Access UI at / or /ui.",
    version="3.0.1"  # Version bump for model clarification
)


class QuestionRequest(BaseModel):
    question: str = Field(..., examples=[
                          "What were the total sales last week?", "Which programs had lower sessions in the last 6 weeks and what might be the reasons considering ad spend, ASP, and inventory? Show weekly trends.", "Tell me about this data."])
    conversation_id: Optional[str] = Field(
        default=None, description="Unique ID for conversation history. Auto-generated if omitted.")
    debug_mode: bool = Field(
        default=False, description="Enable detailed logging.")
    reset_conversation: bool = Field(
        default=False, description="Reset history for the given conversation_id before asking.")
    enable_secondary_reasoning: bool = Field(
        default=True, description="Enable the additional Gemini reasoning step after DataQnA response.")


class DataTableContent(BaseModel):
    data: List[Dict[str, Any]]
    columns: List[str]


class AnswerResponse(BaseModel):
    query: str
    answer: str = Field(
        description="Text answer. If secondary reasoning is enabled, this is the re-reasoned answer.")
    sql_query: Optional[str] = Field(
        default=None, description="Generated SQL query from DataQnA, if applicable.")
    vega_lite_spec: Optional[Dict[str, Any]] = Field(
        default=None, description="Vega-Lite JSON for chart from DataQnA, if applicable.")
    dataframe_content: Optional[DataTableContent] = Field(
        default=None, description="Structured table data from DataQnA, if applicable.")
    conversation_id: Optional[str] = Field(
        default=None, description="Conversation ID used (provided or generated).")


HTML_CONTENT = """
<!DOCTYPE html>
<html>
<head>
    <title>DataQnA Test UI</title>
    <script src="https://cdn.jsdelivr.net/npm/vega@5"></script>
    <script src="https://cdn.jsdelivr.net/npm/vega-lite@5"></script>
    <script src="https://cdn.jsdelivr.net/npm/vega-embed@6"></script>
    <style>
        body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif, "Apple Color Emoji", "Segoe UI Emoji"; margin: 0; padding:0; background-color: #f4f7f6; color: #333; font-size: 16px; line-height:1.6;}
        .container { max-width: 900px; margin: 20px auto; background-color: #fff; padding: 25px; border-radius: 8px; box-shadow: 0 4px 12px rgba(0,0,0,0.1); }
        h1 { color: #2c3e50; text-align: center; margin-bottom: 25px; font-weight: 600;}
        label { display: block; margin-top: 18px; margin-bottom: 5px; font-weight: 600; color: #34495e; }
        input[type="text"], textarea {
            width: 100%;
            padding: 12px;
            margin-top: 5px;
            box-sizing: border-box;
            border: 1px solid #ced4da;
            border-radius: 4px;
            font-size: 0.95em;
            transition: border-color 0.2s ease-in-out, box-shadow 0.2s ease-in-out;
        }
        input[type="text"]:focus, textarea:focus {
            border-color: #80bdff;
            outline: 0;
            box-shadow: 0 0 0 0.2rem rgba(0,123,255,.25);
        }
        textarea { min-height: 100px; resize: vertical; }
        button {
            background-color: #007bff;
            color: white;
            padding: 12px 20px;
            margin-top: 25px;
            border: none;
            border-radius: 4px;
            cursor: pointer;
            font-size: 1em;
            font-weight: 500;
            transition: background-color 0.2s ease-in-out;
        }
        button:hover { background-color: #0056b3; }
        .checkbox-group div { margin-top: 12px; display: flex; align-items: center;}
        .checkbox-group input[type="checkbox"] { margin-right: 8px; width:auto; }
        .checkbox-group label { display: inline-block; margin-left: 0; font-weight: normal; margin-top:0; }

        .response-section { margin-top: 35px; border: 1px solid #e0e0e0; padding: 20px; border-radius: 4px; background-color: #f9fafb; }
        .response-section h3, .response-section h4 { color: #2c3e50; border-bottom: 1px solid #eee; padding-bottom: 8px; margin-bottom: 15px; font-weight: 600;}

        #resAnswer { white-space: pre-wrap; background-color: #fff; padding: 12px; border-radius: 4px; border: 1px solid #e9ecef; margin-bottom:15px; }
        #sqlQuery { background-color: #e8f6fd; color: #0c5460; padding: 12px; white-space: pre-wrap; font-family: 'SFMono-Regular', Consolas, 'Liberation Mono', Menlo, Courier, monospace; border-radius: 4px; border: 1px solid #b0dff6; font-size: 0.9em; overflow-x: auto;}

        table { border-collapse: collapse; width: 100%; margin-top: 15px; box-shadow: 0 1px 3px rgba(0,0,0,0.05); font-size: 0.9em;}
        th, td { border: 1px solid #e0e0e0; padding: 10px 12px; text-align: left; }
        th { background-color: #f2f5f7; color: #34495e; font-weight: 600; }
        td { background-color: #fff; }
        tr:nth-child(even) td { background-color: #f8fafc; }

        .loading { display: none; text-align: center; margin-top: 20px; padding: 15px; font-style: italic; color: #555; }
        .loading img { vertical-align: middle; margin-left: 8px; }
        #vegaChart, #dataTableContainer { margin-top: 10px; padding:10px; border: 1px dashed #ccc; min-height: 50px; background-color: #fff; border-radius: 4px;}
        #dataTableContainer table { margin-top: 0; }
        .error-message { color: #D8000C; background-color: #FFD2D2; border: 1px solid #D8000C; padding: 10px; border-radius: 4px; margin-top: 10px; white-space: pre-wrap;}
        #chartControls { display:none; margin-bottom:10px; padding: 10px; background-color: #f0f0f0; border-radius: 4px; border: 1px solid #ddd;}
        #chartControls label { display:inline-block !important; margin-right:5px; margin-top:0; font-weight:normal; }
        #chartControls select, #chartControls button { margin-top:0; padding: 6px 10px; font-size:0.9em;}
        #chartControls p {font-size:0.8em; color:#666; margin-top:8px; margin-bottom:0;}
        #secondaryReasoningLabel { margin-left: 0; font-weight: normal; }
    </style>
</head>
<body>
    <div class="container">
        <h1>Piagent Test Interface (DataQnA + Gemini Re-reasoning)</h1>

        <label for="question">Your Question:</label>
        <textarea id="question" placeholder="e.g., What were the total product_sales last week for '400 Thread Counts Sheet Set'? Or 'Tell me about this data.'"></textarea>

        <label for="conversationId">Conversation ID (auto-managed, or override):</label>
        <input type="text" id="conversationId" placeholder="Will be auto-filled after first query">

        <div class="checkbox-group">
            <div>
                <input type="checkbox" id="debugMode">
                <label for="debugMode">Enable Debug Mode</label>
            </div>
            <div>
                <input type="checkbox" id="resetConversation">
                <label for="resetConversation">Reset Conversation History (if ID is provided)</label>
            </div>
            <div>
                <input type="checkbox" id="enableSecondaryReasoning" checked>
                <label for="enableSecondaryReasoning" id="secondaryReasoningLabel">Enable Secondary Gemini Reasoning</label>
            </div>
        </div>

        <button onclick="askQuestion()">Ask Question</button>
        <div id="loading" class="loading">
            <p>Processing your question... <img src="https://i.gifer.com/ZZ5H.gif" alt="loading" width="25" height="25"></p>
        </div>


        <div id="responseArea" style="display:none;" class="response-section">
            <h3>Response Details</h3>
            <p><strong>Your Query:</strong> <span id="resQuery"></span></p>
            <p><strong>Conversation ID:</strong> <span id="resConvId"></span></p>

            <h4>Agent's Final Answer (potentially re-reasoned):</h4>
            <div id="resAnswer"></div>

            <h4>Generated SQL Query (from DataQnA):</h4>
            <pre id="sqlQuery">N/A</pre>

            <h4>Chart Visualization (from DataQnA):</h4>
            <div id="chartControls">
                <label for="chartTypeSelector">Try different chart type:</label>
                <select id="chartTypeSelector" onchange="changeChartType()">
                    <option value="">Default (from API)</option>
                    <option value="bar">Bar</option>
                    <option value="line">Line</option>
                    <option value="point">Point (Scatter)</option>
                    <option value="area">Area</option>
                    <option value="arc">Arc (Pie/Donut attempt)</option>
                    <option value="rect">Rect (Heatmap attempt)</option>
                    <option value="tick">Tick</option>
                    <option value="circle">Circle</option>
                    <option value="square">Square</option>
                    <option value="geoshape">Geoshape (for maps)</option>
                </select>
                <button onclick="resetChart()" style="margin-left:10px;">Reset to Original</button>
                <p><i>Note: This is an experimental frontend change. The selected type might not be suitable for the data encodings in the original chart.</i></p>
            </div>
            <div id="vegaChart">N/A</div>

            <h4>Data Table Result (from DataQnA):</h4>
            <div id="dataTableContainer">N/A</div>
        </div>
    </div>

    <script>
        let currentConversationId = '';
        let originalVegaSpec = null;

        function showChartControls(show) {
            document.getElementById('chartControls').style.display = show ? 'block' : 'none';
        }

        function renderVegaChart(spec) {
            const vegaChartDiv = document.getElementById('vegaChart');
            vegaChartDiv.innerHTML = '';
            if (spec && Object.keys(spec).length > 0) {
                vegaEmbed('#vegaChart', spec, { "actions": true }).catch(err => {
                    console.error("Vega Embed Error:", err);
                    vegaChartDiv.innerHTML = '<div class="error-message">Error rendering chart with selected type. See console. Original encodings might be incompatible. <button onclick="resetChart()">Reset to Original</button></div>';
                });
            } else {
                vegaChartDiv.textContent = 'N/A';
            }
        }

        function changeChartType() {
            if (!originalVegaSpec) return;
            const selectedMark = document.getElementById('chartTypeSelector').value;
            if (!selectedMark) {
                renderVegaChart(originalVegaSpec);
                return;
            }
            let modifiedSpec = JSON.parse(JSON.stringify(originalVegaSpec));
            if (typeof modifiedSpec.mark === 'string') {
                modifiedSpec.mark = selectedMark;
            } else if (typeof modifiedSpec.mark === 'object' && modifiedSpec.mark !== null) {
                modifiedSpec.mark.type = selectedMark;
            } else {
                modifiedSpec.mark = { type: selectedMark };
            }
            renderVegaChart(modifiedSpec);
        }

        function resetChart() {
            if (originalVegaSpec) {
                renderVegaChart(originalVegaSpec);
                document.getElementById('chartTypeSelector').value = '';
            }
        }

        async function askQuestion() {
            const question = document.getElementById('question').value;
            let conversationIdInput = document.getElementById('conversationId').value.trim();
            const debugMode = document.getElementById('debugMode').checked;
            const resetConversation = document.getElementById('resetConversation').checked;
            const enableSecondaryReasoning = document.getElementById('enableSecondaryReasoning').checked;

            if (!question) {
                alert("Please enter a question.");
                return;
            }

            let convIdToSend = conversationIdInput || currentConversationId || null;

            const payload = {
                question: question,
                conversation_id: convIdToSend,
                debug_mode: debugMode,
                reset_conversation: resetConversation,
                enable_secondary_reasoning: enableSecondaryReasoning
            };

            document.getElementById('loading').style.display = 'block';
            document.getElementById('responseArea').style.display = 'none';
            originalVegaSpec = null;
            showChartControls(false);

            document.getElementById('resQuery').textContent = '';
            document.getElementById('resAnswer').innerHTML = '';
            document.getElementById('resConvId').textContent = '';
            document.getElementById('sqlQuery').textContent = 'N/A';
            document.getElementById('vegaChart').innerHTML = 'N/A';
            document.getElementById('dataTableContainer').innerHTML = 'N/A';

            try {
                const response = await fetch('/ask', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                        'Accept': 'application/json'
                    },
                    body: JSON.stringify(payload)
                });

                document.getElementById('loading').style.display = 'none';
                document.getElementById('responseArea').style.display = 'block';

                const result = await response.json();

                if (!response.ok) {
                    console.error("API Error:", result);
                    let errorMsgText = `Error: ${response.status} ${response.statusText}`;
                    if (result.detail) {
                         if (typeof result.detail === 'string') errorMsgText += '\\nDetails: ${result.detail}';
                         else if (result.detail.message) errorMsgText += `\\nDetails: ${result.detail.message}`;
                         else errorMsgText += `\\nDetails: ${JSON.stringify(result.detail)}`;
                    }
                    document.getElementById('resAnswer').innerHTML = `<div class="error-message">${errorMsgText.replace(/\\n/g, '<br>')}</div>`;
                    document.getElementById('resQuery').textContent = question;
                    document.getElementById('resConvId').textContent = (result.detail && result.detail.conversation_id) || convIdToSend || 'N/A';
                    return;
                }

                document.getElementById('resQuery').textContent = result.query;
                document.getElementById('resAnswer').textContent = result.answer;

                if (result.conversation_id) {
                    currentConversationId = result.conversation_id;
                    document.getElementById('resConvId').textContent = result.conversation_id;
                    document.getElementById('conversationId').value = result.conversation_id;
                } else {
                     document.getElementById('resConvId').textContent = "N/A (or stateless)";
                     if (!conversationIdInput) {
                        currentConversationId = '';
                        document.getElementById('conversationId').value = '';
                     }
                }

                document.getElementById('sqlQuery').textContent = result.sql_query || 'N/A';

                if (result.vega_lite_spec && Object.keys(result.vega_lite_spec).length > 0) {
                    originalVegaSpec = JSON.parse(JSON.stringify(result.vega_lite_spec));
                    renderVegaChart(originalVegaSpec);
                    showChartControls(true);
                    document.getElementById('chartTypeSelector').value = '';
                } else {
                    document.getElementById('vegaChart').textContent = 'N/A';
                    showChartControls(false);
                }

                const dataTableContainer = document.getElementById('dataTableContainer');
                dataTableContainer.innerHTML = '';
                if (result.dataframe_content && result.dataframe_content.data && result.dataframe_content.data.length > 0) {
                    const table = document.createElement('table');
                    const thead = document.createElement('thead');
                    const tbody = document.createElement('tbody');
                    const headerRow = document.createElement('tr');

                    result.dataframe_content.columns.forEach(colName => {
                        const th = document.createElement('th');
                        th.textContent = colName;
                        headerRow.appendChild(th);
                    });
                    thead.appendChild(headerRow);
                    table.appendChild(thead);

                    result.dataframe_content.data.forEach(rowData => {
                        const tr = document.createElement('tr');
                        result.dataframe_content.columns.forEach(colName => {
                            const td = document.createElement('td');
                            let cellValue = rowData[colName];
                            if (cellValue === null || cellValue === undefined) {
                                cellValue = '';
                            } else if (typeof cellValue === 'object') {
                                cellValue = JSON.stringify(cellValue);
                            }
                            td.textContent = cellValue;
                            tr.appendChild(td);
                        });
                        tbody.appendChild(tr);
                    });
                    table.appendChild(tbody);
                    dataTableContainer.appendChild(table);
                } else if (result.dataframe_content && result.dataframe_content.columns && result.dataframe_content.columns.length > 0 && (!result.dataframe_content.data || result.dataframe_content.data.length === 0)) {
                    const table = document.createElement('table');
                    const thead = document.createElement('thead');
                    const headerRow = document.createElement('tr');
                     result.dataframe_content.columns.forEach(colName => {
                        const th = document.createElement('th');
                        th.textContent = colName;
                        headerRow.appendChild(th);
                    });
                    thead.appendChild(headerRow);
                    table.appendChild(thead);

                    const tbody = document.createElement('tbody');
                    const tr = document.createElement('tr');
                    const td = document.createElement('td');
                    td.setAttribute('colspan', result.dataframe_content.columns.length);
                    td.textContent = '(No data returned for these columns)';
                    td.style.textAlign = 'center';
                    td.style.fontStyle = 'italic';
                    tr.appendChild(td);
                    tbody.appendChild(tr);
                    table.appendChild(tbody);
                    dataTableContainer.appendChild(table);
                }
                else {
                    dataTableContainer.textContent = 'N/A';
                }

            } catch (error) {
                console.error('Fetch Error:', error);
                document.getElementById('loading').style.display = 'none';
                document.getElementById('responseArea').style.display = 'block';
                document.getElementById('resAnswer').innerHTML = `<div class="error-message">An error occurred while fetching the response: ${error.message}</div>`;
            }
        }
    </script>
</body>
</html>
"""


@app.on_event("startup")
async def startup_event():
    global data_qna_instance, vertex_ai_initialized
    print("--- FastAPI Startup: Initializing AI Clients ---", flush=True)
    try:
        print("Initializing Data QnA Client Instance...", flush=True)
        if not all([BQ_PROJECT_ID, BQ_DATASET_ID, BQ_TABLE_ID, DATAQNA_BILLING_PROJECT, LOCATION]):
            raise ValueError(
                "Missing one or more DataQnA environment variables for BQ source or billing project/location.")
        data_qna_instance = DataQnA(
            billing_project=DATAQNA_BILLING_PROJECT,
            location=LOCATION,
            system_instruction=SYSTEM_INSTRUCTION_YAML,
            datasource_bq_project_id=BQ_PROJECT_ID,
            datasource_bq_dataset_id=BQ_DATASET_ID,
            datasource_bq_table_id=BQ_TABLE_ID
        )
        print("DataQnA Client Instance initialized successfully.", flush=True)
    except Exception as e:
        print(
            f"!!! FATAL ERROR initializing DataQnA Client Instance: {e}", flush=True)
        traceback.print_exc()
        data_qna_instance = None
        print("!!! CRITICAL: DataQnA service failed to initialize.", flush=True)

    try:
        print(
            f"Initializing Vertex AI for Project: {VERTEX_PROJECT_ID}, Location: {LOCATION}", flush=True)
        vertexai.init(project=VERTEX_PROJECT_ID, location=LOCATION)
        # Test model instantiation to catch issues early
        print(
            f"Attempting to instantiate reasoning model: {REASONING_MODEL_NAME}")
        _ = GenerativeModel(REASONING_MODEL_NAME)  # Test instantiation
        print(f"Successfully tested instantiation of {REASONING_MODEL_NAME}")
        vertex_ai_initialized = True
        print("Vertex AI for Gemini reasoning initialized successfully.", flush=True)
    except Exception as e:
        print(
            f"!!! FATAL ERROR during Vertex AI (for Gemini reasoning) initialization or model test: {e}", flush=True)
        traceback.print_exc()
        vertex_ai_initialized = False
        print(
            "!!! CRITICAL: Vertex AI for Gemini reasoning failed to initialize.", flush=True)

    if not data_qna_instance:
        print("!!! CRITICAL WARNING: DataQnA service unavailable. Core functionality will be impacted.", flush=True)
    if not vertex_ai_initialized:
        print(
            f"!!! CRITICAL WARNING: Vertex AI (Gemini model {REASONING_MODEL_NAME}) service for secondary reasoning unavailable.", flush=True)

    print("--- FastAPI Startup Complete ---", flush=True)


@app.post("/ask", response_model=AnswerResponse, status_code=status.HTTP_200_OK)
async def ask_data_agent(request_payload: QuestionRequest):
    current_conv_id = request_payload.conversation_id or str(uuid.uuid4())
    if not request_payload.conversation_id and request_payload.debug_mode:
        print(
            f"--- No Client ConvID. Generated new ID: {current_conv_id} ---", flush=True)

    if request_payload.debug_mode:
        print(
            f"Question (ConvID: {current_conv_id}): \"{request_payload.question}\"", flush=True)
        print(
            f"Debug Mode: {request_payload.debug_mode}, Reset: {request_payload.reset_conversation}, Secondary Reasoning: {request_payload.enable_secondary_reasoning}", flush=True)

    if data_qna_instance is None:
        print("!!! ERROR: DataQnA instance not available at request time.", flush=True)
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                            detail={"message": "Core data querying service (DataQnA) is unavailable.", "conversation_id": current_conv_id})

    if request_payload.reset_conversation and request_payload.conversation_id:
        try:
            data_qna_instance.reset_conversation(
                conversation_id=current_conv_id)
            if request_payload.debug_mode:
                print(
                    f"[API Log] Conversation history reset for ID: {current_conv_id}", flush=True)
        except Exception as reset_err:
            print(
                f"Warning: Failed to reset history for ID {current_conv_id}: {reset_err}", flush=True)

    try:
        data_qna_result = data_qna_instance.ask_question(
            question_text=request_payload.question,
            conversation_id=current_conv_id,
            debug_mode=request_payload.debug_mode
        )

        final_answer = data_qna_result.get(
            'answer', "[Error: 'answer' key missing from DataQnA result]")
        dataqna_initial_answer_for_context = final_answer

        if request_payload.enable_secondary_reasoning:
            if not vertex_ai_initialized:
                if request_payload.debug_mode:
                    print(
                        f"[API Log] Secondary reasoning skipped: Vertex AI (model {REASONING_MODEL_NAME}) not initialized.", flush=True)
                final_answer += f" (Note: Secondary reasoning step with {REASONING_MODEL_NAME} skipped as its service is unavailable.)"
            else:
                if request_payload.debug_mode:
                    print(
                        f"[API Log] Performing secondary reasoning with {REASONING_MODEL_NAME} for ConvID: {current_conv_id}", flush=True)

                df_content_for_reasoning = None
                raw_df_from_dataqna = data_qna_result.get("dataframe_content")
                if raw_df_from_dataqna and isinstance(raw_df_from_dataqna, dict) and \
                   "data" in raw_df_from_dataqna and "columns" in raw_df_from_dataqna:
                    df_content_for_reasoning = raw_df_from_dataqna

                refined_answer = reason_on_dataqna_output_with_gemini(
                    nlp_question=request_payload.question,
                    dataqna_sql_query=data_qna_result.get("sql_query"),
                    dataqna_results_df_content=df_content_for_reasoning,
                    dataqna_initial_answer=dataqna_initial_answer_for_context,
                    debug_mode=request_payload.debug_mode
                )
                final_answer = refined_answer
        elif request_payload.debug_mode:
            print("[API Log] Secondary reasoning disabled by request flag.", flush=True)

        df_content_for_response = None
        raw_df_content = data_qna_result.get("dataframe_content")
        if raw_df_content and isinstance(raw_df_content, dict) and "data" in raw_df_content and "columns" in raw_df_content:
            try:
                df_content_for_response = DataTableContent(**raw_df_content)
            except Exception as pydantic_err:
                if request_payload.debug_mode:
                    print(
                        f"[API Warning] Pydantic validation error for DataTableContent: {pydantic_err}. DF content will be null.", flush=True)
        elif raw_df_content and request_payload.debug_mode:
            print(
                f"[API Warning] dataframe_content from DataQnA was not in expected dict format: {type(raw_df_content)}", flush=True)

        response_data = AnswerResponse(
            query=request_payload.question,
            answer=final_answer,
            sql_query=data_qna_result.get("sql_query"),
            vega_lite_spec=data_qna_result.get("vega_lite_spec"),
            dataframe_content=df_content_for_response,
            conversation_id=data_qna_result.get(
                "conversation_id", current_conv_id)
        )
        if request_payload.debug_mode:
            print(
                f"--- Sending API Response (ConvID: {response_data.conversation_id}) ---", flush=True)
            if response_data.sql_query:
                print(
                    f"DataQnA SQL: {response_data.sql_query[:500]}...", flush=True)
            print(f"Final Answer: {response_data.answer[:300]}...", flush=True)
        return response_data

    except HTTPException as http_exc:
        detail = http_exc.detail
        if isinstance(detail, str):
            detail = {"message": detail}
        if isinstance(detail, dict):
            detail["conversation_id"] = detail.get(
                "conversation_id", current_conv_id)
        else:
            detail = {"message": str(
                detail), "conversation_id": current_conv_id}
        raise HTTPException(status_code=http_exc.status_code, detail=detail)
    except Exception as e:
        print(
            f"\n!!! Unhandled Error processing /ask (ConvID: {current_conv_id}): {e} !!!", flush=True)
        traceback.print_exc()
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                            detail={"message": f"An unexpected error occurred: {str(e)}", "conversation_id": current_conv_id})


@app.get("/health", status_code=status.HTTP_200_OK)
async def health_check():
    service_states = {
        "data_query_service_status": "ok" if data_qna_instance else "unavailable",
        "secondary_reasoning_service_status": "ok" if vertex_ai_initialized else f"unavailable (model: {REASONING_MODEL_NAME})"
    }
    overall_status = "ok" if data_qna_instance and vertex_ai_initialized else "error"
    return {"status": overall_status, "services": service_states}


@app.get("/", response_class=HTMLResponse)
@app.get("/ui", response_class=HTMLResponse)
async def serve_ui():
    return HTMLResponse(content=HTML_CONTENT)


if __name__ == "__main__":
    try:
        import google.cloud.dataqna_v1alpha1
        print(
            f"google-cloud-dataqna version: {google.cloud.dataqna_v1alpha1.__version__}")
        from google.cloud.dataqna_v1alpha1.types import AskQuestionRequest
        # print(
        #     f"AskQuestionRequest fields (for reference): {AskQuestionRequest.pb(AskQuestionRequest()).DESCRIPTOR.fields_by_name.keys()}") # Potentially too verbose
        import vertexai
        print(
            f"google-cloud-aiplatform (Vertex AI) version: {vertexai.__version__}")

    except ImportError:
        print("One or more Google Cloud libraries (dataqna, aiplatform) not found.")
    except Exception as e:
        print(f"Error checking library versions: {e}")

    print(
        f"--- Starting Uvicorn Server Locally (DataQnA + Secondary Gemini Reasoning with {REASONING_MODEL_NAME}) ---")
    print("--- Access the Test UI at http://localhost:8089/ or http://localhost:8089/ui ---")
    filename_without_ext = os.path.splitext(os.path.basename(__file__))[0]
    uvicorn.run(f"{filename_without_ext}:app",
                host="0.0.0.0", port=8089, reload=True)
