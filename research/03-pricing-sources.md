# 03 - Pricing sources: books (IN + second country), non-book items, FX, appraisal, caching

Scope: where every price in `claim_packet.json` comes from, how each one stays traceable to a URL and a date, and what one developer can build in about 2 days. Identification and measurement are covered in 01/02.

Researched 2026-10-25. Each claim carries one of three tags:
- **[live 2026-10-25]**: I called the endpoint from this machine today and looked at the response.
- **[docs]**: the provider's own documentation or pricing page.
- **(unverified)**: third-party write-ups only, or not testable without an account. Check these before you build on them.

---

## 0. TL;DR decisions

| Price type | Primary | Fallback 1 | Fallback 2 | Last resort |
|---|---|---|---|---|
| **IN new (replacement)** | **SerpApi `amazon` engine, `amazon_domain=amazon.in`**: price plus the clean `/dp/ASIN` URL | **SerpApi `google_shopping`, `gl=in`**: multi-retailer, merchant URL via the Immersive Product API | **Bookswagon ISBN product page** (`/book/<slug>/<isbn13>`, schema.org Offer) | Convert the second-country new price and label it `converted`. Otherwise mark it "no price" and exclude it from the total |
| **IN used** | **Bookchor product page** when it lists a used-condition offer (schema.org Offer + condition tabs) | **Derived:** sourced IN new price × a versioned condition factor, labelled `derived` | None | Exclude from total |
| **US new** | SerpApi `amazon` (`amazon.com`) | SerpApi `google_shopping` `gl=us` | Google Books `country=US` *ebook* price, flagged as `ebook_equivalent` | Exclude |
| **US used** | **eBay Browse API `EBAY_US`**: active used fixed-price listings, median of matched listings | SerpApi `google_shopping` `gl=us`, used offers (ThriftBooks etc.) | Derived from US new | Exclude |
| **Non-book items (IN)** | SerpApi `google_shopping` `gl=in` → robust P25–P75 range, with each boundary result's merchant URL | SerpApi `amazon` `amazon.in` (stable `/dp/` URLs) | None | `needs_appraisal` / no price |
| **FX** | **ECB reference rate via Frankfurter `v2` pinned to `providers=ECB`** | ECB SDMX data API directly | FBIL (India's official reference rate) via Frankfurter, but it lags by days | None |
| **Rarity gate** | Deterministic rules run **before** pricing: year, spine/voice cues, Open Library edition data, and price signals over a threshold | None | None | None |

**Second country: US/USD.** US has the most data: eBay's largest marketplace, Amazon.com through every SERP vendor, and US Google Shopping is the layout SerpApi supports best. UK/GBP is fully supported by the same code (`EBAY_GB`, `amazon.co.uk`, `gl=uk`) and makes a cheap second check that locale really is a setting.

**Not viable in 2 days:** Amazon Creators API (needs 10 qualifying sales in the last 30 days), PA-API 5 (retired), Flipkart affiliate API (needs an approved affiliate token), the AbeBooks affiliate search API (needs approval through Impact), eBay Marketplace Insights / sold data (limited release), and BookFinder (no API, and robots.txt disallows search).

**Cost per sweep (80 books + 10 items + 10 US books): about 130–150 SERP calls, roughly $2–4 on SerpApi Developer/Starter.** eBay, FX, Open Library, Bookswagon and Bookchor are free. Details in §8.

---

## 1. New book prices in India

### 1.1 Google Books API (`volumes?country=IN`)

- **What `saleInfo` is:** the docs define `saleInfo` as "information about a volume related to the **eBookstore** and/or purchaseability". `saleability` is "whether or not this book is available for sale or offered for free in the **Google eBookstore** for the country listed". `listPrice` is the suggested retail price and `retailPrice` is the actual selling price. **[docs]** https://developers.google.com/books/docs/v1/reference/volumes
- **Consequence:** `listPrice`/`retailPrice` are **Google Play eBook prices**, not print prices. A print-only title comes back `NOT_FOR_SALE` with no price. `filter=paid-ebooks` restricts results to "Google eBooks with a price". **[docs]** https://developers.google.com/books/docs/v1/using
- **Coverage for Indian print books:** I could not measure it (next bullet). Expect gaps: many Indian-publisher titles have no Play edition, and an ebook price does not equal print replacement cost. **(unverified, measure on 20 books once a key exists)**
- **Keys are now effectively required.** Keyless calls from this machine returned `429 RESOURCE_EXHAUSTED`, `quota_limit: defaultPerDayPerProject`, `quota_limit_value: "0"` **[live 2026-10-25]**. The docs say public-data requests "must send an API key" **[docs]**. Forum reports describe a default of about 1,000 requests/day per keyed project and some projects stuck at 0 **(unverified)**: https://discuss.google.dev/t/where-to-start-google-books-api/162388
- **Verdict:** use it for **identification** (ISBN, edition, publisher) and as a **last-resort `ebook_equivalent`** price, flagged into the review queue. Never use it as the primary print replacement cost.

### 1.2 Amazon.in: PA-API 5 and Creators API

- **PA-API 5 is gone.** The old PA-API locale page for India (`webservices.amazon.ca/paapi5/documentation/locale-reference/india.html`) now 302-redirects to Amazon's deprecation notice, which says PA-API 5 "has been deprecated" and that calls get HTTP 403 **[live 2026-10-25]**: https://affiliate-program.amazon.com/creatorsapi/docs/en-us/paapiv5-deprecation. Third-party reports give the dates as 30 Apr 2026 (deprecation) and 15 May 2026 (retirement) **(unverified)**: https://dev.to/th3nate/amazon-pa-api-v5-is-shutting-down-april-30-2026-here-is-what-changes-at-the-auth-layer-22ek
- **Creators API (successor)** requirements: enrolment in Associates for the target marketplace, and "**at least 10 qualifying sales in the past 30 days**". Auth is OAuth2 client credentials. Operations: SearchItems, GetItems, GetVariations, GetBrowseNodes **[docs]** https://affiliate-program.amazon.com/creatorsapi/docs/en-us/introduction. Whether amazon.in is a supported Creators locale is not stated on that page **(unverified)**. A brand-new Associates account also has to reach 3 sales in 180 days just to stay open **(unverified)**.
- **Verdict:** **not obtainable in 2 days.** Get Amazon.in prices through a SERP vendor instead (§1.5). Amazon's Conditions of Use forbid scraping it yourself, so don't.

### 1.3 Flipkart affiliate API

- Official docs list Product Feed, Search and Offers APIs, authenticated with `Fk-Affiliate-Id` + `Fk-Affiliate-Token` headers: https://affiliate.flipkart.com/api-docs/affiliate_index.html. The endpoint is alive: `affiliate-api.flipkart.net/affiliate/1.0/search.json` returns `401` without headers **[live 2026-10-25]**.
- You need an approved affiliate account. The approval time, and whether Flipkart's books catalogue is fully exposed, are unknown **(unverified)**. The terms restrict the data to affiliate promotion.
- **Verdict:** not on the critical path. Flipkart prices already appear inside Google Shopping `gl=in` results.

### 1.4 Indian retailers directly (tested today)

| Site | What works | Robots / ToS | Use |
|---|---|---|---|
| **Bookswagon** | ISBN-addressable product URL `https://www.bookswagon.com/book/<slug>/<isbn13>`. Search `/search-books/<isbn>` links to it. The page has a schema.org `Product`/`Offer` block (its JSON is malformed: a missing quote) plus visible ₹ price and MRP (e.g. Sapiens 9780099590088: ₹559, MRP ₹799) **[live 2026-10-25]** | `/book/` and `/search-books/` are not disallowed for `*` **[live]**. I found no anti-bot clause on the T&C page I fetched (it may render client-side) **(unverified)** | **IN new, fallback 2.** Use low volume (≤1 req/s) and parse visible price + MRP; don't rely on the broken JSON-LD |
| **Crossword** (Shopify) | `/search/suggest.json?q=<isbn>` returns product + price (₹680, compare-at ₹799). `/products/<handle>.json` exposes `variants[].barcode` = ISBN-13 **[live 2026-10-25]** | robots.txt **disallows `/search`** for `*`. ToS forbids "spiders, robots ... intelligent agents" to "navigate or search this Website" **[live]** https://www.crossword.in/policies/terms-of-service | **Don't automate.** Fine for a human hand-check |
| **Sapna Online** (Next.js) | `__NEXT_DATA__` carries `product_unit_price`, ISBNs and stock per result (Sapiens: 646) **[live 2026-10-25]** | robots.txt: `User-agent: * Disallow: /` (only named search bots are allowed) **[live]** | **Don't automate** |
| **Pustakkosh** (WooCommerce) | Public Store API `/wp-json/wc/store/v1/products?search=` returns name, SKU (= ISBN on some items), regular/sale price in INR, and permalink **[live 2026-10-25]** | robots.txt allows it | Sparse catalogue, **no condition attribute**, and listings are inconsistent (two "Sapiens" at ₹465 and ₹630). Optional cross-check only |

### 1.5 Search-result APIs that return shopping prices with URLs

| Vendor | Engines relevant here | Free tier | Paid cost per query | Throughput | Notes / ToS risk |
|---|---|---|---|---|---|
| **SerpApi** | `google_shopping` (`gl=in` listed as supported: https://serpapi.com/google-shopping-countries), `google_immersive_product` (per-store merchant links + prices), `amazon` (`amazon.in`, `amazon.com`, `amazon.co.uk` all listed: https://serpapi.com/amazon-domains), `amazon_product` (by ASIN), `ebay` | **250 searches/month, 50/hour** | Starter $25/1k ($0.025), Developer $75/5k ($0.015), Production $150/15k ($0.010) **[docs]** https://serpapi.com/pricing | 200/h Starter, 1,000/h Developer | Cached identical searches (1 h) don't count. **The US Legal Shield excludes Free/Starter/Developer** **[docs]** https://serpapi.com/legal. SerpApi keeps search data for **31 days only**, so its archive cannot be your audit trail. Lowest-friction option overall |
| **Serper.dev** | `/shopping` (also search, images, etc.) | **2,500 free queries, no card** | $50/50k ($0.001) → $0.30/1k at volume. **Credits expire after 6 months** **[docs]** https://serper.dev | not published | No Amazon engine. Shopping response fields (whether `link` is always present) need confirming with a first call **(unverified)** |
| **SearchAPI.io** | Google Shopping, Amazon Search | 100 free requests | Developer $40/10k ($0.004) **[docs]** https://www.searchapi.io/pricing | n/a | Same model as SerpApi, cheaper. India params need confirming **(unverified)** |
| **DataForSEO** | Merchant API: Google Shopping products/sellers/product_info. Amazon Merchant API | small trial credit; minimum deposit reported as $50 **(unverified)** | $0.001 per SERP (40 products), Standard queue. $0.002 Priority **[docs]** https://dataforseo.com/pricing/merchant/google-shopping-api | async tasks | **Task-based POST then GET. Amazon supports only the Standard (queued) method** **[docs]** https://docs.dataforseo.com/v3/merchant-amazon-overview/. Latency puts the 5-minute packet budget at risk. Cheapest at scale, worst fit for a live demo |
| **Oxylabs Web Scraper API** | `amazon_search`/`amazon_product` with a `domain` (incl. `in`), Google Shopping | trial up to 2,000 results **[docs]** https://developers.oxylabs.io/changelog/july-2025/feature-based-billing-for-web-scraper-api | ~$0.50/1k Amazon, ~$1.00/1k Google (pricing page snapshot, **unverified**) | high | Good backup for Amazon.in, but more setup |
| **ScraperAPI** | Amazon/Google structured endpoints | 7-day trial, 5,000 credits | Amazon 5 credits, Google 25 credits per request. Hobby $49/100k credits **[docs]** https://www.scraperapi.com/pricing/ | 20 threads | **Geotargeting is "US & EU only" on Hobby/Startup.** India needs the $299 Business plan |

**Important Google Shopping caveats (SerpApi release notes, [docs] https://serpapi.com/google-shopping-api/release-notes):**
- Google's old product pages are gone. SerpApi now returns `product_link` to a Google immersive page plus an `immersive_product_page_token`. Merchant URLs are absent ("Remove Product URLs", 13 Nov 2025), and the sample response says "URLs are illustrative and not accessible".
- To cite a **retrievable merchant URL**, make one extra `google_immersive_product` call per product you actually use. It returns `stores[]` with `name`, `link`, `price`, `extracted_price`, `shipping`, `total` (3–5 stores, up to 13 with `more_stores=true`) **[docs]** https://serpapi.com/google-immersive-product-api
- Only do this for the 1–2 results that end up in the packet, not every hit.
- The Amazon engine avoids the problem: each organic result has `link_clean` (a `/dp/ASIN` URL), `asin`, `price` and `extracted_price` **[docs]** https://serpapi.com/amazon-search-api. That is why Amazon.in is the IN-new primary. It is also the source a human hand-checker reaches most easily ("same source type").

**ToS risk ranking (low → high):**
1. Official APIs (eBay Browse, Google Books, ECB).
2. SERP vendors. They carry the scraping risk, and SerpApi's legal shield applies only to plans from Production up.
3. Low-rate fetches of robots-permitted product pages (Bookswagon `/book/`, Bookchor `/book/`).
4. Anything robots.txt or the ToS forbids (Crossword search, Sapna, AbeBooks/BookFinder search, Amazon direct). **Exclude these from the automated pipeline.**

### 1.6 Recommendation (IN new)

1. Resolve the ISBN-13 at identification (Open Library / Google Books). If it is unknown, use title + author + format.
2. **SerpApi `amazon` on `amazon.in`**, query = ISBN-13 (then title + author). Accept a result only if the ISBN appears in the result or the title/author fuzzy match clears a configured threshold *and* the format matches (paperback/hardcover).
3. Store `price` (selling price) as `replacement_cost.amount`. Keep MRP when present as an extra field.
4. If there is no match, use **SerpApi `google_shopping` `gl=in`**: take the lowest-priced *matched new* offer. Then make one `google_immersive_product` call for the store `link`.
5. If there is still nothing, fetch the **Bookswagon ISBN page**.
6. If the exact edition is out of print, price the **current in-print edition of the same work and format** and record its ISBN in `equivalent_isbn`. This is "new or equivalent". Flag it `equivalent_edition`.
7. If no IN price exists anywhere, take the US price (§3) × the ECB rate, set `converted: true` and add a review-queue reason. Indian editions are often far cheaper than US ones, so a converted price overstates the claim.

---

## 2. Used book prices

### 2.1 What exists

| Source | Status in Oct 2026 | Usable? |
|---|---|---|
| **AbeBooks** | **`abebooks.co.in` does not exist** (DNS lookup fails) **[live 2026-10-25]**. abebooks.com prices default to USD, and "preferred currency" shows an approximate conversion only **[docs]** https://www.abebooks.com/docs/currency-preferences. The affiliate program offers "API access to search our site" via Impact Radius after approval **[docs]** https://www.abebooks.com/books/AffiliateProgram/. robots.txt disallows `/servlet/` and `/search/` for `*` **[live]** | **No** for automation in 2 days. Good for a human looking up rare items |
| **BookFinder** | Owned by AbeBooks/Amazon. No public API. robots.txt disallows `/search/`, `/isbn/`, `/book/` for `*` **[live]** | **No** |
| **eBay Browse API** | 16 marketplaces: AT, AU, BE, CA, CH, DE, ES, FR, **GB**, HK, IE, IT, NL, PL, SG, **US**. **No EBAY_IN** (eBay India is "no longer a functioning eBay marketplace") **[docs]** https://developer.ebay.com/api-docs/buy/static/ref-marketplace-supported.html. Active listings only. Sold history lives in **Marketplace Insights, which is a "limited release ... select developers"** **[docs]** https://developer.ebay.com/api-docs/buy/marketplace-insights/overview.html | **Yes for US/UK**, not for India |
| **SerpApi `ebay` engine** | `Sold`/`Complete` filters are "**deprecated and no longer supported**" **[docs]** https://serpapi.com/ebay-search-api | No sold comps here either |
| **Amazon.in used offers** | Amazon.in has had a used-books store with third-party sellers grading Like New / Good / Acceptable since 2016 https://www.medianama.com/2016/08/223-amazon-india-launches-used-book-store/. SerpApi's `amazon_product` documents an `other_sellers` option, but its sample shows no used-condition offers **[docs]** https://serpapi.com/amazon-product-api | **Maybe.** Test 10 ISBNs. Don't depend on it |
| **Bookchor** | Indian new + used bookseller. Product page `https://www.bookchor.com/book/<isbn13>/<slug>` has a schema.org `Offer` with `itemCondition`, plus a condition selector (`setPrice(...)` tabs: New / used grades). The cart shows "Condition: Good ₹189 / MRP ₹399" for used stock **[live 2026-10-25]**. robots.txt allows `/book/` | **Yes, as the only machine-readable IN used source**, but inventory-limited: Sapiens 9781787304123 had only a New offer today |
| **Pustakkosh** | WooCommerce Store API works but has no condition field **[live]** | No for used value |

### 2.2 Is a derived used price acceptable under "a price with no retrievable source counts as fabricated"?

Yes, provided it is labelled honestly and built only from retrievable inputs. The rule targets prices that no one can trace, such as an LLM's memory. A derived value is traceable when:

1. **Its base price is a real, cited, dated new price** (the same record as `replacement_cost`: URL + `retrieved_at` + raw snapshot).
2. **The condition factor is a versioned config value** (e.g. `pricing_rules/condition_factors.v1.yaml`). It is never set per book by a model. The file states its own basis, and the packet records which rule version was applied.
3. **The computation is done in code**, so an adjuster can recompute it: `used = round(new × factor[condition_assumed])`.
4. **The packet labels it.** Set `used_value.source = "derived_from_new"`, `used_value.url = <new-price URL>`, `used_value.method = "condition_factor:v1:good"`, and `condition_assumed` filled. Totals report the market-observed and derived used values separately, so nobody mistakes one for the other.

**How to make the factor itself sourced, not invented:** I found no authoritative published depreciation rate for used books. The only references are anecdotal: buyback at ~50% and resale at ~75% of cover price for college texts (https://liblicense.crl.edu/ListArchives/0508/msg00003.html), and the standard insurance ACV formula of replacement cost × (1 − age/life) (https://acera.ca/understanding-how-depreciation-impacts-insurance-claims/). So **calibrate the factor empirically** and ship the calibration as evidence:
- collect matched new/used pairs with URLs: Bookchor used vs Bookchor/Amazon.in new for India, and eBay used median vs Amazon.com new for the US. Aim for 30+ ISBNs, saved as a dated snapshot;
- set `factor[condition] = median(used/new)` per condition grade, per country;
- keep the calibration set **disjoint from the 15 hand-checked ground-truth books**, so you are not tuning to the test.

**Recommended approach (one choice):** **observed first, derived as fallback, both labelled.**
- *India:* use a Bookchor used offer when one exists (condition from the listing). Otherwise use the derived value from the sourced IN new price with the calibrated factor for the assumed condition. The default assumption is "Good" (a shelved, read book); the user can change it by voice. Stretch goal: per-book condition grading from the frame.
- *US:* use the eBay used median (§3). Otherwise the derived value.

**Why:** India has no API-accessible used market, so "observed only" would leave most used values blank and make `books_used_value` meaningless. A derived value from a sourced price, using a published and calibrated rule, is reproducible and auditable. Exclusion stays available as a setting (`used_value_policy: observed_only | observed_then_derived`). If the evaluator rejects derived values, flip it to `observed_only`.

---

## 3. Second country (US/USD)

| Price | Source | How | Notes |
|---|---|---|---|
| **New** | SerpApi `amazon` with `amazon_domain=amazon.com` | ISBN then title/author. Cite `link_clean` + `extracted_price` | Same code path as IN with a different locale config. This proves locale is a setting |
| New fallback | SerpApi `google_shopping` `gl=us` + `google_immersive_product` for the store link | Lowest matched *new* offer | |
| New last resort | Google Books `country=US` paid-ebooks | Flag as `ebook_equivalent` | §1.1 |
| **Used** | **eBay Browse API**, `X-EBAY-C-MARKETPLACE-ID: EBAY_US`, `item_summary/search` | `q=<title author>` or ISBN keyword, `category_ids=267` (Books & Magazines, **verify**), `filter=conditions:{USED},buyingOptions:{FIXED_PRICE},priceCurrency:USD,deliveryCountry:US` **[docs]** https://developer.ebay.com/api-docs/buy/static/ref-buy-browse-filters.html. Verify the ISBN per hit through `getItem` aspects. Drop titles matching signed/first-edition/lot patterns. Take the **median price + shipping** of ≥3 matched listings, and cite the listing nearest the median (`itemWebUrl`) plus the list of contributing item IDs | `gtin` search accepts UPC only per the docs, so use keyword + aspect verification **(verify)**. These are **asking prices of active listings**, not sold prices, and the packet should say so. Default limit is **5,000 calls/day** **[docs]** https://developer.ebay.com/develop/api/buy/api_call_limits. OAuth client-credentials (application) token. Free |
| Used fallback | SerpApi `google_shopping` `gl=us` | Used offers (ThriftBooks, Better World Books) appear with `second_hand_condition` on some layouts **(unverified)** | |
| Used last resort | Derived from US new with the US-calibrated factor | §2.2 | |
| AbeBooks / BookFinder | Human hand-checks of rare items only | robots.txt disallows automated search | |

eBay licence notes: the 2025 API License Agreement update bars feeding **Restricted API** data (market trends, pricing insights such as Marketplace Insights) into third-party generative AI without consent (https://www.ecommercebytes.com/tag/ebay-license-agreement/). Browse is not a Restricted API. Still, **keep eBay prices out of the LLM context** entirely: the pricing stage is deterministic code, and the LLM only sees the final numbers if it reads the summary aloud. eBay's 2026 User Agreement change bans LLM agents that *place orders* (https://shellypalmer.com/2026/01/ebay-just-banned-ai-shopping-agents/), which doesn't apply here.

UK swap: `EBAY_GB` + `priceCurrency:GBP`, `amazon.co.uk`, `gl=uk`, with the Bank of England/ECB rate.

---

## 4. FX conversion with a source and a date

| Option | Status [live 2026-10-25] | Source of record | Verdict |
|---|---|---|---|
| **Frankfurter v2** `https://api.frankfurter.dev/v2/rates?base=USD&quotes=INR&providers=ECB` | Works, no key. Returned `{"date":"2026-10-23","base":"USD","quote":"INR","rate":94.95}`. `/v2/providers` lists 106 providers incl. **ECB**, **FBIL** (India), **BOE**, FRED | Pin `providers=ECB`. **Without `providers`, the rate is "blended across all providers ... Frankfurter's own calculation"** (dated 2026-10-25, 94.81), which is not a named official source | **Primary** |
| Frankfurter v1 | `api.frankfurter.app` now **301 → `api.frankfurter.dev/v1/`**. v1 still answers (`USD→INR 94.94`, date 2026-10-23) | ECB only | Works, but **legacy host. Use v2 and don't mix paths** https://frankfurter.dev |
| **ECB SDMX data API** `https://data-api.ecb.europa.eu/service/data/EXR/D.INR.EUR.SP00.A?lastNObservations=1&format=csvdata` | Works: INR/EUR **106.272** on 2026-10-23 | ECB itself, the most authoritative | **Fallback / audit cross-check**. Cross-rate USD→INR = (INR/EUR)/(USD/EUR) |
| FBIL via Frankfurter `providers=FBIL` | Works but stale: latest **2026-10-16** (95.0194) | India's official reference rate | Use only if the insurer requires an RBI/FBIL rate. Record the lag |
| exchangerate.host | **Now needs `access_key`** (`missing_access_key` error). Free plan 100 req/month **[docs]** https://exchangerate.host/pricing | apilayer | **Treat as deprecated** as a free, keyless source |
| open.er-api.com | Works, no key, daily update | Data source **not disclosed**. Attribution link required, **redistribution not allowed**, 429 lock for 20 min if over the limit **[docs]** https://www.exchangerate-api.com/docs/free | Not defensible for a claim (unknown source) |

Rules:
- Convert in code: `amount_local = round(amount_src × rate, 2)`.
- Store the original price record untouched, plus `fx: {provider: "ECB via Frankfurter", rate, rate_date, url, retrieved_at, raw_ref}`, and set `converted: true`.
- Use the latest ECB business-day rate on or before `retrieved_at` of the source price. ECB doesn't publish on weekends, so a Saturday sweep uses Friday's rate, and `rate_date` shows that.

---

## 5. Rarity and appraisal gate

This runs **before** pricing. A gated book gets `status: needs_appraisal`, no amounts, and a review-queue reason. Every threshold lives in config, keyed by country/currency.

| Signal | Source | Rule (configurable) |
|---|---|---|
| User says so | Voice ("that's a first edition / signed") | Hard gate |
| Spine/cover text | OCR transcript contains signed, first edition, 1st ed., limited, numbered, "x/500", or an author inscription | Gate |
| Publication age | Open Library edition `publish_date` / work `first_publish_year`. Example: Sapiens 9780099590088 → `publish_date 2015-04-30`, `physical_format Paperback` **[live 2026-10-25]** | Edition year before `antiquarian_year` (setting, e.g. 1950) → gate. "Rare"/"antiquarian" have no precise trade definition (https://thecanadianencyclopedia.ca/en/article/antiquarian-books), so this is a stated assumption |
| No ISBN on a resolved edition | Open Library edition without `isbn_10/13` | Pre-ISBN era (ISBN began around 1970): gate, or at least review |
| Price signal | Any sourced used price ≥ `appraisal_threshold` (e.g. ₹5,000 / $75), or **used ≥ 1.5 × new** (collectible premium), or eBay matches dominated by "first edition/signed" listings | Gate, and discard the auto-price |
| Format cues | Leather/fine binding, slipcase, or a vision "antique" tag | Review |
| Identification confidence | Edition unresolved but the work is old or valuable | Review, no guess |

Open Library etiquette: send `User-Agent: AppName (contact@email)` to get 3 req/s instead of 1 req/s **[docs]** https://openlibrary.org/developers/api. Open Library also returns `physical_dimensions` (e.g. "19.7 x 12.7 x 3.8 centimeters"), a free cross-check for the spine measurements in 02.

Art and portraits default to `needs_appraisal` unless the user says the piece is a print (brief §4).

---

## 6. Non-book items (India)

Query building is deterministic code, not a free-form model prompt: `category + material + size class (+ brand/model if legible)`. Examples: "solid wood 5 tier bookshelf", "table lamp", "coffee machine <brand model>".

1. **Brand and model legible:** SerpApi `google_shopping` `gl=in` with the exact model. Accept ≥1 matched result, then call `google_immersive_product` for the store link → `status: priced`. Use `low = high = lowest matched new offer`, or the min/max across the stores returned.
2. **Generic:** SerpApi `google_shopping` `gl=in` (one page, ~20–40 results), and optionally SerpApi `amazon` `amazon.in` for stable `/dp/` URLs.
   - Filter in code: drop results whose title fails a category keyword match, and drop accessories, used/refurbished items and multipacks.
   - Need **≥5** surviving results. Fewer → `needs_appraisal` / review.
   - Take **low = P25, high = P75** of `extracted_price`. Quartiles are robust to the ₹99 accessory and the ₹2 lakh designer piece; IQR outlier trimming gives the same effect.
   - Resolve the **result nearest P25 and nearest P75** to merchant URLs (2 immersive calls). Put both in `replacement_cost` (`url` = low-end URL; add `url_high`, `retrieved_at`), and keep the full contributing list (title, price, source, product_link) in the raw cache.
   - `status: range`.
3. **Dimensions matter for furniture.** Pass the measured size class into the query ("180 cm bookshelf"), and drop results whose parsed dimensions differ by >50% if the title states them.
4. **Electronics with no model:** give a range, but cap confidence and add a review reason.
5. **Never price** art, portraits or antiques automatically. Use `needs_appraisal`, and have the agent ask "original or print?".

Cost: 10 items × (1 shopping + 2 immersive) ≈ 30 SerpApi searches. Using the `amazon.in` engine for the generic categories instead of the immersive calls brings it down to ~20.

---

## 7. Caching and reproducibility

1. **Raw response cache.** Write `data/price_cache/<sha256(provider|engine|normalized_params|country)>.json.gz` containing:
   - `request_url` (with the API key stripped), `provider`, `engine`, `params`;
   - `retrieved_at` (UTC ISO-8601), HTTP status, `response_sha256`, and the raw body.
   Price records point to this file via `raw_ref`.
2. **Page snapshot of the cited URL.** For each URL that ends up in the packet, GET the page once and save `snapshots/<sha256>.html` + headers + `retrieved_at`. Optionally render it to PDF/PNG with Playwright for the report.
   - This matters because SerpApi deletes search data after **31 days**, eBay listing URLs die when the listing ends, and Google immersive URLs are not stable.
   - Optional public archive: submit to the Wayback Machine (`https://web.archive.org/save/<url>`) and store the returned memento URL. It often fails for Amazon, so treat it as best-effort.
3. **Price record (extends the contract):**
   - `amount`, `currency`, `source`, `source_type` (`retailer_new | marketplace_used_active | derived_from_new | converted | ebook_equivalent`), `url`, `retrieved_at`;
   - `query`, `match` (`isbn_exact | fuzzy:score`), `raw_ref`, `snapshot_ref`;
   - `fx{...}` if converted, and `rule_version` if derived.
4. **TTL.** Reuse cache hits younger than `price_ttl_days` (setting, e.g. 7). The packet then shows the **original** `retrieved_at`, never "now". Re-sweeping the same room is close to free, and the dates stay honest.
5. **Determinism.** Pricing is a pure function of (identified book, locale config, cache). `make reprice --offline` replays from cache and must reproduce the packet byte-for-byte. That is a cheap test to put in the repo.
6. **Totals in code.** Totals cover only `retailer_new` and `converted` for replacement, and observed + derived for used, reported as separate sub-totals. Excluded lines are counted in `excluded_from_totals`.

---

## 8. Final recommended stack and cost per sweep

Assume 80 books, ~75 identified and priceable (the rest unidentified or appraisal), 10 items, and 10 books re-priced for the US.

| Stage | Calls | Unit cost | Cost |
|---|---|---|---|
| IN new: SerpApi `amazon.in` | 75 | $0.015 (Developer) / $0.025 (Starter) | $1.13 / $1.88 |
| IN new fallback: `google_shopping gl=in` + immersive (~25% of books) | ~19 + ~10 | same | $0.44 / $0.73 |
| IN new fallback: Bookswagon page | ~5 | free | $0 |
| IN used: Bookchor page | ~75 HTTP GETs at ≤1/s, in parallel with SERP calls | free | $0 |
| IN used derived | code | free | $0 |
| Items: shopping + immersive | ~30 | same | $0.45 / $0.75 |
| US 10 books new: `amazon.com` | 10 | same | $0.15 / $0.25 |
| US 10 books used: eBay Browse (search + getItem) | ~30 | free (5k/day) | $0 |
| FX: Frankfurter/ECB | 2 | free | $0 |
| Identification lookups (Open Library + Google Books key) | ~160 | free | $0 |
| **Total** | **~145 SERP calls** | | **≈ $2.2 (Developer) / ≈ $3.6 (Starter)** |

- **Throughput matters more than price.** SerpApi Free allows **50 searches/hour**, so a 145-search sweep cannot finish inside the 5-minute packet budget on Free. Starter allows 200/h: one sweep per hour, which is enough for the demo plus the live review. Developer allows 1,000/h. **Buy Starter ($25) for the build week, or Developer ($75) if you run many test sweeps.** At ~8 concurrent requests and ~3 s each, 145 calls take about 60 s.
- **Cheaper alternative:** Serper.dev `/shopping` at $0.001 for the Google Shopping legs (2,500 free credits), keeping SerpApi only for the Amazon engine. That costs about $1.2 per sweep, but Serper's link fields need checking first.
- **Not recommended for the live path:** DataForSEO (async queue latency) and ScraperAPI (India geotargeting needs the $299 plan).

**Build order (fits in about 1 day of the 2):**
1. Locale config + price record schema + cache.
2. SerpApi Amazon (IN/US).
3. Frankfurter-ECB.
4. eBay Browse.
5. Shopping range for items.
6. Bookchor + derived used.
7. Bookswagon fallback.
8. Appraisal gate.
9. Snapshotting.

**Ground-truth protocol for the 25% pass bar ("same source type"):** for each of the 15 hand-checked books, open the **same retailer type** the system cited (Amazon.in product page for new, the Bookchor or eBay listing for used) on the **same day**, record price + URL + screenshot by hand, and compare. Report misses, including books where the system picked a different edition.

---

## 9. Live verification log (2026-10-25, from this machine)

| Probe | Result |
|---|---|
| `googleapis.com/books/v1/volumes?...&country=IN` (no key) | 429, quota_limit_value "0". Key required |
| `api.frankfurter.app/latest` | 301 → `api.frankfurter.dev/v1/latest` |
| `api.frankfurter.dev/v1/latest?base=USD&symbols=INR,GBP` | INR 94.94, GBP 0.74804, date 2026-10-23 |
| `api.frankfurter.dev/v2/rates?base=USD&quotes=INR&providers=ECB` | 94.95, 2026-10-23 (default blended: 94.81, 2026-10-25) |
| `.../v2/rates?...&providers=FBIL` | 95.0194, 2026-10-16 (lagging) |
| ECB SDMX `EXR/D.INR.EUR.SP00.A` | 106.272, 2026-10-23 |
| `api.exchangerate.host/latest` | `missing_access_key` |
| `open.er-api.com/v6/latest/USD` | success, updated 2026-10-25 00:02 UTC |
| `abebooks.co.in` | DNS: no such domain |
| Crossword `search/suggest.json?q=9780099590088` | Sapiens ₹680 (compare-at ₹799). Barcode = ISBN in `products/<handle>.json`. But `/search` is robots-disallowed and the ToS bans bots |
| Bookswagon `/book/sapiens-yuval-noah-harari/9780099590088` | 200. ₹559, MRP ₹799 visible. JSON-LD malformed |
| Sapna `__NEXT_DATA__` | `product_unit_price` 646. robots `Disallow: /` for `*` |
| Bookchor `/book/9781787304123/sapiens` | schema.org Offer INR 807, `NewCondition`. Condition-selector markup present. Used "Condition: Good" seen in cart widget |
| Pustakkosh `wp-json/wc/store/v1/products?search=sapiens` | 2 results (₹465, ₹630), no condition attribute |
| Flipkart `affiliate-api.flipkart.net/.../search.json` | 401 without token (endpoint alive) |
| Open Library `/isbn/9780099590088.json` | Vintage, 2015-04-30, Paperback, 498 pp, 19.7×12.7×3.8 cm |

Not testable without accounts (do these in the first hour of the build): SerpApi `amazon.in` ISBN hit rate, Google Shopping `gl=in` immersive store links, Serper `/shopping` fields, Google Books IN ebook-price coverage, eBay Browse used-listing match rate for 10 sample ISBNs, and Amazon.in used offers via `amazon_product`.
