# 05 - Free pricing stack (no paid services, robots.txt + ToS compliant)

Replaces the SerpApi-based recommendation in `03-pricing-sources.md`.

**New constraint:** the only credential is the user's existing Gemini API key. Every price source must:
- be free, with no card needed;
- be either an official API or a page that robots.txt allows us to fetch;
- not be forbidden by its terms of service.

Each price is parsed in code from a fetched response. A model never produces a number.

All live tests ran on **2026-10-26 from this machine** (Windows, Python 3.12.4 `-I`, httpx 0.28.1, curl 8.19). Tags used below:
- **[live]**: I fetched the endpoint and parsed the response today.
- **[docs]**: taken from the provider's own documentation or terms page, fetched today.
- **[untested]**: needs an account or key I was not allowed to create, or could not be exercised.

The scratch scripts are in `_scratch/pricing/scripts/` (gitignored). The raw responses (295 files) are under `%TEMP%\cozmo_pricing_probe\pages`, because **drive D: was full during the run** (320 KB free at one point).

---

## 0. TL;DR

| Price type | Primary | Fallbacks (in order) | Measured hit rate (10-book core set) |
|---|---|---|---|
| **IN new (replacement)** | **Bookchor** ISBN page, JSON-LD `Offer` with `NewCondition` | 1. **Bookswagon** ISBN page, JSON-LD `Offer`, repaired. *ToS risk, see 2.1*<br>2. Pustakkosh WooCommerce Store API<br>3. Converted from the second country, labelled<br>4. Excluded | Bookchor 8/10, Bookswagon 10/10, Pustakkosh 1/10. **Chain: 10/10** |
| **IN used** | **Bookchor** JSON-LD `Offer` with `UsedCondition` (grade shown as "Good") | 1. Pustakkosh (`sku` ends in `_Used`)<br>2. **Derived** = sourced IN new × versioned condition factor, labelled<br>3. Excluded | Observed 3/10 (Bookchor 2, Pustakkosh 1). Derived covers the other 7/10 |
| **Second country** | **UK/GBP (recommended over US)**: Penguin UK RRP from the book page (`__NEXT_DATA__` `rrp`) | 1. eBay Browse `EBAY_GB` (**[untested]**: account plus partner approval, see 3.3)<br>2. Converted from IN new at the ECB rate, labelled `converted`<br>3. Ebook price (Google Books / iTunes) as a labelled `ebook_equivalent`, outside totals | Penguin UK 3/10. iTunes GB ebook 8/10 (format mismatch). **No free, compliant US/UK print retailer worked** (see 3) |
| **Non-book items (IN)** | Robots-allowed **category listing pages**: IKEA India (`data-price` on cards), Urban Ladder and Vijay Sales (ItemList, then product JSON-LD), Nilkamal (Shopify `products.json`). Range = nearest-rank P25–P75 of matched rows, each bound a real listing with a URL | Fewer than 5 matches: `needs_appraisal`, with the rows kept as evidence | Bookshelf n=34, table lamp n=29, framed wall art n=14, **drip coffee maker 2 true matches → review** |
| **FX** | **Frankfurter v2 `providers=ECB`** (keyless) | ECB SDMX data API | **[live]** GBP→INR 126.93, USD→INR 94.95, both dated 2026-10-23 |
| **Discovery / search APIs** | **Not needed for books**: both IN sources are addressable by ISBN | Items use configured category URLs | Gemini Search grounding is **forbidden by its terms** for this use (see 4) |

**Cost: ₹0 / $0 per sweep.** Every call in the recommended path is keyless. **Latency for about 70 books:** roughly 70–90 s wall-clock at 1 request/s per host, with hosts running in parallel, and it can start during the sweep (see 6).

---

## 1. Test set (verified)

I checked each ISBN with Open Library `/isbn/<isbn>.json` and the ISBN-13 checksum. Two of the supplied entries were wrong:

| # | ISBN-13 | Actual title (publisher) | Published in India? | Note |
|---|---|---|---|---|
| 1 | 9780099590088 | Sapiens (Vintage UK, pb) | | |
| 2 | 9780143127550 | **Everything I Never Told You**, Celeste Ng (Penguin US) | | Not "Into the Water". Kept, because it is a valid, well-known ISBN |
| 3 | 9780141439518 | Pride and Prejudice (Penguin Classics) | | |
| 4 | 9780062316097 | Sapiens (Harper US, hardback) | | |
| 5 | **9789390166268** | The Psychology of Money (**Jaico**, India) | yes | **Replaces 9789354892227**, which fails the ISBN-13 checksum (check digit should be 6, not 7). Open Library 404, Bookswagon search empty, Bookchor redirects to search |
| 6 | 9780143333623 | Grandma's Bag of Stories, Sudha Murty (**Penguin Books India**) | yes | Verified |
| 7 | 9780547928227 | The Hobbit (Mariner/HMH US, pb) | | |
| 8 | 9781847941831 | Atomic Habits (Random House Business UK, pb) | | |
| 9 | 9788173711466 | Wings of Fire, A.P.J. Abdul Kalam (**Universities Press**, India) | yes | Added |
| 10 | 9788129115300 | 2 States, Chetan Bhagat (**Rupa**, India) | yes | Added |
| x | 9780143028574, 9788129104595, 9780143442295 | God of Small Things (Penguin India), Five Point Someone (Rupa), Life's Amazing Secrets (Penguin Ananda) | yes | Extra India-only checks |

Generic items: "wooden bookshelf 5 tier", "table lamp", "drip coffee maker", "framed wall art print".

---

## 2. Candidate results

Legend:
- **Free/card**: whether it is free, and whether a card is needed.
- **Robots**: whether robots.txt allows the specific path for `User-agent: *`.
- **ToS**: whether an automated-access clause was found in the site's terms.

### 2.1 India, new

| Source | Free / card | Robots (path used) | ToS | Price found? | Where the price lives | URL pattern | Hit rate |
|---|---|---|---|---|---|---|---|
| **Bookchor** | Free, no account | `/book/` allowed. `*` group only blocks `/admin/`, `/webservices/`, `/scrappers/`, `*&page*` etc. **[live]** | `/terms-conditions` and `/bc-policies`: **no robot/scrape/crawl clause** **[live]** | Yes | JSON-LD `Product.offers[]` with `@type: Offer`, `itemCondition: https://schema.org/NewCondition`, `price`, `priceCurrency: INR`. Plus `AggregateOffer` (low/high). MRP is the 4th argument of `setPrice('550','31','249','799',…)` | `https://www.bookchor.com/book/<isbn13>/<any-slug>` → 200. `<link rel=canonical>` gives the real slug | **New 8/10** core (10/13 overall). No new offer for Wings of Fire or 2 States (used only) |
| **Bookswagon** | Free | `/book/`, `/search-books/` allowed **[live]** | **Risk.** The Terms of Use say "Harvesting or otherwise collecting information is also prohibited. You must not hyperlink or frame Bookswagon without gaining prior written consent." (https://www.bookswagon.com/termsofuse.aspx) **[live]**. The clause sits in a user-conduct list and is ambiguous, but it is there | Yes | JSON-LD `Product.offers` (`Offer`, `price`, `INR`, `NewCondition`, `availability`). **The JSON is malformed** (`"image":https://…"` is missing its opening quote), so repair it before `json.loads`. Visible labels: `#ctl00_phBody_ProductDetail_lblourPrice` (₹559), `#…lblListPrice` (MRP ₹799), `#…lblBinding` (Paperback). Spec list: `ISBN-13:`, Height (mm), Spine Width (mm) | `https://www.bookswagon.com/book/<any-slug>/<isbn13>` → 200. Canonical has the real slug | **10/10** core, **13/13** overall. One `OutOfStock` (Five Point Someone ₹87) |
| **Pustakkosh** (WooCommerce) | Free | `/wp-json/` not disallowed **[live]** | `/terms-and-condition/`: no clause **[live]** | Sometimes | Store API JSON: `prices.price` (minor units, `currency_minor_unit` 0), `sku` = ISBN (with a `_Used` suffix for used copies), `short_description` "Book Condition: New" | `https://pustakkosh.com/wp-json/wc/store/v1/products?search=<isbn13>`, then filter `sku.startswith(isbn)` | 1/10 new (Sapiens ₹630), 1/10 used |
| Crossword (Shopify) | Free | **`/search` disallowed** **[live]** | Forbids "spiders, robots" (see 03) | not run | n/a | n/a | **Excluded** |
| Sapna Online | Free | **`User-agent: * Disallow: /`** **[live]** | n/a | not run | n/a | n/a | **Excluded** |
| Amazon.in | Free | `/dp/` not disallowed for `*`. `ClaudeBot`, `GPTBot`, `Scrapy` blocked **[live]** | **Forbids** "data mining, robots, or similar data gathering and extraction tools" **[live]** | not run | n/a | n/a | **Excluded** |
| Flipkart | Free | Product pages not disallowed **[live]** | **Forbids** "deep-link, page-scrape, robot, spider or other automatic device" **[live]** https://www.flipkart.com/pages/terms | not run | n/a | n/a | **Excluded** |
| Penguin India (penguin.co.in) | Free | `/book/` and `?s=` allowed **[live]** | **Forbids** "data mining, robots" and "spidering, screen scraping" **[live]** | Yes (MRP ₹250 on the book page) | Visible text only, no Offer JSON-LD | `/?s=<isbn>&filter_search1=book` → `/book/<slug>/` | **Excluded (ToS)** |
| HarperCollins India | Free | `/wp-json/` allowed, `/search/` disallowed **[live]** | **Forbids** "data mining, data harvesting, data extraction" **[live]** https://harpercollins.co.in/terms-of-use/ | Store API works (`sku` = ISBN, `prices.price` in paise) | | `/wp-json/wc/store/v1/products?search=<isbn>` | **Excluded (ToS)**. 0/1 anyway (Jaico title) |
| Pan Macmillan India | Free | Only `/api/`, `/preview` disallowed | not checked | No JSON-LD offers on the homepage | | | Not pursued (own titles only) |
| Kitabay | ? | robots.txt 200 | n/a | **429** on the first homepage hit **[live]** | | | Not testable from this IP |
| Pothi / Notion Press | Free | Allowed (Notion Press has crawl-delay 5) | n/a | Self-published catalogues only: 0 hits for any test ISBN | | | Not useful |
| Google Books API `country=IN` | Free, **needs an API key** (no card) | Official API | n/a | **[untested]**: keyless calls return `429 Quota exceeded … 'Queries per day'` (shared anonymous project) **[live]** | `items[].saleInfo.retailPrice` = **Google Play ebook** price | `GET /books/v1/volumes?q=isbn:<isbn>&country=IN&key=…` | Unknown. Ebook only, so it can never be the print replacement cost |

**Price disagreement between the two IN retailers for the same ISBN (which matters for the 25% bar):**

| ISBN | Bookchor new | Bookswagon new | Diff | MRP Bookchor / Bookswagon |
|---|---|---|---|---|
| 9780099590088 | 550 | 559 | 2% | 799 / 799 |
| 9780143127550 | 1571 | 947 | **66%** | 1747 / 2367 |
| 9780141439518 | 286 | 242 | 18% | 350 / 350 |
| 9780062316097 | 3274 | 2365 | **38%** | 3638 / 3638 |
| 9789390166268 | 302 | 275 | 10% | 399 / 399 |
| 9780143333623 | 242 | 193 | **25%** | 275 / 275 |
| 9780547928227 | 1658 | 1401 | 18% | 1842 / 1843 |
| 9781847941831 | 668 | 649 | 3% | 999 / 999 |
| 9788173711466 | none | 337 | | n/a / 495 |
| 9788129115300 | used only | 708 | | 399 / **1011** (edition mismatch?) |

Consequences:
1. **Hand-checks must use the same retailer** the packet cites, as the brief's "same source type" requires. Selling prices across Indian retailers vary by up to 66% for one ISBN.
2. **MRP is stable:** it matched to the rupee on 7 of 9 shared ISBNs. Store MRP next to the selling price in every IN record. If the two MRPs differ by more than 10%, route the line to the review queue as `edition_mismatch_suspected` (2 States: ₹399 vs ₹1,011).

### 2.2 India, used

| Source | Free / card | Robots | ToS | Price found? | Where | Hit rate |
|---|---|---|---|---|---|---|
| **Bookchor** | Free | Allowed | No clause | Yes | Separate JSON-LD `Offer` with `itemCondition: UsedCondition`. The condition tab shows `<p>Good <span>₹ 661</span></p>` with radio `value="3"` (1 = New) and description "Designated usage on front & back cover" | **2/10** core (EINTY ₹661 Good, 2 States ₹62 Good). **4/13** overall (+ Five Point Someone ₹135, Life's Amazing Secrets ₹135) |
| **Pustakkosh** | Free | Allowed | No clause | Rarely | `sku` = `<isbn>_Used`. `prices.price`. No grade | 1/10 (Wings of Fire ₹251). Catalogue is mostly textbooks |
| OLX India | Free | `*` disallows `*/items/` (search results), `/api/`, `*/q-%..`. Individual `/item/` pages have no discovery route **[live]** | n/a | not run | | **Excluded** |
| AbeBooks | `abebooks.co.in` doesn't exist (see 03) | | | | | **Excluded** |
| Amazon.in used | ToS forbids | | | | | **Excluded** |

**Observed used/MRP ratios** (Bookchor, n=4): 0.38, 0.16, 0.45, 0.45. **Used/new on the same site** (n=2): 0.42, 0.53.

This is far too few points to set a factor. Calibration recipe: probe about 120 ISBNs (Bookchor has used stock for about 30% of them), giving roughly 35 pairs, at 1 request/s, which takes about 2 minutes. Use ISBNs disjoint from the 15 ground-truth books. Set `factor[grade] = median(used/new)` and commit it as `config/pricing/condition_factors.v1.yaml` together with the snapshot it came from.

### 2.3 Second country: US and UK

| Source | Free / card | Robots | ToS | Result today | Verdict |
|---|---|---|---|---|---|
| **Penguin UK** (penguin.co.uk) | Free | Only `/*search-results*` disallowed **[live]** | Terms page not locatable (old URL redirects to `/about/who-we-are`) → **unverified** | `https://www.penguin.co.uk/books/x/x/<isbn13>` **302 → canonical book page** when the ISBN is a PRH UK edition. Otherwise it redirects to `/search-results` (not found). Price is `__NEXT_DATA__` `"rrp":"£12.99"` plus visible "Price: £12.99". No Offer JSON-LD | **3/10** (Sapiens Vintage £12.99, Pride & Prejudice £9.99, Atomic Habits £18.99). **Primary for UK new**, as publisher RRP |
| eBay Browse API (`EBAY_US` / `EBAY_GB`) | Developer Program is free with no card mentioned. Keyset approval about 1 business day **[docs]** | Official API | **"The use of eBay's Buy APIs in production is intended for eBay partners only. You must apply for production access through the eBay Partner Network… There is no guarantee"**, EPN replies "within 10 business days" **[docs]** https://developer.ebay.com/api-docs/buy/buy-requirements.html. The licence also says listing data shown must be ≤ 6 h old (search excerpt of the API License Agreement, **unverified verbatim**) | **[untested]**: no account created | **Apply on day 1, but don't depend on it.** If production Browse works with the default keyset (often reported, **unverified**), it is the best US/UK used source |
| AbeBooks (.com / .co.uk) | Free | `/products/isbn/<isbn>` and `/<isbn>/<slug>/plp` **not** disallowed **[live]** | **Forbids** "data mining, robots, or similar data gathering and extraction tools" **[live]** https://www.abebooks.com/docs/Legal/termsAndConditions.shtml | 1 probe: JSON-LD `AggregateOffer` per condition, `lowPrice` 17.57 USD Used (55 offers), 17.97 USD New (37) | **Excluded (ToS)**, although technically ideal |
| Barnes & Noble | Free | `/s/<isbn>` → `/w/…?ean=` not disallowed | **Forbids** "robot, spider, data miner, crawler, scraper" **[live]** | 1 probe: JSON-LD Offers 32.18 / 16.75 / 15.75 USD (one per format) | **Excluded (ToS)** |
| Better World Books (site + `products.betterworldbooks.com/service.aspx`) | Free | `/product/detail/` allowed | n/a | **403 Cloudflare challenge** for both UAs **[live]** | **Excluded** (bot wall) |
| ThriftBooks | Free | Allowed | Terms page itself returns 406 | **406** for both UAs **[live]** | **Excluded** |
| World of Books | Free | `/search` disallowed. Product handles are not ISBNs | n/a | **429** **[live]** | **Excluded** |
| Bookshop.org (US/UK), Wordery, Hive, Biblio, Powell's, Alibris, HPB, Waterstones, Blackwell's, Foyles, Speedy Hen, HarperCollins US | Free | mixed | n/a | **403 / Cloudflare "Just a moment"** for both UAs **[live]** | **Excluded** (bot walls signal that automated access isn't wanted) |
| HarperCollins UK (Shopify) | Free | `/products/` allowed, `/search` disallowed | n/a | `products.json` 200, but there is **no ISBN lookup** (search disallowed, handles are slugs). `/products/<isbn>` returns 429 | Not usable without a sitemap crawl |
| Penguin Random House US | Free | `/search/` disallowed | n/a | `/books/<isbn>` → 404. Needs the work ID | Not ISBN-addressable |
| Open Library | Free | `/api`, `/search` disallowed for crawlers, but the documented JSON endpoints (`/isbn/<isbn>.json`) are an official API | n/a | **No prices.** `/prices?isbn=` → **404** (removed). Edition pages have no BWB price widget (0 mentions) **[live]** | Identification only. Use the identifying UA: the browser UA got `/verify_human` |
| Apple iTunes Search API | Free, keyless | Official API | Docs: "approximately 20 calls per minute", and the promo content is "only to promote store content" **[docs]** https://performance-partners.apple.com/search-api. Grey area for valuation | `search?term=<title author>&country=us|gb|in&media=ebook&entity=ebook`. `price`, `currency`, `trackViewUrl`. Lookup by print ISBN returns 0 | US 8/10, GB 8/10, **IN 1/10**. **Ebook prices, fuzzy title match** (Pride & Prejudice matched a free public-domain ebook). At most a labelled `ebook_equivalent`, outside totals |
| Google Books API `country=US/GB` | Free with a key | Official | n/a | **[untested]** (keyless 429) | Same role as iTunes, but with an exact ISBN query, so better if a key is available |

**Why UK and not US:** UK has at least one compliant print source today (Penguin UK RRP, 3/10). US has none: PRH US isn't ISBN-addressable, and every retailer is either bot-walled or bans robots in its ToS.

The locale code path is identical. Only `config/locales/GB.yaml` lists different sources. `DEFAULT_COUNTRY=IN`, `SECOND_COUNTRY=GB`. Change `.env.example`, which currently says `SECOND_COUNTRY=US`.

### 2.4 Search / discovery APIs

| Service | Free tier | Card? | Finite? | Returns URLs? | Can we fetch and parse those URLs? | Status |
|---|---|---|---|---|---|---|
| **Gemini "Grounding with Google Search"** | `gemini-2.5-flash` / `-flash-lite`: free up to 500 RPD. **Gemini 3.x Flash: grounding "Not available" on the free tier** **[docs]** https://ai.google.dev/gemini-api/docs/pricing | No | Daily | `groundingChunks[].web.uri` are `vertexaisearch.cloud.google.com/grounding-api-redirect/…` redirects, and the title is just the domain **[docs]** | **No, forbidden by the terms.** "it is a violation of these terms to use Grounding with Google Search to extract or collect one or more of these components for another purpose (for example, using programmatic or automated means to collect Links, using Links to build an index, or **using Links to identify destination pages for crawling or scraping**)". Also: "will not … cache, frame, syndicate, resell, analyze …". Results may be shown only with the Search Suggestions, to the prompting end user **[live, verbatim]** https://ai.google.dev/gemini-api/terms | **Do not use for pricing.** I could not test it live (no key in this project's environment, and I did not borrow keys from other projects) |
| Gemini URL context tool | Free on the free tier, 20 URLs/request **[docs]** | No | n/a | n/a | The model reads the page, but our code never sees the raw HTML, so the price would come from the model. That breaks "parsed in code" | **Do not use for prices** |
| Serper.dev | 2,500 queries, no card **[docs]** https://serper.dev | No | **One-time, expires after 6 months** | Yes (Google SERP) | Technically yes. But Serper resells scraped Google results, which Google's ToS forbids. That is the vendor's risk, yet evaluators may question it | **[untested]** (no account). Optional, off by default |
| Tavily | 1,000 credits/month, no card **[docs]** | No | Monthly reset | Yes (own index) | Yes | **[untested]**. Cleanest search API if discovery is ever needed |
| Exa | $10 credit, "no payment method required", resets monthly (per pricing page) **[docs]** | No | Monthly | Yes (own index) | Yes | **[untested]** |
| Brave Search API | No free plan in 2026. $5/month credit, **card required** ("To prevent abuse") **[docs]** | **Yes** | | | | **Excluded (card)** |
| Jina Reader `r.jina.ai` | Keyless 20 RPM. A free key gives 10M tokens once **[docs]** | No | Finite with a key | n/a (a fetch proxy) | **[live]**: Bookchor via Jina returned 796 chars of markdown with **no price and no JSON-LD**. `X-Return-Format: html` also dropped the JSON-LD | **Not useful**, and it adds a third party to the evidence chain. `s.jina.ai` (search) needs a key |
| DuckDuckGo HTML (`html.duckduckgo.com/html/?q=`) | Free | n/a | n/a | **[live]**: 10 results, targets in `uddg=` params (e.g. the Bookswagon canonical URL) | `duckduckgo.com` robots disallows `/html` and `/lite`. `html.duckduckgo.com` robots says `Allow: /` only "so their noindex tags… are respected". No official web-search API exists. The AUP is JS-rendered and **unverified** | **Excluded** from the automated path |
| Google Custom Search JSON API | 100/day | | | | | **"Closed to new customers"**, sunset 2027-01-01 **[docs]** |

**Conclusion:** books need no search API, because ISBN → URL is deterministic on Bookchor, Bookswagon and Penguin UK. Items use a small, versioned config of category listing URLs per country, which is source configuration and not a price list (see 5).

### 2.5 Non-book items, India

| Source | Free | Robots | ToS | Price location | URL pattern | Status |
|---|---|---|---|---|---|---|
| **IKEA India** | Free | Category pages `/in/en/cat/<slug>-<id>/` and product pages `/in/en/p/…` allowed. **Search (`*/search/?q=*`), `*?filter=*`, `*?sorting=*` and `*priceFilter*` are disallowed**. Don't use the internal search API either **[live]** | T&C page renders client-side, **unverified** | Listing card: `<div class="plp-mastercard" data-product-number data-price="1990" data-currency="INR" data-product-name="BAGGEBO">`. The card's aria-label gives the type and dimensions ("Shelving unit, 60x25x116 cm"). Product page: JSON-LD `Product` (category, width/height/depth, offers) | `bookcases-10382`, `table-lamps-10732`, `wall-decor-10757` (pictures, posters, frames). No coffee makers in India | **[live]** 24 cards/page, about 200 ms |
| **Urban Ladder** | Free | `/collection/<slug>` and `/product/<slug>-<id>` allowed. `/products?q=` disallowed **[live]** | Terms page renders client-side, **unverified** | Collection page JSON-LD `ItemList` → `ListItem.url`. Product page JSON-LD `Offer.price` | `/collection/bookshelves`, `/collection/table-lamps`, `/collection/wall-art` | **[live]** 12 products/listing, product fetch about 400 ms |
| **Nilkamal Furniture** (Shopify) | Free | `/collections/<h>/products.json` allowed **[live]** | `/pages/terms-conditions`: **no clause** **[live]** | `products[].variants[0].price`, `title`, `handle` | `/collections/book-cases/products.json?limit=50` | **[live]** 15 rows. **Got 429 on a fast first call** with the browser UA. Worked with the identifying UA and ≥2 s spacing |
| **Vijay Sales** | Free | `/c/<cat>`, `/p/<id>/<slug>` and `/search?q=` allowed (only cart/checkout/profile disallowed) **[live]** | Terms page not found (`/terms-of-use` → `/faqs`), **unverified** | Listing JSON-LD `ItemList.url`. Product JSON-LD `AggregateOffer` `price` INR `NewCondition` | `/c/coffee-machines`, `/search?q=<terms>` (`/p/` hrefs) | **[live]** but slow: listing 1.2–2.4 s, product about 3.5 s |
| Pepperfry | | **robots.txt itself 403** (Akamai) | | | | **Excluded** |
| Wakefit | | **Cloudflare challenge** on robots.txt | | | | **Excluded** |
| Croma | | **Akamai 403** on robots.txt | | | | **Excluded** |
| Home Centre | Free | `/in/en/c/<cat>` allowed | n/a | Products render client-side. `__NEXT_DATA__` has no product prices | | Not usable without a browser |
| Reliance Digital | Free | `Disallow: /*?*` (every query string) | n/a | Client-rendered. No ItemList | | Not usable |
| Amazon.in / Flipkart via a discovery API | | | ToS forbid fetching the product page | | | **Excluded** (a discovery API doesn't make the fetch allowed) |

---

## 3. Test matrix

### 3.1 Books: India (all [live] 2026-10-26, INR)

| ISBN | Bookchor new | Bookchor used | Bookswagon new (MRP) | Pustakkosh | Chosen IN new | Chosen IN used |
|---|---|---|---|---|---|---|
| 9780099590088 Sapiens (Vintage) | **550** | none | 559 (799) | 630 New | Bookchor 550 | derived |
| 9780143127550 Everything I Never Told You | **1571** | **661 Good** | 947 (2367) | none | Bookchor 1571 (review: retailers differ 66%) | Bookchor 661 |
| 9780141439518 Pride and Prejudice | **286** | none | 242 (350) | none | Bookchor 286 | derived |
| 9780062316097 Sapiens (Harper hb) | **3274** | none | 2365 (3638) | none | Bookchor 3274 | derived |
| 9789390166268 Psychology of Money (Jaico) | **302** | none | 275 (399) | none | Bookchor 302 | derived |
| 9780143333623 Grandma's Bag of Stories | **242** | none | 193 (275) | none | Bookchor 242 | derived |
| 9780547928227 The Hobbit | **1658** | none | 1401 (1843) | none | Bookchor 1658 | derived |
| 9781847941831 Atomic Habits | **668** | none | 649 (999) | none | Bookchor 668 | derived |
| 9788173711466 Wings of Fire | none | none | **337** (495) | 251 Used | Bookswagon 337 | Pustakkosh 251 |
| 9788129115300 2 States | none | **62 Good** | **708** (1011) | none | Bookswagon 708 (review: MRP mismatch 399 vs 1011) | Bookchor 62 |
| *9780143028574 God of Small Things* | 374 | none | 329 (499) | none | | |
| *9788129104595 Five Point Someone* | none | 135 Good | 87 (140) **OutOfStock** | none | | |
| *9780143442295 Life's Amazing Secrets* | 256 | 135 Good | 178 (250) | none | | |

Example URLs:
- `https://www.bookchor.com/book/9780143127550/x`, canonical `https://www.bookchor.com/book/9780099590088/sapiensa-brief-history-of-humankind`
- `https://www.bookswagon.com/book/x/9780099590088`, canonical `https://www.bookswagon.com/book/sapiens-yuval-noah-harari/9780099590088`
- `https://pustakkosh.com/products/wings-of-fire-an-autobiography-of-abdul-kalam/`

### 3.2 Books: second country (live)

| ISBN | Penguin UK RRP (GBP) | iTunes GB ebook | iTunes US ebook | Proposed GB replacement |
|---|---|---|---|---|
| 9780099590088 | **12.99** `penguin.co.uk/books/419795/sapiens-by-harari-yuval-noah/9780099590088` | 5.99 | 14.99 | 12.99 publisher RRP |
| 9780143127550 | not found | 4.99 | 6.99 | converted from IN 1571 / 126.93 = **£12.38** (`converted`) |
| 9780141439518 | **9.99** | Free (wrong edition) | Free | 9.99 |
| 9780062316097 | not found | 5.99 | 14.99 | converted £25.79 |
| 9789390166268 | not found | 7.99 | 10.99 | converted £2.38 (an Indian edition, so the conversion understates the UK price; flag it) |
| 9780143333623 | not found | not found | not found | converted £1.91 |
| 9780547928227 | not found | 4.99 | 11.99 | converted £13.06 |
| 9781847941831 | **18.99** | 5.99 | 12.99 | 18.99 |
| 9788173711466 | not found | 13.99 | 17.99 | converted £2.65 |
| 9788129115300 | not found | not found | not found | converted £5.58 |
| **Hit rate** | **3/10** | 8/10 (ebook) | 8/10 (ebook) | 3 local + 7 converted |

The converted figures use the IN price ÷ ECB GBP→INR 126.93 (2026-10-23). They show the labelling; the code computes them, never a hand. Both sources were fetched in code. eBay GB would replace most of the converted values, and supply UK used values, once production access exists.

### 3.3 Non-book items: India (live, INR, nearest-rank bounds)

| Item | Sources (matched rows) | n | P25 → bound listing | P75 → bound listing | Status |
|---|---|---|---|---|---|
| wooden bookshelf 5 tier | Nilkamal 15, Urban Ladder 12, IKEA 7 | 34 | 3,490 Nilkamal Wales Book Case `nilkamalfurniture.com/products/nilkamal-wales-book-case-classic-walnut` | about 12,742 → 11,990 IKEA BRIMNES 60x190 `ikea.com/in/en/p/brimnes-bookcase-white-10351686/` | `range`. Narrow with material ("solid wood") and size class from vision. "5 tier" isn't a reliable title token |
| table lamp | IKEA 18, Urban Ladder 11 | 29 | about 1,699 → 1,799 Urban Ladder Sage Mosaic `urbanladder.com/product/sage-multi-color-mosaic-glass-iron-table-lamp-7543075` | about 5,245 → 4,990 IKEA HÖGVIND `ikea.com/in/en/p/hoegvind-table-lamp-nickel-plated-grey-glass-20568223/` | `range` |
| drip coffee maker | Vijay Sales: 4 regex matches, **2 true drip** (Wonderchef Onyx Brew 1,799; Regalia Pronto 3-in-1 filter 2,499). The other 2 matched on "drip tray" (espresso). Vijay Sales search for "drip coffee maker" returned 6 products, none of them drip | 2 | | | **`needs_appraisal`**: below the minimum of 5. Add `exclude: drip tray`. Widen to "coffee maker" only if the model is visibly a drip machine |
| framed wall art print | IKEA 7 (PJÄTTERYD "Picture", BILD "Poster"), Urban Ladder 7 | 14 | about 2,742 → 2,990 IKEA PJÄTTERYD 118x78 `ikea.com/in/en/p/pjaetteryd-picture-the-starry-night-june-1889-20578905/` | about 4,131 → 4,091 Urban Ladder metal wall art `urbanladder.com/product/creative-teal-metal-wall-art-wall-decor-8374439` | `range`, **only after the user says it is a print**. Default is `needs_appraisal` |

Filters that mattered: `must` regex on the product type, plus `exclude` for "picture ledge", bare "Frame," and "drip tray". The type comes from IKEA's aria-label (e.g. "Picture, 56x56 cm" vs "Frame, 13x18 cm"), so bare frames don't pass as art. The first run, without `exclude`, priced empty ₹59 frames as wall art.

---

## 4. Recommended free stack (final)

```
identify  -> ISBN-13 (Open Library /isbn/<isbn>.json with identifying UA; Google Books if a key exists)
IN new    -> Bookchor JSON-LD NewCondition Offer
             | Bookswagon JSON-LD Offer (repaired)        [config flag: tos_risk=harvesting_clause]
             | Pustakkosh Store API (sku match, New)
             | converted from GB (Penguin UK RRP x ECB)  [converted=true]
             | none -> excluded_from_totals
IN used   -> Bookchor JSON-LD UsedCondition Offer (grade label from tab, e.g. "Good")
             | Pustakkosh Store API (sku *_Used)
             | derived = IN new x factor[condition_assumed] (condition_factors.v1.yaml)  [source_type=derived_from_new]
             | none -> excluded
GB new    -> Penguin UK book page rrp (exact ISBN)
             | eBay Browse EBAY_GB new FIXED_PRICE median   [only if production access granted]
             | converted from IN new x ECB                  [converted=true]
             | (ebook_equivalent from Google Books/iTunes recorded as extra evidence, never in totals)
GB used   -> eBay Browse EBAY_GB used median               [if access]
             | derived from GB new                           [label; factor flagged "calibrated on IN market"]
             | none -> excluded
items IN  -> config/pricing/item_sources.IN.yaml: category -> [listing URLs]
             -> parse (IKEA data-price | ItemList->product JSON-LD | Shopify products.json)
             -> must/exclude filters -> n>=5 ? nearest-rank P25..P75 : needs_appraisal
FX        -> Frankfurter v2 providers=ECB ; fallback ECB SDMX
```

**Measured hit rates (10-book core):**
- IN new: 10/10 with Bookswagon enabled, or 8/10 Bookchor-only (+1 Pustakkosh would not fill the gaps; Wings of Fire and 2 States would be excluded or converted).
- IN used observed: 3/10, derived: 7/10.
- GB new local: 3/10, converted: 7/10.
- Items: 3/4 priced as ranges, 1/4 to review.

**If the evaluators object to Bookswagon's ToS:** set `bookswagon.enabled: false`. IN new drops to 8/10 (plus whatever Pustakkosh adds). Report that number honestly instead of quietly keeping the source.

---

## 5. Price evidence design

One evidence bundle per quote. The adjuster can open the raw bytes, see exactly which node was parsed, and re-run the parser offline.

```
runs/<sweep_id>/evidence/<quote_id>/
  request.json        {url, final_url, method, headers_sent (UA), robots_url, robots_sha256, robots_rule_matched, fetched_at_utc}
  response.meta.json  {status, headers (date, etag, last-modified, content-type, cf-ray/x-cache if any), bytes, sha256, elapsed_ms}
  response.body.gz    raw bytes exactly as received (HTML or JSON)
  extract.json        {
                        "extractor": "bookchor.jsonld.v1",
                        "path": "$[?(@['@type']=='Product')].offers[?(@.itemCondition=='https://schema.org/UsedCondition')]",
                        "node": { ...the exact JSON-LD Offer object as parsed... },
                        "raw_snippet": "<script type=\"application/ld+json\">…</script>" (first 4 KB of the block),
                        "repairs": ["unquoted_url_value@image"]   (Bookswagon only),
                        "isbn_on_page": "9780143127550", "isbn_check": "match",
                        "canonical_url": "https://www.bookchor.com/book/9780143127550/everything-i-never-told-you",
                        "amount": 661, "currency": "INR", "condition": "used:Good", "mrp": 1747
                      }
  robots.txt          the robots.txt body used for the allow decision (deduplicated by sha256 across quotes)
```

Price record in `claim_packet.json`. It keeps the contract fields and adds new ones:

```json
"used_value": {
  "amount": 661, "currency": "INR",
  "source": "Bookchor", "url": "https://www.bookchor.com/book/9780143127550/everything-i-never-told-you",
  "retrieved_at": "2026-10-26T10:14:03Z", "condition_assumed": "Good (seller grade)",
  "source_type": "marketplace_used_listing",
  "evidence_ref": "evidence/q_0007/", "extractor": "bookchor.jsonld.v1", "response_sha256": "…"
}
```

- **Derived values:** `source_type: derived_from_new`, `url` = the new-price URL, `method: "condition_factor:v1:Good"`, `base_evidence_ref`.
- **Converted values:** `converted: true`, `fx: {provider: "ECB via Frankfurter v2", rate, rate_date, url, retrieved_at, evidence_ref}`, plus `original: {amount, currency, evidence_ref}`.
- **Item ranges:** `low_evidence_ref` and `high_evidence_ref` (the nearest-rank listings), plus `rows_ref` (every matched row with its URL and price). For IKEA, also fetch the two boundary **product pages** and confirm the JSON-LD price matches the listing `data-price`, so the cited URL itself shows the number.

**HTML report:** each price links to the live URL and to a local "evidence" view that renders `extract.json` and the highlighted raw snippet.

**Replay test:** `reprice --offline` re-runs the extractors over `response.body.gz` and must reproduce every amount byte-for-byte. This proves no model touched the numbers.

---

## 6. Rate limits and latency (about 70 books)

Measured per-request latency [live]:

| Host | Typical | Notes / limit we impose |
|---|---|---|
| bookchor.com | 230–560 ms | 1 req/s |
| bookswagon.com | 500–1,600 ms | 1 req/s |
| pustakkosh.com Store API | 370–500 ms | 1 req/s |
| penguin.co.uk | 90 ms (hit) – 1.4 s (not-found redirect) | 1 req/s |
| ikea.com listing | 190–400 ms | 1 req/s |
| urbanladder.com | listing 160–650 ms, product about 400 ms | 1 req/s |
| vijaysales.com | listing 1.2–2.4 s, product about 3.5 s | 1 req/s. The slowest host |
| nilkamalfurniture.com (Shopify) | about 200 ms | **≥2 s spacing**: 429 seen on a cold burst |
| Frankfurter | 340–540 ms | 1 call per sweep (cache by date) |
| iTunes (optional) | about 400 ms | **about 20 calls/min (Apple docs)** → 3.1 s spacing |
| eBay Browse (untested) | n/a | 5,000 calls/day default |

Measured throughput: **13 books × 3 IN sources in 8.4 s** with 4 workers (no per-host throttle).

Budget for 70 books with polite per-host limits (hosts in parallel, 1 req/s each):

| Stage | Requests | Wall time |
|---|---|---|
| Bookchor, all books | 70 | about 70 s |
| Bookswagon, only books Bookchor missed (about 20–30%) | about 20 | about 20 s (parallel with the above) |
| Pustakkosh, books still missing new or used | about 50 | about 50 s (parallel) |
| GB: Penguin UK for the 10 second-country books | 10 | about 10 s |
| Items: about 10 items × (1–3 listings + ≤12 product pages) | about 120 across 4 hosts | about 60 s (Vijay Sales dominates) |
| FX | 1 | <1 s |
| **Total** | about 270 | **about 70–90 s** |

**Start during the sweep:** queue each book as soon as identification emits an ISBN. A 3-minute sweep covers 180 s of the 1 req/s Bookchor budget, so by the time the sweep ends almost all books are priced. The post-sweep tail is the last few ISBNs plus the items, under 60 s, well inside the 5-minute limit.

Cache hits younger than `price_ttl_days` (e.g. 7) are reused with their **original** `retrieved_at`.

Robots handling:
- Fetch each host's robots.txt once per sweep and cache it with its sha256.
- Evaluate it with a parser that supports Google-style `*` and `$` wildcards, such as `protego`. The stdlib `urllib.robotparser` ignores wildcards inside paths, which matters for rules like IKEA `*?filter=*` and Bookchor `*&page*`.
- Honour `Crawl-delay` where present (Notion Press 5 s, Sapna 20 s even though Sapna is excluded).

---

## 7. Staying honest when a book isn't found

Decisions are in code, recorded on each line, and anything unusual goes to the review queue.

1. **Not-found detection must be explicit, never "no number parsed, so try the next regex".**
   - Bookchor: a missing ISBN sends the canonical to `/search/?query=…` and there is no `Product` node.
   - Bookswagon: a missing ISBN gives the generic homepage title, no canonical and no Offer.
   - Penguin UK: it redirects to `/search-results`.
   - Treat each of these as `not_found` and store its evidence too (proof of a negative).
2. **The ISBN must match on the page:** Bookchor `Product.isbn`, Bookswagon `ISBN-13:` spec row, Pustakkosh `sku` prefix, and Penguin UK's final URL ends with the ISBN. A different ISBN on the page means `equivalent_edition`, which is accepted only when the work matches. Record `equivalent_isbn` and send the line to review.
3. **Availability:** `OutOfStock` offers (Bookswagon Five Point Someone) are allowed only as a last resort, with `availability: OutOfStock` recorded and a review reason.
4. **Cross-source sanity:**
   - If two IN retailers both have the ISBN and the selling prices differ by >25%, or the MRPs differ by >10%, keep the primary's price and add a review reason (`retailer_disagreement` / `edition_mismatch_suspected`).
   - Today this fires for EINTY (66%), Harper Sapiens (38%), Grandma's Bag (25%) and 2 States (MRP).
5. **No local price:**
   - Convert from the other configured market at the ECB rate on or before `retrieved_at`.
   - Set `converted: true` and keep both the original record and the FX record.
   - For an Indian edition priced in GB, add the review reason "converted from a low-price market; likely understates local replacement".
6. **No price anywhere:** amounts stay empty, `status` stays `identified`, add a review reason `no_price_source`, and count it in `excluded_from_totals`. Never fall back to an ebook price, or to a model, for totals.
7. **Used:** observed → derived (labelled, with the versioned factor) → empty. Report the observed and derived used sub-totals separately in the report, so `books_used_value` is never mistaken for pure market data.
8. **Rare/appraisal gate first:** if the gate fires, there are no prices at all (see 03 §5).

---

## 8. Code-level notes

- **HTTP**
  - Use one `httpx.Client` per host, with `follow_redirects=True` and `timeout=30`.
  - **Identifying UA:** `LibClaimAgent/0.1 (+mailto:<contact>)` returned 200 on every recommended host today: Bookchor, Bookswagon, Pustakkosh, Penguin UK, IKEA, Urban Ladder, Nilkamal, Vijay Sales, Frankfurter, iTunes and Open Library.
  - A browser UA gained nothing anywhere, and Open Library served `/verify_human` to the browser UA. **Don't spoof a browser.**
  - Send `Accept-Language: en-IN,en;q=0.9` for India.
- **JSON-LD extraction**
  - Regex `<script[^>]*type=["']application/ld\+json["'][^>]*>(.*?)</script>` (DOTALL), then `json.loads`.
  - On failure, apply two repairs and record them in `extract.json.repairs`:
    1. `re.sub(r'(:\s*)(https?://[^"\s,}]*")', r'\1"\2', s)` fixes an unquoted URL value (Bookswagon `"image":https://…"`).
    2. `re.sub(r",\s*([}\]])", r"\1", s)` removes trailing commas.
  - Walk all nested dicts. Match `@type` (str or list) against `Offer` / `AggregateOffer`. Read `price`, `lowPrice`/`highPrice`, `priceCurrency`, `itemCondition`, `availability` and `url`.
- **Bookchor**
  - `GET https://www.bookchor.com/book/{isbn13}/x`.
  - Product: `@type: Product`, `isbn`.
  - New price: Offer with `itemCondition` ending in `NewCondition`. Used price: Offer with `itemCondition` ending in `UsedCondition`.
  - MRP is arg 4 of `setPrice('<price>','<disc%>','<saving>','<mrp>',…)`.
  - Grade labels: inside `condition_chart_book`, `<p>\s*([A-Za-z ]+?)\s*<span>\s*₹\s*([\d,.]+)`. Radio `value` 1 = New, 3 = Good; other grades were not observed today.
  - Cite `<link rel="canonical">`.
- **Bookswagon**
  - `GET https://www.bookswagon.com/book/x/{isbn13}`.
  - Price: the JSON-LD Offer (after repair), or `id="ctl00_phBody_ProductDetail_lblourPrice">₹559`.
  - MRP: `id="ctl00_phBody_ProductDetail_lblListPrice">M.R.P. :<del>₹799`.
  - Binding: `id="ctl00_phBody_ProductDetail_lblBinding">(Paperback)`.
  - ISBN check: `<span class="font-weight-bold">ISBN-13: </span>{isbn}`.
  - Bonus: the spec list has `Height: 198 mm` and `Spine Width: 40 mm`. That is useful as an *independent* sanity check of the spine measurement in the ground-truth sheet, but never feed it into the measurement output.
- **Pustakkosh**
  - `GET https://pustakkosh.com/wp-json/wc/store/v1/products?search={isbn13}`.
  - Keep items where `sku.strip().lower().startswith(isbn)`.
  - Amount: `int(prices.price) / 10**prices.currency_minor_unit`.
  - Condition: the `sku` suffix `_Used`, or `short_description` "Book Condition: X".
  - Cite `permalink`.
- **Penguin UK**
  - `GET https://www.penguin.co.uk/books/x/x/{isbn13}`. It's a hit when `final_url` matches `/books/\d+/.+/{isbn}`.
  - Price: in `<script id="__NEXT_DATA__">`, `"rrp":"£(\d+\.\d\d)"`. Type: RRP (publisher list price). Condition: new.
- **IKEA India**
  - `GET https://www.ikea.com/in/en/cat/{slug-id}/` (page 2 is `?page=2`, which robots allows; `?filter=`, `?sorting=` and `priceFilter` are not).
  - Card regex: `<div class="plp-mastercard[^"]*"[^>]*data-product-number="(\d+)"[^>]*data-price="([\d.]+)"[^>]*data-currency="([A-Z]+)"[^>]*data-product-name="([^"]*)"…aria-label="Compare - Select for product comparison, ([^"]*)"…href="(https://www\.ikea\.com/in/en/p/[^"]+)"`.
  - The product page has a JSON-LD `Product`, with `category`, `width`/`height`/`depth` and offers.
- **Urban Ladder**
  - Listing: `/collection/{slug}` JSON-LD `ItemList.itemListElement[].url`. Skip non-`/product/` entries; the first is a category link.
  - Product: `/product/{slug}-{id}` JSON-LD Offer `price`.
- **Vijay Sales**
  - Listing: `/c/{slug}` JSON-LD ItemList `url`.
  - Search: `/search?q=` hrefs `/p/\d+/[^"]+`.
  - Product JSON-LD: `AggregateOffer.price` (string, INR), `itemCondition: NewCondition`.
- **Nilkamal**
  - `GET /collections/{handle}/products.json?limit=50`.
  - Read `products[].title`, `products[].handle` and `variants[0].price` (string, rupees). Cite `/products/{handle}`. Space requests ≥2 s.
- **Frankfurter v2**
  - `GET https://api.frankfurter.dev/v2/rates?base=GBP&quotes=INR&providers=ECB[&date=YYYY-MM-DD]` returns `[{"date":"2026-10-23","base":"GBP","quote":"INR","rate":126.93}]`.
  - The response is a **list**. A weekend `date` returns the prior business day.
  - Fallback: `https://data-api.ecb.europa.eu/service/data/EXR/D.INR+GBP+USD.EUR.SP00.A?lastNObservations=1&format=csvdata`, which gives EUR-based rates (GBP/EUR 0.83728 on 2026-10-23). Cross-rate them.
- **iTunes (optional, ebook only)**
  - `GET https://itunes.apple.com/search?term={title}+{author}&country=gb&media=ebook&entity=ebook&limit=5`.
  - Read `results[].trackName`, `artistName`, `price`, `currency` and `trackViewUrl`. Space calls at 3.1 s.
- **Config, not catalogue:** `config/pricing/sources.<CC>.yaml` holds the per-host URL templates, extractor name, `enabled`, `min_interval_s` and `tos_note`. `config/pricing/item_sources.IN.yaml` holds the category → listing URLs and the `must`/`exclude` regexes. Neither file contains a price. Unit-test the extractors against the saved raw bodies.

---

## 9. Not tested live, and why

- **Gemini Search grounding:** no Gemini key in this project's environment, and I did not borrow keys from other projects. It is also ruled out by its terms (verbatim quote above), so testing it would not change the decision.
- **Google Books with a key:** keyless calls hit the exhausted shared quota (429). A key from the user's existing Google Cloud project, with Books API enabled (free, no card), should work. Measure the IN/GB ebook coverage before relying on it, and use it only for identification and as `ebook_equivalent`.
- **eBay Browse (US/GB):** needs a developer account (free, about 1 business day). Production Buy APIs are officially "intended for eBay partners only" through an EPN application (≤10 business days, no guarantee). The licence's 6-hour freshness rule is unverified verbatim.
- **Serper, Tavily, Exa:** need signup (all free, no card per their docs). Not needed for the recommended path.
- **Terms pages that render client-side or couldn't be located** (IKEA India, Urban Ladder, Vijay Sales, Penguin UK, DuckDuckGo AUP): no anti-bot clause found, but also **not positively verified**. Read them in a browser before submission and record the result in the README.
- **Kitabay:** 429 from this IP on the first request.
- **Bookchor used-grade vocabulary:** only "Good" and "New" were seen today. Other grades probably exist.
- **Condition-factor calibration set:** recipe in 2.2, not run.
