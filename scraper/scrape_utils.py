import json, requests, pdfplumber
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse
from urllib.robotparser import RobotFileParser
from playwright.sync_api import sync_playwright, TimeoutError as PlaywrightTimeout

HEADERS = {"User-Agent": "NuggetAI-Bot/1.0"}

# per-domain CSS selectors
SITE_SELECTORS = {
    'indianaccent.com': {
        'name': '.desktop-nav-logo',
        'location': '.footer .address',
        'menu_item': '.menu-section .item',
        'item_name': '.item-name',
        'item_desc': '.item-desc',
        'item_price': '.item-price',
        'features': '.features li',
        'hours': '.footer .hours',
        'contact': '.footer .contact',
    },
    'saravanabhavan.com': {
        'name': 'h1.logo',
        'location': '.contact-info .address',
        'menu_item': '.menu-card',
        'item_name': '.card-title',
        'item_desc': '.card-text',
        'item_price': '.price',
        'features': None,
        'hours': '.timing',
        'contact': '.contact-no',
    },
    'bikanervala.com': {
        'name': '.brand-logo',
        'location': '.location',
        'menu_item': '.product-item',
        'item_name': '.title',
        'item_desc': '.desc',
        'item_price': '.amount',
        'features': None,
        'hours': '.store-hours',
        'contact': '.phone',
    },
    'getyellowchilli.com': {
        'name': '.navbar-brand',
        'location': '.restaurant-location',
        'menu_item': '.menu-item',
        'item_name': '.name',
        'item_desc': '.details',
        'item_price': '.cost',
        'features': None,
        'hours': '.open-hours',
        'contact': '.phone-number',
    },
    'sattvik.in': {
        'name': '.site-logo',
        'location': '.address-block',
        'menu_item': '.menu-card',
        'item_name': 'h4',
        'item_desc': 'p',
        'item_price': '.price',
        'features': None,
        'hours': '.opening-hours',
        'contact': '.tel',
    },
}

def can_fetch(url):
    rp = RobotFileParser()
    rp.set_url(urljoin(url, "/robots.txt"))
    rp.read()
    return rp.can_fetch(HEADERS["User-Agent"], url)

def render_page(url, dom_only=True, timeout=8000):
    """Return rendered HTML, but only wait for DOMContentLoaded by default."""
    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page()
        wait = "domcontentloaded" if dom_only else "networkidle"
        page.goto(url, timeout=timeout, wait_until=wait)
        html = page.content()
        browser.close()
    return html

def extract_pdf_menu(soup, base_url):
    # find first PDF link
    a = soup.select_one('a[href$=".pdf"]')
    if not a:
        return []
    pdf_url = urljoin(base_url, a["href"])
    resp = requests.get(pdf_url, headers=HEADERS, timeout=15)
    with open("/tmp/menu.pdf","wb") as f:
        f.write(resp.content)
    items = []
    with pdfplumber.open("/tmp/menu.pdf") as pdf:
        text = "\n".join(page.extract_text() or "" for page in pdf.pages)
    for line in text.split("\n"):
        parts = line.rsplit(" ",1)
        if len(parts)==2 and any(ch.isdigit() for ch in parts[1]):
            items.append({"name": parts[0].strip(), "price": parts[1].strip()})
    return items

def scrape_restaurant(url):
    if not can_fetch(url):
        raise RuntimeError(f"Disallowed by robots.txt: {url}")

    # 1) fetch raw HTML quickly
    try:
        html = render_page(url, dom_only=True, timeout=8000)
    except PlaywrightTimeout:
        html = render_page(url, dom_only=True, timeout=16000)  # one longer try
    soup = BeautifulSoup(html, "html.parser")

    # 2) build skeleton
    data = {"url": url, "name":"", "location":"", "hours":"", "contact":"", "menu":[], "features":[]}

    # 3) try JSON-LD
    for tag in soup.select('script[type="application/ld+json"]'):
        try:
            jd = json.loads(tag.string)
            if jd.get("@type","").lower() in ("restaurant","foodestablishment"):
                data.update({
                    "name": jd.get("name",""),
                    "location": jd.get("address",{}).get("streetAddress",""),
                    "hours": jd.get("openingHours",""),
                    "contact": jd.get("telephone","")
                })
                break
        except:
            pass

    # 4) try PDF menu
    pdf_items = extract_pdf_menu(soup, url)
    if pdf_items:
        data["menu"] = pdf_items

    # 5) CSS fallback only for missing pieces
    domain = urlparse(url).netloc.replace("www.","")
    sel = SITE_SELECTORS.get(domain,{})
    def st(ctx, s): 
        n = ctx.select_one(s) if s else None
        return n.get_text(strip=True) if n else ""

    if not data["name"]:
        data["name"] = st(soup, sel.get("name"))
    if not data["location"]:
        data["location"] = st(soup, sel.get("location"))
    if not data["hours"]:
        data["hours"] = st(soup, sel.get("hours"))
    if not data["contact"]:
        data["contact"] = st(soup, sel.get("contact"))

    # CSS menu fallback
    if not data["menu"] and sel.get("menu_item"):
        for blk in soup.select(sel["menu_item"]):
            data["menu"].append({
                "name":  st(blk, sel.get("item_name")),
                "desc":  st(blk, sel.get("item_desc")),
                "price": st(blk, sel.get("item_price"))
            })

    # features fallback
    if sel.get("features"):
        data["features"] = [li.get_text(strip=True) for li in soup.select(sel["features"])]

    return data