import json
import re
from pathlib import Path
from playwright.sync_api import sync_playwright  # Sync API :contentReference[oaicite:3]{index=3}

INPUT_FILE = "lucknow_top50_restaurants.json"
OUTPUT_FILE = "lucknow_top50_with_menus.json"
MENU_XHR_PATTERN = re.compile(r".*/webroutes/[\w/]*getMenu.*")  # matches Zomato menu endpoint

def load_restaurants(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def save_restaurants(path, data):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

def extract_menu_from_response(response):
    """Parse the JSON body of a menu XHR response into structured menu items."""
    try:
        payload = response.json()
    except Exception:
        return []
    # The payload structure varies; adjust keys as needed.
    items = []
    for section in payload.get("menuSections", []):
        category = section.get("sectionName", "")
        for entry in section.get("menuItems", []):
            items.append({
                "item_name": entry.get("name", "").strip(),
                "description": entry.get("description", "").strip(),
                "price": entry.get("price", "").strip(),
                "category": category,
                "dietary_flags": entry.get("attributes", []),
                "spice_level": entry.get("spiceLevel", "")
            })
    return items

def main():
    restaurants = load_restaurants(INPUT_FILE)
    with sync_playwright() as pw:
        browser = pw.chromium.launch(headless=True)
        page = browser.new_page()
        for rest in restaurants:
            rest["menu"] = []
            for menu_link in rest.get("menu_links", []):
                try:
                    # Wait for the specific XHR response that matches the menu pattern
                    with page.expect_response(lambda resp: MENU_XHR_PATTERN.match(resp.url)) as resp_info:  # :contentReference[oaicite:4]{index=4}
                        page.goto(menu_link, timeout=60000)  # allow time for all XHRs to fire
                    response = resp_info.value
                    menu_items = extract_menu_from_response(response)  # parse JSON :contentReference[oaicite:5]{index=5}
                    rest["menu"].extend(menu_items)
                except Exception as e:
                    print(f"⚠️ Failed to load menu for {rest['restaurant_name']}: {e}")
        browser.close()
    save_restaurants(OUTPUT_FILE, restaurants)
    print(f"✅ Menus extracted and saved to {OUTPUT_FILE}")

if __name__ == "__main__":
    main()
