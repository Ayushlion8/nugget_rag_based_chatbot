import time, json
from .scrape_utils import scrape_restaurant

if __name__ == "__main__":
    urls = [
      "https://www.indianaccent.com/newdelhi/dinner-menu",
      "https://saravanabhavan.com/menu",
      "http://www.bikanervala.com/menu",
      "https://getyellowchilli.com/menu",
      "https://www.sattvik.in/menu"
    ]

    results = []
    for u in urls:
        print("⏳ scraping", u)
        try:
            res = scrape_restaurant(u)
            print("   ✔︎ got", len(res["menu"]), "menu items")
            results.append(res)
        except Exception as e:
            print("   ❌ failed:", e)
        time.sleep(1)

    with open("./data/scraped_restaurants.json","w",encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print("✅ saved", len(results), "records")