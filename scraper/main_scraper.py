import json
from bs4 import BeautifulSoup
import time
import logging
from datetime import datetime
from scraper.scrape_utils import get_html, extract_menu_item # Assuming scrape_utils.py is in the same directory

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# IMPORTANT: Check robots.txt for each site first!
RESTAURANT_URLS = [
        "https://www.indianaccent.com/",
        "https://saravanabhavan.com/",
        "https://www.bikanervalausa.com/",
        "https://getyellowchilli.com/",
        "https://www.sattvik.in/"
]

OUTPUT_FILE = "./data/scraped_restaurants.jsonl" # Store in data directory

def parse_restaurant_site(url, html_content):
    """
    Main parsing logic for a single restaurant site.
    This function needs SIGNIFICANT CUSTOMIZATION per website.
    """
    soup = BeautifulSoup(html_content, 'html.parser')
    restaurant_data = {
        "restaurant_name": "Unknown",
        "location": "Unknown",
        "url": url,
        "menu": [],
        "operating_hours": {}, # Or a string
        "contact_info": {},
        "last_scraped_timestamp": datetime.now().isoformat()
    }

    try:
        # --- TODO: Customize selectors for EACH site ---
        restaurant_data["restaurant_name"] = soup.find('h1', class_='restaurant-title').text.strip() # Example
        restaurant_data["location"] = soup.find('p', class_='address').text.strip() # Example

        # --- Menu Parsing (Most Complex Part) ---
        menu_items_elements = soup.find_all('div', class_='menu-item') # Example selector
        current_category = "Main Course" # Example: Find category headers if available

        for item_element in menu_items_elements:
            # You might need logic here to detect category changes
            # category_header = item_element.find_previous_sibling('h3', class_='category-name') # Example
            # if category_header:
            #    current_category = category_header.text.strip()

            item_details = extract_menu_item(item_element) # Use the helper
            if item_details:
                item_details["category"] = current_category # Assign category
                restaurant_data["menu"].append(item_details)

        # --- Operating Hours Parsing ---
        hours_section = soup.find('div', id='hours') # Example
        if hours_section:
            # TODO: Parse the hours text/structure into a dict or string
             restaurant_data["operating_hours"] = hours_section.text.strip() # Simple example

        # --- Contact Info Parsing ---
        contact_section = soup.find('div', class_='contact') # Example
        if contact_section:
            phone = contact_section.find('a', href=lambda href: href and href.startswith('tel:')) # Example
            if phone:
                restaurant_data["contact_info"]["phone"] = phone.text.strip()
            # TODO: Extract email if available

        logging.info(f"Successfully parsed basic info for {restaurant_data['restaurant_name']} from {url}")
        return restaurant_data

    except AttributeError as e:
        logging.error(f"Parsing failed for {url} - element not found: {e}")
        # Return partially filled data or None depending on requirements
        restaurant_data["restaurant_name"] = f"Parsing Failed for {url}"
        return restaurant_data # Return partial data with URL for context
    except Exception as e:
        logging.error(f"An unexpected error occurred during parsing for {url}: {e}")
        return None


def scrape_sites(urls):
    """Iterates through URLs, scrapes, parses, and saves data."""
    all_data = []
    for url in urls:
        logging.info(f"Attempting to scrape: {url}")
        # Respect robots.txt (manual check or library needed for automated check)
        # print(f"Reminder: Check robots.txt for {url} before proceeding.")

        html = get_html(url)
        if html:
            parsed_data = parse_restaurant_site(url, html)
            if parsed_data:
                all_data.append(parsed_data)
            else:
                logging.warning(f"No data parsed for {url}")
        else:
            logging.warning(f"Could not retrieve HTML for {url}")

        # Respectful delay between requests
        time.sleep(3) # Adjust delay as needed (2-5 seconds is polite)

    # Save data as JSON Lines (append mode)
    try:
        with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
            for entry in all_data:
                json.dump(entry, f)
                f.write('\n')
        logging.info(f"Scraped data saved to {OUTPUT_FILE}")
    except IOError as e:
        logging.error(f"Could not write data to {OUTPUT_FILE}: {e}")

if __name__ == "__main__":
    scrape_sites(RESTAURANT_URLS)