import requests
import time
from requests.exceptions import RequestException
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
}

def get_html(url, retries=3, delay=5):
    """Fetches HTML content from a URL with basic error handling and retries."""
    for attempt in range(retries):
        try:
            response = requests.get(url, headers=HEADERS, timeout=15)
            response.raise_for_status() # Raise HTTPError for bad responses (4xx or 5xx)
            # Check if content type is HTML before returning
            if 'text/html' in response.headers.get('Content-Type', ''):
                return response.text
            else:
                logging.warning(f"Non-HTML content type received from {url}: {response.headers.get('Content-Type')}")
                return None
        except RequestException as e:
            logging.error(f"Attempt {attempt + 1} failed for {url}: {e}")
            if attempt < retries - 1:
                time.sleep(delay * (attempt + 1)) # Exponential backoff
            else:
                logging.error(f"All retries failed for {url}")
                return None
        except Exception as e:
            logging.error(f"An unexpected error occurred for {url}: {e}")
            return None
    return None

def extract_menu_item(item_element):
    """
    Placeholder function to extract details for a single menu item.
    You MUST customize this based on the website's structure.
    """
    try:
        name = item_element.find('h4', class_='menu-item-name').text.strip() # Example selector
        description = item_element.find('p', class_='menu-item-description').text.strip() # Example selector
        price = item_element.find('span', class_='menu-item-price').text.strip() # Example selector

        # Feature extraction - highly custom
        features = []
        if "vegetarian" in description.lower() or item_element.find('span', class_='veg-icon'):
            features.append("vegetarian")
        if "gluten-free" in description.lower() or "GF" in name:
            features.append("gluten-free")
        # Add logic for spice levels, etc.

        return {
            "item_name": name,
            "description": description,
            "price": price,
            "category": "Unknown", # You might get this from a parent element
            "features": features
        }
    except AttributeError as e:
        logging.warning(f"Could not parse menu item fully: {e}. Element: {item_element.prettify()[:200]}...")
        return None
    except Exception as e:
        logging.error(f"Unexpected error parsing menu item: {e}")
        return None

# --- Add more utility functions if needed ---