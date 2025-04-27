# --- Dependencies ---
import requests
from bs4 import BeautifulSoup
import json
import time
import os
import re
from urllib.parse import urljoin, urlparse
import google.generativeai as genai
from PIL import Image # For image handling with Gemini
# import PyPDF2 # Option 1 for PDFs (text-based)
import fitz # PyMuPDF - Often better for text and image extraction from PDFs

# --- Configuration ---

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")


# Configure the Gemini client
try:
    genai.configure(api_key=GEMINI_API_KEY)
    # Model for text generation and understanding
    text_model = genai.GenerativeModel('gemini-1.5-flash') # Or other suitable text model
    # Model for understanding images/PDFs
    vision_model = genai.GenerativeModel('gemini-pro-vision') # Or gemini-1.5-pro for multimodal
    print("Gemini Models configured.")
except Exception as e:
    print(f"Error configuring Gemini: {e}")
    print("Please ensure your API key is correct and you have internet access.")
    text_model = None
    vision_model = None

# Standard User-Agent to mimic a browser
HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
}

# Delay between requests (in seconds) to be polite
REQUEST_DELAY = 3

# Directory to save downloaded files (PDFs, images) temporarily
DOWNLOAD_DIR = "temp_downloads"
if not os.path.exists(DOWNLOAD_DIR):
    os.makedirs(DOWNLOAD_DIR)

# --- Helper Functions ---

def safe_request(url):
    """Sends a GET request with error handling and delay."""
    try:
        time.sleep(REQUEST_DELAY) # Wait before making the request
        response = requests.get(url, headers=HEADERS, timeout=20)
        response.raise_for_status() # Raise an exception for bad status codes (4xx or 5xx)
        return response
    except requests.exceptions.RequestException as e:
        print(f"Error fetching {url}: {e}")
        return None

def clean_text(text):
    """Removes extra whitespace and newlines."""
    if text:
        return ' '.join(text.split()).strip()
    return None

def get_absolute_url(base_url, relative_url):
    """Converts a relative URL to an absolute URL."""
    return urljoin(base_url, relative_url)

def download_file(url, filename):
    """Downloads a file (PDF/Image) from a URL."""
    filepath = os.path.join(DOWNLOAD_DIR, filename)
    response = safe_request(url)
    if response:
        try:
            with open(filepath, 'wb') as f:
                f.write(response.content)
            print(f"Downloaded: {filename}")
            return filepath
        except Exception as e:
            print(f"Error saving file {filename}: {e}")
    return None

# --- Gemini API Interaction Functions ---

def get_info_from_gemini(content, prompt, model_choice="text"):
    """
    Sends content (text, image path, or PDF path) and a prompt to Gemini.
    model_choice: "text" for text_model, "vision" for vision_model
    """
    if not text_model or not vision_model:
        print("Gemini models not configured. Skipping API call.")
        return None

    if GEMINI_API_KEY == "YOUR_GEMINI_API_KEY":
         print("WARNING: Gemini API Key not set. Skipping API call.")
         return None

    try:
        if model_choice == "vision":
            model = vision_model
            # Handle file paths for vision model
            if isinstance(content, str) and os.path.exists(content):
                 # Check file type for Gemini Pro Vision compatibility (adjust as needed)
                mime_type = None
                if content.lower().endswith('.pdf'):
                    # Gemini Pro Vision can handle certain PDF types directly
                    # Adjust mime type based on Gemini documentation if needed
                    # mime_type = "application/pdf" # May not always be required by the library
                    print(f"Processing PDF with Gemini Vision: {content}")
                    file_data = genai.upload_file(path=content) # Upload the file
                    response = model.generate_content([prompt, file_data])
                    # genai.delete_file(file_data.name) # Clean up uploaded file if needed

                elif content.lower().endswith(('.png', '.jpg', '.jpeg', '.webp')):
                    print(f"Processing Image with Gemini Vision: {content}")
                    img = Image.open(content)
                    # Gemini library usually takes PIL Image object directly
                    response = model.generate_content([prompt, img])
                else:
                    print(f"Unsupported file type for Gemini Vision: {content}")
                    return None
            else:
                 print(f"Invalid content for Gemini Vision: {content}")
                 return None

        else: # Default to text model
            model = text_model
            if not isinstance(content, str):
                print(f"Invalid content type for Text Model: {type(content)}")
                return None
            print(f"Processing text with Gemini Text Model...")
            response = model.generate_content([prompt, content]) # Send prompt + text content

        # Attempt to resolve the response safely
        # Sometimes response might need response.resolve() for streaming or complex results
        # For simpler cases, response.text should work. Add error handling.
        try:
            # Check if parts exist and have text
            if response.parts:
                 return response.text
            else:
                 print("Gemini response has no parts.")
                 # Check for safety ratings or other issues
                 print(f"Prompt Feedback: {response.prompt_feedback}")
                 # You might need to inspect response._result for more details in error cases
                 return None
        except Exception as e:
            print(f"Error accessing Gemini response text: {e}")
            print(f"Full response object: {response}") # Log the full response for debugging
            return None

    except Exception as e:
        print(f"Error calling Gemini API: {e}")
        # Specific error handling (e.g., for API key issues, rate limits, content blocking)
        # if "API_KEY_INVALID" in str(e):
        #     print("Invalid Gemini API Key.")
        # Consider more specific exception handling based on the google.api_core.exceptions
        return None


def scrape_menu_from_image(image_path):
    """Uses Gemini Vision to extract menu data from an image file."""
    print(f"--- Attempting to extract menu from IMAGE: {image_path} ---")
    prompt = """
    Analyze the provided image, which contains a restaurant menu. Extract the following information in a structured JSON format:
    - A list of menu items.
    - For each item, include its 'name', 'description' (if available), and 'price' (if available).
    - Group items into categories (e.g., 'Appetizers', 'Main Courses', 'Desserts') if the menu structure suggests them.
    - Note any dietary information mentioned (e.g., 'vegetarian', 'vegan', 'gluten-free', 'spice level').

    Provide the output ONLY as a single JSON object. Do not include any introductory text or explanations before or after the JSON.
    Example Format:
    {
      "categories": [
        {
          "category_name": "Appetizers",
          "items": [
            {
              "name": "Veg Samosa",
              "description": "Crispy pastry filled with spiced potatoes and peas.",
              "price": "$5.99",
              "dietary_notes": ["vegetarian"]
            },
            // ... other items
          ]
        },
        // ... other categories
      ],
      "uncategorized_items": [
         {
           "name": "Special Dish",
           "description": "Chef's special for the day.",
           "price": "Market Price"
         }
      ]
    }
    If no structured data can be extracted, return an empty JSON object: {}
    """
    json_string = get_info_from_gemini(image_path, prompt, model_choice="vision")

    if json_string:
        try:
            # Clean potential markdown code fences
            json_string = json_string.strip().removeprefix("```json").removesuffix("```").strip()
            menu_data = json.loads(json_string)
            print("--- Successfully extracted menu from IMAGE using Gemini ---")
            return menu_data
        except json.JSONDecodeError as e:
            print(f"Error decoding JSON from Gemini (Image): {e}")
            print(f"Received string: {json_string}") # Log the problematic string
            return None
    return None

def scrape_menu_from_pdf(pdf_path):
    """Uses Gemini (Vision or Text) to extract menu data from a PDF file."""
    print(f"--- Attempting to extract menu from PDF: {pdf_path} ---")

    # === Strategy 1: Try Gemini Vision directly with the PDF ===
    # Gemini Pro Vision might handle some PDFs directly.
    prompt_vision = """
    Analyze the provided PDF file, which contains a restaurant menu. Extract the following information in a structured JSON format:
    - A list of menu items.
    - For each item, include its 'name', 'description' (if available), and 'price' (if available).
    - Group items into categories (e.g., 'Appetizers', 'Main Courses', 'Desserts') if the menu structure suggests them.
    - Note any dietary information mentioned (e.g., 'vegetarian', 'vegan', 'gluten-free', 'spice level').

    Provide the output ONLY as a single JSON object. Do not include any introductory text or explanations before or after the JSON.
    Example format is the same as the image prompt.
    If no structured data can be extracted, return an empty JSON object: {}
    """
    json_string_vision = get_info_from_gemini(pdf_path, prompt_vision, model_choice="vision")

    if json_string_vision:
        try:
            json_string_vision = json_string_vision.strip().removeprefix("```json").removesuffix("```").strip()
            menu_data = json.loads(json_string_vision)
            # Basic check if data was extracted
            if menu_data and (menu_data.get("categories") or menu_data.get("uncategorized_items")):
                 print("--- Successfully extracted menu from PDF using Gemini Vision ---")
                 return menu_data
            else:
                 print("Gemini Vision returned empty/invalid JSON for PDF, trying text extraction...")
        except json.JSONDecodeError as e:
            print(f"Error decoding JSON from Gemini Vision (PDF): {e}")
            print(f"Received string: {json_string_vision}")
            print("Falling back to text extraction method...")
        except Exception as e:
             print(f"An unexpected error occurred processing Gemini Vision response for PDF: {e}")
             print("Falling back to text extraction method...")


    # === Strategy 2: Extract text with PyMuPDF and send to Gemini Text model ===
    # This is a fallback if Vision fails or if the PDF is primarily text-based.
    print("--- Extracting text from PDF using PyMuPDF ---")
    extracted_text = ""
    try:
        doc = fitz.open(pdf_path)
        for page_num in range(len(doc)):
            page = doc.load_page(page_num)
            extracted_text += page.get_text("text") + "\n---\n" # Add separator between pages
        doc.close()
        extracted_text = clean_text(extracted_text)

        if not extracted_text or len(extracted_text) < 50: # Basic check if text extraction worked
             print("--- Failed to extract sufficient text from PDF with PyMuPDF ---")
             return None

    except Exception as e:
        print(f"Error extracting text from PDF {pdf_path} using PyMuPDF: {e}")
        return None # Cannot proceed if text extraction fails

    print(f"--- Sending extracted PDF text to Gemini Text Model ---")
    prompt_text = f"""
    Analyze the following text extracted from a restaurant menu PDF. Extract the menu information into a structured JSON format:
    - A list of menu items.
    - For each item, include 'name', 'description' (if available), and 'price' (if available).
    - Group items into categories (e.g., 'Appetizers', 'Main Courses', 'Desserts') if the structure suggests them.
    - Note dietary information (e.g., 'vegetarian', 'vegan', 'gluten-free', 'spice level').

    Provide ONLY the JSON object as output. Do not include explanations.
    Example format is the same as the image prompt.
    If no structured data can be extracted, return an empty JSON object: {{}}

    Extracted Text:
    --- START TEXT ---
    {extracted_text[:10000]}
    --- END TEXT ---
    """ # Limit text size if needed for the model context window

    json_string_text = get_info_from_gemini(extracted_text[:10000], prompt_text, model_choice="text") # Use extracted text

    if json_string_text:
        try:
            json_string_text = json_string_text.strip().removeprefix("```json").removesuffix("```").strip()
            menu_data = json.loads(json_string_text)
            print("--- Successfully extracted menu from PDF TEXT using Gemini ---")
            return menu_data
        except json.JSONDecodeError as e:
            print(f"Error decoding JSON from Gemini (PDF Text): {e}")
            print(f"Received string: {json_string_text}")
            return None
        except Exception as e:
             print(f"An unexpected error occurred processing Gemini Text response for PDF text: {e}")
             return None

    print("--- Failed to extract menu from PDF using all methods ---")
    return None


# --- Web Scraping Functions ---

def find_menu_data(soup, base_url):
    """
    Tries to find menu links (HTML, PDF) or menu images on the page.
    Returns structured menu data if found, otherwise None.
    """
    menu_data = None

    # --- Strategy 1: Look for explicit Menu links (HTML page or PDF) ---
    menu_keywords = ['menu', 'dining', 'food', 'cuisine', 'carte', 'dishes']
    possible_menu_links = []

    for keyword in menu_keywords:
        # Find links containing menu-related keywords (case-insensitive)
        links = soup.find_all('a', href=True, string=re.compile(keyword, re.IGNORECASE))
        for link in links:
            href = link['href']
            # Ignore javascript links or empty hashes
            if href and not href.startswith(('javascript:', '#')):
                 absolute_url = get_absolute_url(base_url, href)
                 if absolute_url not in possible_menu_links: # Avoid duplicates
                    possible_menu_links.append(absolute_url)

    print(f"Found potential menu links: {possible_menu_links}")

    for menu_url in possible_menu_links:
         parsed_link = urlparse(menu_url)
         filename = os.path.basename(parsed_link.path)

         # --- Strategy 1a: Check if it's a PDF link ---
         if menu_url.lower().endswith(".pdf"):
             print(f"Found PDF menu link: {menu_url}")
             pdf_path = download_file(menu_url, filename)
             if pdf_path:
                 menu_data = scrape_menu_from_pdf(pdf_path)
                 # os.remove(pdf_path) # Optional: Clean up downloaded file
                 if menu_data: return menu_data # Return first successful PDF menu
             continue # Try next link if PDF download/scrape failed

         # --- Strategy 1b: Check if it's likely an HTML menu page ---
         # Avoid scraping the same page or external links if possible
         if urlparse(menu_url).netloc == urlparse(base_url).netloc and menu_url != base_url:
             print(f"Found potential HTML menu page: {menu_url}")
             menu_page_response = safe_request(menu_url)
             if menu_page_response:
                 menu_soup = BeautifulSoup(menu_page_response.content, 'html.parser')
                 # --- Attempt to scrape HTML menu directly (NEEDS CUSTOMIZATION) ---
                 # This part is highly site-specific. You need to inspect the HTML.
                 # Example: Look for divs with class 'menu-item', get name/price/desc
                 items = menu_soup.find_all('div', class_=re.compile(r'menu-?item', re.I))
                 if items:
                     print(f"Attempting basic HTML menu parse on {menu_url}...")
                     menu_data = {"categories": [], "uncategorized_items": []}
                     current_category = None
                     # Look for category headers (e.g., h2, h3 tags before items)
                     # This is a very simple example, real menus are more complex
                     all_elements = menu_soup.find_all(['h2', 'h3', 'div'], class_=re.compile(r'(category|section|menu-?item)', re.I))

                     category_items = []
                     for element in all_elements:
                        is_item = 'menu-item' in ' '.join(element.get('class', [])) or element.name == 'div' and element.find(class_=re.compile(r'price|cost', re.I))
                        is_category_header = element.name in ['h2','h3'] and not is_item

                        if is_category_header:
                            # Save previous category's items
                            if current_category and category_items:
                                menu_data["categories"].append({
                                    "category_name": current_category,
                                    "items": category_items
                                })
                            current_category = clean_text(element.get_text())
                            category_items = [] # Reset items for new category
                            print(f"Found Category: {current_category}")

                        elif is_item:
                             item_details = {}
                             # --- CUSTOMIZE SELECTORS HERE ---
                             name_tag = element.find(['h4', 'h5', 'span'], class_=re.compile(r'name|title', re.I))
                             price_tag = element.find(['span', 'div'], class_=re.compile(r'price|cost|amount', re.I))
                             desc_tag = element.find(['p', 'div'], class_=re.compile(r'desc|details', re.I))

                             item_details['name'] = clean_text(name_tag.get_text()) if name_tag else "Unknown Item"
                             item_details['price'] = clean_text(price_tag.get_text()) if price_tag else "N/A"
                             item_details['description'] = clean_text(desc_tag.get_text()) if desc_tag else ""
                             item_details['dietary_notes'] = [] # Placeholder

                             if item_details['name'] != "Unknown Item":
                                 if current_category:
                                     category_items.append(item_details)
                                 else:
                                     menu_data["uncategorized_items"].append(item_details)

                     # Add the last category's items
                     if current_category and category_items:
                         menu_data["categories"].append({
                             "category_name": current_category,
                             "items": category_items
                         })

                     if menu_data.get("categories") or menu_data.get("uncategorized_items"):
                        print(f"--- Successfully parsed basic HTML menu from {menu_url} ---")
                        return menu_data # Return menu data if items found
                     else:
                        print(f"--- Found menu items container, but failed to parse details on {menu_url} ---")

                 # --- Fallback: Check for images on the menu page using Gemini ---
                 print(f"No direct HTML items found or parsed on {menu_url}. Checking for menu images...")
                 images = menu_soup.find_all('img', src=True)
                 for img in images:
                    img_src = img.get('src')
                    if img_src:
                         # Basic check if image might be a menu (keywords in src or alt text)
                         alt_text = img.get('alt', '').lower()
                         if any(k in img_src.lower() for k in menu_keywords) or \
                            any(k in alt_text for k in menu_keywords):
                             img_url = get_absolute_url(menu_url, img_src)
                             filename = os.path.basename(urlparse(img_url).path)
                             if not filename: filename = "menu_image.jpg" # Default name
                             print(f"Found potential menu image: {img_url}")
                             img_path = download_file(img_url, filename)
                             if img_path:
                                 menu_data = scrape_menu_from_image(img_path)
                                 # os.remove(img_path) # Optional cleanup
                                 if menu_data: return menu_data # Return first successful image menu
             else:
                 print(f"Failed to fetch potential HTML menu page: {menu_url}")


    # --- Strategy 2: Look for images directly on the main page using Gemini ---
    print("No menu found via links. Checking for images on the main page...")
    images = soup.find_all('img', src=True)
    for img in images:
        img_src = img.get('src')
        if img_src:
            # Basic check if image might be a menu
            alt_text = img.get('alt', '').lower()
            if any(k in img_src.lower() for k in menu_keywords) or \
               any(k in alt_text for k in menu_keywords) or \
               'menu' in urlparse(img_src).path.lower(): # Check path too
                img_url = get_absolute_url(base_url, img_src)
                filename = os.path.basename(urlparse(img_url).path)
                if not filename: filename = "menu_image_main.jpg"
                print(f"Found potential menu image on main page: {img_url}")
                img_path = download_file(img_url, filename)
                if img_path:
                    menu_data = scrape_menu_from_image(img_path)
                    # os.remove(img_path) # Optional cleanup
                    if menu_data: return menu_data # Return first successful image menu

    print("--- No menu data found using any method ---")
    return None


def scrape_restaurant_website(url):
    """Scrapes a single restaurant website."""
    print(f"\n--- Scraping: {url} ---")
    response = safe_request(url)
    if not response:
        return None

    soup = BeautifulSoup(response.content, 'html.parser')
    data = {"url": url}

    # --- Extract Information (NEEDS CUSTOMIZATION PER SITE) ---
    # These selectors are examples and VERY LIKELY need changing.
    # Use your browser's developer tools (Inspect Element) to find the right ones.

    # 1. Restaurant Name
    name_selectors = ['h1', '.restaurant-name', '#site-title', 'title'] # Common tags/classes
    data['name'] = None
    for selector in name_selectors:
        element = soup.select_one(selector)
        if element:
             name_text = clean_text(element.get_text())
             # Often titles have extra info, try to clean it
             if selector == 'title' and name_text:
                 name_text = re.split(r'[|-]', name_text)[0].strip() # Split by | or - and take first part
             if name_text and len(name_text) > 2: # Basic check
                 data['name'] = name_text
                 break # Stop once found
    if not data['name']: data['name'] = urlparse(url).netloc # Fallback to domain name
    print(f"Name: {data['name']}")


    # 2. Location / Address
    # Look for keywords like 'address', 'location', street names, city names, postal codes
    address_keywords = ['address', 'location', 'directions', 'visit', 'contact']
    location_selectors = ['.address', '#location', '.contact-info', 'footer'] # Common areas
    data['location'] = None
    possible_locations = []
    for selector in location_selectors:
        elements = soup.select(selector)
        for element in elements:
            text = clean_text(element.get_text(separator=' ', strip=True))
            # Use regex to find plausible addresses (very basic example)
            # Looks for a pattern like number, street name, city (Lucknow), optional PIN
            # This regex is basic and might need significant improvement
            matches = re.findall(r'\d+.*?Lucknow.*?\d{6}?', text, re.IGNORECASE | re.DOTALL)
            if matches:
                for match in matches:
                    cleaned_match = clean_text(match)
                    if cleaned_match and len(cleaned_match) > 15 and cleaned_match not in possible_locations: # Basic sanity check
                        possible_locations.append(cleaned_match)
            # Also check for text containing keywords if regex fails
            elif any(keyword in text.lower() for keyword in address_keywords) and len(text) > 15 and "Lucknow" in text:
                 if text not in possible_locations:
                      possible_locations.append(text)

    # Try to select the most likely address (e.g., the longest one found in footer)
    if possible_locations:
         data['location'] = max(possible_locations, key=len) # Simple heuristic
    else:
         # Fallback: Search for "Lucknow" in the whole text
         body_text = clean_text(soup.body.get_text(separator=' ', strip=True))
         if "Lucknow" in body_text:
              # Could try more sophisticated regex here
              data['location'] = "Lucknow (Specific address not found, check website)"
         else:
              data['location'] = "Location not found"
    print(f"Location: {data['location']}")


    # 3. Operating Hours
    # Look for keywords like 'hours', 'opening times', days of the week
    hours_keywords = ['hour', 'opening', 'timing', 'open', 'mon', 'tue', 'wed', 'thu', 'fri', 'sat', 'sun']
    hours_selectors = ['.hours', '.opening-times', '.business-hours', 'footer', '.contact-section']
    data['operating_hours'] = None
    possible_hours = []
    for selector in hours_selectors:
        elements = soup.select(selector)
        for element in elements:
            text = clean_text(element.get_text(separator='\n', strip=True)) # Use newline separator for structure
            # Check if text contains keywords and potentially time format (e.g., AM/PM, colon)
            if any(keyword in text.lower() for keyword in hours_keywords) and re.search(r'\d{1,2}[:.]?\d{0,2}\s*(?:AM|PM|noon|midnight)?', text, re.I):
                 if text not in possible_hours:
                      possible_hours.append(text)

    if possible_hours:
        # Often multiple snippets found, try to combine or take the most detailed one
        data['operating_hours'] = "\n---\n".join(possible_hours) # Join snippets found
    else:
        data['operating_hours'] = "Operating hours not found"
    print(f"Operating Hours: {data['operating_hours']}")

    # 4. Contact Information (Phone, Email)
    data['contact_info'] = {}
    # Regex for typical Indian phone numbers
    phone_regex = re.compile(r'(\+?91[\s-]?)?\(?(\d{3,5})\)?[\s.-]?(\d{3}[\s.-]?\d{4})')
    # Basic email regex
    email_regex = re.compile(r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}')

    phones = set()
    emails = set()

    # Search in links (mailto:, tel:)
    for link in soup.find_all('a', href=True):
        href = link['href']
        if href.startswith('tel:'):
            phone = href.replace('tel:', '').strip()
            if phone not in phones: phones.add(clean_text(phone))
        elif href.startswith('mailto:'):
            email = href.replace('mailto:', '').strip()
            if email not in emails: emails.add(clean_text(email))

    # Search in page text
    body_text = soup.body.get_text()
    found_phones = phone_regex.findall(body_text)
    for match in found_phones:
        # Reconstruct the number (this might need refinement)
        full_number = "".join(filter(None, match)) # Join matched groups
        full_number = re.sub(r'\D', '', full_number) # Remove non-digits
        if len(full_number) >= 10: # Basic validation
            phones.add(full_number)

    found_emails = email_regex.findall(body_text)
    for email in found_emails:
        emails.add(email)

    data['contact_info']['phone'] = list(phones) if phones else "Phone not found"
    data['contact_info']['email'] = list(emails) if emails else "Email not found"
    print(f"Contact Info: {data['contact_info']}")


    # 5. Menu Items (using helper function)
    data['menu'] = find_menu_data(soup, url)
    if not data['menu']:
        data['menu'] = {"error": "Menu could not be found or scraped."}
        print("Menu: Not found or failed to scrape.")
    else:
         print("Menu: Found and processed (details in JSON).")


    # 6. Special Features (Vegetarian, Spice Levels, Allergens) - Often requires menu analysis
    # This part ideally integrates with the menu scraping.
    # If Gemini extracts dietary notes, they will be in data['menu'].
    # We can also do a basic text search on the page as a fallback.
    data['special_features'] = []
    page_text_lower = soup.body.get_text().lower()
    if 'vegetarian' in page_text_lower: data['special_features'].append('Vegetarian options likely available')
    if 'vegan' in page_text_lower: data['special_features'].append('Vegan options possibly available')
    if 'gluten-free' or 'gluten free' in page_text_lower: data['special_features'].append('Gluten-free options possibly available')
    if 'allergen' in page_text_lower: data['special_features'].append('Allergen information might be available')
    if 'spice level' or 'spicy' in page_text_lower: data['special_features'].append('Spice level indication possibly available')
    if not data['special_features']: data['special_features'].append('No specific features explicitly found on main page (check menu details).')

    # You can add more sophisticated checks within the menu data itself later
    print(f"Special Features: {data['special_features']}")

    return data

# --- Main Execution ---

if __name__ == "__main__":
    # ==============================================================
    # IMPORTANT: Populate this list with the ACTUAL website URLs
    # You need to find these via searching/manual checks.
    # Example list (replace with your 50 URLs):
    restaurant_urls = [
    # --- Likely Official Restaurant/Chain Sites ---
    "https://www.barbequenation.com/restaurants/lucknow", # Main page for Lucknow outlets
    "https://www.barbequenation.com/restaurants/lucknow/gomati-nagar", # Specific outlet example
    "http://www.royalcafe.in", # Appears to be the official site
    "https://www.motimahaldelux.com/post/moti-mahal-lucknow", # Specific Lucknow page
    "https://www.tundaykababi.com/", # Official site for the famous Tunday Kababi
    "http://wahidbiryani.com/", # Seems to be official site for Wahid Biryani
    "http://www.piratesofgrill.com/", # Chain website, likely has Lucknow info
    "https://paramparasweets.com/", # Mentioned in Justdial results for Radhelal's
    "https://www.sankalponline.com/saffron-restaurant.aspx", # Saffron Restaurant - mentioned in Justdial
    "http://www.bikanervala.com/", # Major chain, likely has Lucknow outlet info
    "https://www.behrouzbiryani.com/", # Chain website
    "https://www.lapinozpizza.in/", # Chain website
    "https://www.dominos.co.in/", # Chain website
    "https://www.pizzahut.co.in/", # Chain website
    "https://kfc.co.in/", # Chain website
    "https://www.mcdonaldsindia.com/", # Chain website
    "https://www.burgerking.in/", # Chain website
    "https://www.subway.com/en-IN", # Chain website
    "https://www.wowmomo.in/", # Chain website
    "https://sagar-ratna.com/", # Chain website, check for Lucknow
    "https://www.madanrestaurants.com/", # Mentioned for Madhurima, check Lucknow details
    "https://www.chilisgrillandbar.in/", # Check if Lucknow outlet exists
    "https://www.mainlandchinarestaurant.com/", # Chain website, check for Lucknow

    # --- Hotel Websites (Likely Contain Dining Info - Navigate Within Site) ---
    "https://www.tajhotels.com/en-in/taj/taj-mahal-lucknow/", # For Oudhyana, Sahib Cafe
    "https://www.saracahotels.com/lucknow/", # For Azrak, 1936 Ristorante
    "https://www.saracahotels.com/lucknow/dining/azrak.html", # Direct link for Azrak
    "https://www.radissonhotels.com/en-us/hotels/radisson-lucknow-city-center", # For Caprice
    "https://www.hyatt.com/hyatt-regency/en-US/lkorl-hyatt-regency-lucknow", # For Rocca, Lukjin
    "https://www.marriott.com/en-us/hotels/lkobr-renaissance-lucknow-hotel/overview/", # For L-14, Sky Bar, Zaffran
    "https://clarkshotels.com/clarks-avadh-lucknow/", # For Falak Numa (Verify domain)
    "https://www.dayalparadise.com/", # For Jannat, Aadab (Verify domain)
    "https://www.piccadily.co.in/lucknow-hotel/", # For Marine Room, Punjab
    "https://www.piccadily.co.in/lucknow-hotel/dining/marine-room-the-coffee-shop.html", # Direct link for Marine Room
    "https://www.golden-tulip.com/en/hotels/golden-tulip-lucknow", # Check Dining
    "https://www.itchotels.com/in/en/fortuneparkbbalucknow", # Check Dining
    "https://www.lemontreehotels.com/lemontree-hotel/lucknow/lucknow.aspx", # Check Dining
    "https://www.lineagehotels.com/", # Likely for Urban Terrace (Verify domain)
    "https://www.lebua.com/", # Check if separate Lucknow info exists
    "https://all.accor.com/hotel/B9K3/index.en.shtml", # Novotel Lucknow Gomti Nagar - Check Dining

    # --- Other Potential Standalone/Smaller Chain Websites (Verify Vigorously) ---
    "https://www.saddaadda.com/", # Listed on Justdial
    "http://www.dabbuveg.com/", # Listed on Justdial
    "https://vintagemachinelko.com/", # Potential site based on search
    "https://www.theurbandhaba.co.in/", # Potential site based on search
    "https://thecherrytreecafe.in/", # Cafe, potential site
    "http://www.jjbakers.com/", # Bakery/Cafe chain
    "https://mrbrownbakery.com/", # Bakery/Cafe chain - Danbro
    "https://kareemslucknow.com/", # Potential site for Kareem's
    "https://www.baati-chokha.com/", # Chain, check Lucknow details
    "https://nuts-99-restaurant.business.site/", # Google Business Site
    "https://pinkpantherbar.com/" # Listed on Justdial
]

# Note: '//' at the start of a line makes it a comment in Python.
# The above format uses actual string literals for the URLs as intended for the list variable.
# Remember to verify each URL and expect to customize the scraper code.
    # ==============================================================

    if not restaurant_urls:
        print("Error: The 'restaurant_urls' list is empty.")
        print("Please edit the script and add the target website URLs.")
        exit()

    if GEMINI_API_KEY == "YOUR_GEMINI_API_KEY":
        print("*"*60)
        print("WARNING: GEMINI API KEY IS NOT SET.")
        print("The script will attempt to scrape basic info but will skip PDF/Image analysis.")
        print("Set your API key in the GEMINI_API_KEY variable to enable menu extraction from PDFs/Images.")
        print("*"*60)


    all_restaurant_data = []
    failed_urls = []

    for url in restaurant_urls:
        try:
            scraped_data = scrape_restaurant_website(url)
            if scraped_data:
                all_restaurant_data.append(scraped_data)
            else:
                failed_urls.append(url)
        except Exception as e:
            print(f"!!!!!!!!!! UNEXPECTED ERROR scraping {url}: {e} !!!!!!!!!!")
            failed_urls.append(url)
            # Optional: Add more robust error logging here
            import traceback
            traceback.print_exc()


    # --- Save the Data ---
    output_filename = "lucknow_restaurants_data.json"
    try:
        with open(output_filename, 'w', encoding='utf-8') as f:
            json.dump(all_restaurant_data, f, indent=4, ensure_ascii=False)
        print(f"\nSuccessfully scraped {len(all_restaurant_data)} restaurants.")
        print(f"Data saved to {output_filename}")
    except Exception as e:
        print(f"Error saving data to JSON: {e}")

    if failed_urls:
        print("\nFailed to scrape the following URLs:")
        for failed_url in failed_urls:
            print(f"- {failed_url}")

    # Optional: Clean up download directory
    # Consider keeping it for debugging if needed
    # import shutil
    # if os.path.exists(DOWNLOAD_DIR):
    #     try:
    #         shutil.rmtree(DOWNLOAD_DIR)
    #         print(f"Cleaned up temporary download directory: {DOWNLOAD_DIR}")
    #     except Exception as e:
    #         print(f"Error removing download directory {DOWNLOAD_DIR}: {e}")
