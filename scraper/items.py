# scraper/items.py
import scrapy

class RestaurantItem(scrapy.Item):
    # Define the fields for your item here like:
    name = scrapy.Field()
    location = scrapy.Field()
    operating_hours = scrapy.Field()
    contact_info = scrapy.Field()
    website = scrapy.Field()          # URL scraped from
    special_features = scrapy.Field() # List of strings (e.g., ["Vegetarian Friendly", "Outdoor Seating"])
    menu = scrapy.Field()             # List of dictionaries for menu items
    
    # Example structure for menu item dictionary:
    # {
    #   "category": "Appetizers",
    #   "item_name": "Spring Rolls",
    #   "description": "Crispy rolls filled with vegetables.",
    #   "price": "₹250", # Store as string, clean later
    #   "dietary_flags": ["Vegetarian"], # List of strings
    #   "spice_level": "Mild" # Optional string
    # }
    source_spider = scrapy.Field()    # Keep track of which spider scraped the item