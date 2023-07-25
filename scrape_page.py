import requests
from bs4 import BeautifulSoup
import os

# Make sure the "data" folder exists
if not os.path.exists("data"):
    os.makedirs("data")

# URL to scrape
url = "https://snedai.com/e-visa"

# Send a GET request to the URL
response = requests.get(url)

# Parse the HTML content of the page using BeautifulSoup
soup = BeautifulSoup(response.content, "html.parser")

# Find all the links on the page
links = soup.find_all("a")

# Write the text of the URL to the file
with open("data/context.txt", "w") as f:
    for link in links:
        f.write(link.text.strip() + "\n")