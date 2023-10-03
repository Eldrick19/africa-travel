import requests
from bs4 import BeautifulSoup
import os
from get_nationality_destination import get_nationality_destination
import json

def get_visa_info(query):
    # Get nationality and destination from user query
    # query = "What are my visa requirements for travelling to Ivory Coast as a Canadian? Would it be easier travelling to Senegal?"
    # nationality_destination = get_nationality_destination(query)

    # For testing purposes
    test_query = '{"nationality": "CA", "destination": "ghana"}'
    result_dict = json.loads(test_query)


    # Make sure the "data" folder exists
    if not os.path.exists("data"):
        os.makedirs("data")

    # URL to scrape
    nationality = result_dict['nationality']
    destination = result_dict['destination']
    url_options = [
        f"https://www.handyvisas.com/{destination}-visa/?from={nationality}",
        f"https://www.handyvisas.com/global/{destination}-visa/?from={nationality}"
    ]

    url = None
    for option in url_options:
        response = requests.get(option)
        if response.status_code == 200:
            url = option
            web_page = response.content
            break

    if url is None:
        raise ValueError("Could not find a valid URL for the given nationality and destination")
    print(url)
    # Scrape all text from the URL
    soup = BeautifulSoup(web_page, 'html.parser')
    text = soup.get_text()

    # Write the text to a file
    with open("data/visa_requirements.txt", "w") as f:
        for line in text.split("\n"):
            f.write(line + "\n")

    return text

get_visa_info("What are my visa requirements for travelling to Ghana as a Canadian?")