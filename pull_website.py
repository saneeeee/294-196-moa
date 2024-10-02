from urllib.request import urlopen
from bs4 import BeautifulSoup
import re
from tqdm import tqdm
import requests

def clean_text(text):
    return re.sub(r'\s+', ' ', text).strip()

def scrape_main_content(url): # not working many duplications
    html = urlopen(url).read()
    soup = BeautifulSoup(html, features="html.parser")

    for script in soup(["script", "style"]):
        script.extract()

    main_content = soup.find('main', id='main')
    if not main_content:
        main_content = soup.find('div', class_='page-content')
    if not main_content:
        return "Main content not found"

    formatted_content = []
    seen_content = set()

    
    for element in main_content.find_all(['h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'p', 'ul', 'ol', 'div']):
        if element.name.startswith('h'):
            level = int(element.name[1])
            text = clean_text(element.get_text())
            if text not in seen_content:
                formatted_content.append(f"{'#' * level} {text}")
                seen_content.add(text)
        elif element.name in ['p', 'div']:
            
            text = clean_text(element.get_text())
            if text and text not in seen_content:
                formatted_content.append(text)
                seen_content.add(text)
        elif element.name in ['ul', 'ol']:
            list_items = []
            for item in element.find_all('li'):
                text = clean_text(item.get_text())
                if text and text not in seen_content:
                    list_items.append(f"• {text}")
                    seen_content.add(text)
            if list_items:
                formatted_content.append("\n".join(list_items))

    return '\n\n'.join(formatted_content)

def scrape_plain(url):
    
    response = requests.get(url)
    soup = BeautifulSoup(response.content, 'html.parser')
    
    main_content = soup.find('main', id='main')
    
    if main_content:
        text_content = main_content.get_text(separator=' ', strip=True)
        return text_content
    else:
        return "Main content area not found"
    
if __name__ == "__main__":
    OUTPUT_DIR = "./dataset/"
    urls = [
        "https://www.dmv.ca.gov/portal/vehicle-registration/vehicle-registration-renewal/",
        "https://www.dmv.ca.gov/portal/vehicle-registration/vehicle-registration-renewal/frequently-asked-questions-renewing-your-registration/",
        "https://www.dmv.ca.gov/portal/online-change-of-address-coa-system/",
        "https://www.dmv.ca.gov/portal/dmv-virtual-office/title-transfers/",
        "https://www.dmv.ca.gov/portal/vehicle-registration/new-registration/",
        "https://www.dmv.ca.gov/portal/vehicle-registration/vehicle-registration-status/",
        "https://www.dmv.ca.gov/portal/vehicle-registration/insurance-requirements/suspended-vehicle-registration/",
        "https://www.dmv.ca.gov/portal/customer-service/request-vehicle-or-driver-records/online-vehicle-record-request/",
        "https://www.dmv.ca.gov/portal/vehicle-registration/replace-your-registration-card/",
        "https://www.dmv.ca.gov/portal/vehicle-registration/license-plates-decals-and-placards/",
        "https://www.dmv.ca.gov/portal/vehicle-registration/disabled-placards-and-plates/",
        "https://www.dmv.ca.gov/portal/vehicle-registration/updating-information-on-your-registration/",
        "https://www.dmv.ca.gov/portal/vehicle-registration/smog-inspections/",
        "https://www.dmv.ca.gov/portal/vehicle-registration/insurance-requirements/",
        "https://www.dmv.ca.gov/portal/vehicle-registration/registration-fees/",
        "https://www.dmv.ca.gov/portal/about-the-california-department-of-motor-vehicles/renewal-processing-times/",
        "https://www.dmv.ca.gov/portal/vehicle-registration/titles/title-transfers-and-changes/"
    ]
    
    # print(scrape_dmv_page(urls[1]))
    for i in tqdm(range(len(urls))):
        content = scrape_plain(urls[i])
        output_file = OUTPUT_DIR + f"scrape_{i:03d}.txt"
        with open(output_file, 'w', encoding='utf-8') as file:
            file.write(content)