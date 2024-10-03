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
    
def scrape_dmv_handbook_page(url):
    response = requests.get(url)
    soup = BeautifulSoup(response.content, 'html.parser')
    
    handbook_content = soup.find('div', class_='handbook-page-content container container--wide mt-60')
    
    if not handbook_content:
        return "Handbook content not found"
    
    def clean_text(text):
        return re.sub(r'\s+', ' ', text).strip()
    
    formatted_content = []
    
    for element in handbook_content.find_all(['h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'p', 'ul', 'ol']):
        if element.name.startswith('h'):
            level = int(element.name[1])
            text = clean_text(element.get_text())
            formatted_content.append(f"{'#' * level} {text}")
        elif element.name == 'p':
            text = clean_text(element.get_text())
            formatted_content.append(text)
        elif element.name in ['ul', 'ol']:
            for item in element.find_all('li'):
                text = clean_text(item.get_text())
                formatted_content.append(f"• {text}")
    
    return '\n\n'.join(formatted_content)

if __name__ == "__main__":
    OUTPUT_DIR = "./dataset/"
    urls = [
        # "https://www.dmv.ca.gov/portal/vehicle-registration/vehicle-registration-renewal/",
        # "https://www.dmv.ca.gov/portal/vehicle-registration/vehicle-registration-renewal/frequently-asked-questions-renewing-your-registration/",
        # "https://www.dmv.ca.gov/portal/online-change-of-address-coa-system/",
        # "https://www.dmv.ca.gov/portal/dmv-virtual-office/title-transfers/",
        # "https://www.dmv.ca.gov/portal/vehicle-registration/new-registration/",
        # "https://www.dmv.ca.gov/portal/vehicle-registration/vehicle-registration-status/",
        # "https://www.dmv.ca.gov/portal/vehicle-registration/insurance-requirements/suspended-vehicle-registration/",
        # "https://www.dmv.ca.gov/portal/customer-service/request-vehicle-or-driver-records/online-vehicle-record-request/",
        # "https://www.dmv.ca.gov/portal/vehicle-registration/replace-your-registration-card/",
        # "https://www.dmv.ca.gov/portal/vehicle-registration/license-plates-decals-and-placards/",
        # "https://www.dmv.ca.gov/portal/vehicle-registration/disabled-placards-and-plates/",
        # "https://www.dmv.ca.gov/portal/vehicle-registration/updating-information-on-your-registration/",
        # "https://www.dmv.ca.gov/portal/vehicle-registration/smog-inspections/",
        # "https://www.dmv.ca.gov/portal/vehicle-registration/insurance-requirements/",
        # "https://www.dmv.ca.gov/portal/vehicle-registration/registration-fees/",
        # "https://www.dmv.ca.gov/portal/about-the-california-department-of-motor-vehicles/renewal-processing-times/",
        # "https://www.dmv.ca.gov/portal/vehicle-registration/titles/title-transfers-and-changes/",
        "https://www.dmv.ca.gov/portal/handbook/vehicle-industry-registration-procedures-manual-2/general-registration-information/addresses-on-documents/",
        "https://www.dmv.ca.gov/portal/handbook/vehicle-industry-registration-procedures-manual-2/general-registration-information/adhesive-labels-on-documents/",
        "https://www.dmv.ca.gov/portal/handbook/vehicle-industry-registration-procedures-manual-2/general-registration-information/assignment-of-registration-expiration-date/",
        "https://www.dmv.ca.gov/portal/handbook/vehicle-industry-registration-procedures-manual-2/general-registration-information/authority-to-grant-or-refuse-application/",
        "https://www.dmv.ca.gov/portal/handbook/vehicle-industry-registration-procedures-manual-2/general-registration-information/bill-of-sale/",
        "https://www.dmv.ca.gov/portal/handbook/vehicle-industry-registration-procedures-manual-2/general-registration-information/branded-titles/",
        "https://www.dmv.ca.gov/portal/handbook/vehicle-industry-registration-procedures-manual-2/general-registration-information/confidentiality-of-home-address/",
        "https://www.dmv.ca.gov/portal/handbook/vehicle-industry-registration-procedures-manual-2/general-registration-information/co-owners/",
        "https://www.dmv.ca.gov/portal/handbook/vehicle-industry-registration-procedures-manual-2/general-registration-information/definitions-for-clearing-suspense-and-incomplete-applications-rdf/",
        "https://www.dmv.ca.gov/portal/handbook/vehicle-industry-registration-procedures-manual-2/general-registration-information/designating-legal-ownership/",
        "https://www.dmv.ca.gov/portal/handbook/vehicle-industry-registration-procedures-manual-2/general-registration-information/electronic-lien-and-title-elt-program/",
        "https://www.dmv.ca.gov/portal/handbook/vehicle-industry-registration-procedures-manual-2/general-registration-information/highlighters-on-documents/",
        "https://www.dmv.ca.gov/portal/handbook/vehicle-industry-registration-procedures-manual-2/general-registration-information/junk-or-salvage-vehicles-vin-inspections/",
        "https://www.dmv.ca.gov/portal/handbook/vehicle-industry-registration-procedures-manual-2/general-registration-information/leased-vehicles/",
        "https://www.dmv.ca.gov/portal/handbook/vehicle-industry-registration-procedures-manual-2/general-registration-information/legibility-of-writing-or-lettering/",
        "https://www.dmv.ca.gov/portal/handbook/vehicle-industry-registration-procedures-manual-2/general-registration-information/license-plates/",
        "https://www.dmv.ca.gov/portal/handbook/vehicle-industry-registration-procedures-manual-2/general-registration-information/lost-mail-applications/",
        "https://www.dmv.ca.gov/portal/handbook/vehicle-industry-registration-procedures-manual-2/general-registration-information/mail-applications/",
        "https://www.dmv.ca.gov/portal/handbook/vehicle-industry-registration-procedures-manual-2/general-registration-information/name-and-or-address-too-long/",
        "https://www.dmv.ca.gov/portal/handbook/vehicle-industry-registration-procedures-manual-2/general-registration-information/name-statement-requirement/",
        "https://www.dmv.ca.gov/portal/handbook/vehicle-industry-registration-procedures-manual-2/general-registration-information/notary-expiration-date-of-commission-on-documents/",
        "https://www.dmv.ca.gov/portal/handbook/vehicle-industry-registration-procedures-manual-2/general-registration-information/photocopy-fax-copy-of-documents/",
        "https://www.dmv.ca.gov/portal/handbook/vehicle-industry-registration-procedures-manual-2/general-registration-information/q-series-license-plate-numbers/",
        "https://www.dmv.ca.gov/portal/handbook/vehicle-industry-registration-procedures-manual-2/general-registration-information/registration-by-vehicle-identification-number-vin/",
        "https://www.dmv.ca.gov/portal/handbook/vehicle-industry-registration-procedures-manual-2/general-registration-information/rush-title-processing/",
        "https://www.dmv.ca.gov/portal/handbook/vehicle-industry-registration-procedures-manual-2/general-registration-information/signature-by-power-of-attorney-poa/",
        "https://www.dmv.ca.gov/portal/handbook/vehicle-industry-registration-procedures-manual-2/general-registration-information/signature-by-relative-of-military-owner/",
        "https://www.dmv.ca.gov/portal/handbook/vehicle-industry-registration-procedures-manual-2/general-registration-information/signatures-and-endorsements/",
        "https://www.dmv.ca.gov/portal/handbook/vehicle-industry-registration-procedures-manual-2/general-registration-information/statement-to-record-ownership-error-or-erasure/",
        "https://www.dmv.ca.gov/portal/handbook/vehicle-industry-registration-procedures-manual-2/general-registration-information/true-full-name/",
        "https://www.dmv.ca.gov/portal/handbook/vehicle-industry-registration-procedures-manual-2/general-registration-information/unclaimed-certificates-receipts-license-plates-and-stickers/",
        "https://www.dmv.ca.gov/portal/handbook/vehicle-industry-registration-procedures-manual-2/general-registration-information/vehicles-exempt-from-registration/",
        "https://www.dmv.ca.gov/portal/handbook/vehicle-industry-registration-procedures-manual-2/general-registration-information/vehicle-identification-number-vin-plate-assignments/",
        "https://www.dmv.ca.gov/portal/handbook/vehicle-industry-registration-procedures-manual-2/general-registration-information/vehicle-verifications/",
        "https://www.dmv.ca.gov/portal/handbook/vehicle-industry-registration-procedures-manual-2/general-registration-information/vin-check-digit-requirements/",
        "https://www.dmv.ca.gov/portal/handbook/vehicle-industry-registration-procedures-manual-2/general-registration-information/17-digit-vins-on-vehicles-fmvss-regulations-part-565/"

    ]
    
    # print(scrape_dmv_page(urls[1]))
    for i in tqdm(range(len(urls))):
        content = scrape_dmv_handbook_page(urls[i])
        output_file = OUTPUT_DIR + f"scrape_{i:03d}.txt"
        with open(output_file, 'w', encoding='utf-8') as file:
            file.write(content)