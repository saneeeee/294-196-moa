import requests
from bs4 import BeautifulSoup
import json
import os
from tqdm import tqdm
from urllib.parse import urljoin
import fire
from pathlib import Path

def scrape_dmv_handbook_page(url):
    response = requests.get(url)
    soup = BeautifulSoup(response.content, 'html.parser')
    
    handbook_content = soup.find('div', class_='handbook-page-content container container--wide mt-60')
    
    if not handbook_content:
        return "Handbook content not found"
    
    formatted_content = []
    
    for element in handbook_content.find_all(['h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'p', 'ul', 'ol']):
        if element.name.startswith('h'):
            level = int(element.name[1])
            text = element.get_text().strip()
            formatted_content.append(f"{'#' * level} {text}")
        elif element.name == 'p':
            text = element.get_text().strip()
            formatted_content.append(text)
        elif element.name in ['ul', 'ol']:
            for item in element.find_all('li'):
                text = item.get_text().strip()
                formatted_content.append(f"• {text}")
    
    return '\n\n'.join(formatted_content)

def scrape_dmv_handbook(url):
    response = requests.get(url)
    soup = BeautifulSoup(response.content, 'html.parser')
    
    handbook_container = soup.find('div', class_='handbook-container')
    chapters = handbook_container.find_all('ul', class_='toc-menu')
    
    dataset = []
    chapter_number = 0
    
    for chapter in tqdm(chapters, desc="Scraping chapters"):
        chapter_items = chapter.find_all('li', class_='menu-item', recursive=False)
        
        for chapter_item in tqdm(chapter_items, desc="Processing chapter items", leave=False):
            chapter_name = chapter_item.find('a').text.strip()
            chapter_data = {"type": "chapter", "name": chapter_name, "sections": []}
            
            sections = chapter_item.find('ul', class_='sub-menu')
            if sections:
                section_items = sections.find_all('li', class_='menu-item')
                for i, section in enumerate(tqdm(section_items, desc=f"Processing sections for {chapter_name}", leave=False), start=1):
                    section_name = section.find('a').text.strip()
                    section_url = urljoin(url, section.find('a')['href'])
                    text_location = f"ch{chapter_number:02d}/sec{i:02d}.txt"
                    
                    section_data = {
                        "type": "section",
                        "name": section_name,
                        "url": section_url,
                        "text_location": text_location
                    }
                    chapter_data["sections"].append(section_data)
            else:
                # This is for the Introduction, which doesn't have subsections
                section_data = {
                    "type": "section",
                    "name": chapter_name,
                    "url": urljoin(url, chapter_item.find('a')['href']),
                    "text_location": f"ch{chapter_number:02d}/sec00.txt"
                }
                chapter_data["sections"].append(section_data)
            
            dataset.append(chapter_data)
            chapter_number += 1
    
    return dataset

def save_dataset(dataset, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    dataset_description = {"dataset": dataset}
    with open(os.path.join(output_dir, "dataset_description.json"), "w") as f:
        json.dump(dataset_description, f, indent=2)
    
    for chapter in tqdm(dataset, desc="Saving chapters"):
        chapter_num = dataset.index(chapter)
        chapter_dir = os.path.join(output_dir, f"ch{chapter_num:02d}")
        if not os.path.exists(chapter_dir):
            os.makedirs(chapter_dir)
        
        for section in tqdm(chapter["sections"], desc=f"Saving sections for {chapter['name']}", leave=False):
            content = scrape_dmv_handbook_page(section["url"])
            with open(os.path.join(output_dir, section["text_location"]), "w", encoding="utf-8") as f:
                f.write(content)

def main(output_dir, url="https://www.dmv.ca.gov/portal/handbook/vehicle-industry-registration-procedures-manual-2/introduction/"):
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    print("Scraping DMV handbook...")
    dataset = scrape_dmv_handbook(url)
    
    print("\nSaving dataset and content...")
    save_dataset(dataset, output_path)
    
    print("\nDone!")
    
    print(f"Data saved to {output_path}")

if __name__ == "__main__":
    fire.Fire(main)
