from bs4 import BeautifulSoup
import requests
import urllib
import gallery_scrape as gs
import os
from PIL import Image

IMAGE_HEIGHT = 256

def get_html(url):
    response = requests.get(url)
    return response.text


def get_images_for_actor_nm(nm):
    first_page_url = "http://www.imdb.com/name/nm{}/mediaindex?ref_=nm_mv_sm".format(nm)
    first_page_soup = BeautifulSoup(get_html(first_page_url), "lxml")

    h3_tags = first_page_soup.find_all("h3")

    for h3_tag in h3_tags:
        if h3_tag.get("itemprop") == "name":
            name = h3_tag.find_all("a")[0].text.replace(" ", "")
            break

    soups = [first_page_soup]
    remaining_pages = gs.get_remaining_page_list(first_page_soup)

    soups += [BeautifulSoup(get_html(page), "lxml") for page in remaining_pages]
    image_lists = [gs.get_images_from_gallery(soup) for soup in soups]
    return {"images": [y for x in image_lists for y in x], "name": name}


def save_actor_images_to_file(nm):
    actor_image_data = get_images_for_actor_nm(nm)
    name = actor_image_data["name"]
    images = actor_image_data["images"]

    for index, image_url in enumerate(images):
        save_path = os.path.normpath(os.path.join(os.path.abspath(__file__), "../.."))
        save_dir = os.path.join(save_path, "recognition/assets/actors/{}".format(name))
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        save_path = os.path.join(save_dir, "{}.jpg".format(index))
        urllib.request.urlretrieve(image_url, save_path)

        saved_image = Image.open(save_path)
        new_width = saved_image.width / (saved_image.height / IMAGE_HEIGHT)
        new_size = (new_width, IMAGE_HEIGHT)
        saved_image.thumbnail(new_size, Image.ANTIALIAS)
        saved_image.save("{}_comp.jpg".format(save_path.split(".jpg")[0]), "JPEG")

        os.remove(save_path)
