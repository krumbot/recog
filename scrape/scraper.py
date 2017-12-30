from bs4 import BeautifulSoup
import requests
import galleryscrape as gs


def get_html(url):
    response = requests.get(url)
    return response.text


def get_images_for_actor_nm(nm):
    first_page_url = "http://www.imdb.com/name/nm{}/mediaindex?ref_=nm_mv_sm".format(nm)
    first_page_soup = BeautifulSoup(get_html(first_page_url), "lxml")

    soups = [first_page_soup]
    remaining_pages = gs.get_remaining_page_list(first_page_soup)

    soups += [BeautifulSoup(get_html(page), "lxml") for page in remaining_pages]
    return [gs.get_images_from_gallery(soup) for soup in soups]


images = get_images_for_actor_nm(3485845)
print(images)
