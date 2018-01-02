def get_remaining_page_list(first_page):
    page_list = first_page.find_all("span", {"class": "page_list"})[0].find_all("a")
    return ["http://www.imdb.com" + page.get('href') for page in page_list if page.get('href') is not None]


def get_images_from_gallery(gallery):
    images = gallery.find_all("img")
    image_paths = []
    for image in images:
        src = image.get("src")
        if "UY100" in src:
            final_at_sign = src.rfind("@")
            if final_at_sign != -1:
                image_path = src[:final_at_sign + 1] + "._V_.jpg"
                image_paths.append(image_path)
    return image_paths
