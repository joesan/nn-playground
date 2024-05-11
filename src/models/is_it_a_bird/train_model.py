from duckduckgo_search import DDGS
from fastai.vision.utils import *
from fastcore.all import *
from time import sleep

searches = 'forest','bird'
path = Path('../data/raw/bird_or_not')

def search_images(term, max_images=30):
    print(f"Searching for '{term}'")
    with DDGS() as ddgs:
        search_results = ddgs.images(keywords=term)
        # Use itertools.islice to limit the number of results without converting to a list
        image_urls = [result.get("image") for result in itertools.islice(search_results, max_images)]
        return L(image_urls)


def fetch_images(searchTerms = searches):
    for o in searchTerms:
        dest = (path/o)
        dest.mkdir(exist_ok=True, parents=True)
        download_images(dest, urls=search_images(f'{o} photo'))
        sleep(10)  # Pause between searches to avoid over-loading server
        download_images(dest, urls=search_images(f'{o} sun photo'))
        sleep(10)
        download_images(dest, urls=search_images(f'{o} shade photo'))
        sleep(10)
        resize_images(path/o, max_size=400, dest=path/o)