from duckduckgo_search import DDGS
from fastai.data.block import DataBlock, CategoryBlock
from fastai.data.transforms import get_image_files, RandomSplitter, parent_label
from fastai.metrics import error_rate, accuracy
from fastai.vision.augment import Resize
from fastai.vision.data import ImageBlock
from fastai.vision.learner import vision_learner
from fastai.vision.utils import *
from fastcore.all import *
from time import sleep
from torchvision.models import resnet34

import joblib
import os

from src.models.is_it_a_bird import env


def search_images(term, max_images=200):
    print(f"Searching for '{term}'")
    with DDGS() as ddgs:
        search_results = ddgs.images(keywords=term)
        # Use itertools.islice to limit the number of results without converting to a list
        image_urls = [result.get("image") for result in itertools.islice(search_results, max_images)]
        return L(image_urls)


def fetch_images(searchTerms=env.searches):
    for o in searchTerms:
        dest = (env.data_dir / o)
        dest.mkdir(exist_ok=True, parents=True)
        download_images(dest, urls=search_images(f'{o} photo'))
        sleep(10)  # Pause between searches to avoid over-loading server
        download_images(dest, urls=search_images(f'{o} sun photo'))
        sleep(10)
        download_images(dest, urls=search_images(f'{o} shade photo'))
        sleep(10)
        resize_images(env.data_dir / o, max_size=400, dest=env.data_dir / o)


def unlink_failed_images():
    failed = verify_images(get_image_files(env.data_dir))
    failed.map(Path.unlink)
    len(failed)


def data_block():
    return DataBlock(
        blocks=(ImageBlock, CategoryBlock),
        get_items=get_image_files,
        splitter=RandomSplitter(valid_pct=0.2, seed=42),
        get_y=parent_label,
        item_tfms=[Resize(192, method='squish')]
    ).dataloaders(env.data_dir)


dls = data_block()

def train_model_from_data():
    learn = vision_learner(dls, resnet34, metrics=[partial(accuracy, axis=1)])
    learn.fine_tune(6)
    return learn


def dump_model(learner):
    joblib.dump(learner, env.model_path)


# Download Images, Train model and Dump model file
def download_train_dump_model():
    #fetch_images()
    #unlink_failed_images()
    learner = train_model_from_data()
    dump_model(learner)
