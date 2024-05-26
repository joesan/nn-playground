import os
import pytest
from pathlib import Path
from dotenv import unset_key
from src.models.boston_house_price_prediction.env import BostonHousingConfig


@pytest.fixture(scope="function", autouse=True)
def setup_and_teardown_env_file(tmpdir):
    # Set up a temporary .env file for testing
    env_file_path = tmpdir.join(".env")
    with open(env_file_path, 'w') as f:
        f.write("PROJECT_ROOT_FOLDER=/tmp/test_project_root")

    yield str(env_file_path)

    # Clean up environment variable
    unset_key(str(env_file_path), "PROJECT_ROOT_FOLDER")


def test_load_environment_success(setup_and_teardown_env_file):
    config = BostonHousingConfig(env_file_path=setup_and_teardown_env_file)
    config.load_environment()

    assert config.project_root == '/tmp/test_project_root'
    assert config.absolute_path == '/tmp/test_project_root'
    assert config.data_dir == Path('/tmp/test_project_root/data/raw/boston_housing_price').resolve()
    assert config.models_dir == Path('/tmp/test_project_root/models/boston_housing_price')
    assert config.model_path == Path('/tmp/test_project_root/models/boston_housing_price/boston_housing_price_predictor.pkl')


def test_missing_env_file():
    config = BostonHousingConfig(env_file_path="nonexistent.env")

    with pytest.raises(ValueError, match="Failed to load environment file: nonexistent.env"):
        config.load_environment()


def test_missing_project_root_env_var(tmpdir):
    env_file_path = tmpdir.join(".env")
    with open(env_file_path, 'w') as f:
        f.write(f"PROJECT_ROOT_FOLDER={''}")

    config = BostonHousingConfig(env_file_path=str(env_file_path))

    with pytest.raises(ValueError, match="PROJECT_ROOT_FOLDER environment variable is not set."):
        config.load_environment()


def test_invalid_project_root_path(tmpdir, monkeypatch):
    invalid_path = "/invalid/path/to/project_root"
    env_file_path = tmpdir.join(".env")
    with open(env_file_path, 'w') as f:
        f.write(f"PROJECT_ROOT_FOLDER={invalid_path}")

    monkeypatch.setenv("PROJECT_ROOT_FOLDER", invalid_path)
    config = BostonHousingConfig(env_file_path=str(env_file_path))
    config.load_environment()

    assert config.project_root == invalid_path
    assert config.absolute_path == os.path.abspath(invalid_path)
    assert config.data_dir == Path(invalid_path) / 'data' / 'raw' / 'boston_housing_price'
    assert config.models_dir == Path(invalid_path) / 'models' / 'boston_housing_price'
    assert config.model_path == Path(invalid_path) / 'models' / 'boston_housing_price' / config.model_filename
