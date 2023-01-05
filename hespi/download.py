from pathlib import Path
from appdirs import user_cache_dir
from typing import Union
import hashlib
import urllib.request

class DownloadError(Exception):
    pass


def get_cached_path(filename: str) -> Path:
    """
    Returns a path in the ausdex directory in the user's cache.

    File may or may not exist.
    """
    cache_dir = Path(user_cache_dir("hespi"))
    cache_dir.mkdir(exist_ok=True, parents=True)
    return cache_dir / filename


def cached_download(url: str, local_path: Union[str, Path], force: bool = False) -> None:
    """
    Downloads a file if a local file does not already exist.

    Args:
        url (str): The url of the file to download.
        local_path (str, Path): The local path of where the file should be.
            If this file isn't there or the file size is zero then this function downloads it to this location.
        force (bool): Whether or not the file should be forced to download again even if present in the local path.
            Default False.

    Raises:
        DownloadError: Raises an exception if it cannot download the file.
        IOError: Raises an exception if the file does not exist or is empty after downloading.
    """
    local_path = Path(local_path)
    if (not local_path.exists() or local_path.stat().st_size == 0) or force:
        try:
            print(f"Downloading {url} to {local_path}")
            urllib.request.urlretrieve(url, local_path)
        except:
            raise DownloadError(f"Error downloading {url}")

    if not local_path.exists() or local_path.stat().st_size == 0:
        raise IOError(f"Error reading {local_path}")


def get_weights(location:Union[str,Path], force:bool=False) -> Path:
    location = str(location)
    if location.startswith("http"):
        name = location.split("/")[-1]
        extension_location = name.rfind(".")
        if extension_location > 0:
            name_stem = name[:extension_location]
            extension = name[extension_location:]
        else:
            name_stem = name
            extension = ".dat"
        url_hash = hashlib.md5(location.encode()).hexdigest()
        local_path = get_cached_path(f"{name_stem}-{url_hash}{extension}")
        cached_download(location, local_path, force=force)
    else:
        local_path = Path(location)
    
    if not local_path.exists() or local_path.stat().st_size == 0:
        raise IOError(f"Cannot read weights file {local_path}")

    return local_path
