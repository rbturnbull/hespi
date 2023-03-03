import tempfile
import pytest
from unittest.mock import patch
from pathlib import Path
import shutil

from hespi import download

def test_get_cached_path():
    name = "tmpfile.txt"
    tmpfile = download.get_cached_path(name)
    assert tmpfile.parent.name == "hespi"
    assert tmpfile.parent.exists()
    assert tmpfile.name == name


@patch("urllib.request.urlretrieve")
def test_cached_download_exists(mock_urlretrieve):
    download.cached_download("http://www.example.com", Path(__file__).parent / "testdata/test.jpg") # file exists
    mock_urlretrieve.assert_not_called()


@patch("urllib.request.urlretrieve")
def test_cached_download_empty(mock_urlretrieve):
    with pytest.raises(OSError):
        download.cached_download("http://www.example.com", Path(__file__).parent / "testdata/does-not-exist.txt")
    mock_urlretrieve.assert_called_once()


def urlretrieve_fail():
    raise Exception("failed download")

@patch("urllib.request.urlretrieve", urlretrieve_fail)
def test_cached_download_fail():
    with pytest.raises(download.DownloadError):
        download.cached_download("http://www.example.com", Path(__file__).parent / "testdata/does-not-exist.txt")


def test_get_location_empty():
    with pytest.raises(IOError):
        download.get_location(Path(__file__).parent / "testdata/empty.dat")


def test_get_location_path_exists():
    path = Path(__file__).parent / "testdata/test.jpg"
    assert path == download.get_location(path)


@patch("hespi.download.cached_download")
def test_get_location_has_extension(mock_cached_download):
    filename = "test-558d5d8ab1da205b9cfc9754513a9882.jpg"
    local_path = Path(__file__).parent / "testdata" / filename
    with patch("hespi.download.get_cached_path", return_value=local_path) as mock_get_cached_path:
        path = download.get_location("https://raw.githubusercontent.com/rbturnbull/hespi/main/tests/testdata/test.jpg")
        assert path.name == filename
        mock_cached_download.assert_called_once()
        mock_get_cached_path.assert_called_once_with(filename)



@patch("hespi.download.cached_download")
def test_get_location_no_extension(mock_cached_download):
    filename = "test-no-extension-ebaa296923904111a3972b54eba5cf5f.dat"
    local_path = Path(__file__).parent / "testdata" / filename
    with patch("hespi.download.get_cached_path", return_value=local_path) as mock_get_cached_path:
        path = download.get_location("https://raw.githubusercontent.com/rbturnbull/hespi/main/tests/testdata/test-no-extension")
        assert path.name == filename
        mock_cached_download.assert_called_once()
        mock_get_cached_path.assert_called_once_with(filename)


def test_get_location_gz():
    filename = "test-bfc763ba3bb28a80dcdec989b06f055c.txt.gz"
    testdata_path = Path(__file__).parent / "testdata" / filename

    with tempfile.TemporaryDirectory() as tmpdir:        
        tmpdir = Path(tmpdir)
        local_path = tmpdir / filename
        shutil.copy(testdata_path, local_path)
        assert local_path.exists()

        with patch("hespi.download.user_cache_dir", return_value=tmpdir) as mock_user_cache_dir:
            path = download.get_location("https://raw.githubusercontent.com/rbturnbull/hespi/main/tests/testdata/test.txt.gz")
            assert path.name == "test-bfc763ba3bb28a80dcdec989b06f055c.txt"
            assert path.read_text().strip() == "Test Text File"
            mock_user_cache_dir.assert_called_with("hespi")



