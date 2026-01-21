# Copyright (c) TorchGeo Contributors. All rights reserved.
# Licensed under the MIT License.

import asyncio
import os
import shutil
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import aiohttp
import matplotlib.pyplot as plt
import mercantile
import pandas as pd
import pytest
import requests
import torch
import torch.nn as nn
from rasterio.errors import RasterioIOError

from torchgeo.datasets import (
    DatasetNotFoundError,
    IntersectionDataset,
    OpenAerialMap,
    UnionDataset,
)


class TestOpenAerialMap:
    @pytest.fixture
    def dataset(self) -> OpenAerialMap:
        root = os.path.join('tests', 'data', 'openaerialmap')
        transforms = nn.Identity()
        return OpenAerialMap(root, transforms=transforms)

    @pytest.fixture
    def mock_bbox(self) -> tuple[float, float, float, float]:
        return (85.51678, 27.63134, 85.52323, 27.63744)

    def test_getitem(self, dataset: OpenAerialMap) -> None:
        x = dataset[dataset.bounds]
        assert isinstance(x, dict)
        assert isinstance(x['image'], torch.Tensor)
        assert x['image'].shape[0] == 3  # RGB bands

    def test_len(self, dataset: OpenAerialMap) -> None:
        assert len(dataset) == 2

    def test_and(self, dataset: OpenAerialMap) -> None:
        ds = dataset & dataset
        assert isinstance(ds, IntersectionDataset)

    def test_or(self, dataset: OpenAerialMap) -> None:
        ds = dataset | dataset
        assert isinstance(ds, UnionDataset)

    def test_plot(self, dataset: OpenAerialMap) -> None:
        x = dataset[dataset.bounds]
        dataset.plot(x, suptitle='Test')
        plt.close()

    def test_init_validation(self) -> None:
        with pytest.raises(ValueError, match='only 256 and 512 are supported'):
            OpenAerialMap(tile_size=128)

        with pytest.raises(ValueError, match='bbox must be provided when search=True'):
            OpenAerialMap(search=True)

        with pytest.raises(ValueError, match='bbox or image_id must be provided'):
            OpenAerialMap(download=True)

        with pytest.raises(ValueError, match='zoom must be between'):
            OpenAerialMap(bbox=(0, 0, 1, 1), download=True, zoom=5)

    def test_no_data(self, tmp_path: Path) -> None:
        with pytest.raises(DatasetNotFoundError, match='Dataset not found'):
            OpenAerialMap(tmp_path)

    @patch('torchgeo.datasets.openaerialmap.requests.post')
    def test_search(
        self,
        mock_post: MagicMock,
        mock_bbox: tuple[float, float, float, float],
        tmp_path: Path,
    ) -> None:
        mock_post.return_value.status_code = 200
        mock_post.return_value.json.return_value = {
            'features': [
                {
                    'id': 'test_id',
                    'properties': {
                        'start_datetime': '2022-01-01',
                        'oam:platform_type': 'uav',
                        'oam:producer_name': 'test',
                        'gsd': 0.1,
                        'title': 'Test Image',
                    },
                }
            ]
        }

        ds = OpenAerialMap(tmp_path, bbox=mock_bbox, search=True, download=False)
        assert isinstance(ds.search_results, pd.DataFrame)
        assert len(ds.search_results) == 1
        assert ds.search_results.iloc[0]['ID'] == 'test_id'

    @patch('torchgeo.datasets.openaerialmap.requests.post')
    def test_search_empty(
        self,
        mock_post: MagicMock,
        mock_bbox: tuple[float, float, float, float],
        tmp_path: Path,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        mock_post.return_value.status_code = 200
        mock_post.return_value.json.return_value = {'features': []}

        OpenAerialMap(tmp_path, bbox=mock_bbox, search=True, download=False)
        captured = capsys.readouterr()
        assert 'No images found' in captured.out

    @patch('torchgeo.datasets.openaerialmap.requests.post')
    def test_search_failure(
        self,
        mock_post: MagicMock,
        mock_bbox: tuple[float, float, float, float],
        tmp_path: Path,
    ) -> None:
        mock_post.side_effect = requests.RequestException('Search failed')

        with pytest.warns(UserWarning, match='STAC search failed'):
            OpenAerialMap(tmp_path, bbox=mock_bbox, search=True, download=False)

    @patch('torchgeo.datasets.openaerialmap.OpenAerialMap._download_tiles_async')
    @patch('torchgeo.datasets.openaerialmap.requests.post')
    def test_download_flow(
        self,
        mock_post: MagicMock,
        mock_download_tiles: MagicMock,
        mock_bbox: tuple[float, float, float, float],
        tmp_path: Path,
    ) -> None:
        src_dir = os.path.join('tests', 'data', 'openaerialmap')
        valid_file = next(f for f in os.listdir(src_dir) if f.endswith('.tif'))
        shutil.copy(os.path.join(src_dir, valid_file), tmp_path / valid_file)

        mock_post.return_value.status_code = 200
        mock_post.return_value.json.return_value = {
            'features': [
                {
                    'id': 'test_id',
                    'properties': {},
                    'assets': {'visual': {'href': 'http://example.com/image.tif'}},
                }
            ]
        }

        OpenAerialMap(tmp_path, bbox=mock_bbox, zoom=19, download=True)
        assert mock_download_tiles.called

    @patch('torchgeo.datasets.openaerialmap.requests.post')
    def test_download_no_tms(
        self,
        mock_post: MagicMock,
        mock_bbox: tuple[float, float, float, float],
        tmp_path: Path,
    ) -> None:
        mock_post.return_value.status_code = 200
        mock_post.return_value.json.return_value = {'features': []}

        with pytest.warns(UserWarning, match='No TMS imagery found'):
            with pytest.raises(DatasetNotFoundError):
                OpenAerialMap(tmp_path, bbox=mock_bbox, download=True)

        assert (tmp_path / '.downloaded').exists()

    @patch('torchgeo.datasets.openaerialmap.requests.post')
    def test_download_image_id_no_bbox_warning(
        self, mock_post: MagicMock, tmp_path: Path
    ) -> None:
        # Image ID provided, but BBOX missing -> should warn and return from _download
        mock_post.return_value.status_code = 200
        mock_post.return_value.json.return_value = {
            'features': [
                {
                    'id': 'id',
                    'properties': {},
                    'assets': {'visual': {'href': 'http://s'}},
                }
            ]
        }

        with pytest.warns(UserWarning, match='Bounding box .* is required'):
            with pytest.raises(DatasetNotFoundError):
                OpenAerialMap(tmp_path, image_id='test_id', download=True)

    @patch('torchgeo.datasets.openaerialmap.requests.post')
    def test_fetch_tms_url_variations(
        self,
        mock_post: MagicMock,
        dataset: OpenAerialMap,
        mock_bbox: tuple[float, float, float, float],
    ) -> None:
        dataset.bbox = mock_bbox
        dataset.image_id = None
        dataset.tile_size = 512

        mock_post.return_value.json.return_value = {
            'features': [{'properties': {}, 'assets': {'visual': {'href': 'src'}}}]
        }

        url = dataset._fetch_tms_url()
        assert url is not None
        assert '@2x' in url

        mock_post.return_value.json.return_value = {
            'features': [{'properties': {}, 'assets': {}}]
        }
        assert dataset._fetch_tms_url() is None

        mock_post.side_effect = requests.RequestException('Fail')
        with pytest.raises(RuntimeError, match='Failed to query STAC API'):
            dataset._fetch_tms_url()

        mock_post.side_effect = ValueError('JSON error')
        with pytest.raises(RuntimeError, match='Failed to query STAC API'):
            dataset._fetch_tms_url()

        mock_post.side_effect = Exception('General error')
        with pytest.raises(RuntimeError, match='Failed to query STAC API'):
            dataset._fetch_tms_url()

    def test_download_tiles_async(self, dataset: OpenAerialMap, tmp_path: Path) -> None:
        dataset.paths = tmp_path

        async def wrapper() -> None:
            with patch('aiohttp.ClientSession') as mock_session_cls:
                mock_session = mock_session_cls.return_value
                mock_session.__aenter__.return_value = mock_session

                mock_response = AsyncMock()
                mock_response.status = 200
                mock_response.read.return_value = b'fake_tiff_data'
                mock_session.get.return_value.__aenter__.return_value = mock_response

                tile = mercantile.Tile(x=1, y=1, z=1)

                with patch.object(dataset, '_georeference_tile') as mock_geo:
                    await dataset._download_tiles_async(
                        'http://tms/{z}/{x}/{y}', [tile]
                    )
                    assert mock_geo.called
                    assert (tmp_path / 'OAM-1-1-1.tif').exists()

        asyncio.run(wrapper())

    def test_download_single_tile_failures(
        self, dataset: OpenAerialMap, tmp_path: Path
    ) -> None:
        dataset.paths = tmp_path

        async def wrapper() -> None:
            tile = mercantile.Tile(x=2, y=2, z=2)
            filepath = tmp_path / 'OAM-2-2-2.tif'

            filepath.touch()
            with patch('aiohttp.ClientSession') as mock_session:
                await dataset._download_single_tile(mock_session, '', tile)
                assert mock_session.get.call_count == 0

            filepath.unlink()
            mock_session = MagicMock()
            mock_response = AsyncMock()
            mock_response.status = 404
            mock_session.get.return_value.__aenter__.return_value = mock_response

            with pytest.warns(UserWarning, match='Failed to download tile'):
                await dataset._download_single_tile(mock_session, 'url', tile)

            mock_session.get.side_effect = aiohttp.ClientError('Net error')
            with pytest.warns(UserWarning, match='Error downloading tile'):
                await dataset._download_single_tile(mock_session, 'url', tile)

        asyncio.run(wrapper())

    def test_georeference_tile_error(
        self, dataset: OpenAerialMap, tmp_path: Path
    ) -> None:
        filepath = tmp_path / 'test.tif'
        filepath.touch()
        tile = mercantile.Tile(x=1, y=1, z=1)

        with patch('rasterio.open', side_effect=RasterioIOError):
            with pytest.warns(UserWarning, match='Could not georeference'):
                dataset._georeference_tile(str(filepath), tile)
