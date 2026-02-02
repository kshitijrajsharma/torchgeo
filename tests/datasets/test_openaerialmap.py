# Copyright (c) TorchGeo Contributors. All rights reserved.
# Licensed under the MIT License.

import asyncio
import os
import shutil
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import matplotlib.pyplot as plt
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
from torchgeo.datasets.openaerialmap import TileUtils


class TestTileUtils:
    def test_tile_basic(self) -> None:
        tile = TileUtils.tile(0.0, 0.0, 0)
        assert tile.x == 0
        assert tile.y == 0
        assert tile.z == 0

    def test_tile_positive_coords(self) -> None:
        tile = TileUtils.tile(10.0, 20.0, 5)
        assert 0 <= tile.x < 2**5
        assert 0 <= tile.y < 2**5
        assert tile.z == 5

    def test_tile_negative_coords(self) -> None:
        tile = TileUtils.tile(-10.0, -20.0, 5)
        assert 0 <= tile.x < 2**5
        assert 0 <= tile.y < 2**5
        assert tile.z == 5

    def test_tile_with_truncate(self) -> None:
        tile = TileUtils.tile(200.0, 100.0, 10, truncate=True)
        assert 0 <= tile.x < 2**10
        assert 0 <= tile.y < 2**10

    def test_tile_corners(self) -> None:
        tile_nw = TileUtils.tile(-180.0, 85.0, 1)
        tile_se = TileUtils.tile(180.0, -85.0, 1)
        assert tile_nw.x == 0
        assert tile_se.x == 1

    def test_bounds_basic(self) -> None:
        tile = TileUtils.Tile(0, 0, 1)
        bounds = TileUtils.bounds(tile)
        assert bounds.west == -180.0
        assert bounds.east == 0.0
        assert bounds.north > 0
        assert bounds.south < bounds.north

    def test_bounds_roundtrip(self) -> None:
        original_tile = TileUtils.Tile(10, 15, 8)
        bounds = TileUtils.bounds(original_tile)
        center_lng = (bounds.west + bounds.east) / 2
        center_lat = (bounds.north + bounds.south) / 2
        result_tile = TileUtils.tile(center_lng, center_lat, 8)
        assert result_tile == original_tile

    def test_tiles_basic(self) -> None:
        tiles = list(TileUtils.tiles(-1, -1, 1, 1, 0))
        assert len(tiles) == 1
        assert tiles[0].z == 0

    def test_tiles_multiple(self) -> None:
        tiles = list(TileUtils.tiles(-10, -10, 10, 10, 2))
        assert len(tiles) > 1
        assert all(t.z == 2 for t in tiles)

    def test_tiles_with_truncate(self) -> None:
        tiles = list(TileUtils.tiles(-200, -100, 200, 100, 1, truncate=True))
        assert len(tiles) > 0
        assert all(0 <= t.x < 2**1 for t in tiles)
        assert all(0 <= t.y < 2**1 for t in tiles)

    def test_tiles_antimeridian(self) -> None:
        tiles = list(TileUtils.tiles(175, -5, -175, 5, 2))
        assert len(tiles) > 0
        x_values = [t.x for t in tiles]
        assert min(x_values) == 0 or max(x_values) == 3

    def test_tiles_web_mercator_limits(self) -> None:
        tiles = list(TileUtils.tiles(-180, -90, 180, 90, 0, truncate=True))
        assert len(tiles) == 1

    def test_tiles_high_zoom(self) -> None:
        tiles = list(TileUtils.tiles(0, 0, 0.1, 0.1, 19))
        assert len(tiles) > 0
        assert all(t.z == 19 for t in tiles)

    def test_tiles_identical_bounds(self) -> None:
        tiles = list(TileUtils.tiles(10.0, 20.0, 10.0, 20.0, 5))
        assert len(tiles) == 1


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
        assert x['image'].shape[0] == 3

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
        with pytest.raises(ValueError, match='bbox must be provided when search=True'):
            OpenAerialMap(search=True)

        with pytest.raises(ValueError, match='bbox or image_id must be provided'):
            OpenAerialMap(download=True)

        with pytest.raises(ValueError, match='zoom must be between'):
            OpenAerialMap(bbox=(0, 0, 1, 1), download=True, zoom=5)

    def test_no_data(self, tmp_path: Path) -> None:
        with pytest.raises(DatasetNotFoundError, match='Dataset not found'):
            OpenAerialMap(tmp_path)

    def test_search(
        self,
        mock_bbox: tuple[float, float, float, float],
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
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
        monkeypatch.setattr(
            'torchgeo.datasets.openaerialmap.requests.post',
            MagicMock(return_value=mock_response),
        )

        ds = OpenAerialMap(tmp_path, bbox=mock_bbox, search=True, download=False)
        assert isinstance(ds.search_results, pd.DataFrame)
        assert len(ds.search_results) == 1
        assert ds.search_results.iloc[0]['ID'] == 'test_id'

    def test_search_empty(
        self,
        mock_bbox: tuple[float, float, float, float],
        tmp_path: Path,
        capsys: pytest.CaptureFixture[str],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {'features': []}
        monkeypatch.setattr(
            'torchgeo.datasets.openaerialmap.requests.post',
            MagicMock(return_value=mock_response),
        )

        OpenAerialMap(tmp_path, bbox=mock_bbox, search=True, download=False)
        captured = capsys.readouterr()
        assert 'No images found' in captured.out

    def test_search_failure(
        self,
        mock_bbox: tuple[float, float, float, float],
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setattr(
            'torchgeo.datasets.openaerialmap.requests.post',
            MagicMock(side_effect=requests.RequestException('Search failed')),
        )

        with pytest.warns(UserWarning, match='STAC search failed'):
            OpenAerialMap(tmp_path, bbox=mock_bbox, search=True, download=False)

    def test_download_flow(
        self,
        mock_bbox: tuple[float, float, float, float],
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        src_dir = os.path.join('tests', 'data', 'openaerialmap')
        valid_file = next(f for f in os.listdir(src_dir) if f.endswith('.tif'))
        shutil.copy(os.path.join(src_dir, valid_file), tmp_path / valid_file)

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            'features': [
                {
                    'id': 'test_id',
                    'properties': {},
                    'assets': {'visual': {'href': 'http://example.com/image.tif'}},
                }
            ]
        }
        monkeypatch.setattr(
            'torchgeo.datasets.openaerialmap.requests.post',
            MagicMock(return_value=mock_response),
        )

        mock_download = MagicMock()
        monkeypatch.setattr(
            'torchgeo.datasets.openaerialmap.OpenAerialMap._download', mock_download
        )

        OpenAerialMap(tmp_path, bbox=mock_bbox, zoom=19, download=True)
        assert mock_download.called

    def test_download_no_tms(
        self,
        mock_bbox: tuple[float, float, float, float],
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {'features': []}
        monkeypatch.setattr(
            'torchgeo.datasets.openaerialmap.requests.post',
            MagicMock(return_value=mock_response),
        )

        with pytest.warns(UserWarning, match='No imagery found'):
            with pytest.raises(DatasetNotFoundError):
                OpenAerialMap(tmp_path, bbox=mock_bbox, download=True)

        assert (tmp_path / '.downloaded').exists()

    def test_download_image_id_no_bbox_error(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        mock_stac_response = MagicMock()
        mock_stac_response.status_code = 200
        mock_stac_response.json.return_value = {
            'features': [
                {
                    'id': 'test_id',
                    'collection': 'test_collection',
                    'properties': {},
                    'assets': {'visual': {'href': 'http://example.com/image.tif'}},
                }
            ]
        }

        mock_tiles_response = MagicMock()
        mock_tiles_response.status_code = 200
        mock_tiles_response.json.return_value = {
            'tilesets': [
                {
                    'links': [
                        {
                            'rel': 'tile',
                            'href': 'http://example.com/WebMercatorQuad/{z}/{x}/{y}',
                        }
                    ]
                }
            ]
        }

        def mock_requests_func(url: str, **kwargs: Any) -> MagicMock:
            if 'search' in url:
                return mock_stac_response
            else:
                return mock_tiles_response

        monkeypatch.setattr(
            'torchgeo.datasets.openaerialmap.requests.post', mock_requests_func
        )
        monkeypatch.setattr(
            'torchgeo.datasets.openaerialmap.requests.get', mock_requests_func
        )

        with pytest.raises(ValueError, match='is required to calculate tiles'):
            OpenAerialMap(tmp_path, image_id='test_id', download=True)

    def test_fetch_item_id_variations(
        self,
        dataset: OpenAerialMap,
        mock_bbox: tuple[float, float, float, float],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        dataset.bbox = mock_bbox
        dataset.image_id = None

        mock_post = MagicMock()
        mock_get = MagicMock()
        monkeypatch.setattr('torchgeo.datasets.openaerialmap.requests.post', mock_post)
        monkeypatch.setattr('torchgeo.datasets.openaerialmap.requests.get', mock_get)

        mock_post.return_value.json.return_value = {
            'features': [
                {'id': 'test_id', 'collection': 'openaerialmap', 'properties': {}}
            ]
        }
        mock_get.return_value.json.return_value = {
            'tilesets': [
                {
                    'links': [
                        {
                            'rel': 'tile',
                            'href': 'http://api/raster/collections/openaerialmap/items/test_id/tiles/WebMercatorQuad/{z}/{x}/{y}',
                        }
                    ]
                }
            ]
        }
        result = dataset._fetch_item_id()
        assert result is not None
        assert 'WebMercatorQuad' in result

        mock_post.return_value.json.return_value = {'features': []}
        assert dataset._fetch_item_id() is None

        mock_post.side_effect = requests.RequestException('Fail')
        with pytest.raises(RuntimeError, match='Failed to query STAC API'):
            dataset._fetch_item_id()

        mock_post.side_effect = ValueError('JSON error')
        with pytest.raises(RuntimeError, match='Invalid STAC API response'):
            dataset._fetch_item_id()

    @pytest.mark.disable_socket_check
    def test_download_tiles_async(
        self, dataset: OpenAerialMap, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        dataset.paths = tmp_path

        async def wrapper() -> None:
            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_response.content = b'fake_tiff_data'

            monkeypatch.setattr('requests.get', MagicMock(return_value=mock_response))

            mock_geo = MagicMock()
            monkeypatch.setattr(dataset, '_georeference_tile', mock_geo)

            tile = TileUtils.Tile(x=1, y=1, z=1)
            await dataset._download_tiles_async('http://tms/{z}/{x}/{y}', [tile])

            assert mock_geo.called
            assert (tmp_path / 'OAM-1-1-1.tif').exists()

        asyncio.run(wrapper())

    def test_georeference_tile_success(
        self, dataset: OpenAerialMap, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        filepath = tmp_path / 'test.tif'
        filepath.touch()
        tile = TileUtils.Tile(x=1, y=1, z=1)

        mock_ds = MagicMock()
        mock_ds.width = 256
        mock_ds.height = 256
        mock_ds.transform = 'mock_transform'
        mock_ds.crs = 'mock_crs'

        mock_open = MagicMock()
        mock_open.__enter__.return_value = mock_ds

        monkeypatch.setattr('rasterio.open', MagicMock(return_value=mock_open))

        dataset._georeference_tile(str(filepath), tile)

        mock_ds.update_tags.assert_called_once()

    @pytest.mark.disable_socket_check
    def test_download_single_tile_failures(
        self, dataset: OpenAerialMap, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        dataset.paths = tmp_path

        async def wrapper() -> None:
            tile = TileUtils.Tile(x=2, y=2, z=2)
            filepath = tmp_path / 'OAM-2-2-2.tif'

            filepath.touch()
            mock_requests = MagicMock()
            monkeypatch.setattr('requests.get', mock_requests)

            mock_ds = MagicMock()
            mock_ds.crs = MagicMock()
            mock_open = MagicMock(
                return_value=MagicMock(__enter__=MagicMock(return_value=mock_ds))
            )
            monkeypatch.setattr('rasterio.open', mock_open)

            await dataset._download_single_tile('url/{z}/{x}/{y}', tile)
            assert mock_requests.call_count == 0

            filepath.unlink()

            mock_response_404 = MagicMock()
            mock_response_404.status_code = 404
            mock_requests.return_value = mock_response_404

            with pytest.warns(UserWarning, match='Failed to download tile'):
                await dataset._download_single_tile('url/{z}/{x}/{y}', tile)

            mock_requests.side_effect = requests.RequestException('Net error')
            with pytest.warns(UserWarning, match='Error downloading tile'):
                await dataset._download_single_tile('url/{z}/{x}/{y}', tile)

        asyncio.run(wrapper())

    def test_georeference_tile_error(
        self, dataset: OpenAerialMap, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        filepath = tmp_path / 'test.tif'
        filepath.touch()
        tile = TileUtils.Tile(x=1, y=1, z=1)

        monkeypatch.setattr('rasterio.open', MagicMock(side_effect=RasterioIOError))

        with pytest.warns(UserWarning, match='Could not georeference'):
            dataset._georeference_tile(str(filepath), tile)
