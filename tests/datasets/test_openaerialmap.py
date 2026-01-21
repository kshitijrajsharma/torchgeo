# Copyright (c) TorchGeo Contributors. All rights reserved.
# Licensed under the MIT License.

import os
from pathlib import Path
from unittest.mock import MagicMock, patch

import matplotlib.pyplot as plt
import pytest
import torch
import torch.nn as nn

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

    def test_no_data(self, tmp_path: Path) -> None:
        with pytest.raises(DatasetNotFoundError, match='Dataset not found'):
            OpenAerialMap(tmp_path)

    def test_download_no_bbox(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match='bbox or image_id must be provided'):
            OpenAerialMap(tmp_path, download=True)

    @patch('torchgeo.datasets.openaerialmap.requests.post')
    def test_download_empty_features(
        self, mock_post: MagicMock, tmp_path: Path
    ) -> None:
        mock_post.return_value.status_code = 200
        mock_post.return_value.json.return_value = {'features': []}

        # Banepa, Nepal bbox
        bbox = (85.51678, 27.63134, 85.52323, 27.63744)

        with pytest.raises(DatasetNotFoundError, match='Dataset not found'):
            with pytest.warns(UserWarning, match='No TMS imagery found'):
                OpenAerialMap(tmp_path, bbox=bbox, zoom=19, download=True)

    @patch('torchgeo.datasets.openaerialmap.requests.post')
    def test_download_api_error(self, mock_post: MagicMock, tmp_path: Path) -> None:
        mock_post.side_effect = Exception('API Error')

        bbox = (85.51678, 27.63134, 85.52323, 27.63744)

        with pytest.raises(RuntimeError, match='Failed to query STAC API'):
            OpenAerialMap(tmp_path, bbox=bbox, zoom=19, download=True)
