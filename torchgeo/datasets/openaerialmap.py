# Copyright (c) TorchGeo Contributors. All rights reserved.
# Licensed under the MIT License.

"""OpenAerialMap dataset."""

import asyncio
import os
import warnings
from collections.abc import Callable, Iterable
from concurrent.futures import ThreadPoolExecutor
from typing import ClassVar

import aiohttp
import matplotlib.pyplot as plt
import mercantile
import rasterio
import requests
from matplotlib.figure import Figure
from pyproj import CRS
from rasterio.crs import CRS as RioCRS
from rasterio.transform import from_bounds
from tqdm.asyncio import tqdm_asyncio

from .geo import RasterDataset
from .utils import Path, Sample


class OpenAerialMap(RasterDataset):
    """OpenAerialMap dataset.

    The `OpenAerialMap (OAM) <https://openaerialmap.org/>`__ is an open service
    for accessing and sharing aerial imagery. The dataset provides access to
    crowd-sourced aerial imagery from various sources including drones, satellites,
    and aircraft.

    This implementation uses the `STAC API <https://api.imagery.hotosm.org/stac>`__
    to query imagery and download tiles via TMS endpoints. The STAC API returns
    imagery sorted by most recent first.

    Dataset features
    ----------------

    * Aerial imagery from various sources (drones, satellites, aircraft)
    * Global coverage with varying resolution
    * STAC-based querying (most recent imagery first)
    * Tile naming following mercantile standard: OAM-{x}-{y}-{z}.tif
    * Automatic georeferencing using rasterio
    * RGB imagery (3-band)
    * Bbox format compatible with OpenStreetMap for easy dataset combination

    Dataset usage
    -------------

    The dataset can be used in two modes:

    1. **Local mode**: Load pre-downloaded imagery from a local directory
    2. **Download mode**: Query STAC API and download TMS tiles

    For local mode, organize imagery files in a directory structure::

        root/
        ├── OAM-1234-5678-19.tif
        ├── OAM-1235-5678-19.tif
        └── ...

    For download mode, provide a bounding box (same format as OpenStreetMap) and zoom level::

        dataset = OpenAerialMap(
            paths='data/openaerial',
            bbox=(lon_min, lat_min, lon_max, lat_max),
            zoom=19,
            download=True
        )

    If you use this dataset in your research, please cite OpenAerialMap:

    * https://openaerialmap.org/

    .. versionadded:: 0.8
    """

    _stac_api_url: ClassVar[str] = "https://api.imagery.hotosm.org/stac"

    filename_glob = "OAM-*.tif"
    filename_regex = r"^OAM-.*\.tif$"

    all_bands = ("R", "G", "B")
    rgb_bands = ("R", "G", "B")

    def __init__(
        self,
        paths: Path | Iterable[Path] = "data",
        crs: CRS | None = None,
        res: float | tuple[float, float] | None = None,
        bbox: tuple[float, float, float, float] | None = None,
        zoom: int = 19,
        max_items: int = 1,
        transforms: Callable[[Sample], Sample] | None = None,
        cache: bool = True,
        download: bool = False,
        image_id: str | None = None,
    ) -> None:
        """Initialize a new OpenAerialMap dataset instance.

        Args:
            paths: one or more root directories to search or files to load
            crs: :term:`coordinate reference system (CRS)` to warp to
                (defaults to EPSG:3857)
            res: resolution of the dataset in units of CRS
                (defaults to resolution of first file found)
            bbox: bounding box for STAC query as (xmin, ymin, xmax, ymax) in EPSG:4326.
                Same format as OpenStreetMap for easy dataset combination.
                Only used when download=True
            zoom: zoom level for tiles (1-22), only used when download=True.
                Higher zoom = more detail. Typical values: 18-20 for high-res drone imagery
            max_items: maximum number of STAC items to query, only used when download=True.
                STAC API returns most recent imagery first. Use max_items=1 for latest imagery,
                or increase to search through more imagery sources if first result doesn't have TMS tiles.
            transforms: a function/transform that takes an input sample
                and returns a transformed version. Note: CRS transformation is handled
                automatically via the crs parameter.
            cache: if True, cache file handle to speed up repeated sampling
            download: if True, download imagery from STAC API based on bbox
            image_id: optional STAC item ID to download specific imagery

        Raises:
            DatasetNotFoundError: If dataset is not found and download=False.
            ValueError: If download=True but bbox is not provided.
        """
        self.paths = paths
        self.bbox = bbox
        self.zoom = zoom
        self.max_items = max_items
        self.download = download
        self.image_id = image_id

        if download:
            if bbox is None and image_id is None:
                raise ValueError("bbox or image_id must be provided when download=True")
            if not 6 <= zoom <= 22:
                raise ValueError(f"zoom must be between 6 and 22, got {zoom}")
            self._download()

        # If 'crs' is None, it defaults to EPSG:3857 (Web Mercator)
        super().__init__(
            paths, crs or CRS.from_epsg(3857), res, transforms=transforms, cache=cache
        )

    def _download(self) -> None:
        """Download imagery from STAC API and TMS endpoints.

        This method:
        1. Queries STAC API for imagery items within bbox or by ID
        2. Extracts TMS URLs from STAC items
        3. Calculates mercantile tiles for the bbox at specified zoom
        4. Downloads tiles asynchronously with proper georeferencing
        """
        assert isinstance(self.paths, str | os.PathLike)
        os.makedirs(self.paths, exist_ok=True)

        tms_url = self._fetch_tms_url()
        if not tms_url:
            warnings.warn(
                f"No TMS imagery found for bbox {self.bbox} or ID {self.image_id}. "
                "Try a different area or check OpenAerialMap coverage.",
                UserWarning,
                stacklevel=2,
            )
            # create placeholder to avoid DatasetNotFoundError
            placeholder_path = os.path.join(self.paths, ".downloaded")
            with open(placeholder_path, "w") as f:
                f.write("Download attempted but no TMS URLs found.\n")
            return

        # If image_id provided without bbox, we can't calculate tiles easily without parsing
        # the feature geometry. For now, we enforce bbox presence or rely on user providing it.
        if self.bbox is None:
            warnings.warn(
                "Bounding box (bbox) is required to calculate tiles, even when image_id is provided.",
                UserWarning,
                stacklevel=2,
            )
            return

        # Calculate tiles for bbox at specified zoom using mercantile
        tiles = list(mercantile.tiles(*self.bbox, self.zoom, truncate=True))

        def run_in_thread():
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                loop.run_until_complete(self._download_tiles_async(tms_url, tiles))
            finally:
                loop.close()

        with ThreadPoolExecutor() as executor:
            executor.submit(run_in_thread).result()

    def _fetch_tms_url(self) -> str | None:
        """Query STAC API and extract TMS URL from metadata.

        Combines logic for ID-based and BBox-based queries.

        Returns:
            TMS URL template from the specific imagery, or None if not found
        """
        search_url = f"{self._stac_api_url}/search"
        params = {"limit": self.max_items}

        if self.image_id:
            params["ids"] = [self.image_id]
        elif self.bbox:
            params["bbox"] = list(self.bbox)

        try:
            response = requests.post(search_url, json=params, timeout=30)
            response.raise_for_status()
            data = response.json()
        except requests.RequestException as e:
            warnings.warn(f"Failed to query STAC API at {search_url}: {e}", UserWarning)
            return None

        features = data.get("features", [])
        if not features:
            return None

        # Iterate through features to find the first one with a valid TMS
        for feature in features:
            assets = feature.get("assets", {})
            metadata = assets.get("metadata", {})
            href = metadata.get("href")

            if not href:
                continue

            try:
                # Fetch OAM specific metadata which contains the TMS url
                meta_response = requests.get(href, timeout=10)
                meta_response.raise_for_status()
                meta_json = meta_response.json()

                tms = meta_json.get("properties", {}).get("tms")
                if tms and "{z}" in tms and "{x}" in tms and "{y}" in tms:
                    return tms

            except requests.RequestException:
                # If metadata fetch fails for one item, try the next
                continue

        return None

    async def _download_tiles_async(
        self, tms_url: str, tiles: list[mercantile.Tile]
    ) -> None:
        """Download tiles asynchronously with progress bar.

        Args:
            tms_url: TMS URL template with {z}, {x}, {y} placeholders
            tiles: List of mercantile tiles to download
        """
        async with aiohttp.ClientSession() as session:
            tasks = [
                self._download_single_tile(session, tms_url, tile) for tile in tiles
            ]

            await tqdm_asyncio.gather(*tasks, desc="Downloading Tiles", leave=False)

    async def _download_single_tile(
        self,
        session: aiohttp.ClientSession,
        tms_url: str,
        tile: mercantile.Tile,
    ) -> None:
        """Download and georeference a single tile.

        Args:
            session: aiohttp client session
            tms_url: TMS URL template
            tile: mercantile tile to download
        """
        assert isinstance(self.paths, str | os.PathLike)

        url = tms_url.format(z=tile.z, x=tile.x, y=tile.y)
        filename = f"OAM-{tile.x}-{tile.y}-{tile.z}.tif"
        filepath = os.path.join(self.paths, filename)

        if os.path.exists(filepath):
            return

        try:
            async with session.get(url, timeout=30) as response:
                if response.status != 200:
                    warnings.warn(
                        f"Failed to download tile {tile}: HTTP {response.status}",
                        UserWarning,
                    )
                    return

                tile_data = await response.read()

                with open(filepath, "wb") as f:
                    f.write(tile_data)

                self._georeference_tile(filepath, tile)

        except (aiohttp.ClientError, OSError) as e:
            warnings.warn(f"Error downloading tile {tile}: {e}", UserWarning)

    def _georeference_tile(self, filepath: str, tile: mercantile.Tile) -> None:
        """Add georeferencing metadata to a downloaded tile.

        This sets the CRS to EPSG:4326 because mercantile bounds are lat/lon.
        The parent RasterDataset will handle warping to the user's requested CRS.

        Args:
            filepath: path to tile file
            tile: mercantile tile for calculating bounds
        """
        bounds = mercantile.bounds(tile)
        try:
            with rasterio.open(filepath, "r+") as dataset:
                transform = from_bounds(
                    bounds.west,
                    bounds.south,
                    bounds.east,
                    bounds.north,
                    dataset.width,
                    dataset.height,
                )
                dataset.transform = transform
                dataset.crs = RioCRS.from_epsg(4326)
                dataset.update_tags(
                    ns="rio_georeference",
                    georeferencing_applied="True",
                    tile_x=str(tile.x),
                    tile_y=str(tile.y),
                    tile_z=str(tile.z),
                )
        except rasterio.errors.RasterioIOError:
            warnings.warn(
                f"Could not georeference {filepath}. Not a valid raster file.",
                UserWarning,
                stacklevel=2,
            )

    def plot(
        self, sample: Sample, show_titles: bool = True, suptitle: str | None = None
    ) -> Figure:
        """Plot a sample from the dataset.

        Args:
            sample: a sample returned by :meth:`RasterDataset.__getitem__`
            show_titles: flag indicating whether to show titles above each panel
            suptitle: optional string to use as a suptitle

        Returns:
            a matplotlib Figure with the rendered sample
        """
        image = sample["image"]
        # Convert C, H, W -> H, W, C for plotting
        if image.shape[0] >= 3:
            rgb = image[0:3, :, :].permute(1, 2, 0)
        else:
            rgb = image[0, :, :]

        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(4, 4))

        # If data is 2D (grayscale/single band), use gray colormap
        if rgb.ndim == 2:
            ax.imshow(rgb, cmap="gray")
        else:
            ax.imshow(rgb)

        ax.axis("off")
        if show_titles:
            ax.set_title("Image")

        if suptitle is not None:
            plt.suptitle(suptitle)

        return fig
