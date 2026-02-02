# Copyright (c) TorchGeo Contributors. All rights reserved.
# Licensed under the MIT License.

"""OpenAerialMap dataset."""

import asyncio
import math
import os
import warnings
from collections import namedtuple
from collections.abc import Callable, Iterable, Iterator
from typing import Any, ClassVar, Literal, cast

import matplotlib.pyplot as plt
import pandas as pd
import rasterio
import requests
from matplotlib.figure import Figure
from pyproj import CRS
from rasterio.crs import CRS as RioCRS
from rasterio.transform import from_bounds

from .geo import RasterDataset
from .utils import Path, Sample


class TileUtils:
    """Web Mercator tile utilities for XYZ tile calculations.

    Implements standard Web Mercator (EPSG:3857) tile math for converting
    between geographic coordinates and tile indices.

    References:
        * OpenStreetMap Wiki - Slippy map tilenames:
          https://wiki.openstreetmap.org/wiki/Slippy_map_tilenames
        * Web Mercator / Pseudo-Mercator (EPSG:3857):
          https://epsg.io/3857
        * OSGeo Tile Map Service Specification:
          https://wiki.osgeo.org/wiki/Tile_Map_Service_Specification

    The mathematical formulas implemented here are based on the Spherical Mercator
    projection, which is documented in the references above.
    """

    Tile = namedtuple('Tile', ['x', 'y', 'z'])
    LngLatBbox = namedtuple('LngLatBbox', ['west', 'south', 'east', 'north'])

    @classmethod
    def tile(
        cls, lng: float, lat: float, zoom: int, truncate: bool = False
    ) -> 'TileUtils.Tile':
        """Get tile coordinates containing a geographic point.

        Args:
            lng: Longitude in degrees
            lat: Latitude in degrees
            zoom: Zoom level
            truncate: Clamp coordinates to valid ranges

        Returns:
            Tile with x, y, z coordinates
        """
        if truncate:
            lng = max(-180.0, min(180.0, lng))
            lat = max(-90.0, min(90.0, lat))

        x_frac = (lng + 180.0) / 360.0
        lat_rad = math.radians(lat)
        y_frac = (
            1.0 - math.log(math.tan(lat_rad) + (1.0 / math.cos(lat_rad))) / math.pi
        ) / 2.0

        n = 2**zoom
        x = int(min(n - 1, max(0, math.floor(x_frac * n))))
        y = int(min(n - 1, max(0, math.floor(y_frac * n))))

        return cls.Tile(x, y, zoom)

    @classmethod
    def bounds(cls, t: 'TileUtils.Tile') -> 'TileUtils.LngLatBbox':
        """Get geographic bounds of a tile.

        Args:
            t: Tile with x, y, z coordinates

        Returns:
            Bounding box with west, south, east, north in degrees
        """
        n = 2**t.z
        west = t.x / n * 360.0 - 180.0
        east = (t.x + 1) / n * 360.0 - 180.0

        north_rad = math.atan(math.sinh(math.pi * (1 - 2 * t.y / n)))
        south_rad = math.atan(math.sinh(math.pi * (1 - 2 * (t.y + 1) / n)))
        north = math.degrees(north_rad)
        south = math.degrees(south_rad)

        return cls.LngLatBbox(west, south, east, north)

    @classmethod
    def tiles(
        cls,
        west: float,
        south: float,
        east: float,
        north: float,
        zoom: int,
        truncate: bool = False,
    ) -> Iterator['TileUtils.Tile']:
        """Generate tiles covering a bounding box.

        Args:
            west: Western longitude in degrees
            south: Southern latitude in degrees
            east: Eastern longitude in degrees
            north: Northern latitude in degrees
            zoom: Zoom level
            truncate: Clamp coordinates to valid ranges

        Yields:
            Tiles covering the bounding box
        """
        if truncate:
            west = max(-180.0, min(180.0, west))
            south = max(-90.0, min(90.0, south))
            east = max(-180.0, min(180.0, east))
            north = max(-90.0, min(90.0, north))

        if west > east:
            bboxes = [(-180.0, south, east, north), (west, south, 180.0, north)]
        else:
            bboxes = [(west, south, east, north)]

        for w, s, e, n in bboxes:
            w = max(-180.0, w)
            s = max(-85.051129, s)
            e = min(180.0, e)
            n = min(85.051129, n)

            ul_tile = cls.tile(w, n, zoom)
            lr_tile = cls.tile(e, s, zoom)

            for x in range(ul_tile.x, lr_tile.x + 1):
                for y in range(ul_tile.y, lr_tile.y + 1):
                    yield cls.Tile(x, y, zoom)


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

    1. **Search mode**: Query available imagery in a bbox.
    2. **Download mode**: Query STAC API and download TMS tiles.

    For search mode::

        oam = OpenAerialMap(bbox=bbox, search=True)
        # oam.search_results contains the DataFrame

    For download mode, provide a bounding box (same format as OpenStreetMap) and zoom level::

        dataset = OpenAerialMap(
            paths='data/openaerial',
            bbox=(lon_min, lat_min, lon_max, lat_max),
            zoom=19,
            download=True,
        )

    If you use this dataset in your research, please cite OpenAerialMap:

    * https://openaerialmap.org/
    """

    _stac_api_url: ClassVar[str] = 'https://api.imagery.hotosm.org/stac'
    _tile_source_url: ClassVar[str] = (
        'https://titiler.hotosm.org/cog/tiles/WebMercatorQuad/{z}/{x}/{y}@{scale}x?url={source}'
    )

    filename_glob = 'OAM-*.tif'

    all_bands = ('R', 'G', 'B')
    rgb_bands = ('R', 'G', 'B')

    def __init__(
        self,
        paths: Path | Iterable[Path] = 'data',
        crs: CRS | None = None,
        res: float | tuple[float, float] | None = None,
        bbox: tuple[float, float, float, float] | None = None,
        zoom: int = 19,
        max_items: int = 1,
        transforms: Callable[[Sample], Sample] | None = None,
        cache: bool = True,
        download: bool = False,
        search: bool = False,
        image_id: str | None = None,
        tile_size: Literal[256, 512, 768, 1024] = 256,
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
            zoom: zoom level for tiles (6-22), only used when download=True.
                Higher zoom = more detail. Typical values: 18-20 for high-res
                drone imagery. Higher zoom gives higher resolution but covers
                less area per tile. Consider increasing tile_size for better
                quality at the same zoom. Check the GSD of the image before
                downloading.
            max_items: maximum number of STAC items to query.
            transforms: a function/transform that takes an input sample
                and returns a transformed version. Note: CRS transformation is handled
                automatically via the crs parameter.
            cache: if True, cache file handle to speed up repeated sampling
            download: if True, download imagery from STAC API based on bbox
            search: if True, query STAC API for available imagery and return results in
                self.search_results. Skips dataset initialization if download=False.
            image_id: optional STAC item ID to download specific imagery
            tile_size: size of the tiles to download (supported : 256 , 512, 768, 1024 );
                Do verify they exists in the remote image source.

        Raises:
            DatasetNotFoundError: If dataset is not found and download=False.
            ValueError: If download=True but neither bbox nor image_id is provided.
        """
        self.paths = paths
        self.bbox = bbox
        self.zoom = zoom
        self.max_items = max_items
        self.download = download
        self.image_id = image_id
        self.search_results: pd.DataFrame | None = None
        self.tile_size = tile_size

        if search:
            if self.bbox is None:
                raise ValueError('bbox must be provided when search=True')
            self._search_stac()
            # If user only wants to search, return early, because dataset will raise error if it is empty and is instantiated by super init
            if not download:
                return

        if download:
            if bbox is None and image_id is None:
                raise ValueError('bbox or image_id must be provided when download=True')
            if not 6 <= zoom <= 22:
                raise ValueError(f'zoom must be between 6 and 22, got {zoom}')
            self._download()
            print('Download complete.')

        # If 'crs' is None, it defaults to EPSG:3857 (Web Mercator), because 3857 makes logical sense for tiles and web maps.
        super().__init__(
            paths, crs or CRS.from_epsg(3857), res, transforms=transforms, cache=cache
        )

    def _search_stac(self) -> None:
        """Query and display available imagery as a DataFrame."""
        assert self.bbox is not None
        try:
            resp = requests.post(
                f'{self._stac_api_url}/search',
                json={'bbox': list(self.bbox), 'limit': max(self.max_items, 20)},
                timeout=30,
            )
            resp.raise_for_status()
        except requests.RequestException as e:
            warnings.warn(f'STAC search failed: {e}', UserWarning)
            return

        features = resp.json().get('features', [])
        if not features:
            print('No images found in this bounding box.')
            return

        self.search_results = pd.DataFrame(
            [
                {
                    'ID': f['id'],
                    'Date': (
                        f['properties'].get('start_datetime')
                        or f['properties'].get('created')
                        or ''
                    ),
                    'Platform': f['properties'].get('oam:platform_type'),
                    'Provider': f['properties'].get('oam:producer_name'),
                    'GSD': f['properties'].get('gsd'),
                    'Title': f['properties'].get('title'),
                }
                for f in features
            ]
        )

        print(f'Found {len(features)} available images')
        print('\nUse .search_results to view.\n')

    def _download(self) -> None:
        """Download imagery from STAC API and TMS endpoints.

        This method:
        1. Queries STAC API for imagery items within bbox or by ID
        2. Extracts TMS URLs from STAC items
        3. Calculates mercantile tiles for the bbox at specified zoom
        4. Downloads tiles asynchronously with proper georeferencing
        """
        root = cast(str | os.PathLike[str], self.paths)

        if isinstance(root, (str, os.PathLike)):
            os.makedirs(root, exist_ok=True)

        tms_url = self._fetch_tms_url()
        if not tms_url:
            warnings.warn(
                f'No TMS imagery found for bbox {self.bbox} or ID {self.image_id}. '
                'Try a different area or check OpenAerialMap coverage.',
                UserWarning,
                stacklevel=2,
            )
            # create placeholder to avoid DatasetNotFoundError
            with open(os.path.join(root, '.downloaded'), 'w') as f:
                f.write('Download attempted but no TMS URLs found.\n')
            return

        if self.bbox is None:
            raise ValueError(
                'Bounding box (bbox) is required to calculate tiles. '
                'Please provide a bbox when initializing OpenAerialMap, even when image_id is provided.'
            )

        # we use truncate=True to avoid tiles outside the bbox, just to make sure there won't be corner tiles
        tiles = list(TileUtils.tiles(*self.bbox, self.zoom, truncate=True))

        asyncio.run(self._download_tiles_async(tms_url, tiles))

    def _fetch_tms_url(self) -> str | None:
        """Query STAC API and extract TMS URL from metadata.

        Combines logic for ID-based and BBox-based queries.

        Returns:
            TMS URL template from the specific imagery, or None if not found
        """
        params: dict[str, Any] = {'limit': self.max_items}

        if self.image_id:
            params['ids'] = [self.image_id]
        elif self.bbox:
            params['bbox'] = list(self.bbox)

        try:
            response = requests.post(
                f'{self._stac_api_url}/search', json=params, timeout=30
            )
            response.raise_for_status()
            data = response.json()
        except requests.RequestException as e:
            raise RuntimeError(f'Failed to query STAC API: {e}') from e
        except (ValueError, KeyError) as e:
            raise RuntimeError(f'Invalid STAC API response: {e}') from e

        features = data.get('features', [])
        if not features:
            return None

        # Use the first feature available, because it is the most recent
        feature = features[0]
        props = feature.get('properties', {})
        visual_source = feature.get('assets', {}).get('visual', {}).get('href')

        print(f'Using OpenAerialMap image: {props.get("title", "Unknown")}')
        print(f'  ID: {feature.get("id", "Unknown")}')
        print(f'  Date: {props.get("start_datetime", "Unknown")}')
        print(f'  Platform: {props.get("oam:platform_type", "Unknown")}')
        print(f'  Provider: {props.get("oam:producer_name", "Unknown")}')
        print(f'  GSD: {props.get("gsd", "Unknown")}')
        print(f'  License: {props.get("license", "Unknown")}')

        if not visual_source:
            return None

        return self._tile_source_url.format(
            z='{z}',
            x='{x}',
            y='{y}',
            scale=int(self.tile_size / 256),
            source=visual_source,
        )

    async def _download_tiles_async(
        self, tms_url: str, tiles: list[TileUtils.Tile]
    ) -> None:
        """Download tiles asynchronously with progress bar.

        Args:
            tms_url: TMS URL template with {z}, {x}, {y} placeholders
            tiles: List of mercantile tiles to download
        """
        tasks = [self._download_single_tile(tms_url, tile) for tile in tiles]
        total = len(tasks)
        print(f'Starting download of {total} tiles...')

        for i, task in enumerate(asyncio.as_completed(tasks), 1):
            await task

    async def _download_single_tile(self, tms_url: str, tile: TileUtils.Tile) -> None:
        """Download and georeference a single tile.

        Args:
            tms_url: TMS URL template
            tile: mercantile tile to download
        """
        root = cast(str | os.PathLike[str], self.paths)

        url = tms_url.format(z=tile.z, x=tile.x, y=tile.y)
        filename = f'OAM-{tile.x}-{tile.y}-{tile.z}.tif'
        filepath = os.path.join(root, filename)

        if os.path.exists(filepath):
            # it is possible download might be corrupted, so verify georeferencing
            is_valid = False
            try:
                with rasterio.open(filepath) as ds:
                    is_valid = ds.crs is not None
            except (rasterio.errors.RasterioIOError, OSError):
                is_valid = False

            if is_valid:
                return
            os.unlink(filepath)

        try:
            response = await asyncio.to_thread(requests.get, url, timeout=30)
            if response.status_code != 200:
                warnings.warn(
                    f'Failed to download tile {tile}: HTTP {response.status_code}',
                    UserWarning,
                )
                return

            with open(filepath, 'wb') as f:
                f.write(response.content)

            self._georeference_tile(filepath, tile)

        except (requests.RequestException, OSError) as e:
            warnings.warn(f'Error downloading tile {tile}: {e}', UserWarning)

    def _georeference_tile(self, filepath: str, tile: TileUtils.Tile) -> None:
        """Add georeferencing metadata to a downloaded tile.

        This sets the CRS to EPSG:4326 because mercantile bounds are lat/lon.
        The parent RasterDataset will handle warping to the user's requested CRS.

        Args:
            filepath: path to tile file
            tile: mercantile tile for calculating bounds
        """
        bounds = TileUtils.bounds(tile)
        try:
            with rasterio.open(filepath, 'r+') as dataset:
                dataset.transform = from_bounds(
                    bounds.west,
                    bounds.south,
                    bounds.east,
                    bounds.north,
                    dataset.width,
                    dataset.height,
                )
                dataset.crs = RioCRS.from_epsg(4326)
                dataset.update_tags(
                    ns='rio_georeference',
                    georeferencing_applied='True',
                    tile_x=str(tile.x),
                    tile_y=str(tile.y),
                    tile_z=str(tile.z),
                )
        except rasterio.errors.RasterioIOError:
            warnings.warn(
                f'Could not georeference {filepath}. Not a valid raster file.',
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
        image = sample['image']
        #  C, H, W -> H, W, C
        rgb = image[0:3, :, :].permute(1, 2, 0)

        if rgb.is_floating_point() and rgb.max() > 1:
            rgb = rgb / 255.0
            rgb = rgb.clamp(0, 1)

        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(4, 4))
        ax.imshow(rgb, cmap='gray' if rgb.ndim == 2 else None)
        ax.axis('off')

        if show_titles:
            ax.set_title('Image')

        if suptitle is not None:
            plt.suptitle(suptitle)

        return fig
