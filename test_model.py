import os

import matplotlib.pyplot as plt
import segmentation_models_pytorch as smp
import torch
from pyproj import CRS
from torch.utils.data import DataLoader

from torchgeo.datasets import OpenAerialMap, OpenStreetMap
from torchgeo.samplers import RandomGeoSampler

WORK_DIR = "banepa_data"
BBOX = [85.514668, 27.628367, 85.528875, 27.638514]

ZOOM_LEVEL = 18
stac_id = "62d86c65d8499800053796c4"

# Define classes.
# Class 0 = Background
# Class 1 = Building
OSM_CLASSES = [{"name": "building", "selector": [{"building": "*"}]}]


def main():
    print("Initializing OpenAerialMap...")
    oam = OpenAerialMap(
        paths=os.path.join(WORK_DIR, "oam"),
        bbox=BBOX,
        zoom=ZOOM_LEVEL,
        download=True,
        image_id=stac_id,
        crs=CRS.from_epsg(3857),
    )

    # 2. Label Dataset

    print("Initializing OpenStreetMap...")
    osm = OpenStreetMap(
        paths=os.path.join(WORK_DIR, "osm"),
        bbox=BBOX,
        classes=OSM_CLASSES,
        download=True,
    )

    # 3. Intersection
    # This enables the automatic rasterization inside OpenStreetMap to match
    dataset = oam & osm

    # 4. Sampler

    chip_size_px = 256
    print(oam.res)
    chip_size_deg = chip_size_px * oam.res[0]  # pixel count * degrees/pixel

    print(f"Sampling {chip_size_px}px chips (approx {chip_size_deg:.6f} degrees)...")
    sampler = RandomGeoSampler(dataset, size=chip_size_deg, length=20)

    dataloader = DataLoader(
        dataset,
        sampler=sampler,
        batch_size=4,
    )

    print("Dataloader initialized.")
    print(dataloader)

    # 5. Model
    model = smp.Unet(
        encoder_name="resnet18",
        encoder_weights="imagenet",
        in_channels=3,
        classes=2,  # 0: Background, 1: Building
    )

    # 6. Training Loop
    print("Starting Training...")
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    criterion = torch.nn.CrossEntropyLoss()

    for batch in dataloader:
        image = batch["image"]  # Shape: (B, 3, H, W)
        print(image.shape)
        mask = batch["mask"]  # Shape: (B, H, W)
        print(mask.shape)

        # Normalize image to [0, 1]
        image = image.float() / 255.0
        mask = mask.long()

        # Forward pass
        output = model(image)
        loss = criterion(output, mask)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        print(f"Loss: {loss.item():.4f}")
    print("Training complete.")
    # 7. Visualize Result
    fig, ax = plt.subplots(1, 2, figsize=(10, 5))
    ax[0].imshow(image[0].permute(1, 2, 0))
    ax[0].set_title("OAM Image")
    ax[1].imshow(mask[0], cmap="gray")
    ax[1].set_title(f"OSM Rasterized Mask\n(Auto-aligned)")

    plt.show()


if __name__ == "__main__":
    main()
