
This is a toolkit for comprehensive semi-automated vectorisation of historic Ordnance Survey maps series with a built-in feedback loop to iteratively improve prediction capabilities in new context areas. Each feature is an individual binary segmentation model, and any combination of features can be selected and compiled from map patch predictions into a GeoPackage of full map vector layers. Land parcels are derived separately, by partitioning the boundary-line network using the recorded apportionment points as seeds (a point-seeded watershed), so no per-parcel model training is required.

The pipeline targets:
- Solid boundary lines between land parcels with a modified version of [this improved U-Net architecture](). Model weights are retrievable from [here](). 
- Dashed bounary lines using a custom-made U-Net architecture with a pretrained EfficientNetB0 encoder that learns to redraw dashed boundary lines as if they were continuous using a Gaussian-blurred version of the linear annotation training set. 
- Polygonal features (e.g. water courses, build footprints, land cover classes)with [MapSAM](https://arxiv.org/abs/2411.06971). 
- Text with the use of the [MapTextPipeline runner](https://github.com/maps-as-data/MapTextPipeline) with model weights built from David Rumsey historical map collection ([add link to these]()). 

Download's and Pre-requisites:
- This pipeline was built in Windows Subsystem for Linux (WSL) because of package and library agreements. 
- The anaconda environments are set up to work best with an NVIDIA GPU.
- Set up the following conda environments:
```python
conda env create -f envs/maptools.yml   # annotation, patchify, and vectorising
conda env create -f envs/polygons.yml   # used for both polygon (MapSAM) and text steps 
conda env create -f envs/lines.yml      # used for both solid and dashed line steps (U-Net models)
Running through the pipeline:
```
- Clone the necessary repositories into the `models/` folder
- Add model weights into the `models/base/` folder from [my huggingface](https://huggingface.co/ww357)

File structure:
``` bash
repo root/
│
├── data/
│   ├── raw/<SHEET>/<SHEET>.tif        (made by step 00, or [YOU] supply your own GeoTIFF scan)
│   ├── parcel_points/*.gpkg            ← [YOU] land parcel centroids
│   │
│   ├── tiles/<series>/                 (step 00: downloaded-tile cache, shared, safe to delete)
│   ├── areas/<AREA>/aoi.json           (step 00: what was requested + resume state)
│   ├── keys/<series>.png               (step 00 --key: map key images to consult when annotating)
│   ├── map_area_masks/                 (made by step 00 from the tile coverage, or drawn with draw-mask)
│   ├── patches/                        (created by step 01)
│   ├── annotations/                    (created by step 02)
│   ├── predictions/                    (created by steps 03-04)
│   └── outputs/<SHEET>.gpkg            (final GeoPackage created in step 05)
│
├── models/
│   ├── ImprovedLinearUNet/             ← [YOU] clone from GitHub (https://github.com/ww357/Improved-Linear-U-Net)
│   ├── MapSAM/                         ← [YOU] clone from GitHub (https://github.com/ww357/MapSAM)
│   ├── MapTextPipeline/                ← [YOU] clone from GitHub (https://github.com/maps-as-data/MapTextPipeline)
│   │
│   ├── base/
│   │   ├── unet/model_weights.weights.h5
│   │   ├── MapSAM/origional_weights/sam_vit_b_01ec64.pth  
│   │   └── MapTextPipeline/rumsey-finetune.pth      
│   │
│   └── finetuned/                      (generated model weights will land here in steps 03_finetune and 06_feedback)
│
├── steps/                              (pipeline scripts — do not edit)
├── envs/maptools.yml                   ← use to build the maptools conda env
└── config.yaml                         ← pipeline parameters
```

## Step 00 - Download an area of an NLS map series

Gets a chosen area of a georeferenced National Library of Scotland (NLS) Ordnance Survey layer and writes it as pipeline-ready sheets (British National Grid GeoTIFF + map-area mask), so steps 01 onwards work unchanged. Runs in `maptools`.

Series available now (`python run.py download --series help`):
- `6in_1st` - six-inch, England and Wales, 1st edition (1840s-1880s). NLS holds this as one tileset **per county**; see "County-layer series" below.
- `6in_2nd` - six-inch, Great Britain, 2nd edition (1888-1915). One seamless layer.
- `10k_natgrid` - National Grid 1:10,000, Great Britain (1949-1975). One seamless layer.

```bash
conda activate maptools
python run.py download --series help                       # series available (edit/add them in config.yaml > series)

# 1. Preview first: sheets, tiles and download size. Nothing is downloaded.
python run.py download --area Porlock --series 6in_2nd \
       --bbox 285000 143000 291000 149000 --dry-run         # --bbox = BNG metres: minE minN maxE maxN

# 2. Download (same command without --dry-run)
python run.py download --area Porlock --series 6in_2nd --bbox 285000 143000 291000 149000

# 3. Next: patchify the sheet it names at the end
python run.py patchify --sheet Porlock --mask
```

**Choosing the area** (one of): `--bbox MINE MINN MAXE MAXN` (BNG metres), `--centre E N --size-km 5`, `--latlon-bbox MINLON MINLAT MAXLON MAXLAT`, or `--polygon area.gpkg` (any vector file; the map outside the polygon is masked out).

**County-layer series (`6in_1st`).** Because each county is a separate tileset, an area that crosses a county border is stitched automatically:
1. The downloader asks NLS which sheets cover your area (each sheet's county, survey and publication years) and which counties they belong to.
2. Each county's tiles are used **only inside that county** (its historic boundary, c.1860, plus a 75 m tolerance). OS left a neighbouring county's land blank on a sheet even where the sheet extends over it, so using the whole sheet would paste blank paper over real map.
3. The seams therefore follow the county boundaries. Only the tiles actually needed are downloaded.

`--dry-run` prints the plan: counties, sheet counts, survey and publication years, who supplies how much of your area, and the tile count.

```bash
python run.py download --area Border --series 6in_1st --bbox 305000 117000 313000 123000 --dry-run
python run.py download --area Border --series 6in_1st --bbox 305000 117000 313000 123000 --plan      # choose in a window
python run.py download --area Border --series 6in_1st --bbox ... --exclude Devonshire               # leave a county out
```
- `--plan` opens a window (like `draw-mask`): your area over the sheet outlines and county boundaries, with a live preview of who supplies what. Click a county in the list to cycle automatic / preferred / excluded; Enter accepts, Esc cancels.
- `--exclude COUNTY` drops a county (its part of the area is left white and masked out). `--prefer COUNTY` lets it win in the thin strip where two counties both have map. County names are the spellings in the plan table.
- Each sheet also gets `<SHEET>_source_index.tif` (pixel value = which county supplied it; the key is in `source.json` along with every source sheet's survey and publication years). `data/areas/<AREA>/footprints.geojson` keeps the sheet outlines and county boundaries for resuming offline.
- The county lookup is `steps/00_download/nls_tilesets.yaml`, generated from NLS's map viewer. If NLS adds or renames layers: `python run.py download --refresh-index`.
- The map-area mask for county series comes from the sheet outlines and county boundaries (the tiles themselves are opaque, so tile transparency cannot show where there is no map).
- Neighbouring counties' paper tints differ a little, so a faint colour change can be visible at a seam.

**Sheets and names.** A small area becomes one sheet named after `--area`. An area larger than `download.block_px` pixels per side (default 8000) is split into equal, slightly overlapping blocks named `<AREA>_r01c01`, `<AREA>_r01c02`, ... Each block is an ordinary sheet (`--sheet <AREA>_r01c01`), so RAM stays bounded for any area size. With `python run.py`, `--sheet <AREA>` runs the command once per block for `reproject`, `patchify`, `status`, and every `predict-*` and `vectorise-*` step (one failed block doesn't stop the rest). Annotating, training and the interactive tools need one named block. Each sheet's `data/raw/<SHEET>/source.json` records the series, zoom, attribution and the block's un-padded core for the later merge.

**Zoom and resolution.** `--zoom` (default 16, set in `config.yaml > download`). Each zoom step quarters the ground size of a pixel and quadruples the tiles; the dry run prints the ground resolution (zoom 16 is about 1.5 m/px in England and Wales, 17 about 0.75, 18 about 0.37). The output is written at the tiles' native ground resolution unless you pass `--resolution` (e.g. `0.5`). Zoom is capped at each series' maximum (6in_2nd: 17, 10k_natgrid: 16).

**Safe to stop and resume.** Press Ctrl+C, close the window or lose the network at any point, then run the same command again, or just `python run.py download --area Porlock`. Finished tiles and sheets are kept (a sheet's `.tif` is only written once all its tiles are in), failed tiles are retried with backoff, and a block is never built with missing tiles. Add `--recheck-missing` to re-ask the server about tiles it earlier said do not exist. Downloads over 20,000 new tiles (about 1 GB) need `--yes`.

**Map key images.** `python run.py download --key <series-id or key-tileset-URL>` stitches an NLS key (legend) into `data/keys/<name>.png` to consult while working out your annotation classes, e.g. `--key 10k_natgrid` (its key is already set). Add a series' key URL as `key_tiles` in `config.yaml` and `annotate` will point you to the image.

**Label taxonomy.** `config.yaml > taxonomy` (top of the file) defines the classes you annotate: name, line or polygon, which model learns it, and a note. `annotate` offers exactly these labels in labelme. Edit it as you learn the map symbology.

Tiles are NLS's; keep the attribution (`Courtesy of the National Library of Scotland`) with anything you publish. NLS describes these layers as for local desktop use, free for non-commercial use and licensed for commercial use by tiles requested (which includes repeated automated use), so check which applies to you before pulling large areas (maps@nls.uk for bulk requests).

Running through the pipeline:
```python
## Step 00 - Download (see above)
python run.py download --area MyArea --series 6in_2nd --bbox MINE MINN MAXE MAXN

## Step 01 - Patchify
conda activate maptools 
python "steps/01_patchify/draw_mask.py" --sheet MapSheetName
# run this to interactively make mask of map area on the document if necessary 
# to reduce patches for inference (or this mask can be made in another programme):
python "steps/01_patchify/patchify.py" --sheet MapSheetName
# slice GeoTIFF into 512px patches (use --mask flag if no mask is auto-found)

## Step 02 - Annotate
python "steps/02_annotate/annotate.py" --sheet MapSheetName
# open labelme to draw boundary lines and feature polygons
python "steps/02_annotate/export_masks.py" --sheet MapSheetName
# convert labelme JSON to binary mask PNGs per feature label

## Step 03 - Fine-tune (Boundaries - U-Net)
conda activate lines
python "steps/03_finetune/lines/train.py" --sheet MapSheetName --name map_v1
# fine-tune boundary U-Net, checkpoints on path-F1

## Step 03 - Fine-tune (Features - MapSAM)
conda activate polygons
python "steps/03_finetune/polygons/train.py" --sheet MapSheetName --feature FeatureName --name map_v1
# fine-tune SAM DoRA weights for one feature class (repeat per feature)

## Step 04 - Predict
conda activate lines
python "steps/04_predict/lines/predict.py" --sheet MapSheetName
# run U-Net on all patches, skips manually annotated ones
conda activate polygons
python "steps/04_predict/polygons/predict.py" --sheet MapSheetName --feature FeatureName1 FeatureName2 FeatureName3 ...
# run MapSAM on all patches for listed features & run text prediction if "text" is specified

## Step 05 - Vectorise
conda activate maptools
python "steps/05_vectorise/lines/vectorise.py" --sheet MapSheetName # stitch and vectorise boundary lines
python "steps/05_vectorise/polygons/vectorise.py" --sheet MapSheetName # stitch and vectorise polygons
python "steps/05_vectorise/text/text_to_vector.py" --sheet MapSheetName # stitch and vectorise text

## Parcels - point-seeded watershed (run after the boundary lines are vectorised)
conda activate polygons
python "steps/04_predict/parcels/parcel_segment.py" --sheet MapSheetName
# partition the stitched boundary raster into parcels, seeded by apportionment points
python "steps/05_vectorise/parcels/parcel_vectorise.py" --sheet MapSheetName
# join apportionment attributes and write the 'parcels' layer to the GeoPackage

## Step 06 - Feedback loop
conda activate lines
python "steps/06_feedback/lines/feedback.py" --sheet MapSheetName

# Output: data/outputs/MapSheetName.gpkg
```

## Acknowledgements

The boundary U-Net architecture is based on:

> Ran et al. (2022). Raster Map Line Element Extraction Method Based on Improved U-Net Network.
> *ISPRS International Journal of Geo-Information*, 11(8), 439.
> https://doi.org/10.3390/ijgi11080439
> GitHub: https://github.com/FutureuserR/Raster-Map

**MapSAM** — feature segmentation model used for buildings, water, and other polygon features.

> Xue Xia, Daiwei Zhang, Wenxuan Song, Wei Huang, Lorenz Hurni.
> *MapSAM: Adapting Segment Anything Model for Automated Feature Detection in Historical Maps.*
> https://github.com/xiaxue-ethz/MapSAM

**MapTextPipeline** — text detection and recognition pipeline used in step 06.

> Based on DNTextSpotter: Yu Xie et al.
> *DNTextSpotter: Arbitrary-Shaped Scene Text Spotting via Improved Denoising Training.*
> arXiv:2408.00355 (2024). https://github.com/yyyyyxie/DNTextSpotter
>
> MapText fork (maps-as-data): https://github.com/maps-as-data/MapTextPipeline

**MapReader** — map patch management and MapTextPipeline runner used in step 06.

> maps-as-data / The Alan Turing Institute.
> https://github.com/maps-as-data/MapReader