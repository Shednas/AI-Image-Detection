# Raw and unseen datasets

AI Image Detection, Sandesh Thapa, UON ID 24813026

Two separate downloads. Neither is in the repository, and neither is needed to
check a published number: every figure is already in `research/results/`. See
`research/README.md`.

You need these only to rebuild the splits from scratch, or to run the unseen
generator evaluations.

---

## Which zip do you need

| Zip | Contains | Needed for |
|---|---|---|
| `raw.zip` | the seven training sources | rebuilding the splits with `split_dataset.py` |
| `unseen.zip` | Chameleon and MNW | every `test_unseen_*` script |

The processed training data is a third, separate zip with its own README. If you
only want to retrain, take that one instead: it is already split and
preprocessed, and far smaller.

---

## Where to unzip

Both unzip into `research/data/`, which already exists in the repository. Merge
into it rather than replacing it, since `data/metadata/` is tracked.

```text
AI-Image-Detection/
`-- research/
    `-- data/
        |-- raw/          <- raw.zip
        `-- unseen/       <- unseen.zip
```

---

## Expected layout, raw

File counts as they should appear. These count every file, including the
`.gitkeep` markers already in the repository, which is what the status check
below counts.

```text
data/raw/
|-- cifake/
|   |-- FAKE/                                    59,859
|   `-- REAL/                                    50,001
|-- coco/
|   |-- train2017/                              118,288
|   `-- val2017/                                  5,001
|-- flickr30k/                                   31,785
|-- forensynths/
|   `-- FAKE/                                    10,001
|-- genimage/
|   |-- BigGAN/train/ai/                        105,685
|   |-- MidJourney/train/ai/                      2,515
|   |-- stable_diffusion_v_1_4/train/ai/          7,044
|   `-- stable_diffusion_v_1_5/train/ai/          7,061
|-- imagenet/                                    50,001
`-- unsplash/                                     4,003
```

Grouped as the status check reports them:

| Source | Files |
|---|---|
| ImageNet | 50,001 |
| COCO | 123,289 |
| Unsplash | 4,003 |
| GenImage | 122,305 |
| ForenSynths | 10,001 |
| CIFAKE | 109,860 |
| Flickr30k | 31,785 |

### The Unsplash manifest

`extract_unsplash.py` looks for one exact filename:

```text
data/raw/unsplash/photos.csv000
```

Not `photos.csv`, and not in any subfolder. Despite the name it is tab
separated, and the script reads it as such. Without it the download step exits
with `Metadata file not found`.

---

## Expected layout, unseen

```text
data/unseen/
|-- chameleon/
|   |-- fake/                                    11,170
|   `-- real/                                    14,863
`-- MNW/                                         11,251 files
                                                 across 45 generator folders
```

MNW is distributed as a git repository, so clone it rather than copying files:

```powershell
git clone https://github.com/nsail-lab/MNW.git data/unseen/MNW
```

The images are committed directly, so a plain clone is enough.

---

## Checking what you have

```powershell
python scripts/download_dataset.py --status-only
```

All seven raw sources should read `OK` with counts that **should match** the
table above.

**Two limits worth knowing.** The check looks at `data/raw/` only. It has no
knowledge of `data/unseen/`, so it can report everything `OK` while every
`test_unseen_*` script still fails on a missing Chameleon or MNW. It also
reports `OK` for any count above zero, so a half-finished download passes. Read
the counts, do not just look for `OK`.

There is no equivalent check for `data/unseen/`. Count the folders above by hand.

---

## Sizes

| | Approximate |
|---|---|
| raw | 60GB |
| unseen | 12GB |

Both are far larger than the processed training zip, which is about 430MB.
