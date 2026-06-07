# SUFST - DV.ConeDetection

Repo for the SUFST driverless cone detection model

## File structure
```
└─ main.py          - Home screen gui, ML model controller, feature definition
└─ train.py         - ML model training script
└─ data.yaml        - Data file containing key ML model parameters
└─ requirements.txt - Python package/library requirements
└─ data
  └─ images
    └─ train        - Folder for training images
    └─ val          - Folder for validation images
  └─ lables
    └─ train        - Folder for training labels
    └─ val          - Folder for validation labels
```

## Cloning The Repo

- Install git
  - https://git-scm.com/download/win
  - Ensure git is added to PATH
- Clone git repo locally:

```
git clone https://github.com/sufst/DV.ConeDetection.git
```
## How To Use This App

- Make sure all dependencies are installed:

```
pip install -r requirements.txt
```

- Run main.py to launch the UI

### Home Page Navigation

#### Top Control Bar Options

- 'Insert Image'
  - Allows selection of locally stored image into either train or validation datasets
- 'Sample Video'
  - Allows sampling of locally stored videos into either train or validation datasets
- 'Labelled Unlabelled'
  - Opens the annotations tool for all currently unlabelled images
- 'Redraw All'
  - Opens the annotations tool for all currently stored images
  - Overwrites current annotations/labels
- 'Train Model'
  - Opens the model training page, allowing you to start model training with the current database
  - Live view of all model training outputs

#### Per-Image Control Options
- 'Image'
  - Image name, can be used to find image file in the images directory
- 'Set'
  - What data set the image is applied to
  - Either 'train' or 'val' (validation)
- 'Status
  - What is the annotation status of the image
  - Either 'Labelled' or 'Unlabelled'
- 'Redraw'
  - Opens the annotation tool for that image only
  - Overwrites that image's current annotations/labels
- 'Visualise'
  - Displays that images current annotations/labels
- 'Transfer'
  - Transfers the current data set the image is part of
  - Eg. from 'train' to 'val'
- 'Delete'
  - Deletes image from the dataset


### Labeling/Annotating Training Data

The annotator tool is used for labeling training data, classifying defined regoins, which the ML model uses for object detection.

The annotator tool can be opened by:
- Clicking the 'Redraw' button on the home page
- Automatically opens when you sample a video
  - This has the added feature of a slider which allows you to scrub to specific frames

Labels can be drawn by clicking and dragging, this creates a box drawn on the screen.

#### Keyboard Shortcuts:

- 0 - Cone label selection
- 1 - Non-cone label selection
- S - Save current labels/annotations for thta image
- Q - Quit or close the annotation viewer and return to the home page























