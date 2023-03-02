# spivak: <ins>sp</ins>orts <ins>i</ins>ndeed <ins>v</ins>ideo <ins>a</ins>nalysis <ins>k</ins>it
> A toolkit for automatic analysis of sports videos.

## Background

This package implements methods for action spotting and camera
shot segmentation on the
[SoccerNet dataset](https://www.soccer-net.org/). In addition,
it provides a set of tools for visualizing results, and for
visualizing various metrics that are computed against the ground
truth annotations.

Most of the code in this repository deals with implementing the
following two papers, which focus on the task of action spotting
on the SoccerNet dataset.
- [Temporally Precise Action Spotting in Soccer Videos Using
Dense Detection Anchors](https://arxiv.org/abs/2205.10450).
In ICIP, 2022.
- [Action Spotting using Dense Detection Anchors Revisited:
Submission to the SoccerNet Challenge 2022](https://arxiv.org/abs/2206.07846).
arXiv preprint, 2022.

The action spotting method that is implemented here
came in first place in the SoccerNet Challenge 2022. You can
read more about the 2022 challenge and results in the following paper:
- [SoccerNet 2022 Challenges Results](https://arxiv.org/abs/2210.02365).
In MMSports, 2022.

## Setup

### Install

Our models depend on TensorFlow, though this package also includes
evaluation and visualization code that does not. We've currently
only tested our code using TensorFlow 2.3.0, which is thus currently
specified in the [setup.py](setup.py) file. Some visualization
scripts depend on ffmpeg via PyAV (`av` pip package). The rest of
the dependencies are specified in [setup.py](setup.py) and can
likely be directly installed using pip as follows.

```bash
BASE_CODE_DIR="YOUR_BASE_CODE_DIR"  # Wherever the spivak repo is located.
cd $BASE_CODE_DIR/spivak  # The root folder, which contains setup.py.
pip install -e .
```

For the heavier sets of features, the code assumes that a good
amount of CPU RAM is available. We recommend having 256GB or more
available. Our input data pipeline is responsible for consuming most
of the CPU memory and can likely be tweaked to consume less at the cost
of some speed. The relevant code is in
[tf_dataset.py](spivak/models/tf_dataset.py), which is based on
[tf.data](https://www.tensorflow.org/guide/data).

### Get the SoccerNet data

You will most likely want to download the following set of
precomputed features and labels using SoccerNet's pip package.
Please see detailed instructions at <https://www.soccer-net.org/data>.
- ResNet features
(filenames used for downloading: `["1_ResNET_TF2.npy", "2_ResNET_TF2.npy"]`).
- ResNet features projected using PCA
(filenames used for downloading: `["1_ResNET_TF2_PCA512.npy", "2_ResNET_TF2_PCA512.npy"]`).
- Baidu features
(these were also denoted _Combination_ in our papers; filenames
used for downloading:
`["1_baidu_soccer_embeddings.npy", "2_baidu_soccer_embeddings.npy"]`).
- Action spotting labels
(filename used for downloading: `["Labels-v2.json"]`).
- Camera shot segmentation labels
(filename used for downloading: `["Labels-cameras.json"]`).

If you would like to use our video-specific visualization functionality,
you will also need to get the
[SoccerNet videos](https://www.soccer-net.org/data#h.ov9k48lcih5g).
The low-resolution version of the videos should be enough for
visualization purposes.

### Optionally, set up some convenient folders

In order to make it easier to follow our guides, we suggest that you created some
folders to store your models and results, as follows.

```bash
MODELS_DIR="YOUR_MODELS_DIR"
mkdir -p $MODELS_DIR
RESULTS_DIR="YOUR_RESULTS_DIR"
mkdir -p $RESULTS_DIR
REPORTS_DIR="YOUR_REPORTS_DIR"
mkdir -p $REPORTS_DIR
VISUALIZATIONS_DIR="YOUR_VISUALIZATIONS_DIR"
mkdir -p $VISUALIZATIONS_DIR
# We recommend setting FEATURES_DIR to "data/features" as below, since many
# commands in our guides use "data/features" directly.
FEATURES_DIR="data/features"
mkdir -p $FEATURES_DIR
```

We also recommend that you create the symbolic links described below, so
that you can easily access the downloaded SoccerNet data. The symbolic links
will point from the [data/](data) folder to the folders containing the actual
downloaded data.

```bash
cd data/  # This folder will initially just contain the splits/ folder.
ln -s YOUR_LABELS_FOLDER  labels  # For the Labels-v2.json and/or the Labels-cameras.json files.
ln -s YOUR_FEATURES_RESNET_FOLDER  features/resnet  # For the ResNet-based features.
ln -s YOUR_FEATURES_BAIDU_FOLDER  features/baidu  # For the Baidu Combination features.
ln -s YOUR_VIDEOS_224P_FOLDER  videos_224p  # For the low-resolution videos.
```

## Action spotting usage

Please see [Action-spotting-usage.md](Action-spotting-usage.md).

## Camera shot segmentation usage

Stay tuned! Instructions for running the camera shot segmentation code should be coming soon.

## Citations

If you found our models and code useful, please consider citing our works:

```
@inproceedings{soares2022temporally,
  author={Soares, Jo{\~a}o~V.~B. and Shah, Avijit and Biswas, Topojoy},
  booktitle={International Conference on Image Processing (ICIP)},
  title={Temporally Precise Action Spotting in Soccer Videos Using Dense Detection Anchors},
  year={2022},
  pages={2796-2800},
  doi={10.1109/ICIP46576.2022.9897256}
}

@article{soares2022action,
  title={Action Spotting using Dense Detection Anchors Revisited: Submission to the {SoccerNet} {Challenge} 2022},
  author={Soares, Jo{\~a}o~V.~B. and Shah, Avijit},
  journal={arXiv preprint arXiv:2206.07846},
  year={2022}
}
```

## Contribute

Please refer to [the contributing.md file](Contributing.md) for information about how to
get involved. We welcome  issues, questions, and pull requests.

Please be aware that we (the maintainers) are currently busy with other projects, so it
may take some days before we are able to get back to you. We do not foresee big changes
to this repository going forward.

## Maintainers

- Joao Soares: jvbsoares@yahooinc.com
- Avijit Shah: avijit.shah@yahooinc.com

## License

This project is licensed under the terms of the
[Apache 2.0](http://www.apache.org/licenses/LICENSE-2.0.html)
open source license. Please refer to [LICENSE](LICENSE) for
the full terms.

## Acknowledgments

We thank the [SoccerNet team](https://www.soccer-net.org/team) for making their datasets
available and organizing the series of related challenges. We also thank them for
making their code available under open source licenses.
