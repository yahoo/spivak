After you have finished the basic [setup](README.md#setup) steps, you can train and
test action spotting models using the commands below. In particular, you must have
created a particular set of folders and set the environment variables that point to
them, as explained [within the README](README.md#set-up-some-folders).

The commands below are meant to
give you a general idea of how to use the package. If you are specifically interested
in reproducing results from our experiments, please see
[Reproducing-results-from-the-SoccerNet-action-spotting-challenge-2022.md](Reproducing-results-from-the-SoccerNet-action-spotting-challenge-2022.md).

### Training models

Currently, we prefer to use a two-step approach to action spotting, which consists
of training two models. The first one predicts confidence scores, and the second
one predicts temporal displacements.

Train a model that predicts confidence scores using the `ResNet_TF2_PCA512` features:

```bash
CONFIDENCE_MODEL_NAME="RESNET_PCA_CONFIDENCE_LR1e-3_DWD2e-4"
./bin/train.py \
  -dt soccernet_v2 \
  -cd ./configs/soccernet_confidence/ \
  -ld ./data/labels/ \
  -fd ./data/features/resnet/ \
  -sd ./data/splits/ \
  -fn ResNET_TF2_PCA512 \
  -dc dense \
  -cw 1.0 \
  -lr 1e-3 \
  -dwd 2e-4 \
  -m $MODELS_DIR/$CONFIDENCE_MODEL_NAME
```

To monitor the training progress, start Tensorboard, pointing to the `MODELS_DIR` folder:
```bash
tensorboard --logdir="$MODELS_DIR"
```

The training script will save the best model as it runs, based on the average-mAP metric.
If you notice that the model is overfitting, you can kill the run before it finishes and
then use the best saved model.

After training the above model, you will be able to train a temporal displacement
regression model. In order to do that, first run inference on the validation
set using the confidence model. This will produce confidence scores that will be
used when training the temporal displacement model, inside the validation step.

```bash
./bin/test.py \
  -dt soccernet_v2 \
  -cd ./configs/soccernet_confidence/ \
  -ld ./data/labels/ \
  -fd ./data/features/resnet/ \
  -sd ./data/splits/ \
  -fn ResNET_TF2_PCA512 \
  -dc dense \
  -cw 1.0 \
  --test_split validation \
  --evaluate 0 \
  -m  $MODELS_DIR/$CONFIDENCE_MODEL_NAME/best_model \
  -rd $RESULTS_DIR/$CONFIDENCE_MODEL_NAME
```

We can now train the temporal displacement regression model, using the above results
in the validation step. Note that the training step below uses a different configuration
folder: `./configs/soccernet_delta/`, and a different detector setting: `dense_delta`.
Within our code, `delta` refers to the temporal displacements, which were denoted as
_d_ in our [paper](https://arxiv.org/abs/2205.10450).

```bash
DELTA_MODEL_NAME="RESNET_PCA_DELTA_LR2e-3_DWD5e-4"
./bin/train.py \
  -dt soccernet_v2 \
  -cd ./configs/soccernet_delta/ \
  -ld ./data/labels/ \
  -fd ./data/features/resnet/ \
  -sd ./data/splits/ \
  -fn ResNET_TF2_PCA512 \
  -dc dense_delta \
  -dw 1.0 \
  -lr 2e-3 \
  -dwd 5e-4 \
  -rd $RESULTS_DIR/$CONFIDENCE_MODEL_NAME \
  -m  $MODELS_DIR/$DELTA_MODEL_NAME
```

### Testing models

Inference is currently done in two steps, each using its respective model:
first the confidence scores are predicted, and then the temporal displacements.
The results are then saved and evaluated according to different metrics.

```bash
./bin/test.py \
  -dt soccernet_v2 \
  -cd ./configs/soccernet_confidence/ \
  -ld ./data/labels/ \
  -fd ./data/features/resnet/ \
  -sd ./data/splits/ \
  -fn ResNET_TF2_PCA512 \
  -dc dense \
  -cw 1.0 \
  --throw_out_delta 1 \
  --evaluate 0 \
  -m  $MODELS_DIR/$CONFIDENCE_MODEL_NAME/best_model \
  -rd $RESULTS_DIR/$DELTA_MODEL_NAME
```

```bash
./bin/test.py \
  -dt soccernet_v2 \
  -cd ./configs/soccernet_delta/ \
  -ld ./data/labels/ \
  -fd ./data/features/resnet/ \
  -sd ./data/splits/ \
  -fn ResNET_TF2_PCA512 \
  -dc dense_delta \
  -dw 1.0 \
  --throw_out_delta 0 \
  --evaluate 1 \
  -m  $MODELS_DIR/$DELTA_MODEL_NAME/best_model \
  -rd $RESULTS_DIR/$DELTA_MODEL_NAME
```

We could also just evaluate the confidence scores model directly, as below.
In this case, we will not benefit from the predicted temporal displacements,
so we should expect worse metrics at low evaluation tolerances. In most cases
there is a large improvement in the tight average-mAP when adding the temporal
displacements, whereas the loose average-mAP usually does not change much.

```bash
./bin/test.py \
  -dt soccernet_v2 \
  -cd ./configs/soccernet_confidence/ \
  -ld ./data/labels/ \
  -fd ./data/features/resnet/ \
  -sd ./data/splits/ \
  -fn ResNET_TF2_PCA512 \
  -dc dense \
  -cw 1.0 \
  --throw_out_delta 1 \
  --evaluate 1 \
  -m  $MODELS_DIR/$CONFIDENCE_MODEL_NAME/best_model \
  -rd $RESULTS_DIR/$CONFIDENCE_MODEL_NAME
```

### Creating reports with metrics across different methods

After we run evaluation with `./bin/test.py`, a pickle file is created,
containing a bunch of metrics. We can then read that pickle file and
create a report that summarizes those metrics. The
`create_comparison_report.py` script can create a report containing the
metrics from one or more methods, as long as each method has a respective
pickle file. Here is an example of how to create a report from the results
of a single method. It assumes you have already ran the previous steps
that create the results.

```bash
./bin/create_comparison_report.py \
  --pickles $RESULTS_DIR/$DELTA_MODEL_NAME/Evaluation/evaluation_run.pkl \
  --names $DELTA_MODEL_NAME \
  --out_dir $REPORTS_DIR/$DELTA_MODEL_NAME
```

Here is a run that compares the results with and without the temporal displacements .

```bash
./bin/create_comparison_report.py \
  --pickles \
    $RESULTS_DIR/$DELTA_MODEL_NAME/Evaluation/evaluation_run.pkl \
    $RESULTS_DIR/$CONFIDENCE_MODEL_NAME/Evaluation/evaluation_run.pkl \
  --names with_delta without_delta \
  --out_dir $REPORTS_DIR/add_delta/
```

#### Comparing against baseline methods

We can also create reports comparing against other published methods, as long
as they can generate results in standard SoccerNet-v2 JSON format. To do so,
we'll need to run our evaluation code on top of the JSON files, in order to
generate the required pickle files. Below are some steps you can follow in
order to try it out.

In order to generate JSON result files for baseline methods, consult the
documentation for each specific method. For convenience, here are some commands
that run prediction using the NetVLAD++ and CALF models, creating the
desired output JSON files.

```bash
DATA_DIR="$BASE_CODE_DIR/spivak/data/"  # This should point to wherever you have your data.
cd $DATA_DIR
cp -r features/resnet/ features/merged/
# The rsync below does not access the network, it just merges the contents of the two directories.
rsync -r labels/ features/merged/
# Run testing for two methods.
cd $BASE_CODE_DIR
git clone https://github.com/SoccerNet/sn-spotting
pip install SoccerNet
# First, run the temporally aware pooling code (NetVLAD++)
cd $BASE_CODE_DIR/sn-spotting/Benchmarks/TemporallyAwarePooling/
# This will take a while as it will train a new model that uses the ResNET_TF2_PCA512 features.
# (It took around 40 minutes on a V100 for us.) After training, the script runs testing and
# saves the results in the models/NetVLAD++_PCA512/outputs_test/ folder.
python src/main.py \
  --SoccerNet_path=$DATA_DIR/merged/ \
  --features=ResNET_TF2_PCA512.npy \
  --model_name=NetVLAD++_PCA512
# Second, run the context-aware loss function code (CALF)
cd $BASE_CODE_DIR/sn-spotting/Benchmarks/CALF/
# This will use a pre-trained model and save results in the outputs/ folder.
python src/main.py \
  --SoccerNet_path=$DATA_DIR/merged/ \
  --features=ResNET_TF2_PCA512.npy \
  --num_features=512 \
  --model_name=CALF_benchmark \
  --test_only
```

As mentioned above, the necessary pickle files are generated when running `bin/test.py`.
In order to generate similar pickle files for the other methods, we first generate
their outputs in the standard SoccerNet-v2 JSON format as above. Then, we use those JSON
files to generate the evaluation pickles. Here are the examples of how to generate the
evaluation pickle files directly from the JSON files.

```bash
./bin/evaluate_spotting_jsons.py \
  --results_dir $BASE_CODE_DIR/sn-spotting/Benchmarks/TemporallyAwarePooling/models/NetVLAD++_PCA512/outputs_test/ \
  --features_dir ./data/features/resnet/ \
  --labels_dir ./data/labels/ \
  --splits_dir ./data/splits/ \
  --config_dir ./configs/soccernet_confidence/ \
  --output_dir $RESULTS_DIR/NetVLAD++_PCA512/
```

```bash
./bin/evaluate_spotting_jsons.py \
  --results_dir $BASE_CODE_DIR/sn-spotting/Benchmarks/CALF/outputs/ \
  --features_dir ./data/features/resnet/ \
  --labels_dir ./data/labels/ \
  --splits_dir ./data/splits/ \
  --config_dir ./configs/soccernet_confidence/ \
  --output_dir $RESULTS_DIR/CALF_PCA512/
```

The commands above assume that the JSON files are available inside the `--results_dir` folders.

Finally, we can generate a report comparing all the above methods.

```bash
./bin/create_comparison_report.py \
  --pickles \
    $RESULTS_DIR/CALF_PCA512/Evaluation/evaluation_aggregate.pkl \
    $RESULTS_DIR/NetVLAD++_PCA512/Evaluation/evaluation_aggregate.pkl \
    $RESULTS_DIR/$DELTA_MODEL_NAME/Evaluation/evaluation_run.pkl \
    $RESULTS_DIR/$CONFIDENCE_MODEL_NAME/Evaluation/evaluation_run.pkl \
  --names  CALF  NetVLAD++  DU_With_Delta  DU_Without_Delta \
  --out_dir $REPORTS_DIR/multiple
```

#### Visualizing results for each video

We can also create visualizations of the specific predictions for each video.
Here is an example that creates visualizations for a single video.

```bash
GAME_DIR="italy_serie-a/2014-2015/2015-04-29 - 21-45 Juventus 3 - 2 Fiorentina/"
./bin/create_visualizations.py \
  --input_dir "./data/videos_224p/$GAME_DIR" \
  --output_dir "$VISUALIZATIONS_DIR/$DELTA_MODEL_NAME/$GAME_DIR" \
  --results_dir "$RESULTS_DIR/$DELTA_MODEL_NAME/$GAME_DIR" \
  --label_map ./configs/soccernet_confidence/spotting_labels.csv \
  --no_create_videos
```

In order to create the visualizations for all the videos in the test set, you can use
the command below. In order to speed it up, we use `--no_debug_graphs` below, though
the resulting visualizations will be less detailed.

```bash
./bin/create_visualizations.py \
  --input_dir ./data/videos_224p/ \
  --output_dir $VISUALIZATIONS_DIR/$DELTA_MODEL_NAME/ \
  --results_dir $RESULTS_DIR/$DELTA_MODEL_NAME/ \
  --label_map ./configs/soccernet_confidence/spotting_labels.csv \
  --no_create_videos \
  --no_debug_graphs
```

<!--

### Feature generation and transformation

#### Generating new features

SoccerNet has a set of pre-computed features available, including ResNet-based
features and the features provided by the Baidu team. All these features can be
downloaded using the SoccerNet pip package, as mentioned
[here](#get-the-soccernet-data). If instead you would like to generate new features
from video files, make sure to get the feature extraction model files as follows.

```bash
FEATURE_EXTRACTION_MODELS_DIR="YOUR_FEATURE_EXTRACTION_MODELS_DIR"
mkdir -p $FEATURE_EXTRACTION_MODELS_DIR
cd $FEATURE_EXTRACTION_MODELS_DIR
wget "https://github.com/SilvioGiancola/SoccerNetv2-DevKit/raw/main/Features/average_512_TF2.pkl"
wget "https://github.com/SilvioGiancola/SoccerNetv2-DevKit/raw/main/Features/pca_512_TF2.pkl"
wget "https://storage.googleapis.com/tensorflow/keras-applications/resnet/resnet152_weights_tf_dim_ordering_tf_kernels.h5"
```

To generate new features, first extract the basic raw features (before PCA transform)
using the low-resolution videos as follows.

```bash
./bin/extract_features.py TODO-RESNET-FEATURES!!!
```

The feature extraction step takes about 15 hours on a machine with a single
V100 GPU. If you would like to run PCA in order to reduce the dimensionality
of the features, you can create and apply the PCA transform as follows.

```bash
./bin/make_pca_transform.py \
  --features_dir $DATA_DIR \
  --features ResNet
```

```bash
./bin/make_pca_features.py \
  --features_dir $DATA_DIR \
  --pca "${DATA_DIR}/pca_transform_ResNet_512_whiten_False.pkl" \
  --features ResNet
```

Creating the PCA transform takes one or two hours on a machine
with 32 cores. Applying the transform to the complete dataset takes around
5 minutes.

#### Creating normalizers for features

We found that for ResNet features, we achieved good results with the `max_abs` and `min_max`
normalization approaches. We did not experiment with normalization on the Baidu Combination
features.

```bash
./bin/create_normalizer.py \
  --features_dir ./data/features/resnet/ \
  --splits_dir ./data/splits/ \
  --normalizer max_abs \
  --features ResNET_TF2 \
  --out_path ~/challenge/normalizers/resnet_max_abs.pkl
```

```bash
./bin/create_normalizer.py \
  --features_dir ./data/features/resnet/ \
  --splits_dir ./data/splits/ \
  --normalizer min_max \
  --features ResNET_TF2 \
  --out_path ~/challenge/normalizers/resnet_min_max.pkl
```

#### Transforming features using normalization, interpolation in time, and concatenation

For the challenge, we resampled to Baidu Combination features from the original 1 frame per second
to 2 frames per second, using the command below.

```bash
./bin/transform_features.py \
  --input_dirs ./data/features/baidu/ \
  --output_dir ./data/features/baidu_fps2.0_interpolate/ \
  --input_feature_names baidu_soccer_embeddings \
  --factors 2.0 \
  --resampling interpolate
```

To experiment with concatenating the ResNet and Baidu Combination features, the following
command could be used.

```bash
./bin/transform_features.py \
  --input_dirs ./data/features/baidu/ ./data/features/resnet/ \
  --normalizers identity ~/challenge/normalizers/resnet_max_abs.pkl \
  --output_dir ./data/features/baidu_resnet_max_abs_fps2.0_interpolate \
  --input_feature_names baidu_soccer_embeddings ResNET_TF2 \
  --factors 2.0 1.0 \
  --resampling interpolate
```

#### Projecting features to lower dimensions (for speed in later steps)

We experimented a bit with feature projection, in order to reduce the number of dimensions for speed purposes.
Our experiments were not very conclusive, but we found it more practical to not run the projection and instead
use late feature fusion, as described [here](#late-feature-fusion).

```bash
./bin/project_features.py \
  -dt soccernet_v2_challenge_validation \
  -cd ./configs/soccernet_challenge_confidence/ \
  -ld ./data/labels/ \
  -fd ./data/features/baidu_fps2.0_interpolate/ \
  -sd ./data/splits/ \
  -fn transformed \
  -dc dense \
  -lwc 1.0 \
  -cs 112 \
  -bb unet \
  -m ~/challenge/models/validation/Baidu_FPS2.0_INTERP_CS112_PWC0.03_BS256_LR5e-5_DWD5e-5_RHO0.0_MU0.0_UNET_CHALLENGE_VALIDATION_M1/best_model \
  -rd ./data/features/baidu_fps2.0_projected_relu_confidence_validation/
```

```bash
./bin/project_features.py \
  -dt soccernet_v2_challenge_validation \
  -cd ./configs/soccernet_challenge_delta/ \
  -ld ./data/labels/ \
  -fd ./data/features/baidu_fps2.0_interpolate/ \
  -sd ./data/splits/ \
  -fn transformed \
  -dc dense_delta \
  -lwdl 1.0 \
  -cs 112 \
  -bb unet \
  -m ~/challenge/models/validation/Baidu_FPS2.0_INTERP_CS112_PWDL1.0_BS256_LR2e-3_DWD1e-3_RHO0.1_MU0.0_UNET_DD_CHALLENGE_VALIDATION_M1/best_model \
  -rd ./data/features/baidu_fps2.0_projected_relu_delta_validation/
```

```bash
./bin/project_features.py \
  -dt soccernet_v2_challenge_validation \
  -cd ./configs/soccernet_challenge_confidence/ \
  -ld ./data/labels/ \
  -fd ./data/features/resnet_max_abs_fps2.0/ \
  -sd ./data/splits/ \
  -fn transformed \
  -dc dense \
  -lwc 1.0 \
  -cs 112 \
  -bb unet \
  -m ~/challenge/models/validation/Resnet_max_abs_FPS2.0_INTERP_CS112_PWC0.03_BS256_LR5e-4_DWD2e-4_RHO0.02_MU0.0_UNET_CHALLENGE_VALIDATION/best_model \
  -rd ./data/features/resnet_max_abs_projected_relu_confidence_validation/
```

```bash
./bin/project_features.py \
  -dt soccernet_v2_challenge_validation \
  -cd ./configs/soccernet_challenge_delta/ \
  -ld ./data/labels/ \
  -fd ./data/features/resnet_max_abs_fps2.0/ \
  -sd ./data/splits/ \
  -fn transformed \
  -dc dense_delta \
  -lwdl 1.0 \
  -cs 112 \
  -bb unet \
  -m ~/challenge/models/validation/Resnet_max_abs_FPS2.0_INTERP_CS112_PWDL1.0_BS256_LR5e-4_DWD2e-4_RHO0.7_MU0.0_UNET_DD_CHALLENGE_VALIDATION/best_model \
  -rd ./data/features/resnet_max_abs_projected_relu_delta_validation/
```

### Late feature fusion

#### Converting output probabilities into a feature files format

```bash
./bin/create_features_from_results.py \
  --input_dirs \
    ~/challenge/results/validation/Baidu_FPS2.0_INTERP_CS112_PWC0.03_BS256_LR5e-5_DWD5e-5_RHO0.0_MU0.0_UNET_CHALLENGE_VALIDATION_M1/ \
    ~/challenge/results/validation/Resnet_max_abs_FPS2.0_INTERP_CS112_PWC0.03_BS256_LR5e-4_DWD5e-5_RHO0.02_MU0.0_UNET_CHALLENGE_VALIDATION/ \
  --output_name confidence \
  --output_dir ./data/features/results_confidence_validation
```

```bash
./bin/create_features_from_results.py \
  --input_dirs \
    ~/challenge/results/validation/Baidu_FPS2.0_INTERP_CS112_PWDL1.0_BS256_LR2e-3_DWD1e-3_RHO0.1_MU0.0_UNET_DD_CHALLENGE_VALIDATION_M1/ \
    ~/challenge/results/validation/Resnet_max_abs_FPS2.0_INTERP_CS112_PWDL1.0_BS256_LR5e-4_DWD2e-4_RHO0.7_MU0.0_UNET_DD_CHALLENGE_VALIDATION/ \
  --output_name delta \
  --output_dir ./data/features/results_delta_validation
```

#### Creating averaging predictors (who average the output probabilities)

```bash
./bin/create_averaging_predictor.py \
  -sd ./data/splits/ \
  -ld ./data/labels/ \
  -cd ./configs/soccernet_challenge_confidence/ \
  -fd ./data/features/results_confidence_validation/ \
  -dt soccernet_v2_challenge_validation \
  -fn confidence \
  -dc averaging_confidence \
  -m ~/models/challenge/validation/Combined_confidence_validation_logit.pkl
```

```bash
./bin/create_averaging_predictor.py \
  -sd ./data/splits/ \
  -ld ./data/labels/ \
  -cd ./configs/soccernet_challenge_delta/ \
  -fd ./data/features/results_delta_validation/ \
  -dt soccernet_v2_challenge_validation \
  -fn delta \
  -dc averaging_delta \
  --nmsdecay linear \
  -rd ~/challenge/results/validation/Baidu_FPS2.0_INTERP_CS112_PWDL1.0_BS256_LR2e-3_DWD1e-3_RHO0.1_MU0.0_UNET_DD_CHALLENGE_VALIDATION_M1/ -m ~/models/challenge/validation/Combined_delta_soft_validation.pkl
```

#### Testing the averaging predictors

```bash
./bin/test.py \
  -dt soccernet_v2_challenge \
  -cd ./configs/soccernet_challenge_confidence/ \
  -ld ~/SoccerNet/data/labels/ \
  -fd ~/challenge/data.features/outputs_confidence_validation_unlabeled/ \
  -sd ./data/splits/ \
  -fn confidence \
  -dc averaging_confidence \
  -lwc 1.0 \
  -tod 1 \
  -cs 112 \
  -bb unet \
  --testsplit unlabeled \
  -m  ~/challenge/models/validation/Combined_confidence_validation_logit/ \
  -rd ~/challenge/results/validation/Combined_confidence_validation_unlabeled
```

### Running models on new videos

To run models on new videos, you will need to set up the feature extraction model
as described [here](#generating-new-features).

First, get a video and create a couple of folders to store the intermediate
results of the experiment. Set the `EXPERIMENT_DIR` and `VIDEOS_DIR` variables
as below, and move your video into your `VIDEOS_DIR`.

```bash
EXPERIMENT_DIR="YOUR_NEW_FOLDER_FOR_THIS_EXPERIMENT"
mkdir -p $EXPERIMENT_DIR
mkdir -p $EXPERIMENT_DIR/videos
mv YOUR_TEST_VIDEO $EXPERIMENT_DIR/videos
mkdir -p $EXPERIMENT_DIR/splits
```

TODO: explain what to put in the `splits_dir`.

Then, make sure you have the feature extraction models directory `FEATURE_EXTRACTION_MODELS_DIR`
as described [here](#generating-new-features). Now, run prediction:

```bash
./bin/predict_on_videos.py \
  --config_dir ./configs/soccernet_v2 \
  --features ResNet_TF2 \
  --features_models_dir $FEATURE_EXTRACTION_MODELS_DIR \
  --model $MODELS_DIR/$TRAINED_MODEL_NAME/best_model \
  --input_dir $EXPERIMENT_DIR/videos \
  --results_dir $EXPERIMENT_DIR/results/ \
  --features_dir $EXPERIMENT_DIR/features/ \
  --splits_dir $EXPERIMENT_DIR/splits/
```

Currently, the `predict_on_videos.py` script above only accepts a confidence score model
as in the example above. We should incorporate the two-phase model into that script (to
predict the temporal displacement), but we haven't gotten to fixing that yet. Also, the
only feature currently supported is `ResNet_TF2`.

#### Visualizing the results from the new videos

```bash
./bin/create_visualizations.py \
  --input_dir ~/spivak/videos/ \
  --output_dir ~/spivak/output/ \
  --results_dir ~/spivak/recognized/ \
  --label_map ./configs/soccernet_v2/labels.csv \
  --no_create_videos
```

### Sanity-checking our action spotting evaluation code

To make sure our evaluation code is performing similarly to the original evaluation code,
we can run evaluation using the original codebase and make sure the results match
with the evaluation from our code.

#### Check against SoccerNet/sn-spotting/ (for SoccerNet-v3)

The results from spotting_evaluation.py should match the evaluation results from the
[SoccerNet spotting codebase](https://github.com/SoccerNet/sn-spotting). First,
evaluate a model using our codebase, while saving the relevant JSON files.
For example:

```bash
./bin/test.py \
  -dt soccernet_v2 \
  -cd ./configs/soccernet_confidence/ \
  -ld ./data/labels/ \
  -fd ./data/features/resnet/ \
  -sd  ./data/splits/ \
  -fn ResNET_TF2_PCA512 \
  -dc dense \
  --test_save_spotting_jsons 1 \
  -m $MODEL_DIR/best_model \
  -rd $JSON_RESULTS_DIR
```

The [SoccerNet spotting codebase](https://github.com/SoccerNet/sn-spotting)
provides standalone evaluation code that reads the results from files
in a specific JSON format. We can simply run their evaluation code on our
saved results. First, we should make sure that their package is installed
and up-to-date:

```bash
pip install -U SoccerNet
```

Then, we can just run the SoccerNet evaluation code on our results:

```bash
python sn-spotting/Evaluation/EvaluateSpotting.py \
  --SoccerNet_path ./data/labels/ \
  --Predictions_path $JSON_RESULTS_DIR \
  --metric loose
```

```bash
python SoccerNetv2-DevKit/Evaluation/EvaluateSpotting.py \
  --SoccerNet_path ./data/labels/ \
  --Predictions_path $JSON_RESULTS_DIR
  --metric tight
```

In order to view the mAP for each specific matching tolerance, we can add
a print statement within the SoccerNet evaluation code. Modify
`/usr/local/lib/pythonV.V/site-packages/SoccerNet/Evaluation/ActionSpotting.py`,
so that the function `average_mAP` has the additional print statement:
`print(f"mAP (for each tolerance): {mAP}")`.

### Profiling the validation step

The validation step can sometimes be a bit expensive. Since it runs in a separate process,
it is a bit tricky to profile it. We have a script specifically for that purpose.

```bash
./bin/profile_validation.py \
  -dt soccernet_v2_challenge_validation \
  -cd ./configs/soccernet_challenge_confidence/ \
  -ld ./data/labels/ \
  -fd ./data/features/baidu_fps2.0_interpolate/ \
  -sd ./data/splits/ \
  -fn transformed \
  -bs 256 \
  -dc dense \
  -sm uniform \
  -lr 1e-4 \
  -dwd 1e-4 \
  -se 10 \
  -ve 10 \
  -lwc 1.0 \
  -pwc 0.03 \
  -f 2.0 \
  -cs 112 \
  -bb unet \
  -m ~/models/challenge/validation/Baidu_FPS2.0_INTERP_CS112_PWC0.03_BS256_LR1e-4_DWD1e-4_UNET_CHALLENGE_VALIDATION_REPLICATE1
```

```bash
./bin/profile_validation.py \
  -dt soccernet_v2_challenge_validation \
  -cd ./configs/soccernet_challenge_delta/ \
  -ld ./data/labels/ \
  -fd ./data/features/baidu_fps2.0_interpolate/ \
  -sd ./data/splits/ \
  -fn transformed \
  -bs 256 \
  -dc dense_delta \
  -sm uniform \
  -lr 1e-4 \
  -dwd 1e-4 \
  -se 10 \
  -ve 10 \
  -lwdl 1.0 \
  -pwdl 1.0 \
  -f 2.0 \
  -cs 112 \
  -bb unet \
  -rd ~/results/challenge/validation/Baidu_FPS2.0_INTERP_CS112_PWC0.03_BS256_LR5e-5_DWD5e-5_RHO0.0_MU0.1_UNET_CHALLENGE_VALIDATION/ \
  -m ~/models/challenge/validation/Baidu_FPS2.0_INTERP_CS112_PWDL1.0_BS256_LR1e-3_DWD0.0_RHO0.0_MU0.0_UNET_DD_CHALLENGE_VALIDATION/
```

```bash
./bin/profile_validation.py \
  -dt soccernet_v2 \
  -cd ./configs/soccernet_confidence/ \
  -ld ./data/labels/ \
  -fd ./data/features/resnet/ \
  -sd ./data/splits/ \
  -fn ResNET_TF2_PCA512 \
  -bs 256 \
  -dc dense \
  -sm uniform \
  -lr 1e-3 \
  -dwd 2e-4 \
  -se 10 \
  -ve 10 \
  -lwc 1.0 \
  -cs 112 \
  -pwc 0.03 \
  -bb unet \
  -m ~/models/challenge/PCA_CS112_PWC0.03_BS256_LR1e-3_DWD2e-4_UNET
```

-->
