The methods used in our submissions to the action spotting track
in 2022 are described in the following paper.
- [Action Spotting using Dense Detection Anchors Revisited:
Submission to the SoccerNet Challenge 2022](https://arxiv.org/abs/2206.07846).
arXiv preprint, 2022.

In this section, we explain how to reproduce the main results from
the paper.

You can also find a general overview of the 2022 SoccerNet challenges results
in the following paper.
- [SoccerNet 2022 Challenges Results](https://arxiv.org/abs/2210.02365).
In MMSports, 2022.

## Setup

The instructions below assume that you have already set up the package
and SoccerNet data, as described in the [README](README.md#setup). You
should then be able to open a Python terminal and run the commands.
They all require importing
`bin.spotting_challenge_commands`, which maps to the
[bin/spotting_challenge_commands.py](bin/spotting_challenge_commands.py)
file.

In order to run the commands below, you will need to define several
paths. For your convenience, you can define them all in the
[bin/command_user_constants.py](bin/command_user_constants.py) file.
You can also directly pass your paths in as arguments in the function
calls below. These arguments are optional. If they are not explicitly
provided, they will default to the values defined in
[bin/command_user_constants.py](bin/command_user_constants.py).

### Model zoo

You can skip the model training steps by downloading models from [our
model zoo](https://huggingface.co/yahoo-inc/spivak-action-spotting-soccernet).
Since there are several models, we suggest downloading all of them at once using:
```
git clone https://huggingface.co/yahoo-inc/spivak-action-spotting-soccernet  # requires git lfs
```
The instructions below include commands for either training the models
from scratch or for directly running inference and
evaluation using models downloaded from the zoo. If you would like to use
the models from the zoo, the instructions assume
that you have downloaded them into the models folder that is specified within
[bin/command_user_constants.py](bin/command_user_constants.py), which
defaults to `YOUR_MODELS_DIR`.

### Low memory setup

We recommend having 256GB of CPU memory available in general, though our
code can also be tweaked to work with only 64GB. In order to experiment
with a 64GB setup, set `MEMORY_SETUP = MEMORY_SETUP_64GB` in
[bin/command_user_constants.py](bin/command_user_constants.py),
or pass `memory_setup=MEMORY_SETUP_64GB` as an extra input to the
functions below that involve training models. This low memory
configuration is still experimental and is significantly slower than our
standard setup. It also does less shuffling of the input samples, which
could affect its results, though in our initial experiments we have not
noticed any detriment.

## Experiments

The nomenclature used below in the description of the experiments follows the
definitions from [our paper](https://arxiv.org/abs/2206.07846).

### Experimental protocols

This section contains a brief summary  of the different protocols we experimented with.

#### Challenge Validated protocol

The Challenge Validated protocol trains on both the _training and test_ splits, runs
validation on the validation split, and tests on the challenge split. It is useful for
finding hyperparameters and for getting a general idea of performance before submitting
results to the challenge server. However, we achieved better final results by using the
[Challenge protocol](#challenge-protocol), given that it has more training data available:
the Challenge protocol trains on all the labeled splits (training, validation, and test),
has no validation step, and then tests on the challenge split.

#### Challenge protocol

The Challenge protocol trains on all the available labeled data (train, validation
and test splits) and runs prediction over the challenge split. As such, when
using this protocol, we do not have local labels available with which to evaluate its
performance. The commands below will create, among other outputs, a zip file
that contains a set of JSON files, which you then upload to the challenge server.

#### Test protocol

The Test protocol is the standard protocol used with the SoccerNet dataset, in which
training is done using the training split, validation is done on the validation split,
and testing is done on the test split. This protocol does not involve the challenge
split, and is normally used in experiments for publications. When reporting results,
we usually average the resulting metrics over 5 different training runs.

The
[paper related to our challenge submissions](https://arxiv.org/abs/2206.07846) includes
some results using the Test protocol in order to help explain the improvements obtained
relative to our [previous paper](https://arxiv.org/abs/2205.10450), which introduced
our action spotting approach that uses dense detection anchors.

### Feature pre-processing

Feature pre-processing is required in order to run the later commands.

```python
from bin.spotting_challenge_commands import \
    command_resample_baidu, commands_normalize_resnet, print_commands

resample_command = command_resample_baidu()
print(resample_command)
# You can run the command with:
# resample_command.run()
normalize_commands = commands_normalize_resnet()
print_commands(normalize_commands)
# You can run the commands with:
# for command in normalize_commands:
#     command.run()
```

### Experiments using Combination×2 features

#### Challenge Validated protocol

```python
from bin.spotting_challenge_commands import \
    commands_spotting_challenge_validated, print_commands, \
    BAIDU_TWO_FEATURE_NAME, BAIDU_TWO_FEATURES_DIR, RUN_NAME_ZOO

# Commands for running both training and testing:
baidu_two_challenge_validated_commands = commands_spotting_challenge_validated(
    BAIDU_TWO_FEATURES_DIR, BAIDU_TWO_FEATURE_NAME)
print_commands(baidu_two_challenge_validated_commands)

# Commands for running just testing with the models from the zoo:
baidu_two_challenge_validated_commands_zoo = \
    commands_spotting_challenge_validated(
        BAIDU_TWO_FEATURES_DIR, BAIDU_TWO_FEATURE_NAME,
        run_name=RUN_NAME_ZOO, do_train=False)
print_commands(baidu_two_challenge_validated_commands_zoo)
```

#### Challenge protocol

On the challenge set, the models resulting from running the commands below are
expected to result in around 66 to 67 tight average-mAP and 76 to 77 loose
average-mAP.

```python
from bin.spotting_challenge_commands import \
    commands_spotting_challenge, print_commands, BAIDU_TWO_FEATURE_NAME, \
    BAIDU_TWO_FEATURES_DIR, RUN_NAME_ZOO

# Commands for running both training and testing:
baidu_two_challenge_commands = commands_spotting_challenge(
    BAIDU_TWO_FEATURES_DIR, BAIDU_TWO_FEATURE_NAME)
print_commands(baidu_two_challenge_commands)

# Commands for running just testing with the models from the zoo:
baidu_two_challenge_commands_zoo = commands_spotting_challenge(
    BAIDU_TWO_FEATURES_DIR, BAIDU_TWO_FEATURE_NAME,
    run_name=RUN_NAME_ZOO, do_train=False)
print_commands(baidu_two_challenge_commands_zoo)
```

#### Test protocol

In most of our experiments, we apply soft non-maximum suppression (Soft-NMS)
as the post-processing step. However, within this particular experiment
using the Test protocol, we vary the type of non-maximum suppression (NMS)
used during post-processing. We experiment with three NMS strategies:

- Soft-NMS (with a window size that was optimized to maximize tight
average-mAP on the validation set)
- Regular NMS (also with a window size that was optimized to maximize
tight average-mAP on the validation set)
- Regular NMS with a window size of 20 seconds

The commands for experimenting with different NMS approaches are generated
by supplying the `do_nms_comparison=True` flag to the `commands_spotting_test`
function, which generates the list of commands, as shown below.

```python
from bin.spotting_challenge_commands import \
    commands_spotting_test, print_commands, BAIDU_TWO_FEATURE_NAME, \
    BAIDU_TWO_FEATURES_DIR, RUN_NAME_ZOO

# Commands for running both training and testing:
baidu_two_test_commands = commands_spotting_test(
    BAIDU_TWO_FEATURES_DIR, BAIDU_TWO_FEATURE_NAME, do_nms_comparison=True)
print_commands(baidu_two_test_commands)

# Commands for running just testing with the models from the zoo:
baidu_two_test_commands_zoo = commands_spotting_test(
    BAIDU_TWO_FEATURES_DIR, BAIDU_TWO_FEATURE_NAME, do_nms_comparison=True,
    run_name=RUN_NAME_ZOO, do_train=False)
print_commands(baidu_two_test_commands_zoo)
```

### Experiments using ResNet features

#### Challenge Validated protocol

```python
from bin.spotting_challenge_commands import \
    commands_spotting_challenge_validated, print_commands, \
    RESNET_NORMALIZED_FEATURE_NAME, RESNET_NORMALIZED_FEATURES_DIR, \
    RUN_NAME_ZOO

# Commands for running both training and testing:
resnet_normalized_challenge_validated_commands = \
    commands_spotting_challenge_validated(
        RESNET_NORMALIZED_FEATURES_DIR, RESNET_NORMALIZED_FEATURE_NAME)
print_commands(resnet_normalized_challenge_validated_commands)

# Commands for running just testing with the models from the zoo:
resnet_normalized_challenge_validated_commands_zoo = \
    commands_spotting_challenge_validated(
        RESNET_NORMALIZED_FEATURES_DIR, RESNET_NORMALIZED_FEATURE_NAME,
        run_name=RUN_NAME_ZOO, do_train=False)
print_commands(resnet_normalized_challenge_validated_commands_zoo)
```

#### Challenge protocol

```python
from bin.spotting_challenge_commands import \
    commands_spotting_challenge, print_commands, \
    RESNET_NORMALIZED_FEATURE_NAME, RESNET_NORMALIZED_FEATURES_DIR, \
    RUN_NAME_ZOO

# Commands for running both training and testing:
resnet_normalized_challenge_commands = commands_spotting_challenge(
    RESNET_NORMALIZED_FEATURES_DIR, RESNET_NORMALIZED_FEATURE_NAME)
print_commands(resnet_normalized_challenge_commands)

# Commands for running just testing with the models from the zoo:
resnet_normalized_challenge_commands_zoo = commands_spotting_challenge(
    RESNET_NORMALIZED_FEATURES_DIR, RESNET_NORMALIZED_FEATURE_NAME,
    run_name=RUN_NAME_ZOO, do_train=False)
print_commands(resnet_normalized_challenge_commands_zoo)
```

#### Test protocol

```python
from bin.spotting_challenge_commands import \
    commands_spotting_test, print_commands, \
    RESNET_NORMALIZED_FEATURE_NAME, RESNET_NORMALIZED_FEATURES_DIR, \
    RUN_NAME_ZOO

# Commands for running both training and testing:
resnet_normalized_test_commands = commands_spotting_test(
    RESNET_NORMALIZED_FEATURES_DIR, RESNET_NORMALIZED_FEATURE_NAME)
print_commands(resnet_normalized_test_commands)

# Commands for running just testing with the models from the zoo:
resnet_normalized_test_commands_zoo = commands_spotting_test(
    RESNET_NORMALIZED_FEATURES_DIR, RESNET_NORMALIZED_FEATURE_NAME,
    run_name=RUN_NAME_ZOO, do_train=False)
print_commands(resnet_normalized_test_commands_zoo)
```

### Experiments using Combination×2 + ResNet features

This section's models are expected to bring only a minor improvement relative
to the models trained on only the Combination×2 features. By using the late
fusion approach from the current section, we saw an average improvement of
0.8 in the tight average-mAP metric and 1.0 in the loose average-mAP metric,
as presented in [our paper](https://arxiv.org/abs/2206.07846).

We do feature fusion by combining the confidence scores resulting from models
trained on two feature types: Combination×2 and ResNet. We run fusion
on the confidence scores, but not on the temporal displacement predictions, as
fusing the latter did not bring any improvement. The confidence scores
are combined using a weighted average of their logits, which uses a single weight
parameter. When running experiments for the Challenge protocol, which does not have
a validation set available, we use the same parameter that was found from the
Challenge Validated protocol. Thus, you should first run the commands below for
the Challenge Validated protocol in order to find the weight parameter (by
training the weighted averaging model for feature fusion). After that, you can
then run the commands below for the Challenge protocol

#### Challenge Validated protocol

Assuming you have already produced the confidence scores for the Combination×2
and ResNet features with the Challenge Validated protocol (using the commands
from previous sections), you can combine them using our weighted averaging
approach with the commands below.

```python
from bin.spotting_challenge_commands import \
    commands_spotting_challenge_validated_fusion, print_commands, RUN_NAME_ZOO

# Commands for running both validation (to find the fusion weight) and testing:
fusion_challenge_validated_commands = \
    commands_spotting_challenge_validated_fusion()
print_commands(fusion_challenge_validated_commands)

# Commands for running just testing, while reusing the
# fusion weight from the model zoo:
fusion_challenge_validated_commands_zoo = \
    commands_spotting_challenge_validated_fusion(
        run_name=RUN_NAME_ZOO, do_train=False)
print_commands(fusion_challenge_validated_commands_zoo)
```

#### Challenge protocol

Once you have trained the averaging model using the Challenge Validated protocol,
and also produced the confidence scores on the Combination×2 and ResNet features
using the commands from previous sections for the Challenge protocol, then you
can combine them using our averaging approach with the commands below.

```python
from bin.spotting_challenge_commands import \
    commands_spotting_challenge_fusion, print_commands, RUN_NAME_ZOO

# Note that this protocol does not involve a validation step. It reuses
# the fusion weight found in the Challenge Validated protocol above, since the
# Challenge protocol (used here) does not have a validation set.

# Standard commands for the Challenge protocol:
fusion_challenge_commands = commands_spotting_challenge_fusion()
print_commands(fusion_challenge_commands)

# Commands for the Challenge protocol using features computed using the
# models from the zoo.
fusion_challenge_commands_zoo = commands_spotting_challenge_fusion(
    run_name=RUN_NAME_ZOO)
print_commands(fusion_challenge_commands_zoo)
```

#### Test protocol

In order to run the feature fusion experiment commands below, you should have
first run the Test protocol experiments above, so that the confidence scores
produced using the Combination×2 and ResNet features are available to serve
as inputs to the fusion.

```python
from bin.spotting_challenge_commands import \
    commands_spotting_test_fusion, print_commands, RUN_NAME_ZOO

# Commands for running both validation (to find the fusion weight) and testing:
fusion_test_commands = commands_spotting_test_fusion()
print_commands(fusion_test_commands)

# Commands for running just testing, while reusing the
# fusion weight from the model zoo:
fusion_test_commands_zoo = commands_spotting_test_fusion(
    run_name=RUN_NAME_ZOO, do_train=False)
print_commands(fusion_test_commands_zoo)
```
