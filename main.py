"""This is a script that can serve as an initialization for participating in the
ELIAS Bosch AI for Building Optimisation prediction competition.

It contains simple versions of the essential components for
* loading and preprocessing data,
* defining a simple torch pairs dataset for causal prediction (i.e., using only past to predict future),
* defining a simple toy example model and training it,
* and evaluating and creating the submission file of that model on the test input data.

Note that all of the components are just starting points, and many aspects can still be improved:
* take more or all of the covariate variables as predictors (there are hundreds), not just the current few example ones
* also include the 2024 data from source Zenodo, see `README.md` or competition page for details
* some timeseries may rather need linear interpolation instead of the forward fill (ffill) that is currently used
  (reason for the ffill is that, at least for some time series, the recodring is change-triggered).
* and of course the model and training which lies at the core of the competition, and is currently just a toy example
"""

# Copyright (c) 2025 Robert Bosch GmbH
# All rights reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import csv
from datetime import datetime
import glob
from itertools import chain
import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
import torchvision
from tqdm import tqdm

torch.manual_seed(0)
torch.set_default_device("cuda" if torch.cuda.is_available() else "cpu")
print(torch.get_default_device())
torch.set_default_dtype(
    torch.float64
)  # with lower than float64 precision currently the eventual timestamps may be off


DATA_DIR = "data"
OUTPUTS_DIR = "outputs"
TRAIN_DATA_FILE_PATHS = list(
    chain(
        glob.glob(
            f"{DATA_DIR}/kaggle_dl/RBHU-2025-01/RBHU/**/*.parquet", recursive=True
        ),
        glob.glob(
            f"{DATA_DIR}/kaggle_dl/RBHU-2025-02/RBHU/**/*.parquet", recursive=True
        ),
        glob.glob(
            f"{DATA_DIR}/kaggle_dl/RBHU-2025-03/RBHU/**/*.parquet", recursive=True
        ),
        glob.glob(
            f"{DATA_DIR}/kaggle_dl/RBHU-2025-04/RBHU/**/*.parquet", recursive=True
        ),
        glob.glob(
            f"{DATA_DIR}/kaggle_dl/RBHU-2025-05/RBHU/**/*.parquet", recursive=True
        ),
    )
)
TEST_START_DATETIME = datetime(2025, 6, 1)  # start of test set
TEST_INPUT_DATA_FILE_PATHS = list(
    chain(
        glob.glob(
            f"{DATA_DIR}/kaggle_dl/RBHU-2025-05/RBHU/**/*.parquet", recursive=True  # just for the lag
        ),
        glob.glob(
            f"{DATA_DIR}/kaggle_dl/RBHU-2025-06/RBHU/**/*.parquet", recursive=True
        ),
        glob.glob(
            f"{DATA_DIR}/kaggle_dl/RBHU-2025-07/RBHU/**/*.parquet", recursive=True
        ),
    )
)
RESAMPLE_FREQ_MIN = 10  # the frequency in minutes to resample the raw irregularly sampled timeseries to, using ffill
EPS = 1e-6
TARGET_VARIABLE_NAME = "B205WC000.AM02"  # the target variable to be predicted
EXAMPLE_PREDICTOR_VARIABLE_NAMES = [
    "B205WC000.AM01",  # a supply temperature chilled water
    "B106WS01.AM54",  # a external temperature
]  # example predictor variables
SUBMISSION_FILE_PATH = f"{OUTPUTS_DIR}/submission_file.csv"
SUBMISSION_FILE_TARGET_VARIABLE_COLUMN_NAME = "TARGET_VARIABLE"
SUBMISSION_FILE_DATETIME_FORMAT = '%Y-%m-%d_%H:%M:%S'


def simple_load_and_resample_data(
    data_file_paths, generate_sample_plots=None, save_load_df=None
):
    """Load measurement timeseries data from the original data folder structure, turn the original irregularly sampling
    into a regular, common timeline, and join everything into one multivariate timeseries dataframe.

    Important: might have to be adapted for actual models for competition.

    Args:
        data_file_paths: List of file paths to the files containing the monthly measurement data for each sensor,
            should be in the right monthly order and overall original structure.
        generate_sample_plots: If provided, a list of sensor names to plot the timeseries of, and save the plots.
        save_load_df: If provided, the path to save the preprocessed data as a parquet file, and load it from there
            if it already exists.

    Returns:
        A pandas DataFrame containing the multivariate timeseries data, with the columns being the sensors,
        and the index being the datetime.
    """

    save = True
    if save_load_df and os.path.exists(save_load_df):
        print(f"Loading preprocessed data from {save_load_df} ...")
        multivariate_timeseries_df = pd.read_parquet(save_load_df)
        save = False

    else:
        # load, preprocess and group timeseries per sensor (i.e., for each sensor, got multiple periods):
        print("Start loading and preprocessing data ...")
        dataframes_per_sensor = {}
        for path in tqdm(data_file_paths):
            name = path.split("/")[-1].replace(".parquet", "")
            if not name in dataframes_per_sensor.keys():
                dataframes_per_sensor[name] = []
            df_orig = pd.read_parquet(path, columns=["time", "data"])
            df_orig = df_orig.rename(columns={"data": name})
            df_orig["time"] = pd.to_datetime(df_orig["time"])
            df_orig = df_orig.set_index("time", drop=True)
            dataframes_per_sensor[name].append(df_orig)

        # join them along time dimension into one timeseries per sensor, and interpolate to regular frequency sample:
        print("Start joining data ...")
        regular_dataframe_per_sensor = {}
        for name, dfs in tqdm(dataframes_per_sensor.items()):
            # the above forward fill is justified by the fact that prior knowledge indicates, that measurement recordings
            # seem to be change-tirggered:
            resample_freq_str = f"{str(RESAMPLE_FREQ_MIN)}min"
            regular_dataframe_per_sensor[name] = (
                pd.concat(dfs, axis=0)
                .resample(resample_freq_str)
                .ffill()  # interpolate via forward fill
            )

        # join into one multivariate timeseries dataframe, i.e., along column axis:
        multivariate_timeseries_df = pd.concat(
            regular_dataframe_per_sensor.values(), join="outer", axis=1
        ).ffill()  # forward fill again, due to different ends of concatenants otherwise leading to NaNs

    if generate_sample_plots:
        n_plots = len(generate_sample_plots)
        fig, axs = plt.subplots(
            n_plots,
            1,
            sharex=True,
            dpi=500,
        )
        plt.title("Input data timeseries")
        for i, col in enumerate(generate_sample_plots):
            axs[i].plot(multivariate_timeseries_df[col], label=col, linewidth=0.75)
            axs[i].tick_params(axis="x", labelrotation=90)
            axs[i].legend(fontsize="small")
        plt.savefig(f"{OUTPUTS_DIR}/input_data_sample_timeseries_plot.png")
        plt.close(fig)

    if save_load_df and save:
        multivariate_timeseries_df.to_parquet(save_load_df)

    print("Done.")

    return multivariate_timeseries_df


def simple_eval_and_submission_creation(
    loader,
    model,
    loss_fn=None,
    generate_timeseries_prediction=False,
    save_fig=None,
    create_submission_df=False,
):
    """Evaluate the model on the given data loader and optionally create a coherent prediction on the full dataset
    underlying the dataloader, and save it as a figure.

    Important: this might have to be adapted for actual models for competition.

    Args:
        loader: The data loader to run and evaluate the model on. It gives (x, y) pairs, i.e., input variable and target
            variable, where the to-be-predicted target variable value is allowed to be nan, for the sake of just
            producing a prediction for submission (without having the ground truth target variable value available right
            here).
        model: The model to evaluate.
        loss_fn: The loss function to use for evaluation. If None, then this function runs in the mode of just producing
            the prediction for submission.
        generate_timeseries_prediction: Whether to generate a timeseries prediction from the model.
            If True, the predictions will be concatenated and returned as a single tensor.
        save_fig: If provided, the path to save the figure of the predictions.
            If None, no figure will be saved.
        create_submission_df: If True, a DataFrame will be created from the predictions,
            with the index being the datetime of the timestamp in the first column of the tensor `y_pred`. Use this
            option to afterwards create a submission file that can then be submitted to the competition.
            If a datetime is provided, the DataFrame will only contain entries from that datetime onwards. 

    Returns:
        A dictionary containing the evaluation results. If `generate_timeseries_prediction` is True, it will contain
        the predictions as a tensor under the key "ys_pred". If `create_submission_df` is True, it will also contain
        the predictions as a DataFrame under the key "ys_pred_df". If `loss_fn` is provided, it will also
        contain the average loss under the key "avg_loss".
    """
    res = {}

    model.eval()
    ys_true_list = []
    ys_pred_list = []
    if loss_fn is not None:
        cum_batch_loss_list = []
    with torch.no_grad():
        for x_true, y_true in loader:
            assert ~x_true.isnan().any()
            y_pred = model(x_true)
            assert (y_pred[:, 0] == y_true[:, 0]).all(), (
                "Annotated timestamp of prediction does not match to-be-predicted timestamp."
            )
            y_true_core, y_pred_core = (
                y_true[:, 1:],
                y_pred[:, 1:],
            )  # Exclude timestamp from loss calculation
            ys_true_list.append(y_true)
            ys_pred_list.append(y_pred)
            if loss_fn is not None:
                assert loss_fn.reduction == "mean", (
                    "Loss function must have reduction='mean', because the calculation assumes so."
                )
                batch_size = y_true.shape[0]
                assert not torch.isnan(y_true).any(), (
                    "y_true contains NaN values, which is not allowed for loss computation. "
                    "Are you sure the loader is for train/vali, and not submission, "
                    "where the y is just an empty dummy entry?"
                )
                loss = loss_fn(y_pred_core, y_true_core)
                cum_loss = (
                    loss * batch_size
                )  # multiply by batch size to be invariant to batch size
                cum_batch_loss_list.append(cum_loss.unsqueeze(0))
        if loss_fn is not None:
            assert loader.drop_last is False, (
                "Loader must not drop last incomplete batch, otherwise the loss calculation does not work properly."
            )
            avg_loss = torch.cat(cum_batch_loss_list).sum() / len(loader.dataset)
            res["avg_loss"] = avg_loss

    if generate_timeseries_prediction:
        assert isinstance(loader.sampler, torch.utils.data.sampler.SequentialSampler), (
            "Loader must not shuffle data, otherwise the concatenated batch predictions "
            "do not form a properly ordered time series prediction."
        )
        ys_true = torch.cat(ys_true_list, dim=0)
        ys_pred = torch.cat(ys_pred_list, dim=0)
        res["ys_pred"] = ys_pred

        if save_fig is not None or create_submission_df:
            ys_true_df = pd.DataFrame(
                ys_true[:, 1].cpu().numpy(),
                index=pd.to_datetime(ys_true[:, 0].cpu().numpy(), unit="s"),
                columns=["Value(true)"],
            )
            ys_pred_df = pd.DataFrame(
                ys_pred[:, 1].cpu().numpy(),
                index=pd.to_datetime(ys_pred[:, 0].cpu().numpy(), unit="s"),
                columns=[SUBMISSION_FILE_TARGET_VARIABLE_COLUMN_NAME],
            )
            ys_pred_df.index.name = "time"

            if save_fig is not None:
                fig, ax = plt.subplots(dpi=500)
                ax.plot(ys_true_df, label="true", alpha=0.75, linewidth=0.75)
                ax.plot(ys_pred_df, label="pred", alpha=0.75, linewidth=0.75)
                ax.tick_params(axis="x", labelrotation=90)
                ax.legend(fontsize="small")
                plt.savefig(save_fig)
                plt.close(fig)
            if create_submission_df:
                if create_submission_df == True:
                    res["ys_pred_df"] = ys_pred_df
                elif isinstance(create_submission_df, datetime):
                    res["ys_pred_df"] = ys_pred_df[ys_pred_df.index >= create_submission_df]
    return res


def simple_feature_dataset(
    full_multivariate_timeseries_df, add_dummy_y=False, normalize=False
):
    """Create a torch dataset from the multivariate timeseries dataframe, intended for causal prediction (just use past
    to predict future), consisting of samples of predictor features
    (past sequence up to time step t and daytime features) as well as target variable (value of to-be-predicted
    variable/timeseries, at step t plus forecast ahead), each annotatetd with the timestamp of the prediction
    (i.e., the time at which the target variable is to be predicted).

    Important: to be adapted for actual models for competition.

    Args:
        multivariate_timeseries_df: The multivariate timeseries dataframe of measurements.
        add_dummy_y: If True, the to-be-predicted target variable will be set to NaN. This option can be used to create
            a dataset also for the test input data, where no target variable values are available, but nonetheless
            a column needs to exist for the target variable.
            If False, the target variable will be used as is, i.e., the dataset can be used for training or validation
            where the target variable is available in the raw data.
        normalize: normalize selected columns of the timeseries data. 
            if True, take mean and std from the data (and return that info), 
            if a dict, use the contained mean and std.
    Returns:
        A torch dataset containing pairs of input features and target variable values. Note that both, input features'
        and target variable's first entry is the timestamp of the prediction, i.e., the time at which the target
        variable is to be predicted.
    """

    info = {}

    input_seq_len = int(60 / RESAMPLE_FREQ_MIN) * 24
    input_seq_step = 6
    stride = 1  # step size for sliding window
    predict_ahead = int(60 / RESAMPLE_FREQ_MIN) * 0  # lead / ahead time to forecast, in steps of RESAMPLE_FREQ_MIN

    # restrict to only relevant/valid data:
    timeseries_df = full_multivariate_timeseries_df[
        [
            col
            for col in EXAMPLE_PREDICTOR_VARIABLE_NAMES + [TARGET_VARIABLE_NAME]
            if col in full_multivariate_timeseries_df.columns
        ]
    ]
    first_valid_idx = timeseries_df.notna().all(axis=1).idxmax()
    timeseries_df = timeseries_df.loc[first_valid_idx:]

    if add_dummy_y:
        timeseries_df[TARGET_VARIABLE_NAME] = np.nan

    # extract and add some faetures:
    datetime_list = timeseries_df.index.to_pydatetime()
    timeseries_df["timestamp"] = [dtime.timestamp() for dtime in datetime_list]
    timeseries_df["minute_of_day"] = (
        timeseries_df.index.hour * 60 + timeseries_df.index.minute
    )
    timeseries_df["day_of_week"] = timeseries_df.index.dayofweek
    timeseries_df["day_of_year"] = timeseries_df.index.dayofweek
    timeseries_df["yeartime_sin"] = np.sin(
        2 * np.pi * timeseries_df["day_of_year"] / 365
    )
    timeseries_df["yeartime_cos"] = np.cos(
        2 * np.pi * timeseries_df["day_of_year"] / 365
    )
    timeseries_df["daytime_sin"] = np.sin(
        2 * np.pi * timeseries_df["minute_of_day"] / (24 * 60)
    )
    timeseries_df["daytime_cos"] = np.cos(
        2 * np.pi * timeseries_df["minute_of_day"] / (24 * 60)
    )

    column_names = timeseries_df.columns

    if normalize:
        if normalize == True:
            mean = timeseries_df.mean()
            std = timeseries_df.std()
            info.update(
                {
                    "timeseries_df_mean": mean,
                    "timeseries_df_std": std,
                }
            )
        elif isinstance(normalize, dict):
            mean = normalize["timeseries_df_mean"]
            std = normalize["timeseries_df_std"]

        def normalization_fn(x, col):
            return (x - torch.tensor(mean[col])) / (torch.tensor(std[col]) + EPS)
    else:

        def normalization_fn(x, col):
            return x

    timeseries_tensor = torch.tensor(
        timeseries_df.to_numpy(), dtype=torch.get_default_dtype()
    )
    data = timeseries_tensor

    X = []
    Y = []
    for i in range(0, data.shape[0] - input_seq_len - predict_ahead, stride):
        timestamp = data[
            i + input_seq_len + predict_ahead, column_names.get_loc("timestamp")
        ].unsqueeze(0)

        example_predictor_variable_0 = normalization_fn(
            data[
                i : i + input_seq_len : input_seq_step,
                column_names.get_loc(EXAMPLE_PREDICTOR_VARIABLE_NAMES[0]),
            ],
            EXAMPLE_PREDICTOR_VARIABLE_NAMES[0],
        )
        example_predictor_variable_1 = normalization_fn(
            data[
                i : i + input_seq_len : input_seq_step,
                column_names.get_loc(EXAMPLE_PREDICTOR_VARIABLE_NAMES[1]),
            ],
            EXAMPLE_PREDICTOR_VARIABLE_NAMES[1],
        )
        daytime_sin = data[
            i + input_seq_len + predict_ahead, column_names.get_loc("daytime_sin")
        ].unsqueeze(0)
        daytime_cos = data[
            i + input_seq_len + predict_ahead, column_names.get_loc("daytime_cos")
        ].unsqueeze(0)
        day_of_week = torch.nn.functional.one_hot(
            data[
                i + input_seq_len + predict_ahead, column_names.get_loc("day_of_week")
            ].to(dtype=torch.long),
            num_classes=7,
        )
        yeartime_sin = data[
            i + input_seq_len + predict_ahead, column_names.get_loc("yeartime_sin")
        ].unsqueeze(0)
        yeartime_cos = data[
            i + input_seq_len + predict_ahead, column_names.get_loc("yeartime_cos")
        ].unsqueeze(0)

        X.append(
            torch.cat(
                [
                    timestamp,
                    example_predictor_variable_0,
                    example_predictor_variable_1,
                    day_of_week,
                    yeartime_sin,
                    yeartime_cos,
                    daytime_sin,
                    daytime_cos,
                ]
            )
        )

        target_variable = data[
            i + input_seq_len + predict_ahead,
            column_names.get_loc(TARGET_VARIABLE_NAME),
        ].unsqueeze(0)

        Y.append(torch.cat([timestamp, target_variable]))

    X = torch.stack(X)
    Y = torch.stack(Y)

    dataset = TensorDataset(X, Y)

    return dataset, info


def simple_model_and_train(train_loader, vali_loader, loss_fn):
    """Define a simple prediction model and train it on the given training data loader.

    Important: to be adapted for actual models for competition.
    """

    class SimpleAIFBOModel(nn.Module):
        def __init__(self, input_size):
            super().__init__()
            self.mlp = torchvision.ops.MLP(
                in_channels=input_size,
                hidden_channels=[128, 128, 128, 1],
                norm_layer=nn.LayerNorm,
                # dropout=0.1,
            ).to(dtype=torch.get_default_dtype())

        def forward(self, x):
            timestamp_of_prediction, x_core = x[:, :1], x[:, 1:]
            y_core = self.mlp(x_core)
            return torch.cat([timestamp_of_prediction, y_core], dim=1)

    x, _ = next(iter(train_loader))
    input_size = (
        x.shape[-1] - 1
    )  # Get the input size from the first batch, subtract 1 for the timestamp

    model = SimpleAIFBOModel(input_size=input_size)
    optimizer = torch.optim.Adam(model.parameters(), lr=2.5e-4)

    for epoch in range(200):
        model.train()
        # train_losses = []
        cum_batch_loss_list = []
        for _, (x_true, y_true) in enumerate(train_loader):
            assert ~x_true.isnan().any() and ~y_true.isnan().any()
            y_pred = model(x_true)
            assert (y_pred[:, 0] == y_true[:, 0]).all(), (
                "Annotated timestamp of prediction does not match to-be-predicted timestamp."
            )
            y_true_core, y_pred_core = (
                y_true[:, 1:],
                y_pred[:, 1:],
            )  # Exclude timestamp from loss calculation
            loss = loss_fn(y_pred_core, y_true_core)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # train_losses.append(loss.item())
            batch_size = y_true.shape[0]
            cum_loss = (
                loss * batch_size
            )  # multiply by batch size to be invariant to batch size
            cum_batch_loss_list.append(cum_loss.unsqueeze(0))
        # avg_train_loss_running = sum(train_losses) / len(train_losses)
        avg_train_loss_running = torch.cat(cum_batch_loss_list).sum() / len(
            train_loader.dataset
        )

        avg_train_loss_epoch = simple_eval_and_submission_creation(
            train_loader, model, loss_fn
        )["avg_loss"]
        avg_vali_loss = simple_eval_and_submission_creation(
            vali_loader, model, loss_fn
        )["avg_loss"]

        print(
            f"Epoch: {epoch:04d}."
            f"Train Loss: {avg_train_loss_running:.5f}. "
            f"Train Loss Epoch: {avg_train_loss_epoch:.5f}. "
            f"Vali Loss: {avg_vali_loss:.5f}"
        )

    return model


if __name__ == "__main__":
    # Load raw data and prepare it into multivariate dataframes, and create dir for later outputs:
    os.makedirs(OUTPUTS_DIR, exist_ok=True)
    full_train_df = simple_load_and_resample_data(
        TRAIN_DATA_FILE_PATHS,
        generate_sample_plots=[TARGET_VARIABLE_NAME] + EXAMPLE_PREDICTOR_VARIABLE_NAMES,
        save_load_df=f"{OUTPUTS_DIR}/preproc_full_train_df.parquet",
    )
    test_input_df = simple_load_and_resample_data(
        TEST_INPUT_DATA_FILE_PATHS,
        save_load_df=f"{OUTPUTS_DIR}/preproc_test_input_df.parquet",
    )
    tzinfo = full_train_df.index.tzinfo

    # Turn it into torch datasets for simple prediction from past to future, with simple features:
    full_train_dataset, full_train_dataset_info = simple_feature_dataset(
        full_train_df, add_dummy_y=False, normalize=True
    )
    test_input_dataset, _ = simple_feature_dataset(
        test_input_df, add_dummy_y=True, normalize=full_train_dataset_info
    )

    # Turn it into data loaders for training, validation, and submission (where submission loader differs in that
    # it has no target variable values, i.e. y):
    len_full_train_dataset = len(full_train_dataset)
    split_index = int(0.9 * len_full_train_dataset)
    train_dataset = torch.utils.data.Subset(full_train_dataset, range(split_index))
    vali_dataset = torch.utils.data.Subset(
        full_train_dataset, range(split_index, len_full_train_dataset)
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=64,
        shuffle=False,
        generator=torch.Generator(device=torch.get_default_device()),
    )
    vali_loader = DataLoader(vali_dataset, batch_size=64, shuffle=False)
    test_input_loader = DataLoader(test_input_dataset, batch_size=64, shuffle=False)

    # Define loss function, model, and perform training:
    loss_fn = nn.MSELoss()
    model = simple_model_and_train(train_loader, vali_loader, loss_fn)

    # Evaluate model on train, validation, and test data, create plots, and create final prediction submission
    # dataframe (with datetime annotation), and save it as submission file CSV:
    res_eval_train = simple_eval_and_submission_creation(
        train_loader,
        model,
        loss_fn,
        generate_timeseries_prediction=True,
        save_fig=f"{OUTPUTS_DIR}/plot_train.png",
    )
    res_eval_vali = simple_eval_and_submission_creation(
        vali_loader,
        model,
        loss_fn,
        generate_timeseries_prediction=True,
        save_fig=f"{OUTPUTS_DIR}/plot_vali.png",
    )
    res_eval_test_input = simple_eval_and_submission_creation(
        test_input_loader,
        model,
        loss_fn=None,  # no loss function for test set evaluation, because there is no ground truth in raw data
        generate_timeseries_prediction=True,
        save_fig=f"{OUTPUTS_DIR}/plot_test.png",
        create_submission_df=TEST_START_DATETIME,
    )
    test_prediction_df = res_eval_test_input["ys_pred_df"]
    
    test_prediction_df_for_csv = test_prediction_df.copy()
    test_prediction_df_for_csv.index = test_prediction_df.index.tz_localize(tzinfo)
    test_prediction_df_for_csv.index = test_prediction_df.index.strftime(SUBMISSION_FILE_DATETIME_FORMAT)
    test_prediction_df_for_csv.index.name = "ID"
    test_prediction_df_for_csv.to_csv(
        SUBMISSION_FILE_PATH,
        index=True,
        quoting=csv.QUOTE_ALL,
    )

    print("Done.")
