import argparse
import os
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
import pandas as pd
from azureml.core import Dataset, Run
from azureml.data.dataset_factory import TabularDatasetFactory
from sklearn.linear_model import LogisticRegression

# Data is located at:
url = "https://raw.githubusercontent.com/mixmasteru/MLEND-capstone/main/data/mushrooms.csv"
dataset_name = "Mushroom"


def clean_data(dataframe):
    # Clean and one hot encode data
    dataframe.columns = [i.replace('-', '_') for i in dataframe.columns]
    x_df = dataframe.dropna()

    one_hot_cols = ['cap_shape', 'cap_surface', 'cap_color', 'odor', 'gill_spacing', 'gill_size', 'gill_color',
                    'stalk_shape', 'stalk_root', 'stalk_surface_above_ring',
                    'stalk_surface_below_ring', 'stalk_color_above_ring',
                    'stalk_color_below_ring', 'veil_type', 'veil_color', 'ring_number',
                    'ring_type', 'spore_print_color', 'population', 'habitat']

    for col in one_hot_cols:
        col_one_h = pd.get_dummies(x_df.pop(col), prefix=col)
        x_df = x_df.join(col_one_h)

    y_df = x_df.pop("class").apply(lambda s: 1 if s == "e" else 0)

    return x_df, y_df


def main():
    # Add arguments to script
    parser = argparse.ArgumentParser()

    parser.add_argument('--C', type=float, default=1.0,
                        help="Inverse of regularization strength. Smaller values cause stronger regularization")
    parser.add_argument('--max_iter', type=int, default=100, help="Maximum number of iterations to converge")

    args = parser.parse_args()

    run = Run.get_context()
    workspace = run.experiment.workspace
    dataset = Dataset.get_by_name(workspace=workspace, name=dataset_name)
    df = dataset.to_pandas_dataframe()
    x, y = clean_data(df)

    # Split data into train and test sets.
    x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.75, test_size=0.25, random_state=101)

    run.log("Regularization Strength:", np.float(args.C))
    run.log("Max iterations:", np.int(args.max_iter))

    model = LogisticRegression(C=args.C, max_iter=args.max_iter).fit(x_train, y_train)

    accuracy = model.score(x_test, y_test)
    run.log("Accuracy", np.float(accuracy))

    os.makedirs('output', exist_ok=True)
    joblib.dump(model, 'outputs/model.joblib')


if __name__ == '__main__':
    main()
