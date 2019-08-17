# Package imports
import matplotlib.pyplot as plt
import seaborn as sns

import numpy as np
import pandas as pd
import plotly.express as px
import geopandas as gpd

# Have to do this first one as it's an experimental feature as of sklearn v0.21.2
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

from sklearn.linear_model import ElasticNet
from sklearn.neighbors import KNeighborsRegressor
from xgboost.sklearn import XGBRegressor

from sklearn.model_selection import train_test_split
from sklearn.model_selection import RandomizedSearchCV
from sklearn.pipeline import Pipeline

from sklearn.metrics import r2_score


# Consistent random seed for reproducibility and comparability
RANDOM_STATE = 42

def load_data_api(access_token_filepath, save_file=True):
    '''
    TODO: create this for pulling down the latest version of
    all of the API-based data and saving to file with date in filename
    '''

    return None


def load_data_file(filepath):
    '''
    TODO: create this for loading data and properly typecasting everything
    from a CSV file
    '''

    return None


def view_target_distribution(data, target_column, train_size):
    '''
    Plots the target variable distribution for the training,
    testing, and full datasets. This allows for a visual
    understanding of any obvious differences between the samples.


    Parameters
    ----------
    data: pandas DataFrame of all features + target
    
    target_column: str. Column name of the target variable in data

    train_size: float. Fraction of the full dataset you intend to use 
        for training. Must be in the bounds (0.0,1.0)


    Returns
    -------
    Nothing, just plots
    '''

    fig, axes = plt.subplots(figsize=(5,10), nrows=3)
    fig.suptitle('Distributions of Target Variable')

    plots = {'Training Data': target_train, 'Testing Data': target_test, 'All Data': target}


    for i, target_types in enumerate(plots.keys()):
        sns.distplot(plots[target_types], ax=axes[i], axlabel='target value')
        axes[i].set(title=target_types)
        
    plt.tight_layout()
    plt.show();


def long_lat_extract(row):
    '''
    Extracts longitude and latitude from a shapely Point object
    and returns them as DataFrame columns. Intended to be used via
    pandas.Series.apply(axis=1, result_type='expand').
    
    Inputs
    ------
    row: GeoPandas dataframe row, that at a minimum must have a
        'location' column that contains a shapely Point object
    
    
    Returns
    -------
    tuple of floats of the form (longitude, latitude)
    '''
    
    return row['location'].coords[0]


def calculate_split_fractions(validation_size, test_size, print_folds=False):
    '''
    Takes validation and test sizes as values applied to the full dataset and
    calculates the proper fractions so that their fold sizes are properly 
    implemented. 

    For example, val = 0.3 and test = 0.2 indicate that the user wants 30% of 
    the full datasets to be validation and 20% to be for testing, but once the 
    20% for testing is split off via sklearn's train_test_split(), 30% isn't 
    the proper input value for further cross-validation, as any 
    cross-validation algorithm won't know the size of the original dataset but 
    rather the size of the train + validation dataset.

    Parameters
    ----------
    validation_size: float in (0.0, 1.0). Values <= 0 and >= 1 will throw 
        a ValueError.

    test_size: float in (0.0, 1.0). Values <= 0 and >= 1 will throw 
        a ValueError. Also, test_size + validation_size must be in range (0,1).


    Returns
    -------
    float corresponding to value of `cv` parameter that should be used in
    sklearn to achieve the validation_size desired. test_size value entered
    need not be modified to make workflow function
    '''

    if validation_size <= 0.0 or validation_size >= 1.0:
        raise ValueError("validation_size value \
            outside of allowed range (0,1)")

    if test_size <= 0.0 or test_size >= 1.0:
        raise ValueError("test_size value \
            outside of allowed range (0,1)")

    if test_size + validation_size >=1 or test_size + validation_size <= 0:
        raise ValueError("Sum of validation_size and test_size is outside of \
            (0,1).")
                
    result = int((1 - test_size) / validation_size)
    
    if result <= 2:
        raise RuntimeError(f"Calculation resulted in {result} folds, but \
        at least 3 folds should be used for modeling. \
        Please choose a lower value for validation_size, test_size, or both.")
    
    if print_folds:
        fold_size = (1 - test_size) / result
        
        print(f"Calculation indicates that there should be {result} folds \
        for training and validation, with each fold representing {fold_size} \
        of the full dataset.")

    return result



def corr_to_target(data, target_column, title=None, file=None):
    '''
    Produces a sorted plot of correlations of feature values to the target variable, 
    colored like a heatmap. Most positive correlations on top.

    NOTE: code adapted from https://medium.com/better-programming/handy-data-visualization-functions-in-matplotlib-seaborn-to-speed-up-your-eda-241ba0a9c47d
    
    Parameters
    ----------
    data: pandas DataFrame of all features + target
    
    target_column: str. Column name of the target variable in data
    
    title: str. Title to put on the plot
    
    file: str/filepath with filename (e.g. "figs/1.png". Saves the plot
    
    
    Returns
    -------
    Nothing returned, only plots
    
    '''
    # Merge target and features together
    plt.figure(figsize=(4,6))
    sns.set(font_scale=1)
    
    sns.heatmap(data.corr()[[target_column]].sort_values(target_column,
                                                ascending=False)[1:],
                annot=True,
                cmap='coolwarm')
    
    if title: plt.title(f'\n{title}\n', fontsize=18)
    plt.xlabel('')    # optional in case you want an x-axis label
    plt.ylabel('')    # optional in case you want a  y-axis label
    if file: plt.savefig(file, bbox_inches='tight')
    plt.show();


def map_each_hospital(data, target_column, quantile=1.0):
    '''
    Map out the locations of hospitals, with marker sizes and colors 
    reflecting the size of the target variable. Map is interactive.


    Parameters
    ----------
    data: pandas DataFrame of all features + target
    
    target_column: str. Column name of the target variable in data

    quantile: float in range [0.0, 1.0]. Indicates what top % of 
        the target data you want displayed (locations outside of
        the defined quantile will not be plotted). A value of 1.0
        (the default) results in all data being plotted.


    Returns
    -------
    Plots an interactive (plotly-driven) map of all locations with
    marker sizes and color corresponding to the target variable values.

    '''

    data_top_target_quantile = data[data[target_column] >= \
    data.quantile(1-quantile)[target_column]]\
    .dropna(subset=[target_column])

    # Map only top X% of target value with color scale midpoint at median of whole dataset
    fig = px.scatter_mapbox(data_top_target_quantile, 
        lat="Latitude", lon="Longitude", 
        color=target_column, size=target_column,
        size_max=7, zoom=2, opacity=0.25,
        color_continuous_midpoint=data.quantile(0.5)[target_column])

    fig.show()


def state_level_choropleth(data, target_column, statistic='mean'):
    '''
    Create a colored choropleth map (heat map) at the state level 
    that colors states based upon the aggregated `statistic` value
    for the target variable. Map is interactive.


    Parameters
    ----------
    data: pandas DataFrame of all features + target
    
    target_column: str. Column name of the target variable in data

    statistic: str. Can be either "mean" or "median". Dictates what
        descriptive statistic is used for coloring the states in the
        choropleth map.


    Returns
    -------
    Plots an interactive (plotly-driven) map of the US, with each state's
    color corresponding to the target variable's `statistic` value.

    '''

    if statistic == 'mean':
        data_grouped = data.groupby('state', as_index=False).mean()
        title = "Mean Number of Deaths by State"

    elif statistic == 'median':
        data_grouped = data.groupby('state', as_index=False).median()
        title = "Median Number of Deaths by State"

    else:
        raise ValueError("Unallowed value of `statistic` used. \
            Please select one of 'mean' or 'median'.")

    # State-level choropleth map of the target variable
    fig = px.choropleth(data_grouped,
        locations='state', locationmode='USA-states',
        color=target_column, hover_data=[target_columnl],
        scope='usa',
        title=title)
    fig.show()


def model_data(data, target_column, validation_size, test_size, estimator):
    '''
    
    '''

    # Split target and features, and drop any values of target that are null
    target = data.loc[:, target_column].dropna()
    features = data.dropna(subset=[target_column])\
    .drop(columns=[target_column])

    # Split into training and testing data
    features_train, features_test, target_train, target_test =\
    train_test_split(features, target,
        test_size=test_size, random_state=RANDOM_STATE)

    # Setup missing value imputer
    # Need this in pipeline to make sure we aren't using validation
    # (AKA k-1) data to impute training data
    imputer = IterativeImputer(sample_posterior=True,
                              max_iter=73,
                              add_indicator=False,
                              random_state=RANDOM_STATE)

    # Standardize feature data
    scaler = StandardScaler()


    # Use PCA for dimensionality reduction on scaled features
    # Note that PCA's n_components of 31 to 59 correspond to 
    # 75% to 99% explained variance, resp.
    pca = PCA(random_state=RANDOM_STATE)

    fit_params = {}

    # Setup the models and hyperparameter/fit parameter distributions to sample
    if estimator == 'ElasticNet':
        reg = ElasticNet(random_state=RANDOM_STATE, selection='random')

        param_dist = {"pca__n_components": range(31,59),
              "regressor__alpha": np.arange(0.1,1.1,0.1),
              "regressor__l1_ratio": np.arange(0.0,1.1,0.1),
              "regressor__max_iter": range(1000, 10000, 1000)}


    elif estimator == 'kNN':
        reg = KNeighborsRegressor(p=2, n_jobs=4)

        param_dist = {"pca__n_components": range(31,59),
              "regressor__n_neighbors": range(2,100),
              "regressor__weights": ['uniform', 'distance'],
             'regressor__metric': ['euclidean', 'minkowski']}


    # Can't use Pipeline with XGBoost due to incompatibility
    # with early stopping and eval_sets fit parameters, so
    # it requires manual preprocessing
    elif estimator == 'XGBoost':
        reg = XGBRegressor(n_jobs=4, objective='reg:squarederror',
            random_state=RANDOM_STATE)

        param_dist = {"max_depth": range(2,10),
             "learning_rate": np.arange(0.01, 0.25, 0.02),
             "n_estimators": range(50,500, 25),
             'gamma': [0.5, 1, 1.5, 2, 5],
             "min_child_weight": [1, 5, 10],
              "subsample": [0.6, 0.8, 1.0],
              "colsample_bytree": [0.6, 0.8, 1.0],
              "reg_alpha": np.arange(0.1,1.1,0.1),
              "reg_lambda": np.arange(0.1,1.1,0.1)}

        fit_params = {'eval_set': [(features_train_pca, target_train),
                     (features_test_pca, target_test)],
                     'eval_metric': 'rmse',
                     'early_stopping_rounds': 10,
                     'verbose': False
                     }

        features_train = imputer.fit_transform(features_train)
        features_test = imputer.transform(features_test)

        features_train = standard_scaler.fit_transform(features_train)
        features_test = standard_scaler.transform(features_test)

        # Use 45 PCs, as that tested well earlier
        pca = PCA(n_components=45, random_state=RANDOM_STATE)
        features_train = pca.fit_transform(features_train)
        features_test = pca.transform(features_test)

    else:
        raise ValueError("Value of `estimator` not recognized.\
            Please choose one of 'ElasticNet', 'kNN', or 'XGBoost'.")

    # Build the workflow/pipeline and tune all parameters
    # Can't use XGBoost in sklearn Pipeline effectively
    if estimator != 'XGBoost':
        workflow = Pipeline([('imputer', imputer), ('scaler', scaler), 
                     ('pca', pca), ('regressor', reg)], verbose=True)

        parameter_searcher = RandomizedSearchCV(workflow, param_dist, 
            n_iter = 20,
            cv = calculate_split_fractions(validation_size, test_size), 
            iid=False,
            random_state=RANDOM_STATE,
            verbose=1)


    else:
        parameter_searcher = RandomizedSearchCV(reg, param_dist, 
            n_iter = 20,
            cv = calculate_split_fractions(validation_size, test_size), 
            iid=False,
            random_state=RANDOM_STATE,
            verbose=1)

    # Report on results of random search
    best_reg = parameter_searcher.fit(features_train, target_train,
                                 **fit_params).best_estimator_

    # Report on results of random search
    print(f"Best parameters on (CV validation score\
        ={parameter_searcher.best_score_}):")
    print(parameter_searcher.best_params_)

    # Predict data and evaluate using R^2 evaluation with the test data
    predictions = best_reg.predict(features_test)
    print(f"R^2 score on test data of best estimator: {r2_score(target_test, 
        predictions)}")


