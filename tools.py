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

from sklearn.linear_model import ElasticNet, Ridge
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


def is_sparse(data):
    '''
    Check to see if the data are sparse using the common metric of 
    checking if there are 50% or more values that are zero in the matrix.


    Parameters
    ----------
    data: pandas DataFrame of all features + target



    Returns
    -------
    bool. If True, data is sparse.
    '''

    return (data == 0).sum().sum() / (len(data) * len(data.columns)) >= 0.50


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


def split_data(data, target_column, train_size=0.6, validation_size=0.2):
    '''
    Convenience function that takes the full features + target data and 
    splits it up into training and testing features and target.

    Parameters
    ----------
    data: pandas DataFrame of all features + target
    
    target_column: str. Column name of the target variable in data

    train_size: float. Fraction of the full dataset you intend to use 
        for training. Must be in the bounds (0.0,1.0)


    Returns
    -------
    4-tuple of the form features_train, features_test, 
    target_train, target_test
    '''

    target = data.loc[:, target_column].dropna()
    features = data.dropna(subset=[target_column])\
    .drop(columns=[target_column])

    return train_test_split(features, target, 
        train_size=train_size + validation_size, random_state=RANDOM_STATE)


def impute_data(data, target_column, n_iterations, train_size=0.6):
    '''
    Takes a pandas DataFrame, drops any rows that have null values for the 
    target variable, splits the data into training and test sets, and uses multiple imputation to fill in missing values for the features.


    Parameters
    ----------
    data: pandas DataFrame of all features + target
    
    target_column: str. Column name of the target variable in data

    train_size: float. Fraction of the full dataset you intend to use 
        for training. Must be in the bounds (0.0,1.0)


    Returns
    -------
    4-tuple of form (features_train, features_test, target_train,
        target_test)
    '''

    # Split into training and test sets so that imputation criteria for 
    # test set doesn't leak into training set and vice versa

    features_train, features_test, target_train, target_test = \
    split_data(data, target_column, train_size)

    imputer = IterativeImputer(sample_posterior=True,
                          max_iter=n_iterations,
                          add_indicator=False,
                          random_state=RANDOM_STATE)

    features_train_imputed = pd.DataFrame(data=\
        imputer.fit_transform(features_train),
        columns=features_train.columns,
        index=features_train.index)

    # Note that this is transform() only, as it's the test data
    features_test_imputed = pd.DataFrame(data=imputer.transform(features_test),
                                   columns=features_test.columns,
                                   index=features_test.index)


    return [features_train_imputed, features_test_imputed, \
    target_train, target_test]

def view_target_distribution(target_train, target_test):
    '''
    Plots the target variable distribution for the training,
    testing, and full datasets. This allows for a visual
    understanding of any obvious differences between the samples.


    Parameters
    ----------
    target_train: pandas Series of target values (training data only)

    target_test: pandas Series of target values (testing data only)


    Returns
    -------
    Nothing, just plots
    '''

    fig, axes = plt.subplots(figsize=(5,7), nrows=3, sharex=True)
    #fig.suptitle('Distributions of Target Variable')

    target = target_train.append(target_test)

    plots = {'Training Data': target_train, 'Testing Data': target_test,
    'All Data': target}


    for i, target_types in enumerate(plots.keys()):
        sns.distplot(plots[target_types], ax=axes[i], axlabel='target value')
        axes[i].set(title=target_types)
        

    #plt.subplots_adjust(top=0.5)
    plt.tight_layout()
    plt.show();


def long_lat_extract(row, geometry_title='location'):
    '''
    Extracts longitude and latitude from a shapely Point object
    and returns them as DataFrame columns. Intended to be used via
    pandas.DataFrame.apply(axis=1, result_type='expand').
    
    Inputs
    ------
    row: GeoPandas dataframe row, that at a minimum must have a
        column that contains a shapely Point object

    geometry_title: str. Name of the column that is the geometry of the 
        GeoDataFrame
    
    
    Returns
    -------
    tuple of floats of the form (longitude, latitude)
    '''
    
    return row[geometry_title].coords[0]



def corr_to_target(features_train, target_train, title=None, file=None,
    print_corrs=True):
    '''
    Produces a sorted plot of correlations of feature values to the target variable, 
    colored like a heatmap. Most positive correlations on top.

    NOTE: code adapted from https://medium.com/better-programming/handy-data-visualization-functions-in-matplotlib-seaborn-to-speed-up-your-eda-241ba0a9c47d
    
    Parameters
    ----------
    features_train: pandas DataFrame of all features (training data only)
    
    target_train: pandas Series of target values (training data only)
    
    title: str. Title to put on the plot
    
    file: str/filepath with filename (e.g. "figs/1.png". Saves the plot

    print_corrs: bool. If True, prints out the table behind the visual (as
     having too many features will cause items on the visual to be dropped 
     for the sake of saving space)

    
    
    Returns
    -------
    Nothing returned, only plots
    
    '''

    target_column = target_train.name

    plt.figure(figsize=(4,6))
    sns.set(font_scale=1)

    # Merge target_train and feature_train
    data = features_train.copy()
    data[target_column] = target_train
    
    sns.heatmap(data.corr()[[target_column]].sort_values(target_column,
        ascending=False)[1:],
    annot=True,
    cmap='coolwarm')
    
    if title: plt.title(f'\n{title}\n', fontsize=18)
    plt.xlabel('')    # optional in case you want an x-axis label
    plt.ylabel('')    # optional in case you want a  y-axis label
    if file: plt.savefig(file, bbox_inches='tight')
    plt.show();

    if print_corrs:
        print(data.corr()[target_column]\
            .sort_values(ascending=False)\
            .drop(target_column))


def map_each_hospital(features_train, target_train=None, quantile=1.0, 
    quantile_direction='bottom', labels={}, midpoint=None, save_file=None,
    make_plot=True):
    '''
    Map out the locations of hospitals, with marker sizes and colors 
    reflecting the size of the target variable. Map is interactive.


    Parameters
    ----------
    features_train: pandas DataFrame of all features (training data only)
    
    target_train: pandas Series of target values (training data only)

    quantile: float in range [0.0, 1.0]. Indicates what top % of 
        the target data you want displayed (locations outside of
        the defined quantile will not be plotted). A value of 1.0
        (the default) results in all data being plotted. Values less than 0.5 
        are assumed to mean "show me the top X%", whereas values greater 
        than 0.5 are assumed to mean "show me the bottom (1-X)%". Ignored if 
        target_column is None. Use 1.0 if you want to see all data.

    quantile_direction: str. Either 'top' or 'bottom'. Indicates if user 
        wants to see the "top X%" quantile or the "bottom X%" 
        quantile. Use 'bottom' if you want to see all data.

    labels: dict with str keys and str values. By default, column names are 
        used in the figure for axis titles, legend entries and hovers. This 
        parameter allows this to be overridden. The keys of this dict should 
        correspond to column names, and the values should correspond to the 
        desired label to be displayed (e.g. {target_column: 'Heart Attacks'})

    midpoint: float. If set, indicates the value of target_train that you
        want as the midpoint of the color scale used

    save_file: str. Should provide a filepath for the interactive HTML 
        to be saved. If None, file is not saved.

    make_plot: bool. If True, automatically plots the resultant figure


    Returns
    -------
    Plots an interactive (plotly-driven) map of all locations with
    marker sizes and color corresponding to the target variable values. Also returns a plotly go.Figure object

    '''    

    # Pass in my public mapbox token for contextual mapping
    px.set_mapbox_access_token(open("secure_keys/public.mapbox_token").read())

    data = features_train.copy()

    # Setup figure title
    if quantile == 1.0:
        title = "All Data"
    elif quantile_direction == 'top':
        title = f"Top {int(quantile * 100)}%"
    else:
        title = f"Bottom {int(quantile * 100)}%"


    # Combine features and target, if target is specified
    if target_train is not None:
        target_column = target_train.name

        # Combine target and features
        data[target_column] = target_train



        # 'top' -> at or above
        if quantile_direction == 'top':
            print(f"Only showing values at or above \
            {data.quantile(1-quantile)[target_column]}")

            top_target_quantile = data[data[target_column] >= \
            data.quantile(1-quantile)[target_column]]

        # Assume when a large quantile is given, users wants "this % and lower"
        elif quantile_direction == 'bottom':
            print(f"Only showing values at or below \
            {data.quantile(quantile)[target_column]}")

            top_target_quantile = data[data[target_column]\
             <= data.quantile(quantile)[target_column]]

        else: raise ValueError("Invalid value for quantile_direction. Use 'top' or 'bottom'.")

        # Map only top X% of target value with color scale midpoint at median of whole dataset
        if midpoint is None: midpoint = target_train.quantile(0.5)

        fig = px.scatter_mapbox(top_target_quantile, 
            lat="Latitude", lon="Longitude", 
            color=target_column, size=target_column,
            size_max=7, zoom=2, opacity=0.25,
            color_continuous_scale=px.colors.diverging.Portland,
            color_continuous_midpoint=midpoint,
            labels=labels, title=title)

     # No target data provided
    else:
        fig = px.scatter_mapbox(data, 
            lat="Latitude", lon="Longitude", zoom=2,
            labels=labels, title=title, opacity=0.25)

    if save_file:
        fig.write_html(save_file)

    if make_plot: fig.show()

    return fig


def state_level_choropleth(features_train, target_train, 
    statistic='mean', labels={}, midpoint=None):
    '''
    Create a colored choropleth map (heat map) at the state level 
    that colors states based upon the aggregated `statistic` value
    for the target variable. Map is interactive.


    Parameters
    ----------
    features_train: pandas DataFrame of all features (training data only)
    
    target_train: pandas Series of target values (training data only)

    statistic: str. Can be either "mean" or "median". Dictates what
        descriptive statistic is used for coloring the states in the
        choropleth map.

    labels: dict with str keys and str values. By default, column names are 
        used in the figure for axis titles, legend entries and hovers. This 
        parameter allows this to be overridden. The keys of this dict should 
        correspond to column names, and the values should correspond to the 
        desired label to be displayed (e.g. {target_column: 'Heart Attacks'})

    midpoint: float. If set, indicates the value of target_train that you
        want as the midpoint of the color scale used


    Returns
    -------
    Plots an interactive (plotly-driven) map of the US, with each state's
    color corresponding to the target variable's `statistic` value.

    '''

    target_column = target_train.name

    # Pass in my public mapbox token for contextual mapping
    px.set_mapbox_access_token(open("secure_keys/public.mapbox_token").read())

    # Combine target and features
    data = features_train.copy()
    data[target_column] = target_train

    if statistic == 'mean':
        data_grouped = data.groupby('state', as_index=False).mean()
        title = "Mean Value by State"

    elif statistic == 'median':
        data_grouped = data.groupby('state', as_index=False).median()
        title = "Median Value by State"

    else:
        raise ValueError("Unallowed value of `statistic` used. \
            Please select one of 'mean' or 'median'.")

    # State-level choropleth map of the target variable
    fig = px.choropleth(data_grouped,
        locations='state', locationmode='USA-states',
        color=target_column, hover_data=[target_column],
        color_continuous_scale=px.colors.diverging.Portland,
        color_continuous_midpoint=midpoint,
        scope='usa',
        title=title, labels=labels)
    fig.show()


def model_data(data, target_column, 
    validation_size, test_size, estimator, n_pcs_xgb=64,
    return_best_estimator=True):
    '''
    Builds and runs the following pipeline through a randomized search
    parameter optimizer (sklearn's RandomizedSearchCV):

    1. Multivariate imputation
    2. Standardize the data
    3. PCA
    4. Fit and tune the hyperparameters of one of three possible models

    Note that one of the three models possible (XGBoost) doesn't play well
    with RandomizedSearchCV and Pipelines, so it is less robust in its results
    (because it doesn't search over any parameters other than hyperparameters)

    NOTE: XGBoost is currently broken, will be corrected at some point

    NOTE: kNN consistently showed poor results relative to ElasticNet
    in initial testing.


    Parameters
    ----------
    data: pandas DataFrame of all features (only numeric ones) + target. NOTE 
    THAT THIS IS NOT JUST features_train. As we'll be doing k-fold 
    cross-validation here, we'll need the flexibility to split and re-split 
    the data.
    
    target_column: str. Column name of the target variable in data

    validation_size: float. Fraction of the full dataset you intend to use 
        for validation during tuning. Must be in the bounds (0.0,1.0)

    test_size: float. Fraction of the full dataset you intend to use 
        for final testing. Must be in the bounds (0.0,1.0)

    estimator: str. Can be ElasticNet, Ridge, kNN, or XGBoost. Indicates 
        the type of model you want to train

    n_pcs_xgb: int. Dictates how many principal components should be used in
        model training and testing for the XGBoost estimator, since that one
        can't be put into a pipeline to test how many PCs are optimal.
        Note that prior testing has shown that values of 34 to 95 
        correspond to 75% to 99% explained variance, resp. The default
        of 64 corresponds to 95% explained variance

    return_best_estimator: bool. If True, returns the best fitted 
        regressor object so it can be further investigated and used.
        If False, returns the R^2 score.



    Returns
    -------
    R^2 score on the test data as a float. Also reports out on the results of training, parameters ultimately used by printing to console..
    '''

    print(f"Modeling {target_column}...\n")
    
    # Split into training and testing data
    features_train, features_test, target_train, target_test =\
    split_data(data, target_column,
        train_size = 1 - (validation_size + test_size))

    


    # Setup missing value imputer
    # Need this in pipeline to make sure we aren't using validation
    # (AKA k-1) data to impute training data
    imputer = IterativeImputer(sample_posterior=True,
                              max_iter=90,
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

        param_dist = {"pca__n_components": range(34,95),
              "regressor__alpha": np.arange(0.1,1.1,0.1),
              "regressor__l1_ratio": np.arange(0.1,1.1,0.1),
              "regressor__max_iter": range(1000, 10000, 1000)}


    elif estimator == 'Ridge':
        reg = Ridge(random_state=RANDOM_STATE)

        param_dist = {"pca__n_components": range(34,95),
              "regressor__alpha": np.arange(0.1,1.1,0.1),
              "regressor__max_iter": range(1000, 10000, 1000)}

    elif estimator == 'kNN':
        reg = KNeighborsRegressor(p=2, n_jobs=4)

        param_dist = {"pca__n_components": range(34,95),
              "regressor__n_neighbors": range(2,100),
              "regressor__weights": ['uniform', 'distance'],
             'regressor__metric': ['euclidean', 'minkowski']}


    # Can't use Pipeline with XGBoost due to incompatibility
    # with early stopping and eval_sets fit parameters, so
    # it requires manual preprocessing
    elif estimator == 'XGBoost':
        reg = XGBRegressor(n_jobs=-1, objective='reg:squarederror',
            random_state=RANDOM_STATE)

        # Impute
        print("Imputing...")
        features_train = imputer.fit_transform(features_train)
        features_test = imputer.transform(features_test)

        # Scale (standardize)
        print("Standardizing...")
        features_train = scaler.fit_transform(features_train)
        features_test = scaler.transform(features_test)

        print(f"Performing PCA with {n_pcs_xgb} components...")
        pca = PCA(n_components=n_pcs_xgb, random_state=RANDOM_STATE)
        features_train = pca.fit_transform(features_train)
        features_test = pca.transform(features_test)
        print(f"`features_train` now has shape {features_train.shape}")
        print(f"`features_test` now has shape {features_test.shape}")


        # Distributions adapted partly from 
        # https://www.kaggle.com/tilii7/hyperparameter-grid-search-with-xgboost
        param_dist = {"max_depth": range(2,10),
             "learning_rate": np.arange(0.01, 0.25, 0.02),
             "n_estimators": range(50,500, 25),
             'gamma': [0.5, 1, 1.5, 2, 5],
             "min_child_weight": [1, 5, 10],
              "subsample": [0.6, 0.8, 1.0],
              "colsample_bytree": [0.6, 0.8, 1.0],
              "reg_alpha": np.arange(0.1,1.1,0.1),
              "reg_lambda": np.arange(0.1,1.1,0.1)}

        fit_params = {'eval_set': [(features_train, target_train),
                     (features_test, target_test)],
                     'eval_metric': 'rmse',
                     'early_stopping_rounds': 10,
                     'verbose': False
                     }

    else:
        raise ValueError("Value of `estimator` not recognized.\
            Please choose one of 'ElasticNet', 'Ridge', 'kNN', or 'XGBoost'.")

    # Build the workflow/pipeline and train + tune estimators
    # Can't use XGBoost in sklearn Pipeline effectively
    if estimator != 'XGBoost':
        workflow = Pipeline([('imputer', imputer), ('scaler', scaler), 
                     ('pca', pca), ('regressor', reg)], verbose=True)

        parameter_searcher = RandomizedSearchCV(workflow, param_dist, 
            n_iter = 20,
            cv = calculate_split_fractions(validation_size, test_size), 
            iid=False,
            random_state=RANDOM_STATE,
            verbose=1, n_jobs=-1)

    else:
        print("Setting up paramater_searcher for XGBoost...")
        parameter_searcher = RandomizedSearchCV(reg, param_dist, 
            n_iter = 20,
            cv = calculate_split_fractions(validation_size, test_size), 
            iid=False,
            random_state=RANDOM_STATE,
            verbose=1, n_jobs=-1)


    # Report on results of random search
    best_reg = parameter_searcher.fit(features_train, target_train,
                                 **fit_params).best_estimator_

    # Report on results of random search
    print(f"Best parameters on (CV validation score\
        ={parameter_searcher.best_score_}):")
    print(parameter_searcher.best_params_)

    # Predict data and evaluate using R^2 evaluation with the test data
    predictions = best_reg.predict(features_test)
    print(f"R^2 score on test data of best estimator: {r2_score(target_test, predictions)}")


    return best_reg


def interpret_top_components(data, estimator, num_top_components=5,
    return_coeffs = False):
    '''
    Takes a Pipeline that includes a fitted PCA object and a trained 
    linear model and returns the top principal components (by model 
    coefficient) and their weights associated with the original feature names.
    The top components and weights indicate importance to the PCA and thus
    allow for interpreting PC-based model importances/coeffecients in terms
    of feature-space variables

    Recommended usage is 
    interpret_top_components(feature_matrix.drop(columns=['state', 'provider_id',
    target_col_percentile,
    target_col]),
    regressor,
    num_top_components=5)

    NOTE: the returned DataFrame won't be sorted by PC rank, but rather
    by resultant weight. Use groupby() to investigate only a single PC rank

    Parameters
    ----------
    data: pandas DataFrame. The same DataFrame used in model_data() should
        be used here BUT the target column should be dropped too.

    estimator: pipeline-based estimator with PCA and a linear model, 
    at the very least. The PCA element must be named 'pca' and the 
    model in the Pipeline must be named 'regressor'

    num_top_components: int. Indicates how many of the highest-ranked 
        PCs (by model weight) you want to investigate.

    return_coeffs: bool. If True, returns a tuple of the form
        (results_df, model coefficients used)



    Returns
    -------
    pandas DataFrame with columns ['Principal Component Number', 
    'Feature Name', 'Weight']. Note that Principal Component Number
    is essentially a ranking: lower numbers indicate higher model weights.
    Also note that 'Weight' is the product of the linear model weight for the
    relevant PC and the individual PC weight of that feature.
    '''

    # Pull in the weights of each PC and assign the relevant original
    # feature name to each
    pc_components = pd.DataFrame(estimator['pca'].components_, 
        columns = data.columns)

    coeffs = pd.Series(estimator['regressor'].coef_)
    # Sort coeffs so the highest absolute value coefficients are first
    coeffs = coeffs.loc[coeffs.abs().sort_values(ascending=False).index]


    # Multiply PC component weights by relevant top model coefficients
    # then rename PCs by their rank
    # then rename the index column produced from reset_index() to Feature Name
    # then melt so that each original feature weight for a given PC 
        # has its own row
    pc_interpret_results = \
    (pc_components.transpose() * coeffs).loc[:, coeffs[:num_top_components]\
    .index].reset_index()\
    .rename(columns={value: str(i+1) for i,value in enumerate(coeffs.index)})\
    .rename(columns={'index': 'Feature Name'})\
    .melt(id_vars=['Feature Name'], var_name='PC Model Importance Rank', value_name='Weight')

    # Sort by absolute value of weight so we don't ignore large negative values
    pc_interpret_results = \
    pc_interpret_results.loc[pc_interpret_results['Weight'].abs().sort_values(ascending=False).index]

    if return_coeffs:
        return pc_interpret_results, coeffs[:num_top_components]
    else:
        return pc_interpret_results

