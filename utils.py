import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.svm import OneClassSVM
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, RandomizedSearchCV, KFold
from sklearn.svm import OneClassSVM
from sklearn.metrics import make_scorer
from sklearn.decomposition import PCA
from shapely.geometry import shape
import plotly.express as px
import geopandas as gpd
from functools import partial
from joblib import dump, load
import json
import folium
import warnings
import copy
warnings.simplefilter(action='ignore', category=FutureWarning)


def preprocess_data(X_data):
    """
    :param X_data: original clean data with features etc
    :return: scaled features and targets with scaler
    """
    # Create a dummy y array
    y = np.ones(X_data.shape[0])

    # Split into train and test sets
    X_train, X_test, _, _ = train_test_split(X_data, y, test_size=0.2, random_state=42)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    return X_train_scaled, X_test_scaled, scaler


# notice thtat we simply test aganist what's normal
def inlier_fraction_scorer(estimator, X, y=None):
    predictions = estimator.predict(X)
    return (predictions == 1).mean()


def optimize_model_cross_validation(param_grid, X_train_scaled, X_test_scaled, model_type='svm'):
    """

    :param param_grid: params to optimize
    :param X_train_scaled: train data scaled
    :param X_test_scaled: test data scaled
    :param model_type: one-class-svm or isolation forest
    :return:
    """
    if model_type == 'svm':
        model = OneClassSVM()
    elif model_type == 'isolation_forest':
        model = IsolationForest(random_state=42)
    else:
        raise ValueError("Invalid model_type. Expected 'svm' or 'isolation_forest'.")

    # Use a partial function to pass additional arguments to the scorer
    scorer = inlier_fraction_scorer

    # Initialize RandomizedSearchCV
    random_search = RandomizedSearchCV(
        model,
        param_distributions=param_grid,
        n_iter=10,
        cv=KFold(n_splits=5, shuffle=True, random_state=42),
        scoring=scorer,
        random_state=42
    )

    # Perform hyperparameter optimization
    random_search.fit(X_train_scaled)

    # Retrieve the best model
    best_model = random_search.best_estimator_

    # Fit the best model to the training data
    best_model.fit(X_train_scaled)

    # Make predictions and calculate inlier fractions for both training and test sets
    train_predictions = best_model.predict(X_train_scaled)
    test_predictions = best_model.predict(X_test_scaled)
    train_inlier_fraction = (train_predictions == 1).mean()
    test_inlier_fraction = (test_predictions == 1).mean()

    # some sklern models dont have decision function
    if hasattr(best_model, 'decision_function'):
        train_scores = best_model.decision_function(X_train_scaled)
        test_scores = best_model.decision_function(X_test_scaled)
    elif hasattr(best_model, 'score_samples'):
        # Inverse the scores because lower means more abnormal
        train_scores = -best_model.score_samples(X_train_scaled)
        test_scores = -best_model.score_samples(X_test_scaled)

    # average the scores
    train_mean_score = train_scores.mean()
    test_mean_score = test_scores.mean()

    # Compile the results into a dictionary
    results = {
        "best_params": random_search.best_params_,
        "train_mean_score": round(train_mean_score, 2),
        "train_inlier_fraction": round(100 * train_inlier_fraction, 2),
        "test_mean_score": round(test_mean_score, 2),
        "test_inlier_fraction": round(100 * test_inlier_fraction, 2)
    }

    return best_model, results



def plot_svm_decision_boundary_PCA(best_svm_model,X_train_scaled,X_test_scaled,save_dir=None):
    # Apply PCA to reduce the dimensionality to 2D
    pca = PCA(n_components=2)
    X_train_pca = pca.fit_transform(X_train_scaled)
    X_test_pca = pca.transform(X_test_scaled)

    # Train OneClassSVM on the PCA transformed data, messes without copy
    svm_pca_copy = copy.deepcopy(best_svm_model)
    oc_svm_pca = svm_pca_copy.fit(X_train_pca)

    # Create a meshgrid for visualization
    x_min, x_max = X_train_pca[:, 0].min() - 1, X_train_pca[:, 0].max() + 1
    y_min, y_max = X_train_pca[:, 1].min() - 1, X_train_pca[:, 1].max() + 1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 50), np.linspace(y_min, y_max, 50))
    Z = oc_svm_pca.decision_function(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    # Plot the decision boundary
    plt.figure(figsize=(10, 8))
    plt.contourf(xx, yy, Z, levels=np.linspace(Z.min(), 0, 7), cmap=plt.cm.PuBu)
    a = plt.contour(xx, yy, Z, levels=[0], linewidths=2, colors='darkred')
    plt.contourf(xx, yy, Z, levels=[0, Z.max()], colors='palevioletred')
    s = 40
    b1 = plt.scatter(X_train_pca[:, 0], X_train_pca[:, 1], c='white', s=s, edgecolors='k')
    b2 = plt.scatter(X_test_pca[:, 0], X_test_pca[:, 1], c='green', s=s, edgecolors='k')
    plt.axis('tight')
    plt.xlim(x_min, x_max)
    plt.ylim(y_min, y_max)
    plt.legend([a.collections[0], b1, b2],
               ["Decision boundary", "Training data", "Test data"],
               loc="upper left", fontsize=17)
    plt.xlabel("PCA Component 1", fontsize=18)
    plt.ylabel("PCA Component 2", fontsize=18)
    plt.title("OneClassSVM Decision Boundary on 2D PCA-transformed Data", fontsize=18)
    if save_dir is not None:
        plt.savefig(save_dir)
    plt.show()

def plot_iso_forest_decision_function(best_iso_forest_model,X_train_scaled,X_test_scaled,save_dir=None):
    # Apply PCA to reduce the data to two dimensions for visualization
    pca = PCA(n_components=2)
    X_train_pca = pca.fit_transform(X_train_scaled)
    X_test_pca = pca.transform(X_test_scaled)

    # Train the Isolation Forest on the PCA-reduced data
    best_iso_forest_model.fit(X_train_pca)
    # Train OneClassSVM on the PCA transformed data, messes without copy
    iso_pca_copy = copy.deepcopy(best_iso_forest_model)
    oc_iso_pca = iso_pca_copy.fit(X_train_pca)

    # Create a grid for visualization
    xx, yy = np.meshgrid(np.linspace(X_train_pca[:, 0].min(), X_train_pca[:, 0].max(), 50),
                         np.linspace(X_train_pca[:, 1].min(), X_train_pca[:, 1].max(), 50))

    # Predict anomaly scores using the Isolation Forest on the grid
    Z = oc_iso_pca.decision_function(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    # Plot the decision function on a contour plot
    plt.figure(figsize=(10, 8))
    plt.contourf(xx, yy, Z, cmap=plt.cm.Blues_r)

    # Plot the original points on the PCA-reduced data
    plt.scatter(X_train_pca[:, 0], X_train_pca[:, 1], c='white', s=20, edgecolor='k', label='Training data')
    plt.scatter(X_test_pca[:, 0], X_test_pca[:, 1], c='red', s=20, edgecolor='k', label='Test data')

    plt.title('Isolation Forest Decision Function on PCA-Reduced Data', fontsize=20)
    plt.xlabel('Principal Component 1', fontsize=20)
    plt.ylabel('Principal Component 2', fontsize=20)
    plt.legend(fontsize=17)
    plt.axis('tight')
    if save_dir is not None:
        plt.savefig(save_dir)
    plt.show()


def map_promising_zip_codes(data_df, filtered_for_mapping_df, show_highways=False, show_large_airports=False,
                            save_dir=None):
    # Create a base map
    m_states = folium.Map(location=[44, -72], zoom_start=4.5, tiles='OpenStreetMap')

    # Load the GeoJSON data for state boundaries
    with open("by_products/us-states-json.json", "r") as f:
        us_states_geojson = json.load(f)

    # Filter the GeoJSON data to only include the boundaries of the present states and add to base map
    states_to_include = filtered_for_mapping_df.state_id.unique()
    filtered_geojson_features = [feature for feature in us_states_geojson['features'] if
                                 feature['id'] in states_to_include]
    us_states_geojson['features'] = filtered_geojson_features
    folium.GeoJson(us_states_geojson, name='geojson').add_to(m_states)

    # Add state names to the centroids of the states-->maybe useless but let's keep it
    for feature in us_states_geojson['features']:
        state_name = feature['properties']['name']
        # Calculate the centroid using shapely
        geom = shape(feature['geometry'])
        centroid = geom.centroid
        folium.Marker([centroid.y, centroid.x],
                      icon=folium.DivIcon(html=f'<div style="font-size: 10pt"><b>{state_name}</b></div>')).add_to(
            m_states)

    # Add ZIP code points with target=1 to the map-->useful to see the proxy to the existing locations
    all_target1_df = data_df[data_df['target'] == 1]
    for index, row in all_target1_df.iterrows():
        if pd.notna(row['lat']) and pd.notna(row['lng']):
            folium.CircleMarker(
                location=(row['lat'], row['lng']),
                radius=2,
                color="orange",
                fill=True,
                fill_color="orange"
            ).add_to(m_states)

    # show highways if needed
    if show_highways:
        # Load the highways data
        highways_gdf = gpd.read_file('datasets/tl_2019_us_primaryroads/tl_2019_us_primaryroads.shp')

        # Assuming states_gdf is a global variable or defined elsewhere in your code
        # Convert the filtered state boundaries to a geopandas GeoDataFrame
        states_gdf = gpd.GeoDataFrame.from_features(filtered_geojson_features)
        states_gdf.set_crs("EPSG:4269", inplace=True)

        # Filter the highways to only those within the states of interest
        states_names_to_include = filtered_for_mapping_df.state_name.unique()
        filtered_states_gdf = states_gdf[states_gdf['name'].isin(states_names_to_include)]
        intersected_highways = gpd.overlay(highways_gdf, filtered_states_gdf, how='intersection')

        # Add the intersected highways to the map
        geojson_data = intersected_highways.to_json()
        folium.GeoJson(geojson_data, name='Highways',
                       style_function=lambda x: {
                           'color': '#800080',  # Purple color for highways
                           'weight': 2,
                           'opacity': 0.2
                       }).add_to(m_states)

    if show_large_airports:
        # show only large airports
        airports_df = pd.read_csv("datasets/us-airports.csv")
        state_names_to_include = filtered_for_mapping_df.state_name.unique()
        filtered_large_airports_df = airports_df[(airports_df['region_name'].isin(state_names_to_include))
                                                 & (airports_df['type'] == 'large_airport')]
        for index, row in filtered_large_airports_df.iterrows():
            folium.CircleMarker(
                location=(row['latitude_deg'], row['longitude_deg']),
                radius=3,
                color='red',
                fill=True,
                fill_color='red',
                fill_opacity=1.0,
                popup=row['name']
            ).add_to(m_states)

    # Add the corresponding locations
    for index, row in filtered_for_mapping_df.iterrows():
        if pd.notna(row['lat']) and pd.notna(row['lng']):
            folium.CircleMarker(
                location=(row['lat'], row['lng']),
                radius=5,
                color="green",
                fill=True,
                fill_color="green",
                fill_opacity=1.0,
                tooltip=f"""
                        <div style='font-size:16px;'>
                        <strong>zip: {int(row['zip'])}, city: {row['city']},</strong><br>
                        <strong>state:{row['state_id']}, score: {row['common_score']:.2f}</strong>
                        </div>
                        """
            ).add_to(m_states)

    legend_html = """
     <div style="position: fixed; 
                 bottom: 350px; left: 670px; width: 140px; height: 120px; 
                 border:2px solid grey; z-index:9999; font-size:14px; background-color: white;
                 padding: 10px">
                 <strong> Predictions </strong> <br>
                 <span style="color:orange; font-weight:bold;">●</span> Actual Zips<br>
                 <span style="color:green; font-weight:bold;">●</span> Predictions<br>
    """

    # Conditionally add airports and highways to the legend
    if show_highways:
        legend_html += '<span style="color:purple; font-weight:bold;">&mdash;</span> Highways<br>'
    if show_large_airports:
        legend_html += '<span style="color:red; font-weight:bold;">●</span> Large Airports<br>'
    # Close the legend's div tag
    legend_html += "</div>"

    m_states.get_root().html.add_child(folium.Element(legend_html))

    if save_dir is not None:
        m_states.save(save_dir)

    return m_states


def plot_top_scores(data,save_dir=None):
    # Sort the DataFrame by the average anomaly score column
    N = data.shape[0]

    # Create labels for each bar
    labels = [f"{int(row['zip'])}, {row['city']}, {row['state_id']}" for _, row in data.iterrows()]

    # Set up the figure
    fig, ax = plt.subplots(figsize=(10, N * 0.5))

    # Create a bar plot
    bars = ax.barh(labels, data['common_score'], color='skyblue')
    ax.set_title(f'Top {N} ZIP Codes',fontsize=25)
    ax.set_xlabel('Average Score',fontsize=20)
    #ax.set_ylabel('ZIP Code, City, State')
    ax.invert_yaxis()  # To have the highest score at the top

    # Add the score on the bar itself for direct reading
    for bar in bars:
        width = bar.get_width()
        label_x_pos = bar.get_x() + width if width > 0 else bar.get_x()
        ax.text(label_x_pos, bar.get_y() + bar.get_height() / 2, f'{width:.2f}',
                va='center', ha='right' if width < 0 else 'left')

    plt.tight_layout()
    if save_dir is not None:
        plt.savefig(save_dir)
    plt.show()