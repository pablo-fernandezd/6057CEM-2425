import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.datasets import load_iris
from pandas.plotting import parallel_coordinates
from math import pi


def run_eda(X, y):
    """
    Perform Exploratory Data Analysis on the given features X and labels y.
    Prints dataset information and generates plots for distribution, correlation,
    and dimensionality reduction. All output is shown inline (no files are saved).

    Parameters:
    - X: pandas DataFrame or NumPy array of shape (n_samples, n_features) containing feature data.
         (For the Iris dataset, n_features=4).
    - y: pandas Series, NumPy array, or list of labels corresponding to X.
         Labels can be the species names (e.g., "setosa") or numeric codes (0, 1, 2).
    """
    # Ensure X is in DataFrame form (for ease of use with pandas/seaborn)
    if isinstance(X, pd.DataFrame):
        df_features = X.copy()
    else:
        # If X is a numpy array, create DataFrame with appropriate feature names
        iris_data = load_iris()
        feature_names = [name.replace(' (cm)', '') for name in iris_data.feature_names]
        df_features = pd.DataFrame(X, columns=feature_names)
    # Convert y to species names if it's given as numeric codes
    if isinstance(y, pd.Series):
        y_values = y.values
    else:
        y_values = np.array(y)
    if np.issubdtype(y_values.dtype, np.integer):
        # Map integer codes to actual species names using the Iris dataset metadata
        iris_data = load_iris()
        species_names = [iris_data.target_names[int(label)] for label in y_values]
    else:
        # y is already in the form of species names
        species_names = [str(label) for label in y_values]
    # Combine features and labels into one DataFrame for EDA
    df = df_features.copy()
    df['species'] = species_names

    # 1. Dataset Overview (information, descriptive stats, missing values)
    print("Dataset Information:")
    df.info()  # displays info about DataFrame (columns, types, non-null counts)
    print("\nDescriptive Statistics:")
    print(df.describe())
    print("\nMissing Values by Column:")
    print(df.isnull().sum())
    print("\n")  # extra newline for readability

    # 2. Feature Distributions by Species (Histograms for each feature)
    plt.figure(figsize=(15, 10))
    for i, feature in enumerate(df_features.columns):
        plt.subplot(2, 2, i + 1)
        # Plot histogram for each species on the same axes
        for species in sorted(df['species'].unique()):
            sns.histplot(df[df['species'] == species][feature], kde=True, label=species)
        plt.title(f"Distribution of {feature}")
        plt.xlabel(feature)
        plt.ylabel("Frequency")
        plt.legend()
    plt.tight_layout()
    plt.show()

    # 3. Boxplots for Each Feature by Species
    plt.figure(figsize=(15, 10))
    for i, feature in enumerate(df_features.columns):
        plt.subplot(2, 2, i + 1)
        sns.boxplot(x='species', y=feature, data=df)
        plt.title(f"Boxplot of {feature} by Species")
    plt.tight_layout()
    plt.show()

    # 4. Violin Plots for Each Feature by Species
    plt.figure(figsize=(12, 8))
    for i, feature in enumerate(df_features.columns):
        plt.subplot(2, 2, i + 1)
        sns.violinplot(x='species', y=feature, data=df)
        plt.title(f"{feature} Distribution by Species")
    plt.tight_layout()
    plt.show()

    # 5. Correlation Heatmap (feature-to-feature correlation matrix)
    plt.figure(figsize=(10, 8))
    corr_matrix = df_features.corr(numeric_only=True)
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm')
    plt.title("Feature Correlation Matrix")
    plt.show()

    # 6. 3D Feature Space Visualization (using 3 of the features)
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    # Use the first three features for 3D axes (e.g., sepal length, sepal width, petal length)
    x_feat, y_feat, z_feat = df_features.columns[0], df_features.columns[1], df_features.columns[2]
    scatter = ax.scatter(df[x_feat], df[y_feat], df[z_feat],
                         c=df['species'].astype('category').cat.codes, cmap='viridis')
    ax.set_xlabel(f"{x_feat.title()} (cm)")
    ax.set_ylabel(f"{y_feat.title()} (cm)")
    ax.set_zlabel(f"{z_feat.title()} (cm)")
    plt.title("3D Feature Space Representation")
    plt.colorbar(scatter)
    plt.show()

    # 7. Pairplot (pairwise scatter plots and KDE diagonals)
    sns.pairplot(df, hue="species", diag_kind="kde", markers=["o", "s", "D"])
    plt.suptitle("Pairplot of the Iris Dataset", y=1.02)
    plt.show()

    # 8. PCA - Dimensionality Reduction (2D projection)
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(df_features)
    df_pca = pd.DataFrame(X_pca, columns=["PC1", "PC2"])
    df_pca["species"] = df["species"]
    plt.figure(figsize=(8, 6))
    sns.scatterplot(x="PC1", y="PC2", hue="species", data=df_pca, palette="viridis")
    plt.title("PCA - 2D Projection")
    plt.show()

    # 9. t-SNE - Non-linear Dimensionality Reduction (2D projection)
    tsne = TSNE(n_components=2, random_state=42)
    X_tsne = tsne.fit_transform(df_features)
    df_tsne = pd.DataFrame(X_tsne, columns=["TSNE1", "TSNE2"])
    df_tsne["species"] = df["species"]
    plt.figure(figsize=(8, 6))
    sns.scatterplot(x="TSNE1", y="TSNE2", hue="species", data=df_tsne, palette="viridis")
    plt.title("t-SNE - 2D Projection")
    plt.show()

    # 10. Parallel Coordinates Plot (each species plotted across all feature axes)
    plt.figure(figsize=(10, 6))
    parallel_coordinates(df, class_column="species", colormap=plt.get_cmap("viridis"))
    plt.title("Parallel Coordinates Plot")
    plt.xticks(rotation=45)
    plt.show()

    # 11. Radar Chart – Feature Comparison across species means
    categories = list(df_features.columns)
    N = len(categories)
    # Compute the mean of each feature for each species
    data_means = df.groupby("species").mean()
    # Compute the angle each feature will take on the radar chart (in radians)
    angles = [n / float(N) * 2 * pi for n in range(N)]
    angles += angles[:1]  # repeat the first angle at end to close the circle
    # Initialize the radar chart
    fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))
    # Plot each species' average values on the radar
    for species_name in data_means.index:
        values = data_means.loc[species_name].tolist()
        values += values[:1]  # repeat first value at end to close the loop
        ax.plot(angles, values, label=species_name)
        ax.fill(angles, values, alpha=0.2)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, fontsize=10)
    plt.title("Radar Chart for Species Features")
    plt.legend(loc="upper right")
    plt.show()
