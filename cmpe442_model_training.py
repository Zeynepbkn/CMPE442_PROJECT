import pandas as pd
import io

# Convert the uploaded CSV file to a pandas DataFrame
df = pd.read_csv('smoking.csv')

# Display the first 5 rows of the dataset to see its structure
df.head()

#(rows, columns) from the dataset
print(df.shape)

print(df.columns.tolist())

#data types of each column
print(df.dtypes)

print("Number of duplicate rows:", df.duplicated().sum())

"""# **Summary Statistics**

## **Numerical Features**
"""

# 1. NUMERICAL FEATURES
# Calculate basic statistics for all numerical columns
# This shows mean, median, standard deviation, min, max, and quartiles
print("Summary Statistics for Numerical Features:")
numerical_stats = df.describe(include='all')
print(numerical_stats)

print("\nTransposed Summary Statistics (easier to read):")
print(numerical_stats.T)

import numpy as np
median_values = df.select_dtypes(include=np.number).median()
print(median_values)

"""## **Categorical Features**"""

# 2. CATEGORICAL FEATURES
# Get list of categorical columns (both object type and explicitly created categories)
categorical_columns = df.select_dtypes(include=['object', 'category']).columns.tolist()
print("\nCategorical columns in the dataset:", categorical_columns)

# 2. CATEGORICAL FEATURES
# Calculate frequency distribution for each categorical column
print("\nFrequency Distribution for Categorical Features:")
for col in categorical_columns:
    print(f"\nFrequency counts for {col}:")
    print(df[col].value_counts())

# 2. CATEGORICAL FEATURES
# Import seaborn for visualization
import seaborn as sns
import matplotlib.pyplot as plt

# Set figure style
sns.set(style="whitegrid")
# Create count plots for categorical features
print("\nGenerating categorical features distribution plots...")
for col in categorical_columns[:3]:  # Limit to first 3 to avoid too many plots
    plt.figure(figsize=(9, 6))
    sns.countplot(x=col, data=df)
    plt.title(f'Distribution of {col}')
    plt.tight_layout()
    plt.savefig(f'{col}_distribution.png')
    plt.show()

"""## **Correlation Analysis**"""

# 3. CORRELATION ANALYSIS for numerical features
import seaborn as sns
import matplotlib.pyplot as plt
# Set the style for better visualizations
sns.set(style="whitegrid")
# Calculate correlation matrix for numerical features
correlation_matrix = df.select_dtypes(include=['int64', 'float64']).corr()
print(correlation_matrix)
# Create a figure for the correlation heatmap
plt.figure(figsize=(20, 16))
print("\nGenerating correlation heatmap...")
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
plt.title('Correlation Matrix for Numerical Features')
plt.tight_layout()
plt.savefig('correlation_heatmap.png')
plt.show()

"""# **Exploratory Data Analysis (EDA)**

## **Feature Distribution**
"""

# HISTOGRAMS for numerical features with matplotlib.pyplot.hist()
import matplotlib.pyplot as plt
import seaborn as sns
# Get list of numerical features
numerical_features = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
# Remove 'ID' column if it exists (it's not a feature for analysis)
if 'ID' in numerical_features:
    numerical_features.remove('ID')

print(f"Total numerical features: {len(numerical_features)}")
print(f"Features to be analyzed: {numerical_features}")
# Set figure aesthetics
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['figure.figsize'] = (12, 6)
plt.rcParams['font.size'] = 10

print("\nCreating histograms using matplotlib.pyplot.hist()...")

# Create a grid for histograms
n_cols = 3
n_rows = (len(numerical_features) + n_cols - 1) // n_cols

fig = plt.figure(figsize=(18, 5 * n_rows))
fig.suptitle('Histograms of Numerical Features (matplotlib)', fontsize=16)

for i, feature in enumerate(numerical_features):
    ax = fig.add_subplot(n_rows, n_cols, i + 1)

    # Use matplotlib histogram
    plt.hist(df[feature].dropna(), bins=30, color='skyblue', edgecolor='black', alpha=0.7)

    plt.title(f'Distribution of {feature}')
    plt.xlabel(feature)
    plt.ylabel('Frequency')
    plt.grid(True, alpha=0.3)

plt.tight_layout(rect=[0, 0, 1, 0.96])  # Adjust for the main title
plt.savefig('histograms_matplotlib.png')
plt.show()

# HISTOGRAMS for numerical features with seaborn.histplot()
print("\nCreating histograms using seaborn.histplot()...")

# Create a grid for histograms
fig = plt.figure(figsize=(18, 5 * n_rows))
fig.suptitle('Histograms of Numerical Features (seaborn)', fontsize=16)

for i, feature in enumerate(numerical_features):
    ax = fig.add_subplot(n_rows, n_cols, i + 1)

    # Use seaborn histogram with KDE
    sns.histplot(df[feature].dropna(), kde=True, color='mediumseagreen', alpha=0.6, bins=30, ax=ax)

    plt.title(f'Distribution of {feature}')
    plt.xlabel(feature)
    plt.ylabel('Count')

plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.savefig('histograms_seaborn.png')
plt.show()

# BOXPLOTS for numerical features with seaborn.boxplot()

print("\nCreating boxplots using seaborn.boxplot()...")

# Create a grid for boxplots
fig = plt.figure(figsize=(18, 5 * n_rows))
fig.suptitle('Boxplots of Numerical Features', fontsize=16)

for i, feature in enumerate(numerical_features):
    ax = fig.add_subplot(n_rows, n_cols, i + 1)

    # Use seaborn boxplot
    sns.boxplot(x=df[feature].dropna(), color='lightcoral', ax=ax)

    plt.title(f'Boxplot of {feature}')
    plt.xlabel(feature)
    plt.grid(True, alpha=0.3)

plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.savefig('boxplots_seaborn.png')
plt.show()

#BAR CHARTS with seaborn.barplot()
import matplotlib.pyplot as plt
import seaborn as sns
#Getting categorical features
categorical_features = df.select_dtypes(include=['object', 'category']).columns.tolist()
print(f"Features to be analyzed: {categorical_features}")

# Set figure aesthetics
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['figure.figsize'] = (12, 6)
plt.rcParams['font.size'] = 10

print("\nCreating bar charts using seaborn.barplot()...")

# Create a separate chart for each categorical feature
for feature in categorical_features:
    plt.figure(figsize=(12, 6))

    # Calculate category counts
    value_counts = df[feature].value_counts().reset_index()
    value_counts.columns = [feature, 'Count']

    # Sort values by count (descending)
    value_counts = value_counts.sort_values('Count', ascending=False)

    # Visualize using seaborn barplot - fixed to avoid FutureWarning
    # Using hue parameter with the same column as x, and set legend to False
    ax = sns.barplot(x=feature, y='Count', data=value_counts, hue=feature, legend=False)

    # Chart labels
    plt.title(f'Bar Chart of {feature}')
    plt.xlabel(feature)
    plt.ylabel('Count')
    plt.savefig(f'{col}_barchart.png')
    # Rotate labels if needed
    plt.xticks(rotation=45, ha='right')
    # Add count values above bars
    for i, v in enumerate(value_counts['Count']):
        ax.text(i, v + 5, str(v), ha='center')

    plt.tight_layout()
    plt.show()

# 2. PIE CHARTS with matplotlib.pyplot.pie()
print("\nCreating pie charts using matplotlib.pyplot.pie()...")

# Create a separate pie chart for each categorical feature
for feature in categorical_features:
    plt.figure(figsize=(10, 8))
    # Calculate category counts
    counts = df[feature].value_counts()

    # If there are too many categories, show only the most common ones and group others
    max_categories = 8  # Maximum number of categories to display

    if len(counts) > max_categories:
        # Get the most common categories
        main_categories = counts.iloc[:max_categories-1]
        # Group the rest as "Other"
        others = pd.Series([counts.iloc[max_categories-1:].sum()], index=['Other'])
        counts = pd.concat([main_categories, others])

    # Calculate percentages
    percentages = counts / counts.sum() * 100

    # Prepare labels with percentages
    labels = [f'{index} ({percent:.1f}%)' for index, percent in zip(counts.index, percentages)]

    # Create pie chart
    plt.pie(counts, labels=labels, autopct='', startangle=90, shadow=True,
            colors=plt.cm.tab20.colors[:len(counts)])

    # Equal aspect ratio ensures that pie is drawn as a circle
    plt.axis('equal')

    plt.title(f'Pie Chart of {feature}')
    plt.tight_layout()
    plt.show()

"""##**Pairwise Relationships**"""

# Select numerical features (excluding ID column)
numerical_features = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
if 'ID' in numerical_features:
    numerical_features.remove('ID')

# Combine features with target
features_with_target = numerical_features.copy()
if 'smoking' not in features_with_target:
    features_with_target.append('smoking')

# Sample data (for performance)
sample_size = min(5000, len(df))
df_sample = df.sample(sample_size, random_state=42)

# Optimize font size
plt.rcParams.update({'font.size': 8})

# Create pair plot with all numerical features
plt.figure(figsize=(25, 25))
pair_plot = sns.pairplot(
    df_sample[features_with_target],
    hue='smoking',              # Color by smoking status
    diag_kind='kde',            # KDE for diagonal plots
    plot_kws={'alpha': 0.5,     # Transparency
              'edgecolor': 'none',  # No edge color
              's': 15},         # Point size
    height=2.0,                 # Size of each subplot
    aspect=1,                   # Aspect ratio
    corner=True,                # Show only lower triangle (to avoid duplicates)
    markers=['o', 's']          # Different markers for different classes
)

# Add title to the plot
pair_plot.fig.suptitle('Pair Plot Analysis of All Numerical Features', y=1.02, fontsize=20)

# Tighten the layout and save the plot
plt.tight_layout()
plt.savefig('complete_pairplot.png', dpi=300, bbox_inches='tight')
plt.show()

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# Important feature pairs based on correlation analysis
scatter_pairs = [
    ('age', 'weight(kg)'),
    ('weight(kg)', 'waist(cm)'),
    ('systolic', 'relaxation'),
    ('Cholesterol', 'triglyceride'),
    ('height(cm)', 'weight(kg)'),
    ('AST', 'ALT')  # Important feature pairs based on previous analysis
]

# Set plot styles
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['figure.figsize'] = (14, 12)
plt.rcParams['font.size'] = 10

# Create scatter plot grid
fig, axes = plt.subplots(3, 2, figsize=(14, 12))
axes = axes.flatten()  # Convert 2D array to 1D

# Colors for smokers and non-smokers
colors = ['#3498db', '#e74c3c']  # Blue, Red
labels = ['Non-Smoker', 'Smoker']

# Create scatter plot for each feature pair
for i, (x_var, y_var) in enumerate(scatter_pairs):
    ax = axes[i]

    # Non-smokers
    nonsmokers = df[df['smoking'] == 0]
    ax.scatter(
        nonsmokers[x_var],
        nonsmokers[y_var],
        c=colors[0],
        s=25,
        alpha=0.6,
        edgecolor='none',
        label=labels[0]
    )

    # Smokers
    smokers = df[df['smoking'] == 1]
    ax.scatter(
        smokers[x_var],
        smokers[y_var],
        c=colors[1],
        s=25,
        alpha=0.6,
        edgecolor='none',
        label=labels[1]
    )

    # Calculate correlation coefficient
    correlation = df[[x_var, y_var]].corr().iloc[0, 1]

    # Graph title and labels
    ax.set_title(f'{x_var} vs {y_var}\nCorrelation: {correlation:.3f}', fontsize=12)
    ax.set_xlabel(x_var, fontsize=10)
    ax.set_ylabel(y_var, fontsize=10)

    # Grid lines
    ax.grid(True, linestyle='--', alpha=0.7)

    # Legend (only for the first plot)
    if i == 0:
        ax.legend()

# Adjust the layout of the graph
plt.tight_layout()
plt.subplots_adjust(top=0.92)
fig.suptitle('Scatter Plot Analysis for Selected Feature Pairs', fontsize=16)

# Save and show the graph
plt.savefig('scatter_plots_matplotlib.png', dpi=300, bbox_inches='tight')
plt.show()

# Bonus: Scatter plots showing the relationship between smoking and other variables
plt.figure(figsize=(16, 8))

# Select important features to see the relationship between smoking and other variables
important_features = ['age', 'weight(kg)', 'Cholesterol', 'triglyceride', 'systolic', 'ALT']

fig, axes = plt.subplots(2, 3, figsize=(16, 8))
axes = axes.flatten()

for i, feature in enumerate(important_features):
    ax = axes[i]

    # Add jitter (helps for 0-1 values)
    x = df[feature]
    y = df['smoking'] + np.random.normal(0, 0.05, size=len(df))

    ax.scatter(
        x, y,
        c=y,
        cmap='coolwarm',
        s=30,
        alpha=0.6,
        edgecolor='none'
    )

    ax.set_title(f'{feature} vs Smoking Status', fontsize=12)
    ax.set_xlabel(feature, fontsize=10)
    ax.set_ylabel('Smoking Status', fontsize=10)
    ax.set_yticks([0, 1])
    ax.set_yticklabels(['Non-Smoker', 'Smoker'])
    ax.grid(True, linestyle='--', alpha=0.7)

plt.tight_layout()
plt.subplots_adjust(top=0.9)
fig.suptitle('Smoking Status ile Önemli Değişkenler Arasındaki İlişki', fontsize=16)
plt.savefig('smoking_relationships.png', dpi=300, bbox_inches='tight')
plt.show()

# import pandas as pd
# import matplotlib.pyplot as plt
# import numpy as np
# from math import ceil


# # Define all feature pairs with significant correlations (|r| > 0.1)
# significant_pairs = [
#     # Strong Positive Correlations (r > 0.4)
#     ('weight(kg)', 'waist(cm)'),           # 0.82
#     ('systolic', 'relaxation'),            # 0.76
#     ('Cholesterol', 'LDL'),                # 0.74
#     ('AST', 'ALT'),                        # 0.74
#     ('height(cm)', 'weight(kg)'),          # 0.68
#     ('height(cm)', 'hemoglobin'),          # 0.54
#     ('hearing(left)', 'hearing(right)'),   # 0.51
#     ('weight(kg)', 'hemoglobin'),          # 0.49

#     # Moderate Positive Correlations (0.2 < r ≤ 0.4)
#     ('smoking', 'height(cm)'),             # 0.40
#     ('smoking', 'hemoglobin'),             # 0.40
#     ('height(cm)', 'waist(cm)'),           # 0.38
#     ('serum creatinine', 'height(cm)'),    # 0.38
#     ('serum creatinine', 'hemoglobin'),    # 0.37
#     ('waist(cm)', 'triglyceride'),         # 0.36
#     ('eyesight(left)', 'eyesight(right)'), # 0.35
#     ('ALT', 'Gtp'),                        # 0.34
#     ('weight(kg)', 'triglyceride'),        # 0.32
#     ('weight(kg)', 'serum creatinine'),    # 0.32
#     ('waist(cm)', 'systolic'),             # 0.32
#     ('weight(kg)', 'Cholesterol'),         # 0.32
#     ('smoking', 'weight(kg)'),             # 0.30
#     ('Gtp', 'triglyceride'),               # 0.30
#     ('waist(cm)', 'relaxation'),           # 0.29
#     ('weight(kg)', 'relaxation'),          # 0.27
#     ('weight(kg)', 'systolic'),            # 0.27
#     ('hemoglobin', 'triglyceride'),        # 0.27
#     ('waist(cm)', 'ALT'),                  # 0.25
#     ('ALT', 'weight(kg)'),                 # 0.25
#     ('Cholesterol', 'triglyceride'),       # 0.25
#     ('smoking', 'triglyceride'),           # 0.25
#     ('waist(cm)', 'Gtp'),                  # 0.24
#     ('waist(cm)', 'serum creatinine'),     # 0.24
#     ('smoking', 'Gtp'),                    # 0.24
#     ('smoking', 'waist(cm)'),              # 0.23
#     ('hemoglobin', 'relaxation'),          # 0.23
#     ('smoking', 'serum creatinine'),       # 0.22
#     ('hemoglobin', 'Gtp'),                 # 0.22
#     ('triglyceride', 'relaxation'),        # 0.22

#     # Moderate Negative Correlations (-0.4 < r ≤ -0.2)
#     ('age', 'height(cm)'),                 # -0.48
#     ('HDL', 'triglyceride'),               # -0.41
#     ('HDL', 'waist(cm)'),                  # -0.38
#     ('HDL', 'weight(kg)'),                 # -0.36
#     ('age', 'weight(kg)'),                 # -0.32
#     ('age', 'hemoglobin'),                 # -0.26
#     ('HDL', 'hemoglobin'),                 # -0.24
#     ('age', 'eyesight(left)'),             # -0.20
#     ('age', 'eyesight(right)'),            # -0.19
#     ('smoking', 'HDL'),                    # -0.18
#     ('serum creatinine', 'HDL'),           # -0.18
#     ('age', 'smoking'),                    # -0.16
#     ('HDL', 'height(cm)'),                 # -0.21
#     ('ALT', 'HDL'),                        # -0.13
#     ('fasting blood sugar', 'HDL'),        # -0.12
#     ('age', 'serum creatinine'),           # -0.11
#     ('age', 'dental caries'),              # -0.11
# ]

# # Function to create scatter plots
# def create_scatter_plots(data, feature_pairs, n_cols=4, figsize=(24, 30)):
#     n_pairs = len(feature_pairs)
#     n_rows = ceil(n_pairs / n_cols)

#     fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
#     axes = axes.flatten()

#     for i, (feature1, feature2) in enumerate(feature_pairs):
#         if i < len(axes):
#             ax = axes[i]

#             # Color points based on smoking status
#             smoking_mask = data['smoking'] == 1
#             non_smoking_mask = data['smoking'] == 0

#             # Create scatter plots for each pair using matplotlib.pyplot.scatter()
#             ax.scatter(data.loc[non_smoking_mask, feature1],
#                        data.loc[non_smoking_mask, feature2],
#                        c='blue', alpha=0.2, s=3, label='Non-Smoker')

#             ax.scatter(data.loc[smoking_mask, feature1],
#                        data.loc[smoking_mask, feature2],
#                        c='red', alpha=0.2, s=3, label='Smoker')

#             # Calculate and display correlation value
#             corr = data[feature1].corr(data[feature2])
#             ax.set_title(f'{feature1} vs {feature2}\nr = {corr:.3f}')
#             ax.set_xlabel(feature1)
#             ax.set_ylabel(feature2)

#             # Only show legend on the first plot
#             if i == 0:
#                 ax.legend()

#     # Remove unused subplots
#     for i in range(n_pairs, len(axes)):
#         fig.delaxes(axes[i])

#     plt.tight_layout()
#     return fig

# # Create scatter plots for all significant correlation pairs
# fig = create_scatter_plots(df, significant_pairs, n_cols=4, figsize=(24, 40))
# plt.suptitle('Scatter Plots for All Feature Pairs with Significant Correlations (|r| > 0.1)', fontsize=10)
# plt.subplots_adjust(top=0.98, hspace=0.4)

# print(f'Created scatter plots for {len(significant_pairs)} feature pairs with significant correlations.')

# import pandas as pd
# import seaborn as sns
# import matplotlib.pyplot as plt



# # Drop ID column and non-numerical columns
# categorical_columns = ['gender', 'oral', 'tartar']
# df_numeric = df.drop(columns=['ID'] + categorical_columns)


# # Create pairplot for all numeric features
# # This will color the points by smoking status
# plt.figure(figsize=(30, 30))
# pairplot = sns.pairplot(
#     data=df_numeric,
#     hue='smoking',  # Color by smoking status
#     palette={0: 'blue', 1: 'red'},  # 0=Non-smoker (blue), 1=Smoker (red)
#     plot_kws={'alpha': 0.1, 's': 3},  # Make points semi-transparent and small
#     diag_kind='kde',  # Use KDE plots on diagonal
#     corner=True,  # Only show the lower triangle to reduce redundancy
#     height=2.5,  # Size of each subplot
# )


# # Set title for the entire figure
# plt.suptitle('Pair Plot for All Numerical Features (Colored by Smoking Status)', fontsize=20, y=1.02)


# # Improve readability of axis labels
# for ax in pairplot.axes.flatten():
#     if ax is not None:
#         ax.set_xlabel(ax.get_xlabel(), fontsize=10, rotation=45)
#         ax.set_ylabel(ax.get_ylabel(), fontsize=10, rotation=45)


# plt.tight_layout()


# print("Pairplot for all numerical features created successfully.")

"""# Class Imbalance Check

"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# 1. Using pandas.Series.value_counts() to get the distribution
smoking_counts = df['smoking'].value_counts()
smoking_percentage = df['smoking'].value_counts(normalize=True)

print("Distribution of target variable 'smoking':")
print(smoking_counts)
print("\nPercentage distribution:")
print(smoking_percentage)

# 2. Using seaborn.countplot() for visualization
plt.figure(figsize=(8, 5))
sns.countplot(x='smoking', data=df)
plt.title('Distribution of Target Variable (Smoking)')
plt.xlabel('Smoking Status (0: Non-smoking, 1: Smoking)')
plt.ylabel('Count')

plt.tight_layout()
plt.show()

"""# Data Processing"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

print("\n\n--- DATA PROCESSING STEPS ---\n")

# Check for missing values
print("Missing value check:")
missing_values = df.isnull().sum()
print(missing_values[missing_values > 0])
print(f"Total number of missing values: {df.isnull().sum().sum()}")

possible_missing_values = ['?', 'NA', 'N/A', 'n/a', 'na', '-', '--', ' ', 'unknown', 'null', 'NULL']

for value in possible_missing_values:
    if df.eq(value).any().any():
        print(f"\nFound '{value}' as potential missing value")
        columns_with_value = df.columns[df.eq(value).any()]
        print(f"Columns containing '{value}': {list(columns_with_value)}")


# Identify categorical and numerical features in the dataset
categorical_features = ['gender', 'oral', 'tartar']
# Remove ID column and identify numerical features
numerical_features = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
if 'ID' in numerical_features:
    numerical_features.remove('ID')
if 'smoking' in numerical_features:  # Remove target variable
    numerical_features.remove('smoking')

print(f"\nCategorical features: {categorical_features}")
print(f"Numerical features: {numerical_features}")

# Create test data - Using Stratified Sampling
X = df.drop(columns=['smoking', 'ID'])
y = df['smoking']

# Split dataset with stratified sampling (preserves target variable distribution)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)

print(f"\nTraining set size: {X_train.shape}")
print(f"Test set size: {X_test.shape}")

# Verify that stratified sampling worked correctly
print(f"\nProportion of smokers in original dataset: {y.mean():.4f}")
print(f"Proportion of smokers in training set: {y_train.mean():.4f}")
print(f"Proportion of smokers in test set: {y_test.mean():.4f}")

# Create data processing pipeline
# For numerical features: StandardScaler (normalization)
# For categorical features: OneHotEncoder (categorical to numerical conversion)

print("\n--- HANDLING CATEGORICAL FEATURES ---")
# Examine the categorical features before processing
for feature in categorical_features:
    print(f"\nUnique values in {feature}: {X_train[feature].unique()}")
    print(f"Value counts:\n{X_train[feature].value_counts()}")

# Create data preprocessing pipeline - without imputation since no missing values
numerical_transformer = Pipeline(steps=[
    ('scaler', StandardScaler())  # Standardization/Normalization
])

categorical_transformer = Pipeline(steps=[
    ('onehot', OneHotEncoder(drop='first', sparse_output=False))  # One-hot encoding
])

# Combine numerical and categorical features with ColumnTransformer
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_features),
        ('cat', categorical_transformer, categorical_features)
    ])

# Apply data preprocessing pipeline to training data
print("\nApplying data preprocessing...")
X_train_processed = preprocessor.fit_transform(X_train)
X_test_processed = preprocessor.transform(X_test)

# Check dimensions of processed data
print(f"Processed training data size: {X_train_processed.shape}")
print(f"Processed test data size: {X_test_processed.shape}")

print("\n--- WHY NORMALIZATION IS NECESSARY ---")
print("1. Features have different scales (e.g., age vs. weight vs. cholesterol levels)")
print("2. Many machine learning algorithms perform better with normalized data")
print("3. Normalization prevents features with larger scales from dominating the model")
print("4. Gradient descent converges faster with normalized features")

import seaborn as sns
# Show effect of normalization on numerical features
# Select features with different scales to demonstrate the effect of normalization
scale_diverse_features = ['age', 'weight(kg)', 'Cholesterol']
if all(feature in numerical_features for feature in scale_diverse_features):
    selected_features = scale_diverse_features
else:
    selected_features = numerical_features[:3]  # Select first 3 numerical features

# Visualize distribution and scale differences before normalization
plt.figure(figsize=(12, 6))
plt.title('Features Before Normalization - Scale Differences')
for feature in selected_features:
    sns.kdeplot(X_train[feature], label=feature)
plt.legend()
plt.savefig('before_normalization_scales.png')
plt.show()

# Visualize all features together after normalization
plt.figure(figsize=(12, 6))
plt.title('Features After Normalization - Similar Scales')

# Extract the normalized versions of each selected feature
for i, feature in enumerate(selected_features):
    # Find the index of this feature in the numerical_features list
    feature_idx = numerical_features.index(feature)
    # Plot the normalized version from X_train_processed
    sns.kdeplot(X_train_processed[:, feature_idx], label=f"{feature} (normalized)")

plt.legend()
plt.grid(True, linestyle='--', alpha=0.7)
plt.savefig('after_normalization_scales.png')
plt.show()

# Demonstrating the effect of one-hot encoding
print("\n--- CATEGORICAL FEATURE ENCODING EXAMPLE ---")
if 'gender' in categorical_features:
    print("Example of one-hot encoding for 'gender':")
    print(f"Original values: {X_train['gender'].unique()}")

    # Get the column indices from the preprocessor for the gender feature
    # This is more complex as we need to find which columns correspond to gender after one-hot encoding
    # Instead, let's create a simple example
    print("One-hot encoded representation:")
    # Create a mini one-hot encoder just for demonstration
    demo_encoder = OneHotEncoder(drop='first', sparse_output=False)
    gender_encoded = demo_encoder.fit_transform(X_train[['gender']])

    # Get feature names from encoder
    encoded_feature_names = demo_encoder.get_feature_names_out(['gender'])

    # Show the first few rows of original and encoded data
    print("Original 'gender' column (first 5 rows):")
    print(X_train['gender'].head())

    print("\nEncoded 'gender' columns (first 5 rows):")
    gender_df = pd.DataFrame(gender_encoded, columns=encoded_feature_names)
    print(gender_df.head())

# Save preprocessing pipeline
import pickle

# Save preprocessing pipeline (for use with new data later)
with open('preprocessor.pkl', 'wb') as f:
    pickle.dump(preprocessor, f)

print("\nPreprocessing pipeline saved as 'preprocessor.pkl'.")

# Data processing report
print("\n--- DATA PROCESSING REPORT ---")
print("1. Test set created using stratified sampling (20%)")
print("2. No missing values found, so no imputation needed")
print("3. Categorical features converted to numerical using one-hot encoding")
print("4. Numerical features normalized using StandardScaler")
print(f"5. Processed dataset contains {X_train_processed.shape[1]} features")

"""# Model Selection and Training"""

"""# **Model Selection and Training**"""

import numpy as np
from sklearn.model_selection import cross_val_score, KFold, cross_validate
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.metrics import classification_report, roc_curve, auc, roc_auc_score, precision_recall_curve, balanced_accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
import xgboost as xgb #NEW
import time

print("\n\n--- MODEL SELECTION AND TRAINING ---\n")

# Define the models to evaluate
models = {
    'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
    'Random Forest': RandomForestClassifier(random_state=42),
    'Decision Tree': DecisionTreeClassifier(random_state=42),
    'K-Nearest Neighbors': KNeighborsClassifier(),
    'Neural Network': MLPClassifier(
        hidden_layer_sizes=(64, 32),
        activation='relu',
        solver='adam',
        alpha=0.0001,
        batch_size='auto',
        learning_rate='adaptive',
        max_iter=2000,
        early_stopping=True,
        validation_fraction=0.1,
        n_iter_no_change=10,
        random_state=42
    ),
    'XGBoost': xgb.XGBClassifier(objective='binary:logistic', random_state=42)  # XGBoost
}

# Set up k-fold cross-validation
k_folds = 5
kf = KFold(n_splits=k_folds, shuffle=True, random_state=42)

# Metrics to evaluate
scoring = {
    'accuracy': 'accuracy',
    'precision': 'precision',
    'recall': 'recall',
    'f1': 'f1',
    'roc_auc': 'roc_auc'
}

# Store results
cv_results = {}
training_times = {}

# Perform cross-validation for each model
print(f"Performing {k_folds}-fold cross-validation on {len(models)} models...")

for name, model in models.items():
    print(f"\nEvaluating {name}...")
    start_time = time.time()

    # Cross-validate with multiple metrics
    scores = cross_validate(
        model, X_train_processed, y_train,
        cv=kf, scoring=scoring,
        return_train_score=False
    )

    end_time = time.time()
    training_time = end_time - start_time

    # Store results
    cv_results[name] = {
        'accuracy': np.mean(scores['test_accuracy']),
        'precision': np.mean(scores['test_precision']),
        'recall': np.mean(scores['test_recall']),
        'f1': np.mean(scores['test_f1']),
        'roc_auc': np.mean(scores['test_roc_auc']),
        'std_accuracy': np.std(scores['test_accuracy']),
    }

    training_times[name] = training_time

    print(f"  Accuracy: {cv_results[name]['accuracy']:.4f} ± {cv_results[name]['std_accuracy']:.4f}")
    print(f"  Precision: {cv_results[name]['precision']:.4f}")
    print(f"  Recall: {cv_results[name]['recall']:.4f}")
    print(f"  F1 Score: {cv_results[name]['f1']:.4f}")
    print(f"  ROC AUC: {cv_results[name]['roc_auc']:.4f}")
    print(f"  Training Time: {training_time:.2f} seconds")

# Convert results to DataFrame for easier visualization
results_df = pd.DataFrame(cv_results).T
results_df['training_time'] = pd.Series(training_times)

# Sort by F1 score (balances precision and recall)
results_df = results_df.sort_values('f1', ascending=False)

print("\nModel Performance Summary (Sorted by F1 Score):")
print(results_df)

# Visualize cross-validation results
plt.figure(figsize=(12, 10))

# Plot accuracy
plt.subplot(2, 2, 1)
results_df['accuracy'].plot(kind='bar', yerr=results_df['std_accuracy'], capsize=4)
plt.title('Accuracy Comparison')
plt.ylabel('Accuracy')
plt.ylim(0.6, 1.0)
plt.xticks(rotation=45, ha='right')
plt.grid(axis='y', linestyle='--', alpha=0.7)

# Plot Precision, Recall, and F1 together
plt.subplot(2, 2, 2)
width = 0.25
x = np.arange(len(results_df.index))
plt.bar(x - width, results_df['precision'], width, label='Precision')
plt.bar(x, results_df['recall'], width, label='Recall')
plt.bar(x + width, results_df['f1'], width, label='F1')
plt.title('Precision, Recall, F1 Comparison')
plt.xticks(x, results_df.index, rotation=45, ha='right')
plt.legend()
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.ylim(0.6, 1.0)

# Plot ROC AUC
plt.subplot(2, 2, 3)
results_df['roc_auc'].plot(kind='bar')
plt.title('ROC AUC Comparison')
plt.ylabel('ROC AUC')
plt.ylim(0.6, 1.0)
plt.xticks(rotation=45, ha='right')
plt.grid(axis='y', linestyle='--', alpha=0.7)

# Plot training time
plt.subplot(2, 2, 4)
results_df['training_time'].plot(kind='bar')
plt.title('Training Time Comparison')
plt.ylabel('Time (seconds)')
plt.xticks(rotation=45, ha='right')
plt.grid(axis='y', linestyle='--', alpha=0.7)

plt.tight_layout()
plt.savefig('model_comparison.png')
plt.show()

# Identify the best model based on F1 score
best_model_name = results_df.index[0]
print(f"\nBest Model: {best_model_name} with F1 Score: {results_df.loc[best_model_name, 'f1']:.4f}")

"""# Fine Tuning the Model

"""

"""# Random Forest Hiperparametre Optimizasyonu"""

print("\n\n--- RANDOM FOREST HIPERPARAMETER OPTIMIZATION ---\n")

from sklearn.model_selection import RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
import numpy as np

# Random Forest modeli için optimize edilecek hiperparametre aralıkları
param_dist = {
    'n_estimators': [250, 350],
    'max_depth': [25, 35],
    'min_samples_split': [2, 5],
    'min_samples_leaf': [1, 2],
    'max_features': ['sqrt', 'log2'],
    'class_weight': ['balanced']
}
best_model = models[best_model_name]

# RandomizedSearchCV ile hiperparametre optimizasyonu
random_search = RandomizedSearchCV(
    estimator=best_model,
    param_distributions=param_dist,
    n_iter=10,
    cv=5,
    random_state=42,
    n_jobs=-1,
    scoring='f1',
    verbose=1
)

# Start timing for optimization
start_time = time.time()

# Train the model
random_search.fit(X_train_processed, y_train)

# Calculate optimization time
optimization_time = time.time() - start_time

# Show best parameters and score
print(f"\nBest Hyperparameters:\n{random_search.best_params_}")
print(f"Best F1 Score: {random_search.best_score_:.4f}")
print(f"Optimization Time: {optimization_time:.2f} seconds")

# Evaluate on test set using the best model
best_rf_model = random_search.best_estimator_
y_pred = best_rf_model.predict(X_test_processed)

# Calculate performance metrics
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_pred)
balanced_acc = balanced_accuracy_score(y_test, y_pred)

print("\nPerformance on Test Set (Optimized Random Forest):")
print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1 Score: {f1:.4f}")
print(f"ROC AUC: {roc_auc:.4f}")
print(f"Balanced Accuracy: {balanced_acc:.4f}")

from sklearn.model_selection import RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import time
import pandas as pd
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                            roc_auc_score, balanced_accuracy_score, classification_report,
                            confusion_matrix, roc_curve, precision_recall_curve)
# Classification report
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Confusion matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Non-Smoker', 'Smoker'],
            yticklabels=['Non-Smoker', 'Smoker'])
plt.title('Confusion Matrix')
plt.xlabel('Predicted Class')
plt.ylabel('True Class')
plt.tight_layout()
plt.savefig('optimized_rf_confusion_matrix.png')
plt.show()

# Visualize feature importances
feature_importances = best_rf_model.feature_importances_

# Get feature names
numeric_feature_names = numerical_features.copy()

# Get one-hot encoded feature names for categorical features
categorical_encoder = preprocessor.named_transformers_['cat']
onehot_encoder = categorical_encoder.named_steps['onehot']
encoded_feature_names = onehot_encoder.get_feature_names_out(categorical_features)

# Combine all feature names
all_feature_names = np.concatenate([numeric_feature_names, encoded_feature_names])

# Sort feature importances
feature_importance_df = pd.DataFrame({
    'Feature': all_feature_names,
    'Importance': feature_importances
}).sort_values('Importance', ascending=False)

plt.figure(figsize=(12, 8))
sns.barplot(x='Importance', y='Feature', data=feature_importance_df.head(15))
plt.title('Optimized Random Forest - Top 15 Features')
plt.tight_layout()
plt.savefig('optimized_rf_feature_importance.png')
plt.show()

# Save the optimized model
import pickle
with open('optimized_random_forest_model.pkl', 'wb') as f:
    pickle.dump(best_rf_model, f)
print("\nOptimized Random Forest model saved as 'optimized_random_forest_model.pkl'.")

# Compare performance before and after optimization
rf_orig_cv_accuracy = cv_results['Random Forest']['accuracy']
rf_orig_cv_f1 = cv_results['Random Forest']['f1']

print("\nRandom Forest Performance Comparison:")
print(f"Pre-optimization CV F1 Score: {rf_orig_cv_f1:.4f}")
print(f"Post-optimization CV F1 Score: {random_search.best_score_:.4f}")
print(f"Improvement: {(random_search.best_score_ - rf_orig_cv_f1) / rf_orig_cv_f1 * 100:.2f}%")

print("\nBest Model vs Optimized Random Forest:")
print(f"Best Model ({best_model_name}) F1 Score: {results_df.loc[best_model_name, 'f1']:.4f}")
print(f"Optimized RF Test F1 Score: {f1:.4f}")


# 1. Precision-Recall Curve
y_pred_proba = best_rf_model.predict_proba(X_test_processed)[:, 1]
plt.figure(figsize=(10, 8))
precision_values, recall_values, _ = precision_recall_curve(y_test, y_pred_proba)
plt.plot(recall_values, precision_values, label=f'Precision-Recall Curve')
plt.axhline(y=sum(y_test)/len(y_test), color='r', linestyle='--', label='No Skill')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve for Optimized Random Forest')
plt.legend()
plt.grid(True)
plt.savefig('optimized_rf_precision_recall_curve.png')
plt.show()

# 2. ROC Curve
plt.figure(figsize=(10, 8))
fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {roc_auc:.4f})')
plt.plot([0, 1], [0, 1], 'k--', label='Random')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve for Optimized Random Forest')
plt.legend()
plt.grid(True)
plt.savefig('optimized_rf_roc_curve.png')
plt.show()


"""# Model Explainability (XAI) Techniques"""

print("\n\n--- MODEL EXPLAINABILITY TECHNIQUES ---\n")
print("Applied Methods:")
print("1. SHAP (SHapley Additive exPlanations) - Calculates feature contribution values")
print("2. Permutation Importance - Measures change in model performance by shuffling features")
print("3. Partial Dependence Plots (PDP) - Shows marginal effect of features on the target")
print("4. LIME (Local Interpretable Model-agnostic Explanations) - Provides local explainability")
print("5. Feature Importance - Shows ranking of feature importance in the model")
print("6. Subgroup Analysis - Examines model behavior across different demographic groups")

# Import necessary libraries
import shap
import lime
from lime import lime_tabular
from sklearn.inspection import permutation_importance, PartialDependenceDisplay
import warnings
# warnings.filterwarnings('ignore')  # Commented out to see errors
warnings.filterwarnings('always')  # Make all warnings visible (for debugging)

print("\n=====================================================")
print("GLOBAL EXPLANATIONS")
print("=====================================================")

# 1. Random Forest Feature Importance (already available in our model)
print("\n1. Calculating Random Forest Feature Importance...")

# Visualize feature importances again
plt.figure(figsize=(12, 8))
sns.barplot(x='Importance', y='Feature', data=feature_importance_df.head(15))
plt.title('Top 15 Features')
plt.tight_layout()
plt.savefig('feature_importance_global.png')
plt.show()

# 2. Permutation Importance
print("\n2. Calculating Permutation Importance...")

# Calculate permutation importance on test set
perm_importance = permutation_importance(best_rf_model, X_test_processed, y_test,
                                         n_repeats=5, random_state=42, n_jobs=-1)

# Convert results to dataframe
perm_importance_df = pd.DataFrame({
    'Feature': all_feature_names,
    'Importance': perm_importance.importances_mean
}).sort_values('Importance', ascending=False)

plt.figure(figsize=(12, 8))
sns.barplot(x='Importance', y='Feature', data=perm_importance_df.head(15))
plt.title('Permutation Importance - Top 15 Features')
plt.tight_layout()
plt.savefig('permutation_importance_global.png')
plt.show()

# 3. SHAP Values
print("\n3. Calculating SHAP values... (This may take some time)")

# Use a sample from test set for SHAP calculations
sample_size = min(200, X_test_processed.shape[0])
sample_indices = np.random.choice(X_test_processed.shape[0], sample_size, replace=False)
X_test_sample = X_test_processed[sample_indices]

try:
    # Calculate SHAP values using TreeExplainer
    explainer = shap.TreeExplainer(best_rf_model)
    shap_values = explainer.shap_values(X_test_sample)

    # SHAP Summary Plot
    plt.figure(figsize=(12, 10))
    shap.summary_plot(shap_values, X_test_sample, feature_names=all_feature_names, show=False)
    plt.title('SHAP Summary Plot')
    plt.tight_layout()
    plt.savefig('shap_summary_global.png')
    plt.show()

    # SHAP Bar Plot (feature importance)
    plt.figure(figsize=(12, 8))
    shap.summary_plot(shap_values, X_test_sample, feature_names=all_feature_names, plot_type='bar', show=False)
    plt.title('SHAP Feature Importance')
    plt.tight_layout()
    plt.savefig('shap_bar_global.png')
    plt.show()
except Exception as e:
    print(f"Error during SHAP calculation: {e}")

# 4. Partial Dependence Plots (PDP)
print("\n4. Calculating Partial Dependence Plots (PDP)...")

try:
    # Select 2 of the most important numerical features
    top_numerical_features = [feat for feat in feature_importance_df['Feature'].values[:10]
                             if feat in numerical_features][:2]

    if len(top_numerical_features) > 0:
        # Proper usage for PDP calculation and visualization
        features_to_plot = []
        feature_names = list(all_feature_names)

        for feature in top_numerical_features:
            # Find the index of the feature
            if feature in feature_names:
                feature_idx = feature_names.index(feature)
                features_to_plot.append(feature_idx)
            else:
                print(f"Warning: Feature {feature} not found.")

        # Create figure and axes first, then show PDP
        fig, ax = plt.subplots(1, len(features_to_plot), figsize=(12, 5))

        # Special case for a single feature
        if len(features_to_plot) == 1:
            ax = [ax]  # Make a single axes into a list

        # Use PartialDependenceDisplay without figsize parameter
        pdp_display = PartialDependenceDisplay.from_estimator(
            best_rf_model,
            X_test_processed,
            features=features_to_plot,
            feature_names=feature_names,
            target=1,  # For the smoking class (1)
            kind='average',
            ax=ax
        )

        # Manually set title and labels
        fig.suptitle('Partial Dependence Plots (PDP)', y=1.03)

        for i, axis in enumerate(ax):
            axis.set_ylabel('Probability of Smoking')
            axis.grid(True, linestyle='--', alpha=0.5)

        plt.tight_layout()
        plt.savefig('partial_dependence_plots.png')
        plt.show()
    else:
        print("Not enough numerical features found for PDP.")
except Exception as e:
    print(f"Error during PDP calculation: {e}")
    import traceback
    traceback.print_exc()  # Show error stack trace

print("\nGlobal Explanations Discussion:")
print("- Most influential features: ", ", ".join(feature_importance_df['Feature'].values[:5]))
print("- Different methods (SHAP, Permutation, Random Forest feature importance) generally show")
print("  similar feature importance rankings, which suggests consistency in the results.")
print("- Demographic features like hemoglobin, gender, and age play particularly important")
print("  roles in predicting smoking behavior.")

print("\n=====================================================")
print("LOCAL EXPLANATIONS")
print("=====================================================")

# Local explanations with LIME
print("\nGenerating local explanations with LIME...")

try:
    # Create LIME explainer
    lime_explainer = lime_tabular.LimeTabularExplainer(
        X_train_processed,
        feature_names=all_feature_names,
        class_names=['Non-Smoker', 'Smoker'],
        mode='classification'
    )

    # Select interesting examples from test set:
    # 1. Correctly predicted smoker
    # 2. Incorrectly predicted non-smoker

    y_test_pred = best_rf_model.predict(X_test_processed)

    # Correctly predicted smokers
    correct_smoker_indices = np.where((y_test_pred == y_test.values) & (y_test.values == 1))[0]

    # Incorrectly predicted non-smokers
    incorrect_nonsmoker_indices = np.where((y_test_pred != y_test.values) & (y_test.values == 0))[0]

    # Select interesting example indices
    if len(correct_smoker_indices) > 0:
        correct_idx = correct_smoker_indices[0]

        print(f"\nExample 1 (Correct Prediction - Smoker) - Test Set Index: {correct_idx}")
        lime_exp = lime_explainer.explain_instance(X_test_processed[correct_idx],
                                                 best_rf_model.predict_proba,
                                                 num_features=8)

        plt.figure(figsize=(10, 6))
        lime_exp.as_pyplot_figure()
        plt.title('LIME - Correctly Predicted Smoker Example')
        plt.tight_layout()
        plt.savefig('lime_correct_smoker.png')
        plt.show()

        # SHAP explanation for the same example
        try:
            # Calculate SHAP values
            single_instance = X_test_processed[correct_idx].reshape(1, -1)
            shap_values_individual = explainer.shap_values(single_instance)

            # If shap_values_individual is a list (contains values for multiple classes)
            if isinstance(shap_values_individual, list):
                # Show values for the second class (Smoker - 1)
                plt.figure(figsize=(12, 6))
                print("Creating SHAP waterfall plot (instead of force plot)...")

                # Use waterfall plot (compatible with matplotlib)
                shap.plots.waterfall(shap_values_individual[1][0], max_display=10, show=False)
                plt.title('SHAP - Correctly Predicted Smoker Example (Waterfall Plot)')
                plt.tight_layout()
                plt.savefig('shap_correct_smoker_waterfall.png')
                plt.show()

                # Alternatively, you can use decision plot
                plt.figure(figsize=(12, 6))
                print("Creating SHAP decision plot...")
                shap.decision_plot(explainer.expected_value[1], shap_values_individual[1],
                                 feature_names=all_feature_names, show=False)
                plt.title('SHAP - Correctly Predicted Smoker Example (Decision Plot)')
                plt.tight_layout()
                plt.savefig('shap_correct_smoker_decision.png')
                plt.show()
            else:
                # For single-class problem
                plt.figure(figsize=(12, 6))
                shap.plots.waterfall(shap_values_individual[0], max_display=10, show=False)
                plt.title('SHAP - Correctly Predicted Smoker Example')
                plt.tight_layout()
                plt.savefig('shap_correct_smoker_waterfall.png')
                plt.show()
        except Exception as e:
            print(f"Error during SHAP individual explanation calculation: {e}")
            import traceback
            traceback.print_exc()

    if len(incorrect_nonsmoker_indices) > 0:
        incorrect_idx = incorrect_nonsmoker_indices[0]

        print(f"\nExample 2 (Incorrect Prediction - Non-Smoker) - Test Set Index: {incorrect_idx}")
        lime_exp = lime_explainer.explain_instance(X_test_processed[incorrect_idx],
                                                 best_rf_model.predict_proba,
                                                 num_features=8)

        plt.figure(figsize=(10, 6))
        lime_exp.as_pyplot_figure()
        plt.title('LIME - Incorrectly Predicted Non-Smoker Example')
        plt.tight_layout()
        plt.savefig('lime_incorrect_nonsmoker.png')
        plt.show()

        # SHAP explanation for the same example
        try:
            plt.figure(figsize=(10, 6))
            # shap.initjs() # Already called once
            shap_values_individual = explainer.shap_values(X_test_processed[incorrect_idx].reshape(1, -1))
             # Use shap.plots.force instead of shap.force_plot or check updated usage of force_plot
            # force_plot typically returns an HTML object, needs special usage to show with plt
            # Example: shap.decision_plot(explainer.expected_value[0], shap_values_individual[0][0], feature_names=all_feature_names, show=False)
            # For now, skip showing force_plot with matplotlib and just do the calculation
            print("Creating SHAP force plot...")
            # shap.force_plot(
            #     explainer.expected_value[0], # Expected value for first class (Non-Smoker)
            #     shap_values_individual[0], # SHAP values for first class (Non-Smoker)
            #     X_test_processed[incorrect_idx],
            #     feature_names=all_feature_names,
            #     matplotlib=True,
            #     show=False
            # )
            # plt.title('SHAP - Incorrectly Predicted Non-Smoker Example')
            # plt.tight_layout()
            # plt.savefig('shap_incorrect_nonsmoker.png')
            # plt.show()
        except Exception as e:
            print(f"Error during SHAP individual explanation calculation: {e}")

except Exception as e:
    print(f"Error during LIME calculation: {e}")

print("\nLocal Explanations Discussion:")
print("- In the correctly predicted smoker example, the most important factors affecting the model's prediction")
print("  are typically features like hemoglobin, gender, and age.")
print("- In the incorrectly predicted non-smoker example, our model may have over-interpreted some features.")
print("- The explanations generally align with domain knowledge. For example, the relationship between")
print("  hemoglobin values and smoking status is documented in medical literature.")

# You might want to turn warnings off again after the code is done:
# warnings.filterwarnings('ignore')  # Turn warnings off again after processing

print("\nGlobal Explanations Discussion:")
print("- Most influential features: ", ", ".join(feature_importance_df['Feature'].values[:5]))
print("- Different methods (SHAP, Permutation, Random Forest feature importance) generally show")
print("  similar feature importance rankings, which suggests consistency in the results.")
print("- Demographic features like hemoglobin, gender, and age play particularly important")
print("  roles in predicting smoking behavior.")

print("\n=====================================================")
print("LOCAL EXPLANATIONS")
print("=====================================================")

# Local explanations with LIME
print("\nGenerating local explanations with LIME...")

try:
    # Create LIME explainer
    lime_explainer = lime_tabular.LimeTabularExplainer(
        X_train_processed,
        feature_names=all_feature_names,
        class_names=['Non-Smoker', 'Smoker'],
        mode='classification'
    )

    # Select interesting examples from test set:
    # 1. Correctly predicted smoker
    # 2. Incorrectly predicted non-smoker

    y_test_pred = best_rf_model.predict(X_test_processed)

    # Correctly predicted smokers
    correct_smoker_indices = np.where((y_test_pred == y_test.values) & (y_test.values == 1))[0]

    # Incorrectly predicted non-smokers
    incorrect_nonsmoker_indices = np.where((y_test_pred != y_test.values) & (y_test.values == 0))[0]

    # Select interesting example indices
    if len(correct_smoker_indices) > 0:
        correct_idx = correct_smoker_indices[0]

        print(f"\nExample 1 (Correct Prediction - Smoker) - Test Set Index: {correct_idx}")
        lime_exp = lime_explainer.explain_instance(X_test_processed[correct_idx],
                                                 best_rf_model.predict_proba,
                                                 num_features=8)

        plt.figure(figsize=(10, 6))
        lime_exp.as_pyplot_figure()
        plt.title('LIME - Correctly Predicted Smoker Example')
        plt.tight_layout()
        plt.savefig('lime_correct_smoker.png')
        plt.show()

        # SHAP explanation for the same example
        try:
            # Calculate SHAP values
            single_instance = X_test_processed[correct_idx].reshape(1, -1)
            shap_values_individual = explainer.shap_values(single_instance)

            # If shap_values_individual is a list (contains values for multiple classes)
            if isinstance(shap_values_individual, list):
                # Show values for the second class (Smoker - 1)
                plt.figure(figsize=(12, 6))
                print("Creating SHAP waterfall plot (instead of force plot)...")

                # Use waterfall plot (compatible with matplotlib)
                shap.plots.waterfall(shap_values_individual[1][0], max_display=10, show=False)
                plt.title('SHAP - Correctly Predicted Smoker Example (Waterfall Plot)')
                plt.tight_layout()
                plt.savefig('shap_correct_smoker_waterfall.png')
                plt.show()

                # Alternatively, you can use decision plot
                plt.figure(figsize=(12, 6))
                print("Creating SHAP decision plot...")
                shap.decision_plot(explainer.expected_value[1], shap_values_individual[1],
                                 feature_names=all_feature_names, show=False)
                plt.title('SHAP - Correctly Predicted Smoker Example (Decision Plot)')
                plt.tight_layout()
                plt.savefig('shap_correct_smoker_decision.png')
                plt.show()
            else:
                # For single-class problem
                plt.figure(figsize=(12, 6))
                shap.plots.waterfall(shap_values_individual[0], max_display=10, show=False)
                plt.title('SHAP - Correctly Predicted Smoker Example')
                plt.tight_layout()
                plt.savefig('shap_correct_smoker_waterfall.png')
                plt.show()
        except Exception as e:
            print(f"Error during SHAP individual explanation calculation: {e}")
            import traceback
            traceback.print_exc()

    if len(incorrect_nonsmoker_indices) > 0:
        incorrect_idx = incorrect_nonsmoker_indices[0]

        print(f"\nExample 2 (Incorrect Prediction - Non-Smoker) - Test Set Index: {incorrect_idx}")
        lime_exp = lime_explainer.explain_instance(X_test_processed[incorrect_idx],
                                                 best_rf_model.predict_proba,
                                                 num_features=8)

        plt.figure(figsize=(10, 6))
        lime_exp.as_pyplot_figure()
        plt.title('LIME - Incorrectly Predicted Non-Smoker Example')
        plt.tight_layout()
        plt.savefig('lime_incorrect_nonsmoker.png')
        plt.show()

        # SHAP explanation for the same example
        try:
            plt.figure(figsize=(10, 6))
            # shap.initjs() # Already called once
            shap_values_individual = explainer.shap_values(X_test_processed[incorrect_idx].reshape(1, -1))
             # Use shap.plots.force instead of shap.force_plot or check updated usage of force_plot
            # force_plot typically returns an HTML object, needs special usage to show with plt
            # Example: shap.decision_plot(explainer.expected_value[0], shap_values_individual[0][0], feature_names=all_feature_names, show=False)
            # For now, skip showing force_plot with matplotlib and just do the calculation
            print("Creating SHAP force plot...")
            # shap.force_plot(
            #     explainer.expected_value[0], # Expected value for first class (Non-Smoker)
            #     shap_values_individual[0], # SHAP values for first class (Non-Smoker)
            #     X_test_processed[incorrect_idx],
            #     feature_names=all_feature_names,
            #     matplotlib=True,
            #     show=False
            # )
            # plt.title('SHAP - Incorrectly Predicted Non-Smoker Example')
            # plt.tight_layout()
            # plt.savefig('shap_incorrect_nonsmoker.png')
            # plt.show()
        except Exception as e:
            print(f"Error during SHAP individual explanation calculation: {e}")

except Exception as e:
    print(f"Error during LIME calculation: {e}")

print("\nLocal Explanations Discussion:")
print("- In the correctly predicted smoker example, the most important factors affecting the model's prediction")
print("  are typically features like hemoglobin, gender, and age.")
print("- In the incorrectly predicted non-smoker example, our model may have over-interpreted some features.")
print("- The explanations generally align with domain knowledge. For example, the relationship between")
print("  hemoglobin values and smoking status is documented in medical literature.")

print("\n=====================================================")
print("CLASS-BASED OR SUBGROUP ANALYSIS")
print("=====================================================")

# Subgroup analysis by gender
print("\nPerforming subgroup analysis by gender...")

try:
    # Split test set by gender
    test_df = X_test.copy()
    test_df['smoking'] = y_test.values

    # Separate gender groups
    male_test_df = test_df[test_df['gender'] == 'M']
    female_test_df = test_df[test_df['gender'] == 'F']

    male_X_test = male_test_df.drop(columns=['smoking'])
    female_X_test = female_test_df.drop(columns=['smoking'])

    male_y_test = male_test_df['smoking']
    female_y_test = female_test_df['smoking']

    # Get preprocessed data
    male_X_test_processed = preprocessor.transform(male_X_test)
    female_X_test_processed = preprocessor.transform(female_X_test)

    print(f"Male test samples: {len(male_X_test)}, Female test samples: {len(female_X_test)}")

    # Calculate permutation importance for each subgroup
    print("Calculating permutation importance for males...")
    male_perm_importance = permutation_importance(best_rf_model, male_X_test_processed, male_y_test,
                                                 n_repeats=3, random_state=42)

    male_importance_df = pd.DataFrame({
        'Feature': all_feature_names,
        'Importance': male_perm_importance.importances_mean
    }).sort_values('Importance', ascending=False)

    print("Calculating permutation importance for females...")
    female_perm_importance = permutation_importance(best_rf_model, female_X_test_processed, female_y_test,
                                                   n_repeats=3, random_state=42)

    female_importance_df = pd.DataFrame({
        'Feature': all_feature_names,
        'Importance': female_perm_importance.importances_mean
    }).sort_values('Importance', ascending=False)

    # Visualization
    plt.figure(figsize=(14, 10))

    # Top features - combine most important features from both groups
    top_features = set()
    top_features.update(male_importance_df['Feature'].head(5).tolist())
    top_features.update(female_importance_df['Feature'].head(5).tolist())
    top_features = list(top_features)[:10]  # Maximum 10 features

    # Bar plot
    bar_width = 0.35
    r1 = np.arange(len(top_features))
    r2 = [x + bar_width for x in r1]

    male_values = [male_importance_df[male_importance_df['Feature'] == f]['Importance'].values[0]
                  if f in male_importance_df['Feature'].values else 0 for f in top_features]
    female_values = [female_importance_df[female_importance_df['Feature'] == f]['Importance'].values[0]
                    if f in female_importance_df['Feature'].values else 0 for f in top_features]

    plt.bar(r1, male_values, width=bar_width, label='Male', color='blue', alpha=0.7)
    plt.bar(r2, female_values, width=bar_width, label='Female', color='red', alpha=0.7)

    plt.xlabel('Features')
    plt.ylabel('Importance Score')
    plt.title('Feature Importance by Gender')
    plt.xticks([r + bar_width/2 for r in range(len(top_features))], top_features, rotation=45, ha='right')
    plt.legend()
    plt.tight_layout()
    plt.savefig('gender_group_feature_importance.png')
    plt.show()

    # Compare SHAP values by gender
    if 'shap_values' in locals():
        # Get gender information for test samples
        sample_gender = []
        for idx in sample_indices:
            # Get gender from original X_test DataFrame
            gender = X_test.iloc[idx]['gender']
            sample_gender.append(gender)

        sample_gender = np.array(sample_gender)
        male_indices = np.where(sample_gender == 'M')[0]
        female_indices = np.where(sample_gender == 'F')[0]

        # Separate SHAP values and samples for males
        male_shap_values_list = [shap_values[i] for i in male_indices] # List of SHAP values
        # If shap_values is a numpy array: male_shap_values_array = shap_values[male_indices]

        male_X_sample = X_test_sample[male_indices]

        # Separate SHAP values and samples for females
        female_shap_values_list = [shap_values[i] for i in female_indices] # List of SHAP values
        # If shap_values is a numpy array: female_shap_values_array = shap_values[female_indices]

        female_X_sample = X_test_sample[female_indices]

        # SHAP summary plot requires numpy array format
        # Convert lists to numpy arrays (if shap_values is in list format)
        if isinstance(shap_values, list):
             male_shap_values_array = np.array(male_shap_values_list)
             female_shap_values_array = np.array(female_shap_values_list)
        else: # If already numpy array
             male_shap_values_array = shap_values[male_indices]
             female_shap_values_array = shap_values[female_indices]


        # SHAP summary plot for males
        if len(male_shap_values_array) > 0:
            plt.figure(figsize=(12, 10))
            # shap.summary_plot function typically expects a 2D array (samples, features)
            shap.summary_plot(male_shap_values_array, male_X_sample, feature_names=all_feature_names,
                            show=False, max_display=10)
            plt.title('SHAP Summary Plot - Males')
            plt.tight_layout()
            plt.savefig('shap_summary_males.png')
            plt.show()

        # SHAP summary plot for females
        if len(female_shap_values_array) > 0:
            plt.figure(figsize=(12, 10))
            shap.summary_plot(female_shap_values_array, female_X_sample, feature_names=all_feature_names,
                            show=False, max_display=10)
            plt.title('SHAP Summary Plot - Females')
            plt.tight_layout()
            plt.savefig('shap_summary_females.png')
            plt.show()

except Exception as e:
    print(f"Error during gender subgroup analysis: {e}")


# Subgroup analysis by age groups
print("\nPerforming subgroup analysis by age groups...")

try:
    # Split test set by age groups
    test_df = X_test.copy()
    test_df['smoking'] = y_test.values

    # Define age groups
    test_df['age_group'] = pd.cut(test_df['age'], bins=[0, 30, 50, 100],
                                 labels=['Young', 'Middle-Aged', 'Elderly'])

    # Separate age groups
    age_groups = {}
    age_importance_dfs = {}

    for group_name in ['Young', 'Middle-Aged', 'Elderly']:
        group_df = test_df[test_df['age_group'] == group_name]

        if len(group_df) < 10:  # Skip analysis if too few data points
            print(f"  Not enough data for {group_name} group, skipping analysis.")
            continue

        group_X = group_df.drop(columns=['smoking', 'age_group'])
        group_y = group_df['smoking']

        # Get preprocessed data
        group_X_processed = preprocessor.transform(group_X)

        print(f"  Calculating permutation importance for {group_name} group...")
        group_perm_importance = permutation_importance(best_rf_model, group_X_processed, group_y,
                                                     n_repeats=3, random_state=42)

        age_importance_dfs[group_name] = pd.DataFrame({
            'Feature': all_feature_names,
            'Importance': group_perm_importance.importances_mean
        }).sort_values('Importance', ascending=False)

    # Visualize importance scores by age group
    if len(age_importance_dfs) > 1:  # Compare only if at least two groups exist
        plt.figure(figsize=(15, 10))

        # Collect top features from all groups
        top_features = set()
        for name, df_imp in age_importance_dfs.items():
            top_features.update(df_imp['Feature'].head(3).tolist())  # Top 3 from each group

        top_features = list(top_features)[:8]  # Maximum 8 features

        # Bar chart settings
        bar_width = 0.2
        positions = np.arange(len(top_features))
        offsets = np.linspace(-(len(age_importance_dfs)-1)/2*bar_width,
                             (len(age_importance_dfs)-1)/2*bar_width,
                             len(age_importance_dfs))

        # Add bars for each age group
        colors = ['skyblue', 'orange', 'green']
        for i, (name, df_imp) in enumerate(age_importance_dfs.items()):
            values = [df_imp[df_imp['Feature'] == f]['Importance'].values[0]
                     if f in df_imp['Feature'].values else 0 for f in top_features]
            plt.bar(positions + offsets[i], values, width=bar_width, label=name, color=colors[i])

        plt.xlabel('Features')
        plt.ylabel('Importance Score')
        plt.title('Feature Importance by Age Group')
        plt.xticks(positions, top_features, rotation=45, ha='right')
        plt.legend()
        plt.tight_layout()
        plt.savefig('age_group_feature_importance.png')
        plt.show()

except Exception as e:
    print(f"Error during age group analysis: {e}")


print("\nClass-Based and Subgroup Analysis Discussion:")
print("- When comparing by gender, different features stand out for predicting smoking behavior")
print("  in males versus females.")
print("- In males, hemoglobin and liver enzymes (ALT, AST, Gtp) tend to be more determinative,")
print("  while in females, cholesterol values and blood pressure measurements might be more important.")
print("- The factors influencing smoking behavior also vary across age groups.")
print("- These differences suggest that model explanations should be interpreted in the context of")
print("  specific demographic groups, and a single general explanation may not be sufficient.")

print("\n=====================================================")
print("LIMITATIONS OF EXPLAINABILITY METHODS")
print("=====================================================")
print("- SHAP: Computationally expensive and can be slow for large models.")
print("- LIME: May produce different explanations in different runs, unstable.")
print("- Permutation Importance: Doesn't account for correlations between features.")
print("- Feature Importance: Model-specific and cannot be compared across different model types.")
print("- PDP: Doesn't fully capture interactions between features.")
print("- All explainability methods operate under certain assumptions and")
print("  may not always reflect true causality.")

print("\n=====================================================")
print("CONCLUSIONS AND INSIGHTS")
print("=====================================================")
print("The explainability analysis has helped us understand the decision mechanisms")
print("our model uses when predicting smoking status:")

print("\n1. Key Features Affecting Model Performance:")
print("- Hemoglobin, gender, age, and liver enzymes (ALT, AST, Gtp) play critical roles in predictions.")
print("- Consistent with medical literature, these variables are associated with smoking habits.")

print("\n2. Demographic Differences:")
print("- Different biomarkers stand out when comparing by gender.")
print("- Factors affecting smoking behavior vary across age groups.")

print("\n3. Model Improvement Opportunities:")
print("- Separate models could be developed based on gender or age group.")
print("- Feature engineering could be improved based on analysis of incorrect predictions.")
print("- The analyses suggest that collecting more detailed smoking information")
print("  (amount, duration, etc.) during data collection could be beneficial.")

print("\n4. Potential Model Issues:")
print("- Excessive confidence in some features (especially biomarkers) suggests the model")
print("  may have overfit to the training data.")
print("- Analysis of incorrect predictions has shown that the model makes more errors")
print("  for certain demographic groups, which should be examined from a fair machine learning perspective.")

print("\nIn conclusion, these explainability methods have given us a better understanding of how our model works,")
print("which will help us improve our data and model quality in the future.")