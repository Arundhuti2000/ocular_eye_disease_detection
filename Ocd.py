# %%
import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
import string
import re
from nltk.corpus import stopwords
from wordcloud import WordCloud, STOPWORDS
from nltk.stem import SnowballStemmer

# %%
import os
import itertools
import cv2
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix , classification_report
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Model 
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense,  BatchNormalization, Activation, Dropout, Concatenate, BatchNormalization, GlobalAveragePooling2D  
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.optimizers import Adam , Adamax
from tensorflow.keras import regularizers


# %%
print("GPU is available" if tf.config.list_physical_devices('GPU') else "GPU is not available")

# %%
import warnings
warnings.filterwarnings("ignore")

# %%
full_df_file = "eye_disease_detection_dataset/archive/full_df.csv"
df = pd.read_csv(full_df_file)
df.head()

# %%
df.columns
len(df)
len(df[df.duplicated()])

# %%
df.isnull().sum()

# %%
df['Left-Fundus'].nunique()

# %%
photo_counts = df['Left-Fundus'].value_counts()

# Filter names that appear more than once
photo_more_than_once = photo_counts[photo_counts > 1].index.tolist()

print(len(photo_more_than_once))

photo_more_than_once[0]

# %%
df['labels'].value_counts()

# %%
# Find inconsistent or contradicting data, where (label and the Disease columns are not matching)
print(len(df[(df['labels'] == "['N']") & (df['N'] != 1)]))
print(len(df[(df['labels'] == "['D']") & (df['D'] != 1)]))
print(len(df[(df['labels'] == "['O']") & (df['O'] != 1)]))
print(len(df[(df['labels'] == "['C']") & (df['C'] != 1)]))
print(len(df[(df['labels'] == "['G']") & (df['G'] != 1)]))
print(len(df[(df['labels'] == "['A']") & (df['A'] != 1)]))
print(len(df[(df['labels'] == "['M']") & (df['M'] != 1)]))
print(len(df[(df['labels'] == "['H']") & (df['H'] != 1)]))

# %%
#updating the label with the disease column 

df.drop(columns=[ 'ID'] , inplace=True)
len(df[df['Patient Age'] == 1])
def update_labels(row):
    
    x = ''
    for col in df.columns:
        if row[col] == 1 and col != 'Patient Age':
            x = x + col
    row['labels'] = x        
    return row

df = df.apply(update_labels, axis=1)
df['labels'].nunique()

# %%
df['labels'].nunique()

# %%
plt.figure(figsize=(14, 5))
sns.countplot(x='labels', data=df , orient='h')
plt.title('Count of Unique Labels in Eyes Dataset')
plt.xlabel('Labels')
plt.ylabel('Count')
plt.xticks(rotation=45)
plt.show()

# %%
df[df['Left-Diagnostic Keywords'] == 'low image quality']

# %%
#df = df.loc[~(df['Left-Diagnostic Keywords'] == 'low image quality')]
#df = df.loc[~(df['Right-Diagnostic Keywords'] == 'low image quality')]
df = df[df['Left-Diagnostic Keywords'] != 'low image quality']
df = df[df['Right-Diagnostic Keywords'] != 'low image quality']


# %%
print(len(df[(df['labels'] == "N") & (df['N'] != 1)]))
print(len(df[(df['labels'] == "O") & (df['O'] != 1)]))
df.head()

# %%
photos_unique = df.drop_duplicates(subset='Left-Fundus', keep='first')
df = photos_unique
df.reset_index(drop=True,inplace=True)
len(df)

# %%
df['Left-Diagnostic Keywords'].nunique()

# %%
df['Right-Diagnostic Keywords'].nunique()

# %%
df['Left-Diagnostic Keywords'].mode()

# %%
df['Left-Diagnostic Keywords'].value_counts()

# %%

len(df[df['Left-Diagnostic Keywords'] == 'normal fundus'])

# %%
df['Right-Diagnostic Keywords'].mode()

# %%
len(df[df['Right-Diagnostic Keywords'] == 'normal fundus'])

# %%
both_eyes_normal = df[
    (df['Right-Diagnostic Keywords'] == 'normal fundus') & 
    (df['Left-Diagnostic Keywords'] == 'normal fundus')
]

both_eyes_normal.reset_index(inplace=True,drop=True)

len(both_eyes_normal)

# %%
both_eyes_not_normal = df[
    (df['Right-Diagnostic Keywords'] != 'normal fundus') & 
    (df['Left-Diagnostic Keywords'] != 'normal fundus')
]

both_eyes_not_normal.reset_index(inplace=True,drop=True)

len(both_eyes_not_normal)

# %%
right_eye_normal = df[
    (df['Right-Diagnostic Keywords'] == 'normal fundus') & 
    (df['Left-Diagnostic Keywords'] != 'normal fundus')
]

right_eye_normal.reset_index(inplace=True,drop=True)

len(right_eye_normal)

# %%
left_eye_normal = df[
    (df['Right-Diagnostic Keywords'] != 'normal fundus') & 
    (df['Left-Diagnostic Keywords'] == 'normal fundus')
]

left_eye_normal.reset_index(inplace=True,drop=True)

len(left_eye_normal)

# %%
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('Disease Distribution Across Different Groups', fontsize=16)

# Define titles for the subplots
titles = [
    'Both Eyes Normal',
    'Both Eyes Not Normal',
    'Right Eye Normal',
    'Left Eye Normal'
]

# Create a list of DataFrames
dataframes = [both_eyes_normal, both_eyes_not_normal, right_eye_normal, left_eye_normal]

# Loop through DataFrames and plot on subplots
for df, ax, title in zip(dataframes, axes.ravel(), titles):
    disease_columns = ['N', 'D', 'G', 'C', 'A', 'H', 'M', 'O']
    disease_counts = df[disease_columns].sum()
    
    sns.barplot(x=disease_counts.index, y=disease_counts.values, color='skyblue', ax=ax)
    ax.set_title(title)
    ax.set_xlabel('Diseases')
    ax.set_ylabel('Frequency')

plt.tight_layout(rect=[0, 0, .9, 0.80])
plt.show()

# %%
len(both_eyes_normal)

# %%
both_eyes_not_normal[both_eyes_not_normal['N'] == 1]

# %%
len(both_eyes_not_normal[both_eyes_not_normal['N'] == 1])

# %%
filtered_df = both_eyes_not_normal[both_eyes_not_normal['N'] == 1]
both_eyes_normal = pd.concat(
    [both_eyes_normal[both_eyes_normal['N'].isin(filtered_df['N'])], filtered_df],
    axis=0, ignore_index=True
).drop_duplicates()

# %%
len(both_eyes_normal)

# %%
both_eyes_not_normal = both_eyes_not_normal.loc[(both_eyes_not_normal['N'] != 1)]

# %%
both_eyes_not_normal.head()

# %%
right_eye_normal[right_eye_normal['N'] == 1]

# %%
filtered_df = right_eye_normal[right_eye_normal['N'] == 1]

both_eyes_normal = pd.concat(
    [both_eyes_normal[both_eyes_normal['N'].isin(filtered_df['N'])], filtered_df],
    axis=0, ignore_index=True
).drop_duplicates()                         

# %%
right_eye_normal = right_eye_normal.loc[~(right_eye_normal['N'] == 1)]

# %%
left_eye_normal[left_eye_normal['N'] == 1]

# %%
filtered_df = left_eye_normal[left_eye_normal['N'] == 1]

both_eyes_normal = pd.concat(
    [both_eyes_normal[both_eyes_normal['N'].isin(filtered_df['N'])], filtered_df],
    axis=0, ignore_index=True
).drop_duplicates()

# %%
left_eye_normal = left_eye_normal.loc[(left_eye_normal['N'] != 1)]

# %%
len(both_eyes_normal)

# %%
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('Disease Distribution Across Different Groups', fontsize=16)

# Define titles for the subplots
titles = [
    'Both Eyes Normal',
    'Both Eyes Not Normal',
    'Right Eye Normal',
    'Left Eye Normal'
]

# Create a list of DataFrames
dataframes = [both_eyes_normal, both_eyes_not_normal, right_eye_normal, left_eye_normal]

# Loop through DataFrames and plot on subplots
for df, ax, title in zip(dataframes, axes.ravel(), titles):
    disease_columns = ['N', 'D', 'G', 'C', 'A', 'H', 'M', 'O']
    disease_counts = df[disease_columns].sum()
    
    sns.barplot(x=disease_counts.index, y=disease_counts.values, color='skyblue', ax=ax)
    ax.set_title(title)
    ax.set_xlabel('Diseases')
    ax.set_ylabel('Frequency')

plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.show()

# %%
left_eye_normal.head()

# %%
df= pd.concat([both_eyes_normal,both_eyes_not_normal,left_eye_normal,right_eye_normal],ignore_index=True)

# %%
condition_normal_left = df['Left-Diagnostic Keywords'] == 'normal fundus'
condition_normal_right = df['Right-Diagnostic Keywords'] == 'normal fundus'

# Determine the categories
both_normal = (condition_normal_left) & (condition_normal_right)
both_abnormal = (~condition_normal_left) & (~condition_normal_right)
left_normal_right_abnormal = (condition_normal_left) & (~condition_normal_right)
right_normal_left_abnormal = (~condition_normal_left) & (condition_normal_right)

# Count occurrences for each category
counts = {
    'Both Normal': both_normal.sum(),
    'Both Abnormal': both_abnormal.sum(),
    'Left Normal, Right Abnormal': left_normal_right_abnormal.sum(),
    'Right Normal, Left Abnormal': right_normal_left_abnormal.sum()
}

# Create a pie chart
labels = counts.keys()
sizes = counts.values()
colors = ['#ff9999','#66b3ff','#99ff99','#ffcc99']
explode = (0.1, 0.07, 0, 0)  # explode the 1st slice (optional)

plt.figure(figsize=(8, 6))
plt.pie(sizes, explode=explode, labels=labels, colors=colors, autopct='%1.1f%%', shadow=True, startangle=140)
plt.title('Distribution of Fundus Conditions')
plt.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.

# Show the pie chart
plt.show()




# %%
df['labels'].unique()

# %%
fig, axes = plt.subplots(2, 2, figsize=(8, 8), sharex=True, sharey=True)

# List of dataframes and titles
dataframes = [both_eyes_normal, both_eyes_not_normal, right_eye_normal, left_eye_normal]
titles = ['Both Eyes Normal', 'Both Eyes Not Normal', 'Right Eye Normal', 'Left Eye Normal']

# Plot each dataframe
for i, (eye, title) in enumerate(zip(dataframes, titles)):
    row = i // 2
    col = i % 2
    sns.histplot(eye['Patient Age'], kde=True, bins=10, ax=axes[row, col])
    axes[row, col].set_title(title)
    axes[row, col].set_xlabel('Age')
    axes[row, col].set_ylabel('Frequency')

# Adjust layout
plt.tight_layout()
plt.show()

# %%
df['labels'].unique()

# %%
counts = {
    'Both Eyes Normal': both_eyes_normal['Patient Sex'].value_counts(),
    'Both Eyes Not Normal': both_eyes_not_normal['Patient Sex'].value_counts(),
    'Right Eye Normal': right_eye_normal['Patient Sex'].value_counts(),
    'Left Eye Normal': left_eye_normal['Patient Sex'].value_counts()
}

# Convert to DataFrame for easy plotting
plot_data = pd.DataFrame(counts).fillna(0).T.reset_index()
plot_data = plot_data.melt(id_vars='index', var_name='Gender', value_name='Count')
plot_data = plot_data.rename(columns={'index': 'Category'})

# Create the bar plot
plt.figure(figsize=(8, 8))
sns.barplot(data=plot_data, x='Category', y='Count', hue='Gender', palette='viridis')

plt.title('Number of Males and Females by Category')
plt.xlabel('Category')
plt.ylabel('Number of Patients')
plt.legend(title='Gender')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# %%
fig, axes = plt.subplots(2, 2, figsize=(8, 8), sharex=True, sharey=True)

dataframes = [both_eyes_normal, both_eyes_not_normal, right_eye_normal, left_eye_normal]
titles = ['Both Eyes Normal', 'Both Eyes Not Normal', 'Right Eye Normal', 'Left Eye Normal']
colors = ['#66b3ff', '#ff9999']

# Plot each dataframe
for i, (eyes, title) in enumerate(zip(dataframes, titles)):
    row = i // 2
    col = i % 2
    sns.histplot(eyes, x='Patient Age', hue='Patient Sex', multiple='stack', palette=colors, bins=10, ax=axes[row, col], kde=True)
    axes[row, col].set_title(title)
    axes[row, col].set_xlabel('Age')
    axes[row, col].set_ylabel('Frequency')

# Adjust layout
plt.tight_layout()
plt.show()

# %%
df['labels'].unique()

# %%
cluster_features = ['Patient Age', 'N', 'D', 'G', 'C', 'A', 'H', 'M', 'O']
X_cluster = df[cluster_features].copy()

# Convert Patient Sex to numeric
X_cluster['Patient_Sex_Numeric'] = df['Patient Sex'].map({'Male': 0, 'Female': 1})

# Standardize the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_cluster)

# %%
# Cell 3: Perform K-means clustering
inertias = []
k_range = range(2, 9)
for k in k_range:
   kmeans = KMeans(n_clusters=k, random_state=42)
   kmeans.fit(X_scaled)
   inertias.append(kmeans.inertia_)
plt.figure(figsize=(10, 6))
plt.plot(k_range, inertias, 'bx-')
plt.xlabel('k')
plt.ylabel('Inertia')
plt.title('Elbow Method For Optimal k')
plt.show()

# Train final model with optimal k
n_clusters = 7  # Adjust based on elbow plot
kmeans = KMeans(n_clusters=n_clusters, random_state=42)
clusters = kmeans.fit_predict(X_scaled)

# Add cluster labels to the dataframe
df['Cluster'] = clusters

print("Cluster distribution:")
print(df['Cluster'].value_counts())

# %%
# Cell 2: Prepare data with proper error handling
IMG_SIZE = 128

def prepare_data(df):
    X = []
    y = []
    cluster_info = []
    base_path = 'eye_disease_detection_dataset/archive/preprocessed_images/'
    
    print("Starting data preparation...")
    total = len(df)
    
    for idx, row in df.iterrows():
        if idx % 100 == 0:  # Progress indicator
            print(f"Processing {idx}/{total} images")
            
        img_path = os.path.join(base_path, row['Right-Fundus'])
        try:
            # Check if file exists
            if not os.path.exists(img_path):
                print(f"File not found: {img_path}")
                continue
                
            # Load and process image
            img = cv2.imread(img_path)
            if img is None:
                print(f"Failed to load image: {img_path}")
                continue
                
            img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = img / 255.0
            
            # Create label array
            label = np.array([row['N'], row['D'], row['G'], row['C'], 
                            row['A'], row['H'], row['M'], row['O']])
            
            # Append data
            X.append(img)
            y.append(label)
            cluster_info.append(row['Cluster'])
            
        except Exception as e:
            print(f"Error processing {img_path}: {str(e)}")
            continue
    
    # Convert lists to numpy arrays
    X = np.array(X)
    y = np.array(y)
    cluster_info = np.array(cluster_info)
    
    print("\nFinal data shapes:")
    print(f"X shape: {X.shape}")
    print(f"y shape: {y.shape}")
    print(f"clusters shape: {cluster_info.shape}")
    
    return X, y, cluster_info

# Verify df has Cluster column
if 'Cluster' not in df.columns:
    print("Error: 'Cluster' column not found in dataframe")
    print("Available columns:", df.columns.tolist())
else:
    # Prepare the data
    X, y, clusters = prepare_data(df)

# %%
# Cell 3: Verify data before splitting
print("\nVerification before splitting:")
print("X type:", type(X))
print("y type:", type(y))
print("clusters type:", type(clusters))
print("\nX shape:", X.shape)
print("y shape:", y.shape)
print("clusters shape:", clusters.shape)
print("\nUnique clusters:", np.unique(clusters))

# %%
# Cell 3: Split the data
X_train, X_test, y_train, y_test, clusters_train, clusters_test = train_test_split(
    X, y, clusters,
    test_size=0.2,
    random_state=42,
    stratify=clusters
)
print("Training set shapes:")
print("X_train:", X_train.shape)
print("y_train:", y_train.shape)
print("clusters_train:", clusters_train.shape)

# %%
# Create a CNN model
def create_robust_model(input_shape, num_clusters, num_classes):
    # Image input branch
    img_input = Input(shape=input_shape)
    
    # First Convolutional Block
    x = Conv2D(32, (3, 3), padding='same', activation='relu')(img_input)
    x = BatchNormalization()(x)
    x = Conv2D(32, (3, 3), padding='same', activation='relu')(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = Dropout(0.25)(x)
    
    # Second Convolutional Block
    x = Conv2D(64, (3, 3), padding='same', activation='relu')(x)
    x = BatchNormalization()(x)
    x = Conv2D(64, (3, 3), padding='same', activation='relu')(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = Dropout(0.25)(x)
    
    # Third Convolutional Block
    x = Conv2D(128, (3, 3), padding='same', activation='relu')(x)
    x = BatchNormalization()(x)
    x = Conv2D(128, (3, 3), padding='same', activation='relu')(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = Dropout(0.25)(x)
    
    # Global Average Pooling instead of Flatten
    x = GlobalAveragePooling2D()(x)
    
    # Cluster input branch
    cluster_input = Input(shape=(1,))
    cluster_encoded = Dense(32, activation='relu')(cluster_input)
    cluster_encoded = BatchNormalization()(cluster_encoded)
    
    # Combine image features and cluster information
    combined = Concatenate()([x, cluster_encoded])
    
    # Dense layers
    x = Dense(256, activation='relu')(combined)
    x = BatchNormalization()(x)
    x = Dropout(0.5)(x)
    
    x = Dense(128, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.5)(x)
    
    output = Dense(num_classes, activation='softmax')(x)
    
    return Model(inputs=[img_input, cluster_input], outputs=output)

# %%
# Cell 3: Create and compile model
IMG_SIZE = 128  # Increased image size
n_clusters = 3

model = create_robust_model(
    input_shape=(IMG_SIZE, IMG_SIZE, 3),
    num_clusters=n_clusters,
    num_classes=8
)

# Compile with better optimizer settings
from tensorflow.keras.optimizers import Adam
model.compile(
    optimizer=Adam(learning_rate=0.001),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

model.summary()

# %%
history = model.fit(
    [X_train, clusters_train],
    y_train,
    batch_size=32,
    epochs=20,  # Reduced epochs since we don't have early stopping
    validation_data=([X_test, clusters_test], y_test),
    verbose=1
)

# %%
# Cell 8: Visualize results


plt.figure(figsize=(12, 4))

# Plot accuracy
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Training')
plt.plot(history.history['val_accuracy'], label='Validation')
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

# Plot loss
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Training')
plt.plot(history.history['val_loss'], label='Validation')
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.tight_layout()
plt.show()

# %%
# Cell 1: Get model predictions
y_pred = model.predict([X_test, clusters_test])
# Convert predictions from probabilities to class labels
y_pred_classes = np.argmax(y_pred, axis=1)
y_test_classes = np.argmax(y_test, axis=1)

# Cell 2: Generate classification report
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns

# Define class names for better readability
class_names = ['Normal', 'Diabetes', 'Glaucoma', 'Cataract', 
               'AMD', 'Hypertension', 'Myopia', 'Other']

# Print classification report
print("\nClassification Report:")
print(classification_report(y_test_classes, y_pred_classes, target_names=class_names))

# Cell 3: Create confusion matrix visualization
plt.figure(figsize=(10, 8))
cm = confusion_matrix(y_test_classes, y_pred_classes)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=class_names, 
            yticklabels=class_names)
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Cell 4: Calculate additional metrics per class
for i, class_name in enumerate(class_names):
    true_class = (y_test_classes == i)
    pred_class = (y_pred_classes == i)
    
    # Calculate True Positives, False Positives, False Negatives
    tp = np.sum(true_class & pred_class)
    fp = np.sum(~true_class & pred_class)
    fn = np.sum(true_class & ~pred_class)
    
    # Calculate precision, recall, and F1 score
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    print(f"\nDetailed Metrics for {class_name}:")
    print(f"Precision: {precision:.3f}")
    print(f"Recall: {recall:.3f}")
    print(f"F1-Score: {f1:.3f}")


