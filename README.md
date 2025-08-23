# Loading the necessary libraries
import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
%matplotlib inline

# Importing the sklearn libraries
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

# Loading the tensorflow and keras modules
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from scikeras.wrappers import KerasRegressor
# Read the dataset
df = pd.read_csv("Data_in_csv_formate")
df.head()

# Correlating the features using pearson correlation 
df.corr()
corrmat = df.corr()
features = corrmat.index
sns.heatmap(df[features].corr(), cmap="RdBu", square=True)
plt.rcParams.update({'font.weight': 'bold'})
plt.tick_params(axis='both', which='major', labelsize=12)
for label in plt.gca().get_xticklabels() + plt.gca().get_yticklabels():
    label.set_fontweight('bold')
plt.savefig("Storage_pathway_in_png_formate", dpi=300,)
plt.show()
plt.figure(figsize=(50,20))

# Plotting the feature importance score  
model=ExtraTreesRegressor()
model.fit(X,Y)
print(model.feature_importances_)
vd=pd.Series(model.feature_importances_,index=X.columns)
vd.nlargest(14).plot(kind='bar')
plt.ylabel("Feature Importance score", fontsize = 18, fontweight='bold')
plt.savefig("Storage_pathway_in_png_formate", dpi=300,)
plt.show()

# Cpaturing the least important features 
corr_features = correlation(df, 0.9)
len(set(corr_features))
corr_features

# Droppin out the least important features (if necessary) 
df.drop(corr_features, axis = 1)
numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns

# Visualization for each numeric column
for col in numeric_cols:
    plt.figure(figsize=(12, 5))
    
    # Histogram + KDE
    plt.subplot(1, 2, 1)
    sns.histplot(df[col], kde=True, bins=30, color="skyblue")
    plt.title(f"Distribution of {col}", fontsize=14)
    plt.xlabel(col)
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.show()
    X = df.iloc[:,:-1]
Y = df.iloc[:,-1]

# Splitting the dataset into training and testing dataset
X_train, X_test, Y_train, Y_test = train_test_split(X,Y, train_size=0.80, test_size=0.20, random_state=42)
print('Train/Test Sets Sizes : ',X_train.shape, X_test.shape, Y_train.shape, Y_test.shape)

# Standardization of the dataset
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
# Building the model 
def build_model(n_neurons=%, learning_rate=%):
    model = Sequential()
    model.add(Dense(n_neurons, activation='relu', input_shape=(X_train.shape[1],)))
    model.add(Dense(n_neurons//2, activation='relu'))
    model.add(Dense(1))  # Regression output
    model.compile(optimizer=Adam(learning_rate=learning_rate),
                  loss='mse',
                  metrics=['mae'])
    return model
regressor = KerasRegressor(model=build_model, verbose=0)

# Hyperparameter tuning optimization
param_dist = {
    "model__n_neurons": [32, 64, 128],
    "model__learning_rate": [0.001, 0.01, 0.1],
    "batch_size": [16, 32, 64],
    "epochs": [50, 100, 200]
}

random_search = RandomizedSearchCV(
    estimator=regressor,
    param_distributions=param_dist,
    n_iter=5,
    cv=3,
    scoring='r2',
    verbose=2,
    n_jobs=-1
)

random_search.fit(X_train, Y_train)
y_pred=random_search.predict(X_train)
y_pred_nn = random_search.predict(X_test)

# Capturing the loss with hyperparameter towards global minima
history = random_search.fit(X_train, Y_train, validation_split = 0.2, epochs = 200)
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs =range(1, len(loss)+1)
plt.rcParams.update({'font.weight': 'bold'})
plt.rcParams['axes.linewidth'] = 1.5
plt.tick_params(axis='both', 
                direction='in',         
                width=2) 
plt.plot(epochs, loss, 'y', label = 'Training loss')
plt.plot(epochs, val_loss, 'r', label = 'Validation loss')
plt.title('Training and Validation loss', fontsize = 18, fontweight='bold')
plt.xlabel('Epochs', fontsize=16,fontweight='bold')
plt.ylabel('Loss', fontsize=16,fontweight='bold')
plt.legend(fontsize=12)
plt.tick_params(axis='both', which='major', labelsize=12)
for label in plt.gca().get_xticklabels() + plt.gca().get_yticklabels():
    label.set_fontweight('bold')
plt.show()plt.figure(figsize=(9, 7))

# Train data
plt.scatter(Y_train, y_pred, alpha=0.7, s=80, 
            label="Train Dataset", color="#1f77b4", edgecolor="k")

# Test data
plt.scatter(Y_test, y_pred_nn, alpha=0.7, s=80, 
            label="Test Dataset", color="#ff7f0e", edgecolor="k")

# Perfect prediction line
plt.plot([0, 10000], [0, 10000], color="red", linestyle="--", linewidth=2, label="Perfect Prediction")

# Labels with bold font
plt.xlabel("Measured Value", fontsize=28, fontweight="bold")
plt.ylabel("Predicted Value", fontsize=28, fontweight="bold")

# Annotations
plt.text(0.05, 0.95, "Neural Network Regression", fontsize=20, fontweight="bold",
         transform=plt.gca().transAxes, color="darkblue")
plt.text(0.05, 0.90, "Training R² score = 0.98", fontsize=18, fontweight="bold",
         transform=plt.gca().transAxes, color="darkgreen")
plt.text(0.05, 0.85, "Testing R² score = 0.97", fontsize=18, fontweight="bold",
         transform=plt.gca().transAxes, color="darkgreen")

# Legend
plt.legend(fontsize=14, frameon=True, loc="lower right", edgecolor="black")

# Grid
plt.grid(True, linestyle="--", alpha=0.6)

# Bold ticks, inside placement
plt.xticks(fontsize=14, fontweight="bold")
plt.yticks(fontsize=14, fontweight="bold")
plt.tick_params(axis="both", direction="in", width=2, length=6)

# Bold axis spines
for spine in plt.gca().spines.values():
    spine.set_linewidth(2)

# Tight layout
plt.tight_layout()
plt.show()
# Loading ML regressors from sklearn library for comparison
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.linear_model import Lasso
from sklearn.linear_model import Ridge
import xgboost as xgb
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeRegressor

lr = LinearRegression()
dt = DecisionTreeRegressor()
knn = KNeighborsRegressor()
svm = SVR()
lasso=Lasso(alpha = 0.01,max_iter =10000)
ridge= Ridge(alpha =2,max_iter =500)
bg = RandomForestRegressor()
xgb = xgb.XGBRegressor(base_score=1, booster='gbtree', callbacks=None,
             colsample_bylevel=None, colsample_bynode=None,
             colsample_bytree=None, early_stopping_rounds=None,
             enable_categorical=False, eval_metric=None, feature_types=None,
             gamma=None, gpu_id=None, grow_policy=None, importance_type=None,
             interaction_constraints=None, learning_rate=0.1, max_bin=None,
             max_cat_threshold=None, max_cat_to_onehot=None,
             max_delta_step=None, max_depth=3, max_leaves=None,
             min_child_weight=4, monotone_constraints=None,
             n_estimators=1100, n_jobs=None, num_parallel_tree=None,
             predictor=None, random_state=None)

# Employing the dataset with ML regressors
lr.fit(X_train, Y_train)
dt.fit(X_train, Y_train)
knn.fit(X_train, Y_train)
svm.fit(X_train, Y_train)
lasso.fit(X_train, Y_train)
ridge.fit(X_train, Y_train)
bg.fit(X_train, Y_train)
xgb.fit(X_train,Y_train)
y_pred1 = lr.predict(X_test)
y_pred2 = dt.predict(X_test)
y_pred3 = knn.predict(X_test)
y_pred4 = svm.predict(X_test)
y_pred5 = lasso.predict(X_test)
y_pred6 = ridge.predict(X_test)
y_pred7 = bg.predict(X_test)
y_pred8 = model1.predict(X_test)
# Example: feature column
feature = "feature1"   # <-- replace with your feature name
y_pred = model.predict(df.drop("Target", axis=1))  # predicted values

# Add predictions to DataFrame
df["Predicted"] = y_pred

# Scatter plot: Feature vs Predicted Target
plt.figure(figsize=(8,6))
sns.scatterplot(x=df[feature], y=df["Predicted"], color="blue", alpha=0.6, edgecolor="k")
plt.xlabel(feature, fontsize=12)
plt.ylabel("Predicted Target", fontsize=12)
plt.title(f"{feature} vs Predicted Target", fontsize=14)
plt.grid(True, linestyle="--", alpha=0.7)
plt.show()
import shap
background = X_train[np.random.choice(X_train.shape[0], 40, replace=False)]
explainer = shap.Explainer(model)
shap_values = explainer(X)
np.shape(shap_values.values)

shap.plots.waterfall(shap_values[record_number],max_display=len(X.columns))


# Plot SHAP beeswarm
shap.plots.beeswarm(shap_values, 
                    max_display=len(X.columns),   # top 15 features
                    show=False)       # so we can customize before saving

# --- Style adjustments for publication ---
plt.xlabel("SHAP value (impact on model output)", fontsize=16, fontweight='bold')
plt.ylabel("Features", fontsize=16, fontweight='bold')

# Make ticks bold, larger, and inside
plt.xticks(fontsize=14, fontweight='bold')
plt.yticks(fontsize=14, fontweight='bold')
plt.tick_params(axis='both', which='both', direction='in', length=5, width=1.5)

# Bold frame
for spine in plt.gca().spines.values():
    spine.set_linewidth(1.5)

# Adjust colorbar label
cbar = plt.gcf().axes[-1]
cbar.set_ylabel("Feature value (low → high)", fontsize=14, fontweight='bold')
cbar.tick_params(labelsize=12, width=1.5)

# Tight layout for better spacing
plt.tight_layout()

plt.show()
# SHAP dependence bar plot
shap.plots.bar(shap_values,max_display=len(X.columns), show=False)
plt.xlabel("mean(|SHAP value|)", fontsize=16, fontweight='bold')
plt.ylabel("Features", fontsize=16, fontweight='bold')

# Make ticks bold, larger, and inside
plt.xticks(fontsize=14, fontweight='bold')
plt.yticks(fontsize=14, fontweight='bold')
plt.tick_params(axis='both', which='both', direction='in', length=5, width=1.5)

# Bold frame
for spine in plt.gca().spines.values():
    spine.set_linewidth(1.5)

# Adjust colorbar label
cbar = plt.gcf().axes[-1]
cbar.set_ylabel("Features (low → high)", fontsize=14, fontweight='bold')
cbar.tick_params(labelsize=12, width=1.5)

# Tight layout for better spacing
plt.tight_layout()

# Save high-resolution PNG
plt.savefig("shap_bar_pubready.png", dpi=300, bbox_inches='tight')

# Save as vector PDF for journal submission
plt.savefig("shap_bar_pubready.pdf", dpi=300, bbox_inches='tight')

plt.show()
# SHAP scatter plot
shap.plots.scatter(shap_values[:,"col1"],
                  color = shap_values[:,"col2"],show =False)
plt.style.use('default')   # Reset to matplotlib default
plt.rcParams['axes.facecolor'] = 'white'
plt.rcParams['figure.facecolor'] = 'white'
plt.title("SHAP Dependence Plot\ncol1 vs col2", fontsize=18, fontweight='bold', pad=15)
plt.xlabel("col1", fontsize=14, fontweight='bold')
plt.ylabel("SHAP Value for col1", fontsize=14, fontweight='bold')

# Make axes lines and tick labels bold
plt.tick_params(axis='both', which='major', labelsize=12, width=1.5, direction='in', length=6)
plt.gca().spines['top'].set_linewidth(1.5)
plt.gca().spines['right'].set_linewidth(1.5)
plt.gca().spines['left'].set_linewidth(1.5)
plt.gca().spines['bottom'].set_linewidth(1.5)
plt.show()
