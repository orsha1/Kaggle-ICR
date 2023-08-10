# %%
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import networkx as nx


from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import KNNImputer


# %%
sns.set_style("ticks")

# %%
# Load data

train = pd.read_csv("./data/train.csv")
greeks = pd.read_csv("./data/greeks.csv")
test = pd.read_csv("./data/test.csv")

# %%
# Set index

train = train.set_index("Id")
test = test.set_index("Id")
greeks = greeks.set_index("Id")

# Separate target and features
X = train.drop("Class", axis=1)
y = train["Class"]

# %% [markdown]
# <h3><center> Data Organization / Cleaning </center></h3>

# %%
# Get features list
violin_features = list(X.columns)
categorial_features = X.columns[X.dtypes != float].values.tolist()

# Remove non-float features
violin_features.remove(*categorial_features)

print(categorial_features)

# %% [markdown]
# One-hot encoding for 'EJ' categorial feature

# %%
# Encode categorial object 'EJ' feature

enc = OneHotEncoder()
enc.fit(X["EJ"].values.reshape(-1, 1))
onehot_cols = ["EJ" + "_" + x for x in enc.categories_[0]]

X[onehot_cols] = enc.transform(X["EJ"].values.reshape(-1, 1)).toarray()
X = X.drop("EJ", axis=1)

# test[onehot_cols] = enc.transform(test['EJ'].values.reshape(-1, 1)).toarray()
# test = test.drop('EJ', axis = 1)

# violin_features.extend(onehot_cols)

# %% [markdown]
# Check and fill missing values using KNN Imputer

# %%
# Check which features are missing and how many missing data points are there

nas = X.isnull().sum()[X.isnull().sum() > 0]
nas

# %%
# Since the features which contain missing values are not normally-distributed, we will use iterative imputer.

imp = KNNImputer()  # weight by distance but use many neighbors
X2 = pd.DataFrame(imp.fit_transform(X), index=X.index, columns=X.columns)

# %%
# Plot violin plots of non-categorial features

# fig, axs = plt.subplots(nrows = 7, ncols = 1, figsize = (15, 20))
#
# axs = np.ravel(axs)
#
# for row in range(7):
#    features = violin_features[row*8:(row+1)*8]
#    #df_plot = pd.concat([X2[features], y], axis = 1)
#    df_plot = pd.concat([X2[features], y], axis = 1)
#    df_plot = df_plot / df_plot.max(axis = 0) # Normalize them for plotting purposes
#    df_plot = df_plot.melt(id_vars = ['Class'], value_vars = features)
#    sns.violinplot(data = df_plot, x = 'variable', y = 'value', hue = 'Class', split = True,palette = 'viridis', ax = axs[row])
#    sns.stripplot(data = df_plot, x = 'variable', y = 'value', hue = 'Class', dodge = True,palette = 'viridis', alpha = 0.2, ax = axs[row])
#    axs[row].get_legend().remove()
# sns.despine()
##plt.tight_layout()

# %% [markdown]
# Semi-insights:
# *Highly concentrated AY, BC, BR, BZ, DU, EH, EU, FR  <br>
# * Large AM -> Highly likely Green <br>
# * Large CR -> Slightly likely blue <br>
# * Large DH -> Slightly likely blue <br>
# * Large DA -> Slightly likely blue <br>
# * Large EE -> Slightly likely blue <br>
# * Large FI -> Slightly likely blue <br>
# <br>
#
# Before studying the correlations, let us complete the missing values.

# %%
# Plot categorial

# %%
df_corr = X2.corr().abs()
corr_pairs = df_corr.unstack().dropna().sort_values(ascending=False)
selected_corr_pairs = corr_pairs[(corr_pairs < 1) & (corr_pairs > 0.75)]
selected_corr_pairs = selected_corr_pairs.iloc[::2]  # drop the identical
selected_corr_pairs

# %% [markdown]
# Spaces?

# %%
# Plot calibration plots of highly correlated categorial features

# fig, axs = plt.subplots(nrows = 4, ncols = 3, figsize = (12, 8))
#
# axs = np.ravel(axs)
#
# for i, pair in enumerate(selected_corr_pairs.index):
#    f1, f2 = pair
#    sns.scatterplot(data = pd.concat([X2[[f1, f2]], y], axis = 1), x = f1, y = f2, hue = 'Class', ax = axs[i], palette='viridis')
#    #sns.regplot(data = train[[f1, f2, 'Class']], x = f1, y = f2, ax = axs[i])
#
#    axs[i].get_legend().remove()
#    axs[i].set_xticks([])
#    axs[i].set_yticks([])
# sns.despine()


# %%
df_categorial = train[["GL", "EJ", "Class"]]


# %%
# fig, ax = plt.subplots(ncols = 2, nrows = 1, figsize = (10, 4))
# sns.histplot(data = train[train['Class'] == 0], x = 'GL', hue = 'EJ', ax = ax[0], hue_order = ['A', 'B'], bins = 20)
# sns.histplot(data = train[train['Class'] == 1], x = 'GL', hue = 'EJ', ax = ax[1], hue_order = ['A', 'B'], bins = 20)
# plt.tight_layout()

# %%
print(
    "Value count with GL over 18\n", train.loc[train["GL"] >= 18, "EJ"].value_counts()
)
print(
    "Value count with GL under 18\n", train.loc[train["GL"] < 18, "EJ"].value_counts()
)

# %%
print(
    "The single value with B ("
    + train.loc[(train["GL"] > 18) & (train["EJ"] == "B")].index[0]
    + ") is not the same as the missing value in GL ("
    + X["GL"][X["GL"].isnull()].index[0]
    + ")"
)

# %% [markdown]
# Thus, smaller GL will be always in Class B, and larger A will always be in class A.
# <br>
# Let us map the correlated features in a network.

# %%
selected_corr_pairs = corr_pairs[(corr_pairs < 1) & (corr_pairs > 0.6)]
selected_corr_pairs = selected_corr_pairs.iloc[::2]  # drop the identical
selected_corr_pairs
G = nx.DiGraph(selected_corr_pairs.index.to_list())


# %%
def generate_points_on_circle(nodes, radius=1):
    points = {}
    num_points = len(nodes)
    for i in range(num_points):
        angle = 2 * np.pi * i / num_points
        x = radius * np.cos(angle)
        y = radius * np.sin(angle)
        points[nodes[i]] = (x, y)
    return points


# Generate 15 points on the circle with radius 1
points_on_circle = generate_points_on_circle(list(list(G.nodes())))


# %%
# params = {
#    "font_size": 16,
#    "node_size": 700,
#    "node_color": "white",
#    "edgecolors": "black",
#    "linewidths": 0.5,
#    "width": 0.5,
# }
# nx.draw_networkx(G, points_on_circle, **params)
# ax = plt.gca()
# ax.margins(0.2)
# plt.axis("off")
# plt.show()

# %%
# Add clusters

# %% [markdown]
# <h3><center> Dimensionality Reduction </center></h3>

# %%
from sklearn.decomposition import PCA

# %% [markdown]
# Let us try PCA.
#

# %%
pca = PCA(n_components=3)
X2_reduced = pd.DataFrame(
    pca.fit_transform(X2 / X2.max()), index=X2.index
)  # PCA on scaled data
X2_reduced = X2_reduced / X2_reduced.max()
print(
    "The 3 PCA components explain {:.2f}%, {:.2f}% ,and {:.2f}%, respectievly.".format(
        *100 * pca.explained_variance_ratio_
    )
)


## %%
# fig, axs = plt.subplots(ncols = 2, nrows = 1, figsize = (8, 3.5))
#
# sns.histplot(data = pd.concat([X2_reduced, y], axis = 1), x = 0, hue = 'Class', ax = axs[0])
# sns.scatterplot(data = pd.concat([X2_reduced, y], axis = 1), x = 0, y = 1, hue = 'Class', ax = axs[1])
# plt.tight_layout()

## %%
# fig = plt.figure(1, figsize=(8, 6))
# ax = fig.add_subplot(projection="3d", elev=30, azim=50)
# ax.scatter(
#    X2_reduced[0],
#    X2_reduced[1],
#    X2_reduced[2],
#    c=y,
#    cmap='bwr',
#    edgecolor="k",
#    s=40,
# )
# ax.set_xlabel('0')
# ax.set_ylabel('1')
# ax.set_zlabel('2')

# %% [markdown]
# * The first component can clearly cluster the data to two clusters.
# * The distribution is not related to the class itslef.
# * The second component does not help to further cluster the data. The third is even less useful.<br>
#
# These observations are in line with the relatively high explained variance ratio of the first component compared to the second one. <br>
# Let us examine the components.

# %%
pca_importance = (
    pd.DataFrame(pca.components_[0, :], index=X2.columns, columns=["Importance"])
    .abs()
    .sort_values(ascending=False, by="Importance")
    .reset_index()
)
pca_importance = pca_importance.rename({"index": "Features"}, axis=1)

# %%
pca_importance["Importance cumsum"] = pca_importance["Importance"].cumsum()

# %%
# ax = sns.barplot(data = pca_importance[0:10], y = 'Importance', x = 'Features')
# ax2 = ax.twinx()
# sns.lineplot(data = pca_importance[0:10], y = 'Importance cumsum', x = 'Features', ax = ax2)

# %% [markdown]
# <h3><center> Baseline Models </center></h3>

# %%
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
import lightgbm as lgb
import xgboost as xgb
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_sample_weight, compute_class_weight
from sklearn.metrics import f1_score, accuracy_score


# %%
def balanced_logloss(y, p):
    """
    Calculation of balance logloss according to the competition metric.
    Input:
        y   -   Real target
        p   -   Predicted probabilities
    Output:
        L   -   Balanced logloss
    """
    # Clip the propabilities to avoid function extremes
    p = np.clip(p, 1e-15, 1 - 1e-15)

    # Count targets
    N_0 = y[y == 0].shape[0]
    N_1 = y[y == 1].shape[0]

    # Arrays of ones for samples which belong to each target 0
    y0 = np.zeros_like(y)
    y0[y == 0] = 1
    y1 = np.zeros_like(y)
    y1[y == 1] = 1

    # Probability array for each target
    p0 = p[:, 0]
    p1 = p[:, 1]

    # logloss of target, normalized to the number of samples for each target
    logloss_0 = -np.sum(y0 * np.log(p0)) / (N_0 + 1e-15)
    logloss_1 = -np.sum(y1 * np.log(p1)) / (N_1 + 1e-15)

    L = (logloss_0 + logloss_1) / 2

    return L


# test
# y_tmp = np.array([0, 0, 0, 0, 0])
# p_tmp = np.array([[0.91, 0.09], # 0
#                  [0.89, 0.11], # 0
#                  [0.07, 0.93], # 1
#                  [0.02, 0.98], # 1
#                  [0.84, 0.16]]) # 0
# balanced_logloss(y_tmp, p_tmp)


# %%
def preprocess(X_train, X_test, y_train):  #
    """
    A function which imputes (knn), scales (standard scaler), and calculate class weights.
    Input:
        X_train   -   Training X dataset (df).
        X_test    -   Testing X dataset (df).
        y_train   -   Training target (df) for weighting purposes.
    Output:
        X_train   -   Scaled and imputed training X dataset (df).
        X_test    -   Scaled and imputed testing X dataset (df).
        sw        -   Samples weight based on the training target.
    """
    # Add missing values using KNN Imputer
    imp = KNNImputer(weights="distance", n_neighbors=20).fit(X_train)

    X_train = pd.DataFrame(
        imp.transform(X_train), index=X_train.index, columns=X_train.columns
    )
    X_test = pd.DataFrame(
        imp.transform(X_test), index=X_test.index, columns=X_test.columns
    )

    # Scale data using Standard Scaler
    scaler = StandardScaler().fit(X_train)

    X_train = pd.DataFrame(
        scaler.transform(X_train), index=X_train.index, columns=X_train.columns
    )
    X_test = pd.DataFrame(
        scaler.transform(X_test), index=X_test.index, columns=X_test.columns
    )

    # Class and sample weights
    cw = compute_class_weight(
        class_weight="balanced", classes=np.unique(y_train), y=y_train
    )
    cw = dict(zip(np.unique(y_train), cw))
    sw = compute_sample_weight(cw, y_train)

    return X_train, X_test, imp, scaler, sw, cw


# %%
def pred_score_store(i, X_train, y_train, X_test, y_test, models, models_names):
    """
    A function which predicts, scores predictions, and calculate scores for the different models.
    The function treats a single split.
    Inputs:
        i              -   Split number (int)
        X_train        -   Training X dataset (df).
        X_test         -   Testing X dataset (df).
        y_train        -   Training target (df).
        y_test         -   Testing target (df).
        models         -   Trained models to be evaluated (string)
        models_names   -   Trained models to be evaluated
    Outputs:
        score          -   Scores
        y_hats         -   Predictions
        ps             -   Predicted probabilities
    """
    score, y_hats, ps = {}, {}, {}

    # Iterate over models, store their predictions, probabilities and scores
    for model, model_name in zip(models, models_names):
        model_train_name = (model_name, "train", str(i))
        model_test_name = (model_name, "test", str(i))

        # Store predicted probabilities
        ps[model_train_name] = model.predict_proba(X_train)
        ps[model_test_name] = model.predict_proba(X_test)

        # Store predicted targets
        y_hats[model_train_name] = model.predict(X_train)
        y_hats[model_test_name] = model.predict(X_test)

        # Calculate scores
        score[(model_name, str(i))] = [
            balanced_logloss(y_test, ps[model_test_name]),
            balanced_logloss(y_train, ps[model_train_name]),
            f1_score(y_test, y_hats[model_test_name]),
            f1_score(y_train, y_hats[model_train_name]),
            accuracy_score(y_test, y_hats[model_test_name]),
            accuracy_score(y_train, y_hats[model_train_name]),
        ]

    return score, y_hats, ps


# %%
def split_train(X_in, y_in, random_state):
    skf = StratifiedKFold(n_splits=5, random_state=random_state, shuffle=True)

    scores = {}

    for i, (train_index, test_index) in enumerate(skf.split(X_in, y_in)):
        # Split
        X_train, X_test = X_in.iloc[train_index, :], X.iloc[test_index, :]
        y_train, y_test = y_in.iloc[train_index], y.iloc[test_index]

        X_train, X_test, imp, scaler, sw, cw = preprocess(X_train, X_test, y_train)

        # xgb4
        rf = RandomForestClassifier(criterion="log_loss")
        rf.fit(X_train, y_train, sample_weight=sw)

        # GB
        gb = GradientBoostingClassifier(loss="log_loss")
        gb.fit(X_train, y_train, sample_weight=sw)

        # xgboost
        xgb_clf = xgb.XGBClassifier(n_jobs=28)
        xgb_clf.fit(X_train, y_train, sample_weight=sw)

        # Predict, score and store
        score_split, y_hats, ps = pred_score_store(
            i, X_train, y_train, X_test, y_test, [rf, gb, xgb_clf], ["rf", "gb", "xgb"]
        )
        scores = {**scores, **score_split}

    scores_cols = [
        "balanced_logloss",
        "balanced_logloss_train",
        "f1_score",
        "f1_score_train",
        "accuracy_score",
        "accuracy_score_train",
    ]
    df_scores = pd.DataFrame(scores, index=scores_cols).T.sort_index()

    return (df_scores, y_hats, ps)


# %%
scores = []
for split in [42]:  # [0, 1, 2, 7, 13, 25, 42, 67, 73]:
    score, y_hats, ps = split_train(X, y, split)
    scores.append(score)

# %%
df_scores = (
    pd.concat(scores)
    .reset_index()
    .groupby("level_0")
    .describe()
    .loc[:, (slice(None), ["count", "mean", "std"])]
)
df_scores

# %% [markdown]
# <h3><center> Models Optimization using Optuna  </center></h3>

# %%
import optuna

# from sklearn.metrics import make_scorer
from sklearn.model_selection import cross_val_score, cross_validate
from sklearn.metrics import confusion_matrix


# %%
def optuna_balanced_logloss(clf, X, y):
    """
    Calculation of balance logloss according to the competition metric.
    Input:
        y   -   Real target
        p   -   Predicted probabilities
    Output:
        L   -   Balanced logloss
    """
    # Calculate p
    p = clf.predict_proba(X)

    # Clip the propabilities to avoid function extremes
    p = np.clip(p, 1e-15, 1 - 1e-15)

    # Count targets
    N_0 = y[y == 0].shape[0]
    N_1 = y[y == 1].shape[0]

    # Arrays of ones for samples which belong to each target 0
    y0 = np.zeros_like(y)
    y0[y == 0] = 1
    y1 = np.zeros_like(y)
    y1[y == 1] = 1

    # Probability array for each target
    p0 = p[:, 0]
    p1 = p[:, 1]

    # logloss of target, normalized to the number of samples for each target
    logloss_0 = -np.sum(y0 * np.log(p0)) / (N_0 + 1e-15)
    logloss_1 = -np.sum(y1 * np.log(p1)) / (N_1 + 1e-15)

    L = (logloss_0 + logloss_1) / 2

    return L


# test
# y_tmp = np.array([0, 0, 0, 0, 0])
# p_tmp = np.array([[0.91, 0.09], # 0
#                  [0.89, 0.11], # 0
#                  [0.07, 0.93], # 1
#                  [0.02, 0.98], # 1
#                  [0.84, 0.16]]) # 0
# balanced_logloss(y_tmp, p_tmp)


# %%
def split_optimize_train(X_in, y_in, n_trials, random_state):
    skf = StratifiedKFold(n_splits=5, random_state=random_state, shuffle=True)

    xgb4_studies = {}
    xgb4_bests = {}

    xgb2_studies = {}
    xgb2_bests = {}

    xgb3_studies = {}
    xgb3_bests = {}

    for i, (train_index, val_index) in enumerate(skf.split(X_in, y_in)):
        X_train, X_val = X_in.iloc[train_index, :], X_in.iloc[val_index, :]
        y_train, y_val = y_in.iloc[train_index], y_in.iloc[val_index]

        X_train, X_val, imp, scaler, sw, cw = preprocess(X_train, X_val, y_train)
        cw = cw[0] / cw[1]

        print(random_state, i, "xgb2")

        # Optimize xgb2
        def objective_xgb2(trial):
            xgb2_params = {
                "eval_metric": "logloss",
                "n_jobs": -1,
                "verbosity": 0,
                "random_state": random_state,
                "verbose_eval": False,
                "objective": "binary:logistic",
                "booster": "gbtree",
                "n_estimators": 600,
                "early_stopping_rounds": 20,
                "scale_pos_weight": cw,
                "max_depth": trial.suggest_int(
                    "max_depth", 1, 9
                ),  # trial.suggest_categorical('random_state', [1, 2, 3, 4, 5, 6, 7, 8, 9, None])
                "learning_rate": trial.suggest_float(
                    "learning_rate", 1e-4, 1e-2, log=True
                ),
                "eta": trial.suggest_float("eta", 1e-5, 1.0, log=True),
                "lambda": trial.suggest_float("lambda", 1e-5, 1.0, log=True),
                "gamma": trial.suggest_float("gamma", 1e-5, 1.0, log=True),
                "alpha": trial.suggest_float("alpha", 1e-5, 1.0, log=True),
                "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
                "grow_policy": trial.suggest_categorical(
                    "grow_policy", ["depthwise", "lossguide"]
                ),
            }

            xgb2_clf = xgb.XGBClassifier(**xgb2_params)

            xgb2_clf.fit(
                X_train,
                y_train,
                eval_set=[(X_val, y_val)],
                # random_state = random_state,
                verbose=False,
                sample_weight=sw,
            )

            p_val = xgb2_clf.predict_proba(X_val)

            scores = balanced_logloss(y_val, p_val)

            return scores

        xgb2_study = optuna.create_study(direction="minimize")
        xgb2_study.optimize(objective_xgb2, n_trials=n_trials, n_jobs=-1)

        xgb2_best = xgb.XGBClassifier(**xgb2_study.best_params).fit(
            X_train, y_train, sample_weight=sw
        )

        train_loss = balanced_logloss(y_train, xgb2_best.predict_proba(X_train))

        xgb2_studies[(i, random_state)] = {
            **{"train_score": train_loss},
            **{"val_score": xgb2_study.best_value},
            **xgb2_study.best_params,
        }

        xgb2_bests[(i, random_state)] = xgb2_best

        print(random_state, i, "xgb3")

        # Optimize xgb3
        def objective_xgb3(trial):
            xgb3_params = {
                "eval_metric": "logloss",
                "n_jobs": -1,
                "verbosity": 0,
                "random_state": random_state,
                "verbose_eval": False,
                "objective": "binary:logistic",
                "booster": "gbtree",
                "n_estimators": 900,
                "early_stopping_rounds": 20,
                "scale_pos_weight": cw,
                "max_depth": trial.suggest_int(
                    "max_depth", 1, 9
                ),  # trial.suggest_categorical('random_state', [1, 2, 3, 4, 5, 6, 7, 8, 9, None])
                "learning_rate": trial.suggest_float(
                    "learning_rate", 1e-4, 1e-2, log=True
                ),
                "eta": trial.suggest_float("eta", 1e-5, 1.0, log=True),
                "lambda": trial.suggest_float("lambda", 1e-5, 1.0, log=True),
                "gamma": trial.suggest_float("gamma", 1e-5, 1.0, log=True),
                "alpha": trial.suggest_float("alpha", 1e-5, 1.0, log=True),
                "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
                "grow_policy": trial.suggest_categorical(
                    "grow_policy", ["depthwise", "lossguide"]
                ),
            }

            xgb3_clf = xgb.XGBClassifier(**xgb3_params)

            xgb3_clf.fit(
                X_train,
                y_train,
                eval_set=[(X_val, y_val)],
                verbose=False,
                sample_weight=sw,
            )

            p_val = xgb3_clf.predict_proba(X_val)

            scores = balanced_logloss(y_val, p_val)

            return scores

        xgb3_study = optuna.create_study(direction="minimize")
        xgb3_study.optimize(objective_xgb3, n_trials=n_trials, n_jobs=-1)

        xgb3_best = xgb.XGBClassifier(**xgb3_study.best_params).fit(
            X_train, y_train, sample_weight=sw
        )

        train_loss = balanced_logloss(y_train, xgb3_best.predict_proba(X_train))

        xgb3_studies[(i, random_state)] = {
            **{"train_score": train_loss},
            **{"val_score": xgb3_study.best_value},
            **xgb3_study.best_params,
        }

        xgb3_bests[(i, random_state)] = xgb3_best

        # Optimize xgb4
        def objective_xgb4(trial):
            xgb4_params = {
                "eval_metric": "logloss",
                "n_jobs": -1,
                "verbosity": 0,
                "random_state": random_state,
                "verbose_eval": False,
                "objective": "binary:logistic",
                "booster": "gbtree",
                "n_estimators": 1200,
                "early_stopping_rounds": 20,
                "scale_pos_weight": cw,
                "max_depth": trial.suggest_int(
                    "max_depth", 1, 9
                ),  # trial.suggest_categorical('random_state', [1, 2, 3, 4, 5, 6, 7, 8, 9, None])
                "learning_rate": trial.suggest_float(
                    "learning_rate", 1e-4, 1e-2, log=True
                ),
                "eta": trial.suggest_float("eta", 1e-5, 1.0, log=True),
                "lambda": trial.suggest_float("lambda", 1e-5, 1.0, log=True),
                "gamma": trial.suggest_float("gamma", 1e-5, 1.0, log=True),
                "alpha": trial.suggest_float("alpha", 1e-5, 1.0, log=True),
                "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
                "grow_policy": trial.suggest_categorical(
                    "grow_policy", ["depthwise", "lossguide"]
                ),
            }

            xgb4_clf = xgb.XGBClassifier(**xgb4_params)

            xgb4_clf.fit(
                X_train,
                y_train,
                eval_set=[(X_val, y_val)],
                verbose=False,
                sample_weight=sw,
            )

            p_val = xgb4_clf.predict_proba(X_val)

            scores = balanced_logloss(y_val, p_val)

            return scores

        xgb4_study = optuna.create_study(direction="minimize")
        xgb4_study.optimize(objective_xgb4, n_trials=n_trials, n_jobs=-1)

        xgb4_best = xgb.XGBClassifier(**xgb4_study.best_params).fit(
            X_train, y_train, sample_weight=sw
        )

        train_loss = balanced_logloss(y_train, xgb4_best.predict_proba(X_train))

        xgb4_studies[(i, random_state)] = {
            **{"train_score": train_loss},
            **{"val_score": xgb4_study.best_value},
            **xgb4_study.best_params,
        }

        xgb4_bests[(i, random_state)] = xgb4_best

    return (
        xgb2_studies,
        xgb2_best,
        xgb3_studies,
        xgb3_best,
        xgb4_studies,
        xgb4_best,
    )


# %%
xgb4_studies_all = []
xgb4_bests_all = []
xgb3_studies_all = []
xgb3_bests_all = []
xgb2_studies_all = []
xgb2_bests_all = []

for split in [0, 13]:  # [0, 13, 25, 42, 67]:
    (
        xgb2_studies,
        xgb2_best,
        xgb3_studies,
        xgb3_best,
        xgb4_studies,
        xgb4_best,
    ) = split_optimize_train(X, y, 3, split)

    xgb2_studies = pd.DataFrame(xgb2_studies).T
    xgb2_studies_all.append(xgb2_studies)
    xgb2_bests_all.append(xgb2_best)

    xgb4_studies = pd.DataFrame(xgb4_studies).T
    xgb4_studies_all.append(xgb4_studies)
    xgb4_bests_all.append(xgb4_best)

    xgb3_studies = pd.DataFrame(xgb3_studies).T
    xgb3_studies_all.append(xgb3_studies)
    xgb3_bests_all.append(xgb3_best)

    xgb2_studies.to_csv("xgb2-" + str(split) + ".csv")
    xgb3_studies.to_csv("xgb3-" + str(split) + ".csv")
    xgb4_studies.to_csv("xgb4-" + str(split) + ".csv")

# %%
pd.concat(xgb2_studies_all).to_csv("xgb2_studies.csv")
pd.concat(xgb4_studies_all).to_csv("xgb4_studies.csv")
pd.concat(xgb3_studies_all).to_csv("xgb3_studies.csv")

# %%
