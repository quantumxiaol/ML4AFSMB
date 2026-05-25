# %%
import os
import pickle
import warnings
from time import time
import shap
import matplotlib.colors as mcolors
import joblib
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import numpy as np
import pandas as pd
import seaborn as sns
from lightgbm import LGBMRegressor
from matplotlib import cm
from matplotlib.ticker import MaxNLocator
from mlxtend.feature_selection import SequentialFeatureSelector as SFS
from sklearn.decomposition import PCA
from sklearn.ensemble import AdaBoostRegressor, RandomForestRegressor
from sklearn.feature_selection import RFECV
from sklearn.inspection import PartialDependenceDisplay, permutation_importance
from sklearn.manifold import TSNE, trustworthiness
from sklearn.metrics import make_scorer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, silhouette_score
from sklearn.model_selection import KFold, LeaveOneGroupOut, cross_val_score, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeRegressor
from sklearn.utils import resample
from statsmodels.stats.outliers_influence import variance_inflation_factor
from xgboost import XGBRegressor
warnings.filterwarnings("ignore")

if not os.path.exists('./graph'):
    os.makedirs('./graph')

if not os.path.exists('./data'):
    os.makedirs('./data')

def graph_path(filename):
    return os.path.join('./graph', filename)

# %%
df=pd.read_excel('./Electrolytes.xlsx')

# %%
df.columns

# %%
feature=[ 'sC', 'sO', 'sF', 'aC', 'aO', 'aF', 'InOr',
       'MolLogP', 'TPSA', 'NumHAcceptors', 'NumValenceElectrons',
       'MaxAbsPartialCharge', 'MaxPartialCharge', 'MinAbsPartialCharge',
       'MinPartialCharge', 'Balabanj', 'PartialChargeGap', 'DipoleMoment',
       'Polarizability', 'ESPmin', 'ESPmax', 'ESPGap', 'PoSA', 'NeSA', 'MPI',
       'NonSA', 'PolarSA', 'HOMO', 'LUMO', 'MOGap', 'PoSAratio', 'NeSAratio',
       'NonSAratio', 'PolarSAratio']
DGBMEcolor='#C55645'
DGDEcolor='#767877'
G2color='#354898'
color1='#D06262'
color2='#4871B3'
color3='#73ABCF'
color4='#972D36'
color5='#FAAE5F'
color6='#6B6B6B'

# %%
plt.rcParams['font.family'] = 'Arial' 

# %% [markdown]
# # Revision

# %%
features = ['aO', 'sO', 'NonSA']
target = 'LCE'
seed = 42

rf = RandomForestRegressor(
    n_estimators=200,
    max_depth=5,
    random_state=seed,
    n_jobs=-1
)

kf = KFold(n_splits=5, shuffle=True, random_state=seed)
X = df[features]
y = df[target]

# --- Training and Saving to Excel ---
save_path = './LCE_CV_Results.xlsx'
with pd.ExcelWriter(save_path) as writer:
    for i, (train_idx, test_idx) in enumerate(kf.split(X)):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
        
        rf.fit(X_train, y_train)
        y_train_pred = rf.predict(X_train)
        y_test_pred = rf.predict(X_test)
        
        # Calculate MSE
        mse_train = mean_squared_error(y_train, y_train_pred)
        mse_test = mean_squared_error(y_test, y_test_pred)
        
        # Combine data for current fold
        # Use a flag to distinguish Train and Test set within the same sheet
        df_train = pd.DataFrame({'Actual': y_train, 'Predicted': y_train_pred, 'Type': 'Train'})
        df_test = pd.DataFrame({'Actual': y_test, 'Predicted': y_test_pred, 'Type': 'Test'})
        df_fold = pd.concat([df_train, df_test])
        
        # Store MSE in sheet metadata (as the first two rows or via separate columns)
        df_fold['MSE_Train'] = mse_train
        df_fold['MSE_Test'] = mse_test
        
        # Save to specific sheet
        df_fold.to_excel(writer, sheet_name=f'Fold_{i+1}', index=False)

print(f"Cross-validation completed. Results saved to {save_path}")

# %%
def plot_density_cv(file_path):
    """
    Reads CV results from Excel and plots 2D Density Plots for each fold.
    """
    # Load all sheet names
    xl = pd.ExcelFile(file_path)
    sheet_names = xl.sheet_names
    num_folds = len(sheet_names)
    
    fig, axes = plt.subplots(1, num_folds, figsize=(5 * num_folds, 5))
    if num_folds == 1: axes = [axes]
    
    for i, sheet in enumerate(sheet_names):
        data = pd.read_excel(file_path, sheet_name=sheet)
        
        train_data = data[data['Type'] == 'Train']
        test_data = data[data['Type'] == 'Test']
        mse_test = data['MSE_Test'].iloc[0]
        
        ax = axes[i]
        
        # Plot 2D Density for Test Set (Main visualization)
        # Using 'fill=True' and 'cmap' to show density by color depth
        sns.kdeplot(
            x=test_data['Actual'], 
            y=test_data['Predicted'], 
            fill=True, 
            thresh=0, 
            levels=20, 
            cmap="Blues", 
            ax=ax
        )
        
        # Overlay a light scatter of Test points for clarity
        ax.scatter(test_data['Actual'], test_data['Predicted'], color='black', s=5, alpha=0.3, label='Test Points')
        
        # Identity line (y=x)
        min_val = min(data['Actual'].min(), data['Predicted'].min())
        max_val = max(data['Actual'].max(), data['Predicted'].max())
        ax.plot([min_val, max_val], [min_val, max_val], 'r--', lw=1.5)
        
        # Formatting
        ax.set_title(f'{sheet}\nTest MSE: {mse_test:.4f}')
        ax.set_xlabel('Actual LCE')
        if i == 0:
            ax.set_ylabel('Predicted LCE')
        
        ax.set_xlim(min_val, max_val)
        ax.set_ylim(min_val, max_val)

    plt.tight_layout()
    plt.show()


# %%



X5 = df[['aO', 'PoSA', 'NonSA', 'sO', 'NeSA']]
X3 = df[['aO', 'NonSA', 'sO']]

# %% [markdown]
# ## Five-Feature Comparison Experiment

# %%
def apply_publication_style(ax, xlabel, ylabel):
    ax.set_xlabel(xlabel, fontsize=24)
    ax.set_ylabel(ylabel, fontsize=24)
    # Tick parameters
    ax.tick_params(axis='both', width=1.8, length=5, labelsize=22, direction='out')
    # Spine linewidth
    for spine in ax.spines.values():
        spine.set_linewidth(1.8)

def save_and_show(save_name):
    save_path = graph_path(save_name)
    plt.tight_layout()
    plt.savefig(save_path, transparent=True, bbox_inches='tight', dpi=300)
    print(f"Figure saved to: {save_path}")
    plt.show()
def run_ablation_mse(X5, X3, y):
    cv = KFold(n_splits=5, shuffle=True, random_state=42)
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    
    # Scikit-learn scoring returns negative MSE
    mse_5 = -cross_val_score(model, X5, y, cv=cv, scoring='neg_mean_squared_error')
    mse_3 = -cross_val_score(model, X3, y, cv=cv, scoring='neg_mean_squared_error')
    fig, ax = plt.subplots(figsize=(8, 7))
    sns.boxplot(data=[mse_5, mse_3], ax=ax, palette="Set2", width=0.6, linewidth=1.8)
    ax.set_xticklabels(['5 Features', '3 Features'])
    ax.set_ylim(-0.05,0.45)
    apply_publication_style(ax, '', 'MSE')
    save_and_show('Ablation_Study_MSE.pdf')
    
    return mse_5, mse_3
def plot_permutation_importance(result, feature_names):
    fig, ax = plt.subplots(figsize=(8, 7))
    sorted_idx = result.importances_mean.argsort()
    
    bp = ax.boxplot(result.importances[sorted_idx].T, vert=False, 
                    labels=feature_names[sorted_idx], patch_artist=True)
    
    for element in ['boxes', 'whiskers', 'fliers', 'means', 'medians', 'caps']:
        plt.setp(bp[element], color='black', linewidth=1.8)
    for patch in bp['boxes']:
        patch.set_facecolor('#8da0cb')

    apply_publication_style(ax, 'Importance Score', 'Descriptors')
    save_and_show('Permutation_Importance.pdf')
def plot_correlation(X5):
    fig, ax = plt.subplots(figsize=(9, 8))
    corr = X5.corr()
    
    sns.heatmap(corr, annot=True, cmap='RdBu_r', center=0, ax=ax,
                annot_kws={"size": 18}, 
                cbar_kws={'shrink': 0.8}, linewidths=1.2)
    
    ax.tick_params(axis='both', labelsize=22)
    for spine in ax.spines.values():
        spine.set_linewidth(1.8)
        
    save_and_show('Correlation_Matrix.pdf')


# %%
mse_5, mse_3 = run_ablation_mse(X5, X3, y)
model_5 = RandomForestRegressor(n_estimators=200, random_state=42,max_depth=5).fit(X5, y)
result = permutation_importance(model_5, X5, y, n_repeats=10, random_state=42)
plot_permutation_importance(result, np.array(X5.columns))
plot_correlation(X5)

# %%
def apply_publication_style(ax, xlabel, ylabel):
    ax.set_xlabel(xlabel, fontsize=24)
    ax.set_ylabel(ylabel, fontsize=24)
    ax.tick_params(axis='both', width=1.8, length=5, labelsize=22, direction='out')
    for spine in ax.spines.values():
        spine.set_linewidth(1.8)

def save_and_show(save_name):
    save_path = graph_path(save_name)
    plt.tight_layout()
    plt.savefig(save_path, transparent=True, bbox_inches='tight', dpi=300)
    print(f"Heatmap saved to: {save_path}")
    plt.show()

def plot_bootstrap_rank_heatmap(X5, y, n_iterations=100, cbar_fontsize=20):
    """
    Perform bootstrap and plot rank stability heatmap.
    cbar_fontsize: Controls the font size of the colorbar ticks and label.
    """
    feature_names = X5.columns
    rank_list = []

    print(f"Running {n_iterations} bootstrap iterations on {len(y)} samples...")
    
    for i in range(n_iterations):
        # 1. Bootstrapping (Sampling with replacement)
        X_res, y_res = resample(X5, y, random_state=i)
        
        # 2. Fit model
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X_res, y_res)
        
        # 3. Calculate ranks (1 = highest importance)
        importances = model.feature_importances_
        ranks = pd.Series(importances).rank(ascending=False).values
        rank_list.append(ranks)

    # Aggregate rank frequencies
    df_rank = pd.DataFrame(rank_list, columns=feature_names)
    rank_counts = df_rank.apply(lambda x: x.value_counts()).fillna(0).T
    rank_counts = rank_counts / n_iterations  # Convert to frequency (0-1)

    # --- Plotting ---
    fig, ax = plt.subplots(figsize=(8, 7))
    
    # cbar_kws used for the colorbar label size
    sns.heatmap(rank_counts, annot=True, cmap="YlGnBu", fmt=".2f", 
                annot_kws={"size": 20}, 
                cbar_kws={'label': 'Frequency'}, ax=ax)

    # Adjust colorbar tick labels size
    cbar = ax.collections[0].colorbar
    cbar.ax.tick_params(labelsize=cbar_fontsize)
    cbar.set_label('Frequency', fontsize=cbar_fontsize)

    # Standard formatting
    apply_publication_style(ax, 'Rank', '')
    
    # Spine thickness for heatmap
    for spine in ax.spines.values():
        spine.set_linewidth(1.8)

    save_and_show('Bootstrap_Rank_Heatmap.pdf')

# --- Usage ---
plot_bootstrap_rank_heatmap(X5, y, n_iterations=100, cbar_fontsize=22)

# %% [markdown]
# ## External Electrolyte Validation

# %%
validation_df = pd.read_excel('./Validation_dataset.xlsx', sheet_name='Sheet1')

# %%
model = joblib.load('./rf_model_LCE.pkl')
scaler = joblib.load('./scaler_LCE.pkl')
features = ['aO', 'NonSA', 'sO']
X_test = validation_df[features]
y_true = validation_df['LCE']
X_test_scaled = scaler.transform(X_test)
y_pred = model.predict(X_test_scaled)
plt.rcParams.update({'font.size': 12, 'font.family': 'Arial'})
fig, ax = plt.subplots(figsize=(5, 5))
sns.scatterplot(x=y_true, y=y_pred, alpha=1.0, color=color2, edgecolor='k', s=60, ax=ax)
lims = [
    np.min([ax.get_xlim(), ax.get_ylim()]),
    np.max([ax.get_xlim(), ax.get_ylim()])
]   
ax.plot(lims, lims, 'k--', alpha=0.75, zorder=1)
margin = np.max(np.abs(y_true - y_pred))
lims_arr = np.array(lims)
ax.fill_between(lims_arr, lims_arr - margin, lims_arr + margin, 
                color='gray', alpha=0.1, zorder=0)
mse = mean_squared_error(y_true, y_pred)
ax.set_xlabel('Actual LCE')
ax.set_ylabel('Predicted LCE')
ax.set_xlim(lims)
ax.set_ylim(lims)
ax.set_aspect('equal')
stats_text = f'MSE = {mse:.3f}'
ax.text(0.05, 0.95, stats_text, transform=ax.transAxes, 
        verticalalignment='top')

plt.tight_layout()
plt.savefig(graph_path('LCE_Validation.pdf'), dpi=300, transparent=True, bbox_inches='tight')
plt.show()

# %% [markdown]
# ## Train-Test Sample Split

# %%
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
train_families = df.loc[X_train.index, 'Solvent_Family'].value_counts().rename('Train')
test_families = df.loc[X_test.index, 'Solvent_Family'].value_counts().rename('Test')
split_df = pd.DataFrame([train_families, test_families]).T.fillna(0)
plot_df = split_df.stack().reset_index()
plot_df.columns = ['Solvent_Family', 'Split', 'Count']
plt.figure(figsize=(12, 8), dpi=300)
ax = sns.barplot(
    data=plot_df, 
    x='Solvent_Family', 
    y='Count', 
    hue='Split', 
    palette=[G2color,DGBMEcolor],
    edgecolor='black',
    linewidth=0.8
)
for spine in ax.spines.values():
    spine.set_linewidth(2)
plt.xticks(rotation=20, ha='right', fontsize=14)
plt.yticks(fontsize=14)
plt.xlabel('Solvent Family', fontsize=20)
plt.ylabel('Number of Samples', fontsize=20)
ax.set_ylim(0, 30)
for p in ax.patches:
    val = int(p.get_height())
    if val > 0:
        ax.annotate(f'{val}', 
                    (p.get_x() + p.get_width() / 2., p.get_height()), 
                    ha='center', va='center', 
                    xytext=(0, 5), 
                    textcoords='offset points', 
                    fontsize=16)
ax.tick_params(
    axis='both',
    width=2,
    length=5,
    labelsize=16
)
plt.legend(title='Data Split', title_fontsize='20', fontsize='18', loc='upper right', frameon=False)
sns.despine()
plt.tight_layout()
plt.savefig(graph_path('Data_Split_Distribution.pdf'), transparent=True, bbox_inches='tight', dpi=300)

# %%
X = df[['aO', 'sO', 'NonSA']]
y = df['LCE']
groups = df['Solvent_Family']

logo = LeaveOneGroupOut()
rf = RandomForestRegressor(n_estimators=100, random_state=42)

y_true, y_pred = [], []
for train_idx, test_idx in logo.split(X, y, groups=groups):
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
    
    rf.fit(X_train, y_train)
    y_true.extend(y_test)
    y_pred.extend(rf.predict(X_test))
lofo_r2 = r2_score(y_true, y_pred)
lofo_mse = mean_squared_error(y_true, y_pred)

print(f"Leave-One-Family-Out R2: {lofo_r2:.3f}")
print(f"Leave-One-Family-Out MSE: {lofo_mse:.3f}")

# %%
X = df[['aO', 'sO', 'NonSA']]
y = df['LCE']
groups = df['Solvent_Family']

# ==============================================================================
# 2. LEAVE-ONE-FAMILY-OUT CROSS-VALIDATION (LOFO-CV)
# ==============================================================================
logo = LeaveOneGroupOut()
rf = RandomForestRegressor(n_estimators=100, random_state=42)

# Arrays to store results
y_true_all = []
y_pred_all = []
family_all = []

# Loop through each distinct chemical family as the test set
for train_idx, test_idx in logo.split(X, y, groups=groups):
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
    
    rf.fit(X_train, y_train)
    
    y_true_all.extend(y_test)
    y_pred_all.extend(rf.predict(X_test))
    family_all.extend(groups.iloc[test_idx])

# Create a summary dataframe for results
results_df = pd.DataFrame({
    'Actual_LCE': y_true_all,
    'Predicted_LCE': y_pred_all,
    'Solvent_Family': family_all
})

# ==============================================================================
# 3. PRINT GLOBAL AND GROUP-WISE METRICS FOR REBUTTAL INSIGHTS
# ==============================================================================
global_r2 = r2_score(results_df['Actual_LCE'], results_df['Predicted_LCE'])
global_mse = mean_squared_error(results_df['Actual_LCE'], results_df['Predicted_LCE'])

print("=" * 50)
print(f"GLOBAL LOFO PERFORMANCE:")
print(f"  Global $R^2$: {global_r2:.3f}")
print(f"  Global MSE: {global_mse:.3f}")
print("=" * 50)
print("GROUP-WISE PERFORMANCE (DECOUPLING ERROR SOURCE):")

for fam in results_df['Solvent_Family'].unique():
    sub = results_df[results_df['Solvent_Family'] == fam]
    # R2 can be volatile if a group has very few samples (e.g., Ionic Liquids)
    if len(sub) > 1:
        fam_r2 = r2_score(sub['Actual_LCE'], sub['Predicted_LCE'])
        fam_mse = mean_squared_error(sub['Actual_LCE'], sub['Predicted_LCE'])
        print(f"  Family: {fam:<25} | Samples: {len(sub):<3} | R2: {fam_r2:>6.3f} | MSE: {fam_mse:.3f}")
    else:
        fam_mse = mean_squared_error(sub['Actual_LCE'], sub['Predicted_LCE'])
        print(f"  Family: {fam:<25} | Samples: {len(sub):<3} | R2:   N/A  | MSE: {fam_mse:.3f}")
print("=" * 50)

# ==============================================================================
# 4. HIGH-QUALITY JACS-STYLE VISUALIZATION
# ==============================================================================
plt.figure(figsize=(8, 6), dpi=300) # JACS standard high resolution
# Professional academic palette for 6 distinct solvent families
custom_palette = {
    'Linear_Ethers':color1,
    'Conventional_Carbonates': color2,
    'Fluorinated_Systems': color3,
    'Organophosphates_Sulfones': color4,
    'Cyclic_Ethers': color5,
    'Ionic_Liquids': color6
}

# Plot the scatter points categorized by family
sns.scatterplot(
    data=results_df,
    x='Actual_LCE',
    y='Predicted_LCE',
    hue='Solvent_Family',
    palette=custom_palette,
    s=80,
    edgecolor='black',
    linewidth=0.8,
    alpha=0.9
)

# Draw the perfect prediction baseline (y = x)
min_val = min(results_df['Actual_LCE'].min(), results_df['Predicted_LCE'].min()) - 0.2
max_val = max(results_df['Actual_LCE'].max(), results_df['Predicted_LCE'].max()) + 0.2
plt.plot([min_val, max_val], [min_val, max_val], color='grey', linestyle='--', linewidth=1.5)

# Refine axis labels and limits
plt.xlabel('Actual LCE', fontsize=24)
plt.ylabel('Predicted LCE', fontsize=24)
plt.xlim(min_val, max_val)
plt.ylim(min_val, max_val)
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)

# Format the legend to look elegant
plt.legend(fontsize='14', loc='upper left', frameon=False)

# Add statistical information box inside the plot
stats_text = f"MSE = {global_mse:.3f}"
plt.text(
    0.95, 0.05, stats_text,
    transform=plt.gca().transAxes,
    fontsize=18, 
    va='bottom', ha='right'
    
)

plt.tight_layout()
plt.savefig(graph_path('Lofo.pdf'), transparent=True, bbox_inches='tight', dpi=300)
plt.show()

# %% [markdown]
# ## TGC

# %%


# %% [markdown]
# # CE and LCE Distribution Plots

# %%
df['totalSalt']=df['Ratio1']+df['Ratio2']+df['Ratio3']

# %%
fig, ax = plt.subplots(figsize=(8,6))
sns.histplot(df, x='totalSalt', bins=20, kde=True,ax=ax,color='#b88640',
             line_kws={'color':'#C87740','linewidth':2})                                                                         
ax.set_xlabel("CE (%)", fontsize=18, fontweight='medium')
ax.set_ylabel("Count", fontsize=18, fontweight='medium')
ax.tick_params(axis='x', labelsize=16)
ax.tick_params(axis='y', labelsize=16)
for spine in ax.spines.values():
    spine.set_linewidth(1.8)  
ax.tick_params(
    axis='both',
    width=1.8,
    length=5,
    labelsize=16
)
ax.set_ylim(0,60)
plt.tight_layout()
plt.savefig(graph_path('Salt_con.pdf'),transparent=True,dpi=300,bbox_inches="tight")
plt.show()

# %%
fig, ax = plt.subplots(figsize=(8,6))
sns.histplot(df, x='CE(%)', bins=20, kde=True,ax=ax,color='#436BA9',
             line_kws={'color':'#436BA9','linewidth':2,'alpha':0.6})                                                                         
ax.set_xlabel("CE (%)", fontsize=18, fontweight='medium')
ax.set_ylabel("Count", fontsize=18, fontweight='medium')
ax.tick_params(axis='x', labelsize=16)
ax.tick_params(axis='y', labelsize=16)
for spine in ax.spines.values():
    spine.set_linewidth(1.8)  
ax.tick_params(
    axis='both',
    width=1.8,
    length=5,
    labelsize=16
)
ax.set_ylim(0,35)
plt.tight_layout()
plt.savefig(graph_path('CE_dis.pdf'),transparent=True,dpi=300,bbox_inches="tight")
plt.show()

# %%
fig, ax = plt.subplots(figsize=(8,6))
sns.histplot(df, x='LCE', bins=20, kde=True,ax=ax,color='#506E96',
             line_kws={'color':'#436BA9','linewidth':2,'alpha':0.6})                                                                         
ax.set_xlabel("LCE", fontsize=18, fontweight='medium')
ax.set_ylabel("Count", fontsize=18, fontweight='medium')
ax.tick_params(axis='x', labelsize=16)
ax.tick_params(axis='y', labelsize=16)
for spine in ax.spines.values():
    spine.set_linewidth(1.8)  
ax.tick_params(
    axis='both',                    
    width=1.8,                      
    length=5,                       
    labelsize=16                    
)
ax.set_ylim(0,35)
plt.tight_layout()
plt.savefig(graph_path('LCE_dis.pdf'),transparent=True,dpi=300,bbox_inches="tight")
plt.show()

# %% [markdown]
# # t-SNE Dimensionality Reduction

# %% [markdown]
# ## LCE

# %%
feature_df = df[feature]
feature_df = feature_df.apply(pd.to_numeric, errors='coerce').fillna(0)
features = feature_df.values
performance = df['LCE'].values
tsne = TSNE(n_components=2, perplexity=30)
tsne_results = tsne.fit_transform(features)
k = 10
trust_score = trustworthiness(features, tsne_results, n_neighbors=k)
print(f"Trustworthiness (k={k}): {trust_score:.4f}")
tsne_data = {
    'tsne_results': tsne_results,
    'performance': performance,
    'trust_score': trust_score
}
with open('./data/tsne_lce.pkl', 'wb') as f:
    pickle.dump(tsne_data, f)
print("t-SNE data saved to data/tsne_lce.pkl")

# %%
with open('./data/tsne_lce.pkl', 'rb') as f:
    tsne_data = pickle.load(f)
tsne_results = tsne_data['tsne_results']
performance = tsne_data['performance']
fig, ax = plt.subplots(figsize=(9.5,6.5))
scatter = plt.scatter(tsne_results[:, 0], tsne_results[:, 1], c=performance, cmap='viridis')
cbar=plt.colorbar(scatter,ax=ax,pad=0.05)
cbar.set_label('LCE', fontsize=14, fontweight='medium')
cbar.ax.tick_params(
    labelsize=14,    
    width=1.8,       
    length=5         
)
plt.xlabel('t-SNE1',fontsize=18)
plt.ylabel('t-SNE2',fontsize=18)
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
for spine in ax.spines.values():
    spine.set_linewidth(1.8)  
ax.tick_params(
    axis='both',                    
    width=1.8,                      
    length=5,                       
    labelsize=16                    
)
plt.savefig(graph_path('t_SNE_LCE_dis.pdf'),transparent=True,dpi=300,bbox_inches="tight")
plt.show()

# %% [markdown]
# ##  CE

# %%
feature_df = df[feature]
feature_df = feature_df.apply(pd.to_numeric, errors='coerce').fillna(0)
features = feature_df.values
performance = df['CE'].values
tsne = TSNE(n_components=2, perplexity=30)
tsne_results = tsne.fit_transform(features)
k = 10
trust_score = trustworthiness(features, tsne_results, n_neighbors=k)
print(f"Trustworthiness (k={k}): {trust_score:.4f}")
tsne_data = {
    'tsne_results': tsne_results,
    'performance': performance,
    'trust_score': trust_score
}
with open('./data/tsne_ce.pkl', 'wb') as f:
    pickle.dump(tsne_data, f)
print("t-SNE data saved to data/tsne_ce.pkl")

# %%
with open('./data/tsne_ce.pkl', 'rb') as f:
    tsne_data = pickle.load(f)
tsne_results = tsne_data['tsne_results']
performance = tsne_data['performance']
fig, ax = plt.subplots(figsize=(9.5,6.5))
scatter = plt.scatter(tsne_results[:, 0], tsne_results[:, 1], c=performance, cmap='viridis')
cbar=plt.colorbar(scatter,ax=ax,pad=0.05)
cbar.set_label('CE (%)', fontsize=14, fontweight='medium')
cbar.ax.tick_params(
    labelsize=14,    
    width=1.8,       
    length=5         
)
plt.xlabel('t-SNE1',fontsize=18)
plt.ylabel('t-SNE2',fontsize=18)
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
for spine in ax.spines.values():
    spine.set_linewidth(1.8)  
ax.tick_params(
    axis='both',                    
    width=1.8,                      
    length=5,                       
    labelsize=16                    
)
plt.savefig(graph_path('t_SNE_CE_dis.pdf'),transparent=True,dpi=300,bbox_inches="tight")
plt.show()

# %% [markdown]
# # PCA Analysis

# %%
X=df[feature]
y=df['LCE']
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
pca = PCA()
principal_components = pca.fit_transform(X_scaled)
var_ratios = pca.explained_variance_ratio_
print("-" * 30)
for i, vr in enumerate(var_ratios[:10], start=1):
    print(f"PC{i}: {vr:.1%} (Cumulative: {np.cumsum(var_ratios)[i-1]:.1%})")
print("-" * 30)
pca_data = {
    'explained_variance_ratio': pca.explained_variance_ratio_,
    'cumulative_variance_ratio': np.cumsum(pca.explained_variance_ratio_),
    'components': pca.components_,
    'feature_names': feature,
    'transformed_data': principal_components,
    'targets': y
}

with open('./data/pca_results.pkl', 'wb') as f:
    pickle.dump(pca_data, f)

print("PCA results saved to ./data/pca_results.pkl, including transformed coordinates and target labels")

# %%
with open('./data/pca_results.pkl', 'rb') as f:
    pca_data = pickle.load(f)

explained_var_ratio = pca_data['explained_variance_ratio']
cumulative_var_ratio = pca_data['cumulative_variance_ratio']
X_pca = pca_data['transformed_data']
y_target = pca_data['targets']
G2color = '#708cac'
DGBMEcolor = '#AC7070'
cmap_colors = ["#495E77", "#f7f7f7", "#824D4D"]
my_cmap = mcolors.LinearSegmentedColormap.from_list("LCE_Gradient", cmap_colors)
fig, ax = plt.subplots(figsize=(12.8, 5.8))
x_all = range(1, len(explained_var_ratio) + 1)
ax.bar(x_all, explained_var_ratio, label='Individual Var.', color=G2color)
ax.plot(x_all, cumulative_var_ratio, 'o-', linewidth=2.5, label='Cumulative Var.', 
        color=DGBMEcolor, markersize=10)
ax.axhline(y=0.9, color='#6B6B6B', alpha=0.6, linestyle='--', linewidth=1.8)
ax_ins = ax.inset_axes([0.5, 0.25, 0.35, 0.52])
scatter = ax_ins.scatter(X_pca[:, 0], X_pca[:, 1], c=y_target, cmap=my_cmap, 
                         s=40, alpha=0.9, edgecolors='#444444', linewidth=0.5)
ax_ins.set_xlabel('PC1', fontsize=18, labelpad=1)
ax_ins.set_ylabel('PC2', fontsize=18, labelpad=1)
ax_ins.grid(True, linestyle=':', alpha=0.5)
ax_ins.tick_params(axis='both', width=1.8, length=5, labelsize=18)
for spine in ax_ins.spines.values():
    spine.set_linewidth(1.8)
#ax_ins.patch.set_alpha(0.85)
cax = ax_ins.inset_axes([1.05, 0, 0.05, 1]) # [x, y, width, height]
cbar = plt.colorbar(scatter, cax=cax)
cbar.ax.tick_params(labelsize=14)
cbar.set_label('LCE', rotation=270, labelpad=10, fontsize=18)
ax.set_ylabel('Explained Variance Ratio', fontsize=24)
ax.set_xlabel('Principal Components', fontsize=24)
ax.set_xlim(0, len(explained_var_ratio)+1)
ax.set_ylim(0, 1.05)
ax.tick_params(axis='both', width=1.8, length=5, labelsize=22)
for spine in ax.spines.values():
    spine.set_linewidth(1.8)
ax.legend(loc='upper left', frameon=False, fontsize=20, bbox_to_anchor=(0.15, 0.28))
plt.tight_layout()
plt.savefig(graph_path('PCA_Analysis_Final.pdf'), transparent=True, dpi=300, bbox_inches='tight')
plt.show()

# %% [markdown]
# # Pearson Correlation Matrix

# %%
correlation_matrix = df[['aC', 'aO', 'aF','sC', 'sO', 'sF', 'InOr',
       'MolLogP', 'TPSA', 'NumHAcceptors', 'NumValenceElectrons',
       'MaxAbsPartialCharge', 'MaxPartialCharge', 'MinAbsPartialCharge',
       'MinPartialCharge', 'Balabanj', 'PartialChargeGap', 'DipoleMoment',
       'Polarizability', 'ESPmin', 'ESPmax', 'ESPGap', 'PoSA', 'NeSA', 'MPI',
       'NonSA', 'PolarSA', 'HOMO', 'LUMO', 'MOGap', 'PoSAratio', 'NeSAratio',
       'NonSAratio', 'PolarSAratio']].corr(method='pearson')

# %%
fig,ax=plt.subplots(figsize=(20, 16))
heat = sns.heatmap(
    correlation_matrix,
    cmap='RdBu_r',
    annot_kws={"size": 12, "color": "black"},
    cbar_kws={"shrink": 1, "format": "%.2f",'pad':0.02},
    ax=ax
)
cbar = heat.collections[0].colorbar
#cbar.set_label('Pearson coefficient', fontsize=20)
cbar.ax.tick_params(labelsize=20, width=1.8, length=5)
ax.set_xticklabels(
    ax.get_xticklabels(),
    rotation=90,
    fontsize=18
)
ax.set_yticklabels(
    ax.get_yticklabels(),
    rotation=0,
    fontsize=18
)
for spine in ax.spines.values():
    spine.set_linewidth(1.8)
plt.tight_layout()
plt.savefig(graph_path('Pearson_34.pdf'),transparent=True, dpi=300,bbox_inches='tight')

# %%
correlation_matrix = df[['sO', 'aO','NonSA']].corr(method='pearson')

# %%
fig, ax = plt.subplots(figsize=(8, 6))
heat = sns.heatmap(
    correlation_matrix,
    annot=True,
    fmt='.2f',
    cmap='RdBu_r',
    vmin=-1,
    vmax=1,
    linewidths=1.5,
    linecolor='black',
    annot_kws={"size": 20, "color": "black"},
    cbar_kws={
        "shrink": 1,
        "format": "%.2f",
        "pad": 0.02
    },
    ax=ax
)
# cbar = heat.collections[0].colorbar
# cbar.set_clim(-1, 1)
cbar = heat.collections[0].colorbar
cbar.ax.tick_params(labelsize=22, width=1.8, length=5)
ax.set_xticklabels(
    ax.get_xticklabels(),
    fontsize=24
)
ax.set_yticklabels(
    ax.get_yticklabels(),
    rotation=0,
    fontsize=24
)
for spine in ax.spines.values():
    spine.set_linewidth(1.8)

plt.tight_layout()
plt.savefig(
    graph_path('3person.pdf'),
    transparent=True,
    dpi=300,
    bbox_inches='tight'
)
plt.show()

# %% [markdown]
# # Wrapper Feature Selection

# %% [markdown]
# ## Full-Descriptor Random Forest with Error Bars

# %%
X = df[feature]
y = df['LCE']
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)
seeds = [0, 1, 2, 3, 4]
mse_train_list = []
r2_train_list  = []
mse_test_list  = []
r2_test_list   = []
y_train_preds = []  # list of arrays, shape (n_train,)
y_test_preds  = []  # list of arrays, shape (n_test,)

for seed in seeds:
    rf = RandomForestRegressor(
        n_estimators=200,
        max_depth=5,
        random_state=seed,
        n_jobs=-1
    )
    rf.fit(X_train_scaled, y_train)
    y_tr_pred = rf.predict(X_train_scaled)
    y_te_pred = rf.predict(X_test_scaled)
    y_train_preds.append(y_tr_pred)
    y_test_preds .append(y_te_pred)
    mse_train_list.append(mean_squared_error(y_train, y_tr_pred))
    r2_train_list .append(r2_score(y_train, y_tr_pred))
    mse_test_list .append(mean_squared_error(y_test, y_te_pred))
    r2_test_list  .append(r2_score(y_test, y_te_pred))
y_train_preds = np.vstack(y_train_preds).T  # shape (n_train, 5)
y_test_preds  = np.vstack(y_test_preds ) .T  # shape (n_test,  5)
for i, seed in enumerate(seeds):
    print(f"Seed={seed} → Train MSE={mse_train_list[i]:.3f}, "
          f"Train R²={r2_train_list[i]:.3f} | "
          f"Test MSE={mse_test_list[i]:.3f}, "
          f"Test R²={r2_test_list[i]:.3f}")
train_mean = y_train_preds.mean(axis=1)
train_std  = y_train_preds.std(axis=1)
test_mean  = y_test_preds.mean(axis=1)
test_std   = y_test_preds.std(axis=1)
mean_r2_train = np.mean(r2_train_list)
mean_r2_test  = np.mean(r2_test_list)

# %%
fig, ax = plt.subplots(figsize=(8, 6))
ax.errorbar(
    y_train, train_mean,
    yerr=train_std,
    fmt='o', ecolor='k', mec='k', mfc='#436BA9', ms=12, elinewidth=1.5,
    label='Train'
)
ax.errorbar(
    y_test, test_mean,
    yerr=test_std,
    fmt='o', ecolor='k', mec='k', mfc='#D95F02', ms=12, elinewidth=1.5,
    label='Test'
)
ax.plot([y.min(), y.max()], [y.min(), y.max()], 'k--', lw=2)
ax.set_xlabel('Actual LCE', fontsize=24)
ax.set_ylabel('Predicted LCE', fontsize=24)
ax.tick_params(axis='both', which='major', labelsize=22, width=1.8, length=5)
for spine in ax.spines.values():
    spine.set_linewidth(1.8)
ax.legend(fontsize=20, loc='lower right', frameon=False)
ax.text(
    0.05, 0.95,
    f'Train $R^2$ = {mean_r2_train:.2f}\n'
    f'Test  $R^2$ = {mean_r2_test:.2f}',
    fontsize=20,
    verticalalignment='top',
    transform=ax.transAxes
)
plt.tight_layout()
plt.savefig(graph_path('RF_34.pdf'),dpi=300,transparent=True,bbox_inches='tight')
plt.show()

# Save scores for later comparison
Model_34_train = r2_train_list.copy()
Model_34_test = r2_test_list.copy()
print(f"Model_34 train R2 mean: {np.mean(Model_34_train):.4f}, test R2 mean: {np.mean(Model_34_test):.4f}")

# %% [markdown]
# ## PCA-Selected Descriptors

# %% [markdown]
# ### 15 Descriptors

# %%
X = df[['aO', 'aC', 'sC', 'ESPmin', 'aF', 'InOr', 'NumHAcceptors', 'PoSA', 'Polarizability', 'LUMO', 'sF',\
        'NonSA', 'sO', 'NeSA', 'DipoleMoment']]
y = df['LCE']
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)
seeds = [0, 1, 2, 3, 4]
mse_train_list = []
r2_train_list  = []
mse_test_list  = []
r2_test_list   = []
y_train_preds = []  # list of arrays, shape (n_train,)
y_test_preds  = []  # list of arrays, shape (n_test,)

for seed in seeds:
    rf = RandomForestRegressor(
        n_estimators=200,
        max_depth=5,
        random_state=seed,
        n_jobs=-1
    )
    rf.fit(X_train_scaled, y_train)
    y_tr_pred = rf.predict(X_train_scaled)
    y_te_pred = rf.predict(X_test_scaled)
    y_train_preds.append(y_tr_pred)
    y_test_preds .append(y_te_pred)
    mse_train_list.append(mean_squared_error(y_train, y_tr_pred))
    r2_train_list .append(r2_score(y_train, y_tr_pred))
    mse_test_list .append(mean_squared_error(y_test, y_te_pred))
    r2_test_list  .append(r2_score(y_test, y_te_pred))
y_train_preds = np.vstack(y_train_preds).T  # shape (n_train, 5)
y_test_preds  = np.vstack(y_test_preds ) .T  # shape (n_test,  5)
for i, seed in enumerate(seeds):
    print(f"Seed={seed} → Train MSE={mse_train_list[i]:.3f}, "
          f"Train R²={r2_train_list[i]:.3f} | "
          f"Test MSE={mse_test_list[i]:.3f}, "
          f"Test R²={r2_test_list[i]:.3f}")
train_mean = y_train_preds.mean(axis=1)
train_std  = y_train_preds.std(axis=1)
test_mean  = y_test_preds.mean(axis=1)
test_std   = y_test_preds.std(axis=1)
mean_r2_train = np.mean(r2_train_list)
mean_r2_test  = np.mean(r2_test_list)

# %%
fig, ax = plt.subplots(figsize=(8, 6))
ax.errorbar(
    y_train, train_mean,
    yerr=train_std,
    fmt='o', ecolor='k', mec='k', mfc='#436BA9', ms=12, elinewidth=1.5,
    label='Train'
)
ax.errorbar(
    y_test, test_mean,
    yerr=test_std,
    fmt='o', ecolor='k', mec='k', mfc='#D95F02', ms=12, elinewidth=1.5,
    label='Test'
)
ax.plot([y.min(), y.max()], [y.min(), y.max()], 'k--', lw=2)
ax.set_xlabel('Actual LCE', fontsize=24)
ax.set_ylabel('Predicted LCE', fontsize=24)
ax.tick_params(axis='both', which='major', labelsize=22, width=1.8, length=5)
for spine in ax.spines.values():
    spine.set_linewidth(1.8)
ax.legend(fontsize=20, loc='lower right', frameon=False)
ax.text(
    0.05, 0.95,
    f'Train $R^2$ = {mean_r2_train:.2f}\n'
    f'Test  $R^2$ = {mean_r2_test:.2f}',
    fontsize=20,
    verticalalignment='top',
    transform=ax.transAxes
)
plt.tight_layout()
plt.savefig(graph_path('RF_15.pdf'),dpi=300,transparent=True,bbox_inches='tight')
plt.show()

# Save scores for later comparison
Model_15_train = r2_train_list.copy()
Model_15_test = r2_test_list.copy()
print(f"Model_15 train R2 mean: {np.mean(Model_15_train):.4f}, test R2 mean: {np.mean(Model_15_test):.4f}")

# %% [markdown]
# ## Five Descriptors

# %%
X = df[['aO', 'PoSA', 'NonSA', 'sO', 'NeSA']]
y = df['LCE']
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)
seeds = [0, 1, 2, 3, 4]
mse_train_list = []
r2_train_list  = []
mse_test_list  = []
r2_test_list   = []
y_train_preds = []  # list of arrays, shape (n_train,)
y_test_preds  = []  # list of arrays, shape (n_test,)

for seed in seeds:
    rf = RandomForestRegressor(
        n_estimators=200,
        max_depth=5,
        random_state=seed,
        n_jobs=-1
    )
    rf.fit(X_train_scaled, y_train)
    y_tr_pred = rf.predict(X_train_scaled)
    y_te_pred = rf.predict(X_test_scaled)
    y_train_preds.append(y_tr_pred)
    y_test_preds .append(y_te_pred)
    mse_train_list.append(mean_squared_error(y_train, y_tr_pred))
    r2_train_list .append(r2_score(y_train, y_tr_pred))
    mse_test_list .append(mean_squared_error(y_test, y_te_pred))
    r2_test_list  .append(r2_score(y_test, y_te_pred))
y_train_preds = np.vstack(y_train_preds).T  # shape (n_train, 5)
y_test_preds  = np.vstack(y_test_preds ) .T  # shape (n_test,  5)
for i, seed in enumerate(seeds):
    print(f"Seed={seed} → Train MSE={mse_train_list[i]:.3f}, "
          f"Train R²={r2_train_list[i]:.3f} | "
          f"Test MSE={mse_test_list[i]:.3f}, "
          f"Test R²={r2_test_list[i]:.3f}")
train_mean = y_train_preds.mean(axis=1)
train_std  = y_train_preds.std(axis=1)
test_mean  = y_test_preds.mean(axis=1)
test_std   = y_test_preds.std(axis=1)
mean_r2_train = np.mean(r2_train_list)
mean_r2_test  = np.mean(r2_test_list)

# %%
fig, ax = plt.subplots(figsize=(8, 6))
ax.errorbar(
    y_train, train_mean,
    yerr=train_std,
    fmt='o', ecolor='k', mec='k', mfc='#436BA9', ms=12, elinewidth=1.5,
    label='Train'
)
ax.errorbar(
    y_test, test_mean,
    yerr=test_std,
    fmt='o', ecolor='k', mec='k', mfc='#D95F02', ms=12, elinewidth=1.5,
    label='Test'
)
ax.plot([y.min(), y.max()], [y.min(), y.max()], 'k--', lw=2)
ax.set_xlabel('Actual LCE', fontsize=24)
ax.set_ylabel('Predicted LCE', fontsize=24)
ax.tick_params(axis='both', which='major', labelsize=22, width=1.8, length=5)
for spine in ax.spines.values():
    spine.set_linewidth(1.8)
ax.legend(fontsize=20, loc='lower right', frameon=False)
ax.text(
    0.05, 0.95,
    f'Train $R^2$ = {mean_r2_train:.2f}\n'
    f'Test  $R^2$ = {mean_r2_test:.2f}',
    fontsize=20,
    verticalalignment='top',
    transform=ax.transAxes
)
plt.tight_layout()
plt.savefig(graph_path('RF_5.pdf'),dpi=300,transparent=True,bbox_inches='tight')
plt.show()

# %% [markdown]
# ## Three Descriptors

# %%
X = df[['aO','NonSA', 'sO']]
y = df['LCE']
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)
seeds = [0, 1, 2, 3, 4]
mse_train_list = []
r2_train_list  = []
mse_test_list  = []
r2_test_list   = []
y_train_preds = []  # list of arrays, shape (n_train,)
y_test_preds  = []  # list of arrays, shape (n_test,)

for seed in seeds:
    rf = RandomForestRegressor(
        n_estimators=200,
        max_depth=5,
        random_state=seed,
        n_jobs=-1
    )
    rf.fit(X_train_scaled, y_train)
    y_tr_pred = rf.predict(X_train_scaled)
    y_te_pred = rf.predict(X_test_scaled)
    y_train_preds.append(y_tr_pred)
    y_test_preds .append(y_te_pred)
    mse_train_list.append(mean_squared_error(y_train, y_tr_pred))
    r2_train_list .append(r2_score(y_train, y_tr_pred))
    mse_test_list .append(mean_squared_error(y_test, y_te_pred))
    r2_test_list  .append(r2_score(y_test, y_te_pred))
y_train_preds = np.vstack(y_train_preds).T  # shape (n_train, 5)
y_test_preds  = np.vstack(y_test_preds ) .T  # shape (n_test,  5)
for i, seed in enumerate(seeds):
    print(f"Seed={seed} → Train MSE={mse_train_list[i]:.3f}, "
          f"Train R²={r2_train_list[i]:.3f} | "
          f"Test MSE={mse_test_list[i]:.3f}, "
          f"Test R²={r2_test_list[i]:.3f}")
train_mean = y_train_preds.mean(axis=1)
train_std  = y_train_preds.std(axis=1)
test_mean  = y_test_preds.mean(axis=1)
test_std   = y_test_preds.std(axis=1)
mean_r2_train = np.mean(r2_train_list)
mean_r2_test  = np.mean(r2_test_list)

# %%
Model_3_train=[0.876,0.878,0.877,0.877,0.881]
Model_3_test=[0.891,0.883,0.887,0.878,0.893]

# %%
fig, ax = plt.subplots(figsize=(8, 6))
ax.errorbar(
    y_train, train_mean,
    yerr=train_std,
    fmt='o', ecolor='k', mec='k', mfc='#436BA9', ms=12, elinewidth=1.5,
    label='Train'
)
ax.errorbar(
    y_test, test_mean,
    yerr=test_std,
    fmt='o', ecolor='k', mec='k', mfc='#D95F02', ms=12, elinewidth=1.5,
    label='Test'
)
ax.plot([y.min(), y.max()], [y.min(), y.max()], 'k--', lw=2)
ax.set_xlabel('Actual LCE', fontsize=24)
ax.set_ylabel('Predicted LCE', fontsize=24)
ax.tick_params(axis='both', which='major', labelsize=22, width=1.8, length=5)
for spine in ax.spines.values():
    spine.set_linewidth(1.8)
ax.legend(fontsize=20, loc='lower right', frameon=False)
ax.text(
    0.05, 0.95,
    f'Train $R^2$ = {mean_r2_train:.2f}\n'
    f'Test  $R^2$ = {mean_r2_test:.2f}',
    fontsize=20,
    verticalalignment='top',
    transform=ax.transAxes
)
plt.tight_layout()
plt.savefig(graph_path('RF_3__with_errorbars.pdf'),dpi=300,transparent=True,bbox_inches='tight')
plt.show()

# %% [markdown]
# # Three-Descriptor Random Forest and Model Persistence

# %%
X = df[['aO', 'NonSA', 'sO']]
y = df['LCE']
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)
rf = RandomForestRegressor(
    n_estimators=200,
    max_depth=5,
    random_state=42,
    n_jobs=-1
)
rf.fit(X_train_scaled, y_train)
y_tr_pred = rf.predict(X_train_scaled)
y_te_pred = rf.predict(X_test_scaled)

mse_train = mean_squared_error(y_train, y_tr_pred)
r2_train  = r2_score(y_train, y_tr_pred)
mse_test  = mean_squared_error(y_test, y_te_pred)
r2_test   = r2_score(y_test, y_te_pred)

print(f"Train MSE={mse_train:.3f}, Train R²={r2_train:.3f} | Test MSE={mse_test:.3f}, Test R²={r2_test:.3f}")
joblib.dump(rf, './rf_model_LCE.pkl')
joblib.dump(scaler, './scaler_LCE.pkl')
print("Model and Scaler saved successfully.")

# %% [markdown]
# # Single-Descriptor Model

# %%
X = df[['sO']]
y = df['LCE']
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)
seeds = [0, 1, 2, 3, 4]
mse_train_list = []
r2_train_list  = []
mse_test_list  = []
r2_test_list   = []
y_train_preds = []  # list of arrays, shape (n_train,)
y_test_preds  = []  # list of arrays, shape (n_test,)

for seed in seeds:
    rf = RandomForestRegressor(
        n_estimators=200,
        max_depth=5,
        random_state=seed,
        n_jobs=-1
    )
    rf.fit(X_train_scaled, y_train)
    y_tr_pred = rf.predict(X_train_scaled)
    y_te_pred = rf.predict(X_test_scaled)
    y_train_preds.append(y_tr_pred)
    y_test_preds .append(y_te_pred)
    mse_train_list.append(mean_squared_error(y_train, y_tr_pred))
    r2_train_list .append(r2_score(y_train, y_tr_pred))
    mse_test_list .append(mean_squared_error(y_test, y_te_pred))
    r2_test_list  .append(r2_score(y_test, y_te_pred))
y_train_preds = np.vstack(y_train_preds).T  # shape (n_train, 5)
y_test_preds  = np.vstack(y_test_preds ) .T  # shape (n_test,  5)
for i, seed in enumerate(seeds):
    print(f"Seed={seed} → Train MSE={mse_train_list[i]:.3f}, "
          f"Train R²={r2_train_list[i]:.3f} | "
          f"Test MSE={mse_test_list[i]:.3f}, "
          f"Test R²={r2_test_list[i]:.3f}")
train_mean = y_train_preds.mean(axis=1)
train_std  = y_train_preds.std(axis=1)
test_mean  = y_test_preds.mean(axis=1)
test_std   = y_test_preds.std(axis=1)
mean_r2_train = np.mean(r2_train_list)
mean_r2_test  = np.mean(r2_test_list)

# %%
fig, ax = plt.subplots(figsize=(8, 6))
ax.errorbar(
    y_train, train_mean,
    yerr=train_std,
    fmt='o', ecolor='k', mec='k', mfc='#436BA9', ms=12, elinewidth=1.5,
    label='Train'
)
ax.errorbar(
    y_test, test_mean,
    yerr=test_std,
    fmt='o', ecolor='k', mec='k', mfc='#D95F02', ms=12, elinewidth=1.5,
    label='Test'
)
ax.plot([y.min(), y.max()], [y.min(), y.max()], 'k--', lw=2)
ax.set_xlabel('Actual LCE', fontsize=24)
ax.set_ylabel('Predicted LCE', fontsize=24)
ax.tick_params(axis='both', which='major', labelsize=22, width=1.8, length=5)
for spine in ax.spines.values():
    spine.set_linewidth(1.8)
ax.legend(fontsize=20, loc='lower right', frameon=False)
ax.text(
    0.05, 0.95,
    f'Train $R^2$ = {mean_r2_train:.2f}\n'
    f'Test  $R^2$ = {mean_r2_test:.2f}',
    fontsize=20,
    verticalalignment='top',
    transform=ax.transAxes
)
plt.tight_layout()
plt.savefig(graph_path('RF_sO.pdf'),dpi=300,transparent=True,bbox_inches='tight')
plt.show()

# %% [markdown]
# # Recursive Feature Elimination

# %%
X = df[['aO', 'aC', 'sC', 'ESPmin', 'aF', 'InOr', 'NumHAcceptors', 'PoSA', 'Polarizability', 'LUMO', 'sF',\
        'NonSA', 'sO', 'NeSA', 'DipoleMoment']]
y = df['LCE']
rf = RandomForestRegressor(random_state=42)
cv = KFold(n_splits=5, shuffle=True, random_state=42)

rfecv = RFECV(
    estimator=rf,
    step=1,
    cv=cv,
    scoring='neg_mean_squared_error',
    n_jobs=-1
)

rfecv.fit(X, y)

print(f"Optimal number of features: {rfecv.n_features_}")
selected_feats = X.columns[rfecv.support_].tolist()
print("Selected features: ", selected_feats)
mse_scores = -rfecv.cv_results_['mean_test_score']
with open('./data/rfecv_mse_scores.pkl', 'wb') as f:
    pickle.dump(mse_scores, f)
print("RFECV MSE scores saved to data/rfecv_mse_scores.pkl")
rfecv_mse_df = pd.DataFrame({
    'Feature_Number': range(1, len(mse_scores) + 1),
    'Average_MSE': mse_scores
})
rfecv_mse_df.to_excel('./data/rfecv_mse_data.xlsx', index=False)
print("RFECV MSE data saved to data/rfecv_mse_data.xlsx")

# %%
with open('./data/rfecv_mse_scores.pkl', 'rb') as f:
    mse_scores = pickle.load(f)
fig, ax = plt.subplots(figsize=(13.5,6))
ax.plot(
    range(1, len(mse_scores) + 1),
    mse_scores,
    linestyle='-',linewidth=2.6,
    marker='o',markersize=12,color='#436BA9',alpha=1,markeredgewidth=1.5,markeredgecolor='k')
plt.xlabel("Features number",fontsize=24)
plt.ylabel("Average MSE",fontsize=24)
for spine in ax.spines.values():
    spine.set_linewidth(1.8)
ax.tick_params(axis='both', width=1.8, length=5, labelsize=22)
ax.text(0.8,0.95,'5 fold CV', transform=ax.transAxes,fontsize=20,va='top')
plt.savefig(graph_path('REF.pdf'),transparent=True,dpi=300,bbox_inches='tight')
plt.show()

# %%
feat_names = ['34', '15', '3']

train_scores = [
    Model_34_train,  # 34
    Model_15_train,
    Model_3_train
]

test_scores = [
    Model_34_test,
    Model_15_test,
    Model_3_test
]
train_means = [np.mean(s) for s in train_scores]
train_stds  = [np.std(s)  for s in train_scores]
test_means  = [np.mean(s) for s in test_scores]
test_stds   = [np.std(s)  for s in test_scores]
x = np.arange(len(feat_names))
width = 0.35

fig, ax = plt.subplots(figsize=(8, 6))
ax.bar(x - width/2, train_means, width, yerr=train_stds,
       label='Train R²', capsize=8, color='#436BA9', alpha=1)
ax.bar(x + width/2, test_means,  width, yerr=test_stds,
       label='Test R²',  capsize=8, color='#D95F02', alpha=1)
ax.set_xticks(x)
ax.set_xticklabels(feat_names, fontsize=22)
ax.set_xlabel('Features number',fontsize=24)
ax.set_ylabel(r'R$^\mathregular{2}$ Score', fontsize=24)
ax.set_ylim(0.6, 1.0)
ax.yaxis.set_major_formatter(mtick.FormatStrFormatter('%.2f'))
ax.legend(fontsize=18, frameon=False)
for spine in ax.spines.values():
    spine.set_linewidth(1.5)
ax.tick_params(axis='y', width=1.5, length=5, labelsize=22)
ax.tick_params(axis='x', width=1.5, length=5, labelsize=22)

plt.tight_layout()
plt.savefig(graph_path('34_15_3.pdf'),transparent=True, dpi=300, bbox_inches='tight')

# %% [markdown]
# # Feature Interpretation

# %% [markdown]
# ### Five-Descriptor SHAP Analysis

# %%
X = df[['aO', 'PoSA', 'NonSA', 'sO','NeSA']]
y = df['LCE']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scaler = StandardScaler() 
X_train_scaled = scaler.fit_transform(X_train) 
X_test_scaled = scaler.transform(X_test)

# %%
rf_model = RandomForestRegressor(n_estimators=200, random_state=42,max_depth=5)
rf_model.fit(X_train, y_train)

# %%
import shap
selected_features=['aO', 'PoSA', 'NonSA', 'sO','NeSA']
explainer = shap.TreeExplainer(rf_model)
X_sel=df[selected_features]
shap_values = explainer.shap_values(X_sel)
plt.figure(figsize=(12,6))
shap.summary_plot(shap_values, X_sel, show=False)
plt.tight_layout()
plt.savefig(graph_path('SHAP_5.pdf'),transparent=True,dpi=300,bbox_inches='tight')

# %% [markdown]
# ### Three-Descriptor SHAP Analysis

# %%
X = df[['aO', 'NonSA', 'sO']]
y = df['LCE']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scaler = StandardScaler() 
X_train_scaled = scaler.fit_transform(X_train) 
X_test_scaled = scaler.transform(X_test)
rf_model = RandomForestRegressor(n_estimators=200, random_state=42,max_depth=5)
rf_model.fit(X_train, y_train)

# %%
# ==========================================
# ==========================================
colors = [
    (0.00, "#495E77"),
    (0.45, "#6D84A1"),
    (0.50, "#8C8090"),
    (0.55, "#B57B7B"),
    (1.00, "#824D4D"),
]
my_cmap = mcolors.LinearSegmentedColormap.from_list("ShapVisible", colors, N=256)

# ==========================================
# ==========================================
selected_features = ['aO', 'NonSA', 'sO']
explainer = shap.TreeExplainer(rf_model)
X_sel = df[selected_features]
shap_values = explainer.shap_values(X_sel)
if isinstance(shap_values, list):
    shap_values = shap_values[1]
shap.summary_plot(shap_values, X_sel, show=False, cmap=my_cmap, alpha=0.9)

# ==========================================
# ==========================================
fig = plt.gcf()
fig.set_size_inches(3, 2)
try:
    cbar_ax = fig.axes[-1]
    cbar_ax.tick_params(labelsize=8)
    cbar_ax.set_ylabel("Feature Value", fontsize=9)
except:
    pass

plt.tight_layout()
plt.savefig(graph_path('SHAP_3.pdf'), transparent=True, dpi=300, bbox_inches='tight')
plt.show()

# %% [markdown]
# # Three-Descriptor Random Forest and PDP Visualization

# %%
X = df[['aO', 'NonSA', 'sO']]
y = df['LCE']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scaler = StandardScaler() 
X_train_scaled = scaler.fit_transform(X_train) 
X_test_scaled = scaler.transform(X_test)

# %%
rf_model = RandomForestRegressor(n_estimators=300, max_depth=5, random_state=42)
rf_model.fit(X_train, y_train)
feature_importances = rf_model.feature_importances_
features = X.columns
sorted_indices = np.argsort(feature_importances)[::-1]
sorted_features = features[sorted_indices]

print("Features sorted by importance (High to Low):")
for feat_name, importance in zip(sorted_features, feature_importances[sorted_indices]):
    print(f'{feat_name}: {importance:.4f}')

# %%

def compute_2d_pdp_manual(model, X_df, feat_x, feat_y, grid_resolution=50,
                          pct=(0,100), sample_size=500, verbose=True):
    """
    Compute a manual 2D partial dependence grid by replacing two descriptor columns and averaging predictions.
    Return x values, y values, and Z with shape (ny, nx).
    """
    assert feat_x in X_df.columns and feat_y in X_df.columns
    lo_x, hi_x = np.percentile(X_df[feat_x], pct[0]), np.percentile(X_df[feat_x], pct[1])
    lo_y, hi_y = np.percentile(X_df[feat_y], pct[0]), np.percentile(X_df[feat_y], pct[1])
    x_vals = np.linspace(lo_x, hi_x, grid_resolution)
    y_vals = np.linspace(lo_y, hi_y, grid_resolution)
    if sample_size is None or sample_size >= len(X_df):
        X_sample = X_df.copy().reset_index(drop=True)
    else:
        X_sample = X_df.sample(n=min(sample_size, len(X_df)), random_state=42).reset_index(drop=True)

    X_tmp_vals = X_sample.values  # ndarray
    cols = list(X_df.columns)
    ix = cols.index(feat_x)
    iy = cols.index(feat_y)

    ny = len(y_vals)
    nx = len(x_vals)
    Z = np.zeros((ny, nx), dtype=float)

    total = ny * nx
    t0 = time()
    cnt = 0
    for j, yv in enumerate(y_vals):        # rows (y)
        for i, xv in enumerate(x_vals):    # cols (x)
            Xc = X_tmp_vals.copy()
            Xc[:, ix] = xv
            Xc[:, iy] = yv
            preds = model.predict(Xc)
            Z[j, i] = preds.mean()
            cnt += 1
        if verbose and ( (j+1) % max(1, ny//5) == 0):
            print(f'  progress: {cnt}/{total} grid points, elapsed {time()-t0:.1f}s')
    if verbose:
        print('Finished grid computation. Z.shape =', Z.shape)
    return x_vals, y_vals, Z
feat_x = 'sO'
feat_y = 'NonSA'
X_for_pdp = X_train.copy()

xv, yv, Z = compute_2d_pdp_manual(
    rf_model,
    X_for_pdp,
    feat_x,
    feat_y,
    grid_resolution=50,
    pct=(0, 100),
    sample_size=500,
    verbose=True
)
Xg, Yg = np.meshgrid(xv, yv)

# %%
def plot_2d_pdp_heatmap(Xg, Yg, Z, X_points=None, xlabel='sO', ylabel='NonSA',
                              title='2D PDP heatmap', cmap='plasma', clevels=None,
                              xlim=None, ylim=None, symmetric_cbar=False, savepath=None):
    """
    Plot a corrected 2D PDP heatmap with colorbar formatting and shape checks.
    Xg, Yg: meshgrid arrays (expected shape (ny, nx))
    Z: 2D array (expected shape (ny, nx))
    X_points: optional DataFrame for scatter overlay (columns matching xlabel,ylabel)
    """
    if not hasattr(Z, 'ndim') or Z.ndim != 2:
        raise ValueError(f'Z must be 2D array, got ndim={getattr(Z,"ndim",None)}')
    ny, nx = Z.shape
    if Xg.shape != (ny, nx) or Yg.shape != (ny, nx):
        if Xg.shape == (nx, ny) and Yg.shape == (nx, ny) and Z.T.shape == (ny, nx):
            print('Detected transpose mismatch between grid and Z: transposing Z automatically.')
            Z = Z.T
        else:
            raise ValueError(f'Grid shapes incompatible: Xg {Xg.shape}, Yg {Yg.shape}, Z {Z.shape}')

    fig, ax = plt.subplots(figsize=(8,6))
    cf = ax.contourf(Xg, Yg, Z, levels=clevels or 50, cmap=cmap)
    #cs = ax.contour(Xg, Yg, Z, levels=8, colors='k', linewidths=0.6, alpha=0.6)
    cbar = fig.colorbar(cf, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('Partial dependence', fontsize=22)
    cbar.ax.yaxis.set_major_locator(MaxNLocator(nbins=5, integer=False))
    cbar.ax.tick_params(labelsize=16)
    ax.set_xlabel(xlabel, fontsize=24)
    ax.set_ylabel(ylabel, fontsize=24)
    if xlim is not None:
        ax.set_xlim(*xlim)
    if ylim is not None:
        ax.set_ylim(*ylim)

    ax.tick_params(axis='both', width=1.8, length=5, labelsize=22)
    for spine in ax.spines.values():
        spine.set_linewidth(1.8)

    if savepath:
        fig.savefig(savepath, dpi=300, bbox_inches='tight',transparent=True)
        print(f'Figure saved to: {savepath}')

    plt.show()
    return fig, ax
plot_2d_pdp_heatmap(
    Xg, Yg, Z,
    xlabel='sO', ylabel='NonSA',
    cmap='bwr',
    clevels=100,
    xlim=(13,40),
    ylim=(18,145),
    symmetric_cbar=False,
    savepath=graph_path('2D_PDP.pdf')
)

# %% [markdown]
# ## One-Dimensional PDP

# %%
import numpy as np
import matplotlib.pyplot as plt
from sklearn.inspection import PartialDependenceDisplay

features = ['aO', 'sO', 'NonSA']
colors   = ['#6689C2', '#C2695A', '#ECA634']
x_refs   = [1.0, 22.0, 90.0]

for feat, color, x_ref in zip(features, colors, x_refs):
    fig, ax = plt.subplots(figsize=(8, 6), dpi=300)
    disp = PartialDependenceDisplay.from_estimator(
        rf_model,
        X,
        [feat],
        ax=ax,
        kind='average',
        grid_resolution=500,
        line_kw={'linewidth': 2.5, 'color': color}
    )
    pdp_ax = disp.axes_[0, 0]
    pdp_ax.set_xlabel(feat, fontsize=24)
    pdp_ax.set_ylabel('Partial dependence', fontsize=24)
    pdp_ax.tick_params(axis='both',
                       which='major',
                       labelsize=22,
                       width=1.8,
                       length=5)
    for spine in pdp_ax.spines.values():
        spine.set_linewidth(1.5)
    plt.tight_layout()
    plt.savefig(
        graph_path(f'PDP_{feat}.pdf'),
        transparent=True,
        dpi=300,
        bbox_inches='tight'
    )
    plt.show()


# %% [markdown]
# # Benchmark Experiments

# %% [markdown]
# ## Full-Descriptor AdaBoost

# %%
try:
    X = df[feature]
    y = df['LCE']
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled  = scaler.transform(X_test)
    seeds = [0, 1, 2, 3, 4]
    mse_train_list = []
    r2_train_list  = []
    mse_test_list  = []
    r2_test_list   = []
    y_train_preds = []  # list of arrays, shape (n_train,)
    y_test_preds  = []  # list of arrays, shape (n_test,)

    base_estimator = DecisionTreeRegressor(max_depth=5, random_state=42)

    for seed in seeds:
        ab_model = AdaBoostRegressor(
            estimator=base_estimator,
            n_estimators=200,
            learning_rate=0.1,
            random_state=seed
        )
        model=ab_model
        
        model.fit(X_train_scaled, y_train)
        y_tr_pred = model.predict(X_train_scaled)
        y_te_pred = model.predict(X_test_scaled)
        y_train_preds.append(y_tr_pred)
        y_test_preds .append(y_te_pred)
        mse_train_list.append(mean_squared_error(y_train, y_tr_pred))
        r2_train_list .append(r2_score(y_train, y_tr_pred))
        mse_test_list .append(mean_squared_error(y_test, y_te_pred))
        r2_test_list  .append(r2_score(y_test, y_te_pred))
    y_train_preds = np.vstack(y_train_preds).T  # shape (n_train, 5)
    y_test_preds  = np.vstack(y_test_preds ) .T  # shape (n_test,  5)
    for i, seed in enumerate(seeds):
        print(f"Seed={seed} → Train MSE={mse_train_list[i]:.3f}, "
              f"Train R²={r2_train_list[i]:.3f} | "
              f"Test MSE={mse_test_list[i]:.3f}, "
              f"Test R²={r2_test_list[i]:.3f}")
    train_mean = y_train_preds.mean(axis=1)
    train_std  = y_train_preds.std(axis=1)
    test_mean  = y_test_preds.mean(axis=1)
    test_std   = y_test_preds.std(axis=1)
    mean_r2_train = np.mean(r2_train_list)
    mean_r2_test  = np.mean(r2_test_list)
except Exception as e:
    print(f"[SKIPPED] AdaBoost section failed: {e}")

# %%
Ada_train= [0.975, 0.979, 0.977, 0.977, 0.975]
Ada_test = [0.823, 0.827, 0.829, 0.828, 0.833]

# %%
try:
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.errorbar(
        y_train, train_mean,
        yerr=train_std,
        fmt='o', ecolor='k', mec='k', mfc='#436BA9', ms=12, elinewidth=1.5,
        label='Train'
    )
    ax.errorbar(
        y_test, test_mean,
        yerr=test_std,
        fmt='o', ecolor='k', mec='k', mfc='#D95F02', ms=12, elinewidth=1.5,
        label='Test'
    )
    ax.plot([y.min(), y.max()], [y.min(), y.max()], 'k--', lw=2)
    ax.set_xlabel('Actual LCE', fontsize=24)
    ax.set_ylabel('Predicted LCE', fontsize=24)
    ax.tick_params(axis='both', which='major', labelsize=22, width=1.8, length=5)
    for spine in ax.spines.values():
        spine.set_linewidth(1.8)
    ax.legend(fontsize=20, loc='lower right', frameon=False)
    ax.text(
        0.05, 0.95,
        f'AdaBoost\n'
        f'Train $R^2$ = {mean_r2_train:.2f}\n'
        f'Test  $R^2$ = {mean_r2_test:.2f}',
        fontsize=16,
        verticalalignment='top',
        transform=ax.transAxes
    )
    plt.tight_layout()
    plt.savefig(graph_path('Adaboost_with_errorbars.pdf'),dpi=300,transparent=True,bbox_inches='tight')
    plt.show()
except Exception as e:
    print(f"[SKIPPED] AdaBoost plot failed: {e}")

# %% [markdown]
# ## Full-Descriptor XGBoost

# %%
X = df[feature]
y = df['LCE']
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)
seeds = [0, 1, 2, 3, 4]
mse_train_list = []
r2_train_list  = []
mse_test_list  = []
r2_test_list   = []
y_train_preds = []  # list of arrays, shape (n_train,)
y_test_preds  = []  # list of arrays, shape (n_test,)

for seed in seeds:
    xgb_model = XGBRegressor(
        n_estimators=200,
        learning_rate=0.1,
        max_depth=5,
        random_state=seed,
        subsample=0.8,
        colsample_bytree=0.8,
        objective='reg:squarederror',
        use_label_encoder=False,
        eval_metric='rmse'       
    )
    model=xgb_model
    model.fit(X_train_scaled, y_train)
    y_tr_pred = model.predict(X_train_scaled)
    y_te_pred = model.predict(X_test_scaled)
    y_train_preds.append(y_tr_pred)
    y_test_preds .append(y_te_pred)
    mse_train_list.append(mean_squared_error(y_train, y_tr_pred))
    r2_train_list .append(r2_score(y_train, y_tr_pred))
    mse_test_list .append(mean_squared_error(y_test, y_te_pred))
    r2_test_list  .append(r2_score(y_test, y_te_pred))
y_train_preds = np.vstack(y_train_preds).T  # shape (n_train, 5)
y_test_preds  = np.vstack(y_test_preds ) .T  # shape (n_test,  5)
for i, seed in enumerate(seeds):
    print(f"Seed={seed} → Train MSE={mse_train_list[i]:.3f}, "
          f"Train R²={r2_train_list[i]:.3f} | "
          f"Test MSE={mse_test_list[i]:.3f}, "
          f"Test R²={r2_test_list[i]:.3f}")
train_mean = y_train_preds.mean(axis=1)
train_std  = y_train_preds.std(axis=1)
test_mean  = y_test_preds.mean(axis=1)
test_std   = y_test_preds.std(axis=1)
mean_r2_train = np.mean(r2_train_list)
mean_r2_test  = np.mean(r2_test_list)

# %%
fig, ax = plt.subplots(figsize=(8, 6))
ax.errorbar(
    y_train, train_mean,
    yerr=train_std,
    fmt='o', ecolor='k', mec='k', mfc='#436BA9', ms=12, elinewidth=1.5,
    label='Train'
)
ax.errorbar(
    y_test, test_mean,
    yerr=test_std,
    fmt='o', ecolor='k', mec='k', mfc='#D95F02', ms=12, elinewidth=1.5,
    label='Test'
)
ax.plot([y.min(), y.max()], [y.min(), y.max()], 'k--', lw=2)
ax.set_xlabel('Actual LCE', fontsize=24)
ax.set_ylabel('Predicted LCE', fontsize=24)
ax.tick_params(axis='both', which='major', labelsize=22, width=1.8, length=5)
for spine in ax.spines.values():
    spine.set_linewidth(1.8)
ax.legend(fontsize=20, loc='lower right', frameon=False)
ax.text(
    0.05, 0.95,
    f'XGBBoost\n'
    f'Train $R^2$ = {mean_r2_train:.2f}\n'
    f'Test  $R^2$ = {mean_r2_test:.2f}',
    fontsize=16,
    verticalalignment='top',
    transform=ax.transAxes
)
plt.tight_layout()
plt.savefig(graph_path('XGB_with_errorbars.pdf'),dpi=300,transparent=True,bbox_inches='tight')
plt.show()

# %%
XGb_train= ['0.9918', '0.9918', '0.9918', '0.9918', '0.9917']
XGb_test= ['0.7803', '0.8074', '0.8030', '0.814', '0.798']

# %% [markdown]
# ## Full-Descriptor LightGBM

# %%
X = df[feature]
y = df['LCE']
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)
seeds = [0, 1, 2, 3, 4]
mse_train_list = []
r2_train_list  = []
mse_test_list  = []
r2_test_list   = []
y_train_preds = []  # list of arrays, shape (n_train,)
y_test_preds  = []  # list of arrays, shape (n_test,)

for seed in seeds:
    lgb_model = LGBMRegressor(
        n_estimators=200,
        learning_rate=0.1,
        max_depth=5,
        random_state=seed,
        metric='l2'
    )
    model=lgb_model
    model.fit(X_train_scaled, y_train)
    y_tr_pred = model.predict(X_train_scaled)
    y_te_pred = model.predict(X_test_scaled)
    y_train_preds.append(y_tr_pred)
    y_test_preds .append(y_te_pred)
    mse_train_list.append(mean_squared_error(y_train, y_tr_pred))
    r2_train_list .append(r2_score(y_train, y_tr_pred))
    mse_test_list .append(mean_squared_error(y_test, y_te_pred))
    r2_test_list  .append(r2_score(y_test, y_te_pred))
y_train_preds = np.vstack(y_train_preds).T  # shape (n_train, 5)
y_test_preds  = np.vstack(y_test_preds ) .T  # shape (n_test,  5)
for i, seed in enumerate(seeds):
    print(f"Seed={seed} → Train MSE={mse_train_list[i]:.3f}, "
          f"Train R²={r2_train_list[i]:.3f} | "
          f"Test MSE={mse_test_list[i]:.3f}, "
          f"Test R²={r2_test_list[i]:.3f}")
train_mean = y_train_preds.mean(axis=1)
train_std  = y_train_preds.std(axis=1)
test_mean  = y_test_preds.mean(axis=1)
test_std   = y_test_preds.std(axis=1)
mean_r2_train = np.mean(r2_train_list)
mean_r2_test  = np.mean(r2_test_list)

# %%
fig, ax = plt.subplots(figsize=(8, 6))
ax.errorbar(
    y_train, train_mean,
    yerr=train_std,
    fmt='o', ecolor='k', mec='k', mfc='#436BA9', ms=12, elinewidth=1.5,
    label='Train'
)
ax.errorbar(
    y_test, test_mean,
    yerr=test_std,
    fmt='o', ecolor='k', mec='k', mfc='#D95F02', ms=12, elinewidth=1.5,
    label='Test'
)
ax.plot([y.min(), y.max()], [y.min(), y.max()], 'k--', lw=2)
ax.set_xlabel('Actual LCE', fontsize=24)
ax.set_ylabel('Predicted LCE', fontsize=24)
ax.tick_params(axis='both', which='major', labelsize=22, width=1.8, length=5)
for spine in ax.spines.values():
    spine.set_linewidth(1.8)
ax.legend(fontsize=20, loc='lower right', frameon=False)
ax.text(
    0.05, 0.95,
    f'LightGBM\n'
    f'Train $R^2$ = {mean_r2_train:.2f}\n'
    f'Test  $R^2$ = {mean_r2_test:.2f}',
    fontsize=16,
    verticalalignment='top',
    transform=ax.transAxes
)
plt.tight_layout()
plt.savefig(graph_path('LightGBM_with_errorbars.pdf'),dpi=300,transparent=True,bbox_inches='tight')
plt.show()

# %%
LGBM_train= ['0.86', '0.86', '0.86', '0.86', '0.86']
LGBM_test=['0.84', '0.84', '0.84', '0.84', '0.84']

# %% [markdown]
# ## Summary

# %%
RF_train= Model_3_train
RF_test=  Model_3_test

Ada_train=Ada_train
Ada_test=Ada_test

XGB_train=XGb_train
XGB_test= XGb_test

Lgb_train=LGBM_train
Lgb_test=LGBM_test

# %%
# Models and stats
models = ['This work', 'AdaBoost', 'XGBoost', 'LightGBM']
train_scores = [RF_train, Ada_train, XGB_train, Lgb_train]
test_scores  = [RF_test, Ada_test, XGB_test, Lgb_test]
train_scores = [np.array(scores, dtype=float) for scores in train_scores]
test_scores  = [np.array(scores, dtype=float) for scores in test_scores]
# Compute means and standard deviations
train_means = [np.mean(scores) for scores in train_scores]
train_stds  = [np.std(scores)  for scores in train_scores]
test_means  = [np.mean(scores) for scores in test_scores]
test_stds   = [np.std(scores)  for scores in test_scores]

# Create bar chart with error bars
x = np.arange(len(models))
width = 0.35

fig, ax = plt.subplots(figsize=(8, 6))
ax.bar(x - width/2, train_means, width, yerr=train_stds, label='Train R²', capsize=8,color='#436BA9',alpha=1)
ax.bar(x + width/2, test_means,  width, yerr=test_stds,  label='Test R²',  capsize=8,color='#D95F02',alpha=1)

# Labels and title
ax.set_xticks(x)
ax.set_xticklabels(models, fontsize=22)
ax.set_ylabel('R² Score', fontsize=24)
ax.legend(fontsize=20, frameon=False)
ax.set_ylim(0.6,1.1)
ax.yaxis.set_major_formatter(mtick.FormatStrFormatter('%.2f'))
# Beautify axes
for spine in ax.spines.values():
    spine.set_linewidth(1.5)
ax.tick_params(axis='y', width=1.5, length=5, labelsize=22)
ax.tick_params(axis='x', width=1.5, length=5, labelsize=22)
plt.tight_layout()
plt.savefig(graph_path('Comparation.pdf'),transparent=True,dpi=300,bbox_inches='tight')

# %% [markdown]
# 

# %%



