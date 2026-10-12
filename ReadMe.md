# Lesotho points of interest: data pipeline and GNN model

This folder holds the OpenStreetMap points of interest (POIs) for Lesotho and the files derived from them for the
GNN (graph neural network) business-location study, the GNN itself, and a command-line tool that applies it. Every
derived file comes from the one OSM export; no outside data are added at any stage. How the model is evaluated is in
[Stage 5.3](#53-metrics). Terms, codes, categories and every column are explained in the [Dictionary](#dictionary)
at the end.

```
points_of_interest.geojson     12,536 OSM features (HOT export, snapshot 2026-09-08)
        │  Stage 1: flatten to a table (geometry kept as WKT + a centroid)
        ▼
points_of_interest.csv         12,536 rows x 31 columns
        │  Stage 2: poi_eda.ipynb (clean, label, merge duplicates, node features, CV folds)
        ▼
processed_poi.csv              6,083 nodes x 43 columns
        │  Stage 3: poi_feature_engineering.ipynb (model-ready features)
        ▼
poi_engineered.csv             6,083 nodes x 92 columns (66 model features)
        │  Stage 5: gnn_location_model.ipynb (dual-mode GraphSAGE, evaluation, ablations, RO2/RO3)
        ▼                (Stage 4, the first design, was retired; see git history)
model_outputs_v2/              25 trained models, predictions in both modes, market gaps, metrics
        │  Stage 6: predict_poi.py (score a candidate site, suggest sites, market gaps)
```

## Files

| File | What it is |
|---|---|
| `points_of_interest.geojson` | Original export, from `../hotosm_lso_points_of_interest_osm_geojson.zip` |
| `points_of_interest.csv` | Stage 1 output: the GeoJSON as a flat table |
| `poi_eda.ipynb` | Stage 2 notebook: exploratory analysis and cleaning |
| `processed_poi.csv` | Stage 2 output: one cleaned row per graph node |
| `poi_feature_engineering.ipynb` | Stage 3 notebook: feature engineering |
| `poi_engineered.csv` | Stage 3 output: the table the GNN is trained on |
| `gnn_location_model.ipynb` | Model iteration 2: one GraphSAGE trained in two modes (known POI / candidate site), with architecture comparison, tuning, ablations, RO2 (what shapes an area) and RO3 (market gaps) |
| `model_outputs_v2/` | Iteration 2 outputs: `gnn_dual_models.pt` (25 models), out-of-fold and unlabelled predictions in both modes, `market_gaps.csv`, `metrics.json`, training log |
| `predict_poi.py` | Command-line tool on the iteration 2 model: `site`, `suggest`, `gaps`, `new`, `existing`, `batch`, and `--check` against the notebook (run `python predict_poi.py -h`) |

---

## Stage 1: `points_of_interest.geojson` → `points_of_interest.csv`

**Source.** The GeoJSON is the Humanitarian OpenStreetMap Team (HOT) "Points of Interest" export for Lesotho,
produced with `oex` 0.4.13 on 2026-09-09 from the OSM snapshot of 2026-09-08. It contains every OSM feature tagged
with `amenity`, `shop`, `tourism` or `man_made` inside the geoBoundaries outline of Lesotho: 12,536 features
(5,372 points, 6,880 polygons, 284 lines) in WGS84 longitude/latitude. The export tool attached the district
(`adm1_*`) and sub-district (`adm2_*`) codes and names.

**What the conversion did.** Each GeoJSON feature became one CSV row, in the same order. The 27 attribute columns were
copied unchanged (OSM `null` becomes an empty cell), and four columns were added so the geometry can be used without
a GIS library:

| Added column | Content |
|---|---|
| `geometry_type` | `Point`, `Polygon` or `LineString` |
| `longitude`, `latitude` | Centroid of the geometry in degrees, rounded to 7 decimals. For a point, the point itself |
| `geometry_wkt` | The full geometry as WKT text, with the original coordinates |

The file is UTF-8 with a byte-order mark (read it with `encoding="utf-8-sig"`).

**Checked against the GeoJSON.** Same 12,536 ids in the same order; every attribute value identical; every WKT
geometry identical to the GeoJSON geometry. The centroid falls inside its polygon for 99.6% of polygons, but on its
line for only 36% of lines; Stage 2 removes the long lines for this reason.

**Reproducing it.** The conversion script was not saved in this folder. The code below reproduces the file: every
attribute and geometry byte-identical, and the centroid identical except on 415 of 12,536 rows, where
`longitude` or `latitude` differs by one unit in the 7th decimal (about 1 cm), a rounding difference.

```python
import pandas as pd
import geopandas as gpd

gdf = gpd.read_file("points_of_interest.geojson")
out = pd.DataFrame(gdf.drop(columns="geometry"))
out["geometry_type"] = gdf.geom_type
centroid = gdf.geometry.centroid              # planar centroid in degrees
out["longitude"] = centroid.x.round(7)
out["latitude"] = centroid.y.round(7)
out["geometry_wkt"] = gdf.geometry.apply(lambda g: g.wkt)
out.to_csv("points_of_interest.csv", index=False, encoding="utf-8-sig")
```

---

## Stage 2: `points_of_interest.csv` → `processed_poi.csv` (`poi_eda.ipynb`)

The GNN is semi-supervised node classification: each POI is a node, the target is its category, and uncategorised
POIs are unlabelled nodes. This stage turns raw OSM features into clean nodes with labels, basic node features and
spatial cross-validation folds.

### 2.1 Labels from OSM tags

- Tag values are lower-cased and trimmed.
- Each feature takes its label from one tag, in the order `tourism` > `amenity` > `shop` > `man_made` (22 features
  carry two keys; a fuel station also tagged `shop=convenience` stays a fuel station).
- 202 raw tag values map to 36 subcategories, which roll up into the 8 categories of the proposal: Infrastructure,
  Educational, Retail, Other, Food & Beverage, Tourism, Religious, Healthcare.
- `man_made=yes` and `amenity=fixme` say nothing about the feature and become `Uncategorized` (unlabelled nodes).
- Retail, Food & Beverage, Accommodation, Finance and Other services are flagged `is_business`.

### 2.2 Rows removed or moved

| Step | Rows after | Removed | Why |
|---|---:|---:|---|
| Raw export | 12,536 | | |
| Drop placeholders (`shop=vacant`, `man_made=proposed`) | 12,534 | 2 | Closed or not yet built |
| Drop linear networks (`pipeline`, `goods_conveyor`, `embankment`) | 12,372 | 162 | Lines up to 44 km long whose centroid is not a place; includes the only row without a district (a water-transfer tunnel) |
| Move `amenity=toilets` out of the node set | 7,899 | 4,473 | Household pit latrines, not public facilities (median 9 m² outline, 0.1% named, 51% from one mapping effort in Mafeteng). Kept as the context feature `latrines_1km` |
| Merge duplicates and repeated units | **6,083** | 1,816 | See 2.3 |

### 2.3 Merging duplicates

OSM often holds one place twice, and two copies of one place would become each other's nearest neighbour with the
same label, inflating accuracy. Features within 100 m of each other are linked when:

- **A.** they share a name and a category (750 pairs; 706 are a point plus a building outline);
- **B.** they touch (within 1 m), share a subcategory and have no conflicting names (360 pairs), e.g. a school point
  inside the school outline;
- **C.** both are unnamed, share a subcategory and are under 50 m apart (3,904 pairs): kiosk stalls of one market,
  storage tanks of one water works, silos of one farm.

Each linked group becomes one node at the members' mean position, taking the tags of its best-described member
(a named point first). `n_units` records how many features were merged. Result: 7,899 features → 6,083 nodes, 1,129
of them merged groups, and no same-name, same-category pair left within 100 m.

### 2.4 Node features (19)

- **Location (11), from positions only:** UTM 35S coordinates (`x_km`, `y_km`), POI counts within 500 m / 2 km /
  10 km, distance to the nearest POI and mean distance to the 8 nearest, household latrines within 1 km, an urban
  flag, and distances to the nearest town centre and to Maseru. Towns are found from the data: a node is urban if at
  least 50 other nodes lie within 2 km; urban nodes within 2 km of each other form one town (13 towns).
- **Attributes (8), from the OSM record:** has name / address / opening hours / building outline, log outline area,
  mapped as a point, a completeness score, and `osm_id_rank` (percentile of the OSM id, kept as a bias covariate;
  it mixes mapping date with point-versus-outline, see the Dictionary).
- `name`, the source tags and `n_units` are **not** features: 66% of names contain a keyword of their own category,
  and the tags and `n_units` are derived from the label.

### 2.5 Spatial cross-validation folds

Two POIs within about 700 m share a category more often than chance, so random splits would overstate accuracy.
Lesotho is cut into 5 km blocks (840 blocks); whole blocks are assigned to 5 folds, keeping each class within 6.3
percentage points of an even 20% per fold. `fold_boundary` flags the 15% of nodes that have a node from another fold
within 700 m, for an optional training buffer.

### 2.6 Output

`processed_poi.csv`: 6,083 nodes, 43 columns, features left unscaled. 5,952 labelled nodes and 131 unlabelled:

| Infrastructure | Educational | Retail | Other | Food & Beverage | Tourism | Religious | Healthcare | Uncategorized |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1,792 | 1,666 | 1,181 | 334 | 293 | 252 | 236 | 198 | 131 |

Columns by role: `node_id`, `osm_id` (id); `osm_ids`, `n_units`, `name`, `tag_key`, `tag_value`, `lon`, `lat`,
`nearest_town` (info, not features); `category`, `label` (0-7, -1 = unlabelled), `is_labelled`, `subcategory`,
`is_business` (labels); the 19 features; `adm1_*`, `adm2_*` (graph); `fold`, `block_id`, `fold_boundary` (split).

**Findings that shaped the graph.** A k-nearest-neighbour graph with k = 8 is the smallest k tested that connects
every node (median edge 0.84 km). 39% of its edges join two nodes of the same category against 22% by chance.

---

## Stage 3: `processed_poi.csv` → `poi_engineered.csv` (`poi_feature_engineering.ipynb`)

This stage turns the 19 node features into a model-ready feature set. It follows four rules:

1. **Only positions, district codes and the nodes' own OSM attributes are used.** No feature is computed from any
   node's label, including neighbours' labels. Label-based context (neighbour label shares, thematic edges) has to be
   computed inside each training fold in the model notebook.
2. **Nothing is fitted on the data.** Only fixed transforms (logs, ratios, counts) are applied, so the standard
   scaler is still fitted on the training folds in the model notebook.
3. **Every feature column starts with a group prefix**, so the model selects them with
   `df.filter(regex=r"^(loc|adm|struct|attr|ctx)_")`.
4. **The EDA's numbers are recomputed first as a check.** Densities, distances and towns match `processed_poi.csv`
   (up to the 0.1 m rounding of the stored coordinates).

### 3.1 What was engineered

| Step | Group | What it adds |
|---|---|---|
| Transform the 19 EDA features | `loc_`, `attr_` | `log1p` on the skewed counts and distances (skew 1.1–5.5 → mostly below 0.6); binaries, coordinates, `dist_maseru_km`, `osm_id_rank` and log outline area unchanged |
| Density at more scales | `loc_` | POI counts within 100 m, 250 m, 1 km, 5 km, 20 km; distance to the 3rd, 16th and 32nd nearest POI |
| Shape of the neighbourhood | `loc_` | For the 8 and 32 nearest POIs: linearity (on a line along a road or valley), offset from their centroid (centre or edge of a cluster), widest empty sector |
| Towns and access | `loc_` | Size of the nearest town (POIs within 5 km of its centre), distance relative to that town's radius, distance to the second-nearest town, gravity-style town-access index at 5 km and 25 km |
| Administrative context | `adm_` | Nodes in the sub-district and district, the sub-district's urban share, the node's density rank and distance from the centre within its sub-district, 10 district indicators |
| Mapping context | `ctx_` | For the 32 nearest POIs: share named / outlined / mapped as points / with address / with hours, mean and spread of their mapping date; the node's outline size and mapping date relative to them |
| Structural encodings | `struct_` | On the k = 8 graph: in-degree, reciprocity, clustering coefficient, random-walk return probabilities at 2–16 steps |

### 3.2 Redundancy check

80 candidate features were built. A feature was dropped when its Spearman correlation with an already kept feature
was 0.95 or higher (the EDA's own features were kept first). This check does not look at labels. 14 were dropped,
mostly near-identical random-walk steps, leaving **66 features**:

| `loc_` | `adm_` | `struct_` | `attr_` | `ctx_` |
|---:|---:|---:|---:|---:|
| 27 | 15 | 7 | 8 | 9 |

### 3.3 Checks

- The CSV is read back and compared with the in-memory features. All values are finite and no label-derived column
  is among the features. The carried-over columns are identical to `processed_poi.csv`.
- A PyTorch Geometric graph is built for one split (fold 0 held out): scaler fitted on training folds only, k = 8
  edges from `x_km`/`y_km` (31,452 undirected edges, no self-loops, no isolated nodes). One forward pass of an
  untrained GraphSAGE gives finite outputs.
- The notebook runs top to bottom in a fresh kernel in under a minute and reproduces the CSV byte for byte.

### 3.4 Output

`poi_engineered.csv`: 6,083 nodes, 92 columns = 26 columns carried over from `processed_poi.csv` (id, info, label,
graph and split columns, including `x_km`/`y_km` for building edges) + 66 prefixed features. The 17 raw feature
columns are not repeated; they appear only in their transformed, prefixed form. The notebook prints a full data
dictionary.

**Caveat for modelling.** `attr_osm_id_rank` is the single most label-informative feature, for two reasons that
both come from OSM history rather than from the places:

- OSM numbers points (nodes) and outlines (ways) in separate sequences, and way ids are far lower than most node ids.
  The rank therefore largely says whether a node came from an outline (correlation −0.82 with `attr_has_footprint`),
  and some categories are mostly outlines (86% of Religious and 85% of Educational nodes).
- Within each sequence, ids follow mapping date, and categories were mapped at different times: Religious, Retail,
  Food & Beverage and Other nodes early (median rank about 0.3 within their sequence), Educational and
  Infrastructure later (about 0.6).

Results should be reported with and without the three columns whose names contain `osm_id_rank`.

---

## Stage 4: model iteration 1 (retired)

The first model was a GraphSAGE network (with a separate weight for the node itself, residual layers and jumping
knowledge; GAT was tested and lost) trained on all 66 features, with the label trick and validation-fitted class
offsets. Under the spatial CV of Stage 2.5 it reached **test macro-F1 0.375 ± 0.021**, three times chance.

Its ablations showed where that score came from. Without the 8 `attr_` features it lost 0.051, and **geography alone
(`loc_` + `adm_` + `struct_`) reached only 0.235**. The model classified POIs mostly by how volunteers had mapped them
(named or not, outline or point, address, opening hours). Those attributes do not exist for an empty candidate site,
so this model could not score a location where no business stands yet, which is what the study's aim (market gaps,
suitable areas) needs. In design-science terms, its evaluation fed back into a second design (Stage 5). The notebook
(`model.ipynb`) and its outputs (`model_outputs/`) have been removed from the folder; they remain in the git
history (last present in commit `47b2787`) as the record of that first cycle.

---

## Stage 5: the dual-mode GNN (`gnn_location_model.ipynb` → `model_outputs_v2/`)

### 5.1 Design: one model, two input modes

| Mode | The node's own mapping attributes | Used for |
|---|---|---|
| **Known POI mode** | visible | classifying a mapped POI (RO1, RO4) |
| **Site mode** | hidden, and a flag `own_attrs_hidden` set to 1 | what a location and its neighbourhood imply, e.g. for a candidate site (RO2, RO3, RO4) |

The same weights serve both modes. The **own-node columns** hidden in site mode are the 8 `attr_` features plus
`ctx_osm_id_rank_vs_k32` and `ctx_footprint_vs_k32`, which compare the node with its neighbours. The other `ctx_`
features describe the 32 nearest *other* nodes, so they exist for a candidate site too. A hidden column is set to 0
(the training mean of a standardised column), and the flag tells the model "hidden" apart from "absent".

**Architecture.** A linear encoder takes the node's 66 features, the flag and its visible category (one-hot, zeros
if hidden). It is followed by 2 residual GraphSAGE layers with batch norm, ReLU and dropout, and a classifier on the
concatenation of all layers (jumping knowledge). The GraphSAGE layer averages the neighbours with **distance
weights** (a softmax of −distance / 250 m over each node's incoming edges, so a shop 50 m away counts more than a
village 4 km away) and keeps a separate weight for the node itself.

**One training epoch.**

1. The training nodes are shuffled. A share *r*, drawn uniformly from 0.1 to 0.9 each epoch, shows its category as
   an input (the **label trick**); the loss is taken on the other nodes, so a node never sees its own answer.
2. Of those loss nodes, 30% (the **mask rate**) are put in site mode.
3. Loss: cross-entropy with class weights `(n / (8 n_c)) ** 0.75` and label smoothing 0.05; AdamW (learning rate
   0.01, weight decay 0.05), full batch, dropout 0.5, hidden width 128.
4. **Early stopping** on the mean of the known-mode and site-mode validation macro-F1 (patience 50, at most 300
   epochs); the best epoch's weights are kept.

**After training**, two more steps are applied:
- **Label reuse:** confident predictions (softmax ≥ τ) of the nodes outside the training folds are fed back in as
  visible categories and the model runs again. It was selected on validation (1 round, τ = 0.5) but changes test F1
  by less than 0.004.
- **Temperature scaling:** one temperature per model and mode, fitted on the validation fold, divides the logits so
  that the probabilities match observed frequencies.

### 5.2 Evaluation protocol

- **Spatial 5-fold CV** on the 5 km blocks of Stage 2.5. Split *f* tests on fold *f*, validates on fold
  (*f* + 1) mod 5 and trains on the other three. Only training-fold categories are ever visible as inputs. The 131
  unlabelled nodes stay in the graph as neighbours.
- **Every choice is made on validation:** the architecture, the hyperparameters, the label-reuse setting and the
  temperatures. The test folds play no part in any choice.
- **Final model:** 5 seeds per split, 25 models in total. Each test fold is predicted by the mean of its 5 models'
  temperature-scaled probabilities, so every labelled node gets one **out-of-fold** prediction from models that never
  trained on its area.
- **Evaluation in both modes.** The nodes outside the training folds are split into 10 random groups. Each group is
  predicted with the other groups present as neighbours. In site mode the group's own attributes are hidden, so a node's
  attributes are not present anywhere in the graph when it is scored.
- **Chance level:** guessing at the class frequencies gives an expected macro-F1 of 0.125, because each class's F1
  equals its share and the 8 shares average to 1/8.

### 5.3 Metrics

| Metric | Definition | Why it is used | Where |
|---|---|---|---|
| **Macro-F1** (main metric) | F1 = 2·precision·recall / (precision + recall), computed per category and averaged with equal weight | The classes are unbalanced (Infrastructure is 9 times Healthcare); accuracy would reward ignoring the small classes. Chapter 1 specifies it | Selection (validation), headline result (test), ablations |
| **Precision, recall, F1 per category** | Precision: share of the nodes predicted as *c* that are *c*. Recall: share of the *c* nodes predicted as *c* | Shows which categories the model can and cannot recognise; chapter 1 asks for per-class recall | Results (5.5) |
| **Accuracy** | Share of nodes given the right category | Familiar summary; reported next to macro-F1, never instead of it | Results |
| **Top-2 / top-3 accuracy** | Share of nodes whose true category is among the model's 2 or 3 most likely | The tool gives a ranked shortlist, so a correct second or third choice is still useful | Results, per category |
| **Log loss** | Mean of −ln(probability given to the true category) | Measures the quality of the whole probability vector, not just the top class. Far less noisy than macro-F1 | Temperature fitting, permutation importance (RO2) |
| **Expected calibration error (ECE)** | Nodes are binned by their top probability (15 bins). ECE is the weighted mean gap between each bin's average probability and its accuracy | RO3 sums probabilities into expected counts, which is only valid if they are calibrated | Results, RO3 |
| **Fold SD** | Standard deviation of the 5 held-out folds' scores | Shows how much performance varies between areas; differences smaller than about one SD are treated as noise | Every table |
| **Pooled score** | The metric over all 5,952 out-of-fold predictions at once | One number per category without averaging over folds | Results |
| **Validation score** | Mean of known-mode and site-mode validation macro-F1 | One criterion that values both modes equally | Architecture and tuning selection |
| **z-score of a market gap** | (expected − observed) / √Σ p(1 − p) per sub-district and category | A significance test of a gap. It relies on the calibration measured by ECE; it is not a performance metric | RO3 (5.8) |

The second half of RO4, how useful business owners find the recommendations, is measured by interviews, not here.

### 5.4 Selection and tuning

| Step | Result |
|---|---|
| **Layer type** (same settings, 3 seeds) | GraphSAGE validation score 0.315, GAT 0.313, GCN 0.312: tied. **GraphSAGE** was kept (best on validation, 4 times faster than GAT) |
| **Tuning** (12 random configurations + the start, 3 seeds each) | Validation scores 0.306 to 0.320; the top eight lie within 0.008. Selected: hidden 128, 2 layers, dropout 0.5, class-weight power 0.75, mask rate 0.3, label rate 0.1 to 0.9, distance scale 250 m. **The margin over the starting configuration (0.005) is within noise**: the limit is the information in the data, not the settings. The one clear effect is the mask rate, which trades the two modes against each other |

### 5.5 Results (final 25-model ensemble, test folds)

| | Known mode | Site mode |
|---|---|---|
| **Macro-F1, mean ± SD over 5 folds** | **0.356 ± 0.039** | **0.245 ± 0.036** |
| Macro-F1, pooled | 0.360 | 0.255 |
| Accuracy | 0.552 | 0.388 |
| Top-2 / top-3 accuracy | 0.715 / 0.826 | 0.597 / 0.718 |
| ECE before / after temperature scaling | 0.038 / 0.038 | 0.032 / 0.036 |
| Log loss before / after | 1.335 / 1.315 | 1.765 / 1.739 |
| Median temperature | 0.87 | 0.97 |

Per held-out fold (macro-F1, known / site): fold 0 0.388 / 0.261, fold 1 0.360 / 0.251, fold 2 0.372 / 0.255,
fold 3 0.383 / 0.282, **fold 4 (Maseru) 0.280 / 0.176**, the weakest in both modes.

Per category (out-of-fold, all 5,952 labelled nodes):

| Category | Support | Known: precision | recall | F1 | top-3 | Site: precision | recall | F1 | top-3 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Infrastructure | 1,792 | 0.740 | 0.679 | 0.708 | 0.869 | 0.515 | 0.360 | 0.424 | 0.741 |
| Educational | 1,666 | 0.852 | 0.843 | **0.847** | 0.923 | 0.620 | 0.606 | **0.613** | 0.806 |
| Retail | 1,181 | 0.477 | 0.312 | 0.377 | 0.850 | 0.453 | 0.392 | **0.420** | 0.804 |
| Other | 334 | 0.163 | 0.174 | 0.168 | 0.506 | 0.120 | 0.156 | 0.136 | 0.485 |
| Food & Beverage | 293 | 0.134 | 0.215 | 0.165 | 0.703 | 0.069 | 0.085 | 0.076 | 0.560 |
| Tourism | 252 | 0.187 | 0.246 | 0.213 | 0.591 | 0.146 | 0.214 | 0.174 | 0.476 |
| Religious | 236 | 0.266 | 0.360 | 0.306 | 0.720 | 0.094 | 0.153 | 0.117 | 0.513 |
| Healthcare | 198 | 0.068 | 0.141 | 0.092 | 0.621 | 0.054 | 0.136 | 0.077 | 0.434 |

**Reading.** From location and neighbourhood alone (site mode), the model is twice chance on areas it never saw,
and the true category is in its top 3 for 72% of nodes. It recognises Educational, Infrastructure and Retail.
Retail scores *higher* in site mode than in known mode: shops are defined more by their surroundings than by how they
were mapped. Food & Beverage, Healthcare, Religious and Other are near chance in site mode and useful only as a
top-3 shortlist. Known mode is 0.02 below iteration 1, mostly on fold 4: part of that is the cost of sharing weights
with site mode, and part comes from the selected configuration.

### 5.6 Ablations (RO1)

Each row changes one thing in the selected configuration (3 seeds; reference test F1 0.349 known / 0.241 site).

| Ablation | Test F1 known | Test F1 site | What it shows |
|---|---:|---:|---|
| **Local features only** (coordinates, district, own attributes: 20), **graph on** | 0.363 | **0.218** | Message passing adds **+0.08** in site mode and +0.04 in known mode: **the GNN learns the neighbourhood from the graph** |
| Local features only, **graph off** | 0.326 | 0.140 | |
| All 66 features, graph off | 0.356 | 0.240 | With the engineered neighbourhood features (computed from the same graph), message passing adds nothing more |
| Neighbour categories off (no label trick) | 0.358 | 0.242 | Co-location adds nothing on top of the features |
| Neighbour categories only (no features) | 0.097 | 0.095 | Below chance: 42 to 73% of held-out nodes have no labelled neighbour under spatial CV |
| Mask rate 0 / 0.5 / 0.7 / 1 (selected: 0.3) | 0.348 / 0.349 / 0.337 / 0.246 | 0.230 / 0.257 / 0.248 / 0.242 | Training in site mode is what makes site mode work. It costs known mode nothing up to 0.5. 0.5 is best on test, but validation preferred 0.3 (within noise) |
| No `osm_id_rank` (4 columns) | 0.356 | 0.254 | No dependence on OSM mapping history |
| Buffered CV (700 m) | 0.351 | 0.249 | No leakage across fold edges |

### 5.7 RO2: what shapes the composition of an area

Both analyses use the final models in site mode, on the held-out folds.

- **Permutation importance** (a feature is shuffled across all nodes; importance is the rise in log loss).
  Location characteristics dominate. Shuffling the `loc_` group raises log loss by 0.31 and drops macro-F1 from
  0.247 to 0.154, while hiding all neighbour categories changes log loss by 0.002. The most important single inputs
  describe settlement density and spacing: POIs within 100 m, the distance to the 16th and the mean distance to the
  8 nearest POIs, density within 2 km, the distance to the nearest POI and density within 250 m. Next come the
  sub-district's urban share, the named share of the 32 nearest POIs and household latrines within 1 km.
- **Neighbour-type influence (occlusion).** The categories of all type-A POIs are hidden, and the change in each
  predicted probability is measured for held-out nodes with a type-A neighbour, with every other category visible
  (the deployment condition, where site-mode macro-F1 is 0.256). The effects are at most 1.4 percentage points but
  structured:
  - like attracts like (Infrastructure +1.4, Retail +1.0, Tourism +0.7, Educational +0.6);
  - Retail and Food & Beverage go together (+0.2 each way), and a Retail neighbour lowers Infrastructure (−0.7);
  - schools sit near churches (Educational neighbour → Religious +0.3);
  - Healthcare follows no neighbour pattern.

### 5.8 RO3: market gaps per sub-district

- **Expected against observed.** For every mapped POI, the out-of-fold site-mode probabilities say what its location
  resembles. Their sum over a sub-district is the **expected** count per category; the **observed** count is what is
  mapped there.
- **Prior correction.** Class weights inflate the small classes, so one factor per category is fitted (iterative
  proportional fitting) to make the national expected totals equal the observed totals. Gaps are therefore relative
  to the national pattern. The corrected probabilities are used only for counting (ECE 0.035).
- **Flags.** A **gap** is flagged when z ≥ 2 and expected − observed ≥ 2 POIs; a **cluster** when z ≤ −2 and
  observed − expected ≥ 2. Only sub-districts with at least 20 labelled POIs are flagged. A sub-district is marked
  **thin** when its mean `attr_quality_score` is in the lowest quarter, because missing data can look like a gap.
- **Results.** 67 of the 78 sub-districts qualify: **17 gaps and 40 clusters**.
  - Business gaps: Healthcare (70 mapped against 107 expected) and Retail (433 against 479) in Maseru town; Retail in
    Quthing town (33 against 48) and in Qiloane (0 against 8, thinly mapped); Tourism in Ngoajane; Food & Beverage in
    Leribe.
  - Clusters: Food & Beverage in Teyateyaneng and Maputsoe; Retail in Butha-Buthe.
  - 5 of the 17 gaps are in thinly mapped sub-districts.

A gap means that the locations in an area resemble where a category is found elsewhere in Lesotho, but the category is
mapped there less often than expected. It is a hypothesis about unmet demand or missing data, not evidence of
profitability. It is the input to the interviews with business owners (RO4).

### 5.9 Output (`model_outputs_v2/`)

| File | Content |
|---|---|
| `gnn_dual_models.pt` | The 25 trained models and everything needed to use them (D10) |
| `oof_predictions.csv` | Every labelled node with its predictions in both modes (D7) |
| `unlabelled_predictions.csv` | The 131 unlabelled nodes with predictions in both modes, averaged over the 5 splits (D8) |
| `market_gaps.csv` | Expected and observed counts per sub-district and category, with flags (D9) |
| `metrics.json` | Every table of the notebook (D10) |
| `training_progress.log` | The training log, shown live in a separate window while the notebook ran |

### 5.10 Limits

- Site mode is twice chance overall but reliable only for Educational, Infrastructure and Retail.
- The urban Maseru fold is the weakest in both modes.
- The tuned configuration won by a margin within noise.
- Neighbour categories add only a small co-location signal.
- Market gaps inherit OSM's uneven coverage and say nothing about profitability (chapter 1, Section 1.6).

---

## Stage 6: the location-intelligence tool (`predict_poi.py`)

A command-line tool on the Stage 5 model. Run it with no arguments for an interactive menu.

| Command | What it does | Time |
|---|---|---|
| `site --lat -29.3142 --lon 27.4833` | What a candidate location suits (site mode), with the market signals of its sub-district | about 1 s |
| `suggest --category Retail --lat -30.4005 --lon 27.7013 --radius-km 1.5` | Scores a grid of up to about 49 candidate sites (points more than 1.5 km from any mapped POI are skipped) around a location and ranks them for one category, with the number of that category already within 1 km | about 1 s per site |
| `gaps --area Quthing` (`--all` for unflagged rows) | The `market_gaps.csv` rows of the sub-districts or districts matching a name or code | instant |
| `new --lat ... --lon ... --name --hours --footprint-m2 120` | A POI described as it would be mapped (known mode) | about 5 s |
| `existing --query node/4340647091` | The notebook's out-of-fold predictions for a mapped POI, both modes | instant |
| `batch sites.csv [--known]` | Every row of a CSV (`lat`, `lon`; optional `adm2_pcode`, or the mapping attributes with `--known`) | as above, per row |
| `--check` | Verifies the tool against the notebook (below) | about 8 min |

**How a location is scored.** The location is added to the graph of the 6,083 POIs as one more node, and its 66
features are computed with the code of Stages 2 and 3.
- **Candidate site (site mode):** a **one-way node**. It receives messages from its k-NN partners but changes nothing
  for the existing POIs, and only its own random-walk encoding is computed. Only its 2-hop neighbourhood is run
  through the models.
- **New POI (known mode):** fully inserted, and every affected feature is recomputed.

In both cases all 25 models score it, each with its split's training categories visible, and their
temperature-scaled probabilities are averaged. Label reuse is not applied (it changed test F1 by less than 0.004).
Because training weights the small categories up, their probabilities run high: read the ranking, not the
percentages, as odds. Every output shows each category's out-of-fold F1 and top-3 accuracy in that mode next to its
probability.

**`--check`** runs four tests:
1. All 66 features of the 6,083 POIs are rebuilt and match `poi_engineered.csv` (largest difference 1e-6).
2. Live predictions match the notebook's out-of-fold predictions: macro-F1 0.359 against 0.360 (known) and 0.255
   against 0.255 (site), the same class for 98% and 96% of nodes.
3. POIs removed and added back as new POIs get the same features and probabilities (differences below 1e-4).
4. Spots of removed POIs scored as candidate sites agree with the notebook's site mode for about 90% of nodes, with
   the same top-3 accuracy (73% against 70% on 30 nodes). The notebook's site-mode scores therefore hold for real
   candidate sites.

---

## Reproducing the pipeline

1. Stage 1: run the snippet above (only needed if `points_of_interest.csv` is missing).
2. Run `poi_eda.ipynb` top to bottom → `processed_poi.csv`.
3. Run `poi_feature_engineering.ipynb` top to bottom → `poi_engineered.csv`.
4. Run `gnn_location_model.ipynb` top to bottom → `model_outputs_v2/`. On 4 CPU threads this takes about 4 hours
   (architecture comparison 25 min, tuning 70 min, final ensemble 15 min, ablations 2 hours, RO2 and RO3 15 min);
   a terminal window follows `training_progress.log` while it runs. Seeds are fixed, but results can differ in the
   third decimal between machines.
5. Run `python predict_poi.py --check` to confirm the tool matches the notebook.

Environment used: Python 3.14, pandas 3.0, NumPy 2.4, GeoPandas 1.1, SciPy 1.17, scikit-learn 1.9, matplotlib 3.10,
PyTorch 2.14 (CPU) and PyTorch Geometric 2.8, pyproj for `predict_poi.py`.

---

## Dictionary

What each term, code, category and column in this folder means. D1 explains the terms used above and in the
notebooks; D2 lists the codes; D3 defines the categories; D4 to D6 describe every column of the three data CSV files, and D7 to D10 the model
outputs.

### D1. Terms

#### OpenStreetMap and geography

| Term | Meaning |
|---|---|
| OpenStreetMap (OSM) | A free world map built by volunteers. Anyone can add or edit, so how complete an area is depends on where mappers have worked. |
| HOT, HDX | The Humanitarian OpenStreetMap Team, which produced the export, and the Humanitarian Data Exchange, where it is published. |
| `oex` | The export tool HOT used (version 0.4.13). |
| Snapshot | The date of the OSM database copy the export was taken from (2026-09-08). Later edits are not in the data. |
| Feature | One mapped object in the raw export: one row of `points_of_interest.csv`. |
| OSM element: node, way, relation | OSM's three object types. A **node** is a single point. A **way** is a line through several points, or an outline when it closes on itself. A **relation** groups other elements (here, 2 multi-part outlines). An OSM node is not the same thing as a graph node (D1, Data preparation). |
| OSM id | Element type and number, e.g. `node/4340647091` or `way/297456013`. Each type has its own number sequence, issued in the order elements are created. |
| Tag, key, value | OSM describes a feature with `key=value` tags, e.g. `amenity=school`. This export kept features with any of four keys: `amenity` (public facilities and services), `shop` (shops), `tourism` (accommodation and sights) and `man_made` (built structures such as tanks, masts and bridges). |
| POI | Point of interest: any place people might look for or travel to, such as a shop, school, clinic or water tank. |
| Point, Polygon, LineString | Geometry types. A Point is one location; a Polygon is a closed outline, usually of a building; a LineString is an open line such as a pipeline or a pier. |
| Outline, footprint | A Polygon drawn around a building or site. Its area is the footprint. |
| WKT | Well-Known Text: a plain-text way to write a geometry, e.g. `POINT (27.447 -30.149)` or `POLYGON ((x1 y1, x2 y2, ...))`, with longitude before latitude. |
| Centroid | The geometric centre of a shape. For a Point it is the point itself. For a concave outline or a bent line it can fall outside the shape. |
| WGS84 (EPSG:4326, CRS84) | The longitude/latitude system used by GPS and web maps. Units are degrees. |
| UTM zone 35S (EPSG:32735) | A flat map projection in metres covering Lesotho. Every distance and area in this folder is measured in it. `x_km` is the easting and `y_km` the northing, in km. Southern-hemisphere northings count from a false origin 10,000 km south of the equator, which is why `y_km` values are near 6,700. |
| geoBoundaries, ADM0 / ADM1 / ADM2 | An open dataset of administrative boundaries. ADM0 is the country, ADM1 the 10 districts, ADM2 the 78 sub-district units used here. ADM3 and ADM4 are empty for Lesotho in this export. |
| P-code | Place code: a fixed identifier for an administrative unit, e.g. `LSA` for Maseru district and `LSA12` for one of its sub-districts. Some sub-districts share a name with a district or town (`Qacha's Nek` is both), and two carry a suffix: `Maseru (Maseru District)` and `Maseru (Berea Distict)` (spelled as in the source). |
| ODbL | The Open Database Licence that OSM uses. Use is free with attribution, and derived databases must be shared under the same terms. |

#### Data preparation

| Term | Meaning |
|---|---|
| Node (graph node) | One cleaned POI in `processed_poi.csv` and `poi_engineered.csv`: either a single OSM feature or a group of duplicate features merged into one. Each node is one row, numbered by `node_id`. |
| Category | The 8-class target: Infrastructure, Educational, Retail, Other, Food & Beverage, Tourism, Religious, Healthcare (D3). |
| Subcategory | One of 36 finer types that roll up into the categories, e.g. Small grocery → Retail (D3). |
| Labelled node | A node with one of the 8 categories (5,952 nodes). |
| Uncategorized, unlabelled node | A node whose tag says nothing about its type (`man_made=yes`, `amenity=fixme`); 131 nodes. It stays in the graph, so its neighbours still see it, but it has no target (`label = -1`). |
| Primary tag, tag precedence | When a feature has more than one of the four keys, its label comes from the first present in the order `tourism`, `amenity`, `shop`, `man_made`. `tag_key` and `tag_value` hold the chosen tag. |
| Business (`is_business`) | The private-business subcategories: all of Retail and Food & Beverage, plus Accommodation, Finance and Other services (1,756 nodes). |
| Placeholder | A tag for a place that is not operating: `shop=vacant` (closed) or `man_made=proposed` (not yet built). Dropped. |
| Linear network | Pipelines, goods conveyors and embankments: long lines whose centre is not a place. Dropped. |
| Household latrine | An `amenity=toilets` feature. These are private pit latrines (median outline 9 m², almost none named), half of them from one Mafeteng mapping effort. They were removed from the nodes and counted around each node instead (`latrines_1km`). |
| Duplicate rules A, B, C | The three ways features were linked for merging (Stage 2.3). A: same name and category within 100 m. B: touching, same subcategory, no conflicting names. C: both unnamed, same subcategory, under 50 m apart. |
| Merged group, representative member | Linked features become one node at their mean position. The representative is the member whose tags the node takes: a named point first, then a named outline, then the largest outline. 1,129 nodes are merged groups. |
| Urban node | A node with at least 50 other nodes within 2 km (`is_urban = 1`). |
| Town (data-derived) | A cluster of urban nodes lying within 2 km of each other; 13 were found. Each town is named after the sub-district holding most of its urban nodes, so `Manonyane` is the Roma area and `Tayateyaneng` is Teyateyaneng. |
| Town centre | The urban node in a town with the most other nodes within 2 km. Maseru's centre is the reference point for `dist_maseru_km`. |
| Town radius | The distance from the centre within which 90% of a town's urban nodes lie, with a minimum of 1 km (Stage 3). It is 5.6 km for Maseru and 1 to 1.8 km for the other towns. |
| Density | The number of other nodes within a straight-line distance, e.g. `density_500m`. It counts nodes, not raw features, and excludes latrines. |
| k-th nearest neighbour distance | The straight-line distance to the k-th closest other node: small in towns, large in the mountains. |
| Label leakage | A column that gives away the answer rather than predicting it, e.g. `name` ("... Primary School"), the source tags, or `n_units`. These columns are kept for reading results but never used as model inputs. |
| Mapping bias | Differences in how thoroughly, when and how (point or outline) volunteers mapped areas and types, e.g. Mafeteng's latrine campaign. It shows up in the data but says nothing about the places. |
| Bias covariate | A column kept so the model can be checked for dependence on mapping bias, rather than because it describes the place. |
| `osm_id_rank` | The percentile (0 to 1) of the smallest OSM id number among the node's features. It mixes two signals. Within one element type, a higher id was created later, so the rank orders mapping date. Across types, way ids go up to about 1.6 billion while node ids go up to about 14 billion. As a result, every node whose smallest id is a way (an outline) ranks at or below 0.62, while point-based nodes have a median rank of 0.80, and the rank correlates −0.82 with `has_footprint`. Read it as "mapping date and point-versus-outline together". |
| Spatial cross-validation (CV) | Testing the model on whole areas held out from training, rather than on random nodes, so test nodes do not sit next to training nodes of the same type. |
| Block | A 5 km × 5 km square of the UTM grid; there are 840 with at least one node. |
| Fold | One of 5 groups of blocks (0 to 4). Each CV split trains on four folds and tests on the fifth. |
| Fold boundary, buffer | A node is a fold-boundary node when a node of another fold lies within 700 m, the distance up to which nearby POIs share a category more often than chance. Leaving these nodes out of training when the neighbouring fold is the test fold gives a buffered CV. |

#### Graph and model

| Term | Meaning |
|---|---|
| GNN | Graph neural network: a model that predicts a node's label from its own features and from the features of its neighbours, passed along the graph's edges. |
| Semi-supervised node classification | The task here: every node is in the graph, only some have labels, and the model predicts the category of the others. |
| Transductive setting | All nodes, including test nodes and their features, are in the graph during training; only the test labels are hidden. The Stage 3 check builds the graph this way. |
| Edge | A link between two nodes along which the GNN passes information. |
| k-NN graph (k = 8) | Each node is linked to its 8 nearest nodes, and a link is kept if either node lists the other (undirected). Each node therefore has 8 to 20 neighbours (median 10). k = 8 is the smallest k tested that joins every node into one connected graph. |
| Self-loop, isolated node, connected component | An edge from a node to itself; a node with no edges; a set of nodes that can all reach each other. The k = 8 graph has no self-loops, no isolated nodes and a single component. |
| Administrative edges | Optional edges between nodes in the same sub-district or district, built from `adm2_idx` and `adm1_idx`. |
| Thematic edges | Optional edges between nodes of categories that tend to sit together, e.g. Retail and Food & Beverage. They use labels, so they must be computed from the training folds only. |
| Message passing | What one GNN layer does: each node collects its neighbours' features (e.g. their mean) and combines them with its own. Two layers reach the neighbours of neighbours. |
| GCN, GraphSAGE, GAT | Common GNN layer types. GCN averages a node with its neighbours; GraphSAGE keeps the node's own features separate from its neighbours'; GAT learns how much weight to give each neighbour. |
| Homophily | The tendency of linked nodes to share a label. 39% of k = 8 edges join same-category nodes, against 22% for random pairs. |
| Co-location quotient (CLQ) | How much more often category B appears among category A's 8 nearest neighbours than B's overall share predicts. Above 1 means the two attract (Retail, Food & Beverage, Other and Healthcare: 1.3 to 1.8); below 1 means they avoid each other. |
| Class weight | A weight in the training loss that makes rare classes count more. With "balanced" weights, a Healthcare node (smallest class) counts about 9 times as much as an Infrastructure node (largest). |
| Macro-F1 | The F1 score (the balance of precision and recall) computed per class and averaged with equal weight, so small classes count as much as large ones. |
| Standard scaler | Rescales each feature to mean 0 and standard deviation 1. It is fitted on the training folds' rows only and then applied to every row. |
| log1p | `ln(1 + x)`: compresses large values and keeps 0 at 0. Undo it with `exp(v) − 1`. Every column with `log` in its name is stored this way. |
| Skew | How lopsided a distribution is. A long tail of large values (a few town-centre nodes with hundreds of neighbours) gives a skew above 1. |
| Percentile rank | A value's position among all values, from 0 (lowest) to 1 (highest). |
| One-hot indicator | One 0/1 column per possible value, with exactly one 1 per row (the `adm_district_*` columns). |
| Spearman correlation (ρ) | The correlation between the ranks of two variables, from −1 to 1. Stage 3 drops a feature when its ρ with a feature already kept is 0.95 or more, or −0.95 or less. |
| Mutual information | How much knowing a feature reduces uncertainty about the category, in nats (0 = no information). Stage 3 uses it only to describe features, not to choose them. |
| Linearity | `1 − λ2/λ1` from the spread of the nearest nodes' positions: near 1 when they lie along a line (a road or valley), near 0 when they are spread evenly. |
| Offset | The distance from a node to the centre of its nearest nodes, divided by their mean distance: 0 in the middle of a cluster, near 1 at its edge. |
| Angular gap | The widest direction with no neighbour, as a share of 360°. Above 0.5 means every neighbour lies on one side of the node. |
| Gravity index (town access) | The sum over the 13 towns of town size × `exp(−distance / β)`: high next to a big town, small but above zero between towns. The file keeps β = 5 km. |
| Structural encoding | A feature that describes a node's place in the structure of the graph rather than on the map. It is given to a GNN because message passing cannot work it out. |
| In-degree | How many nodes list this node among their 8 nearest (0 to 19). A high value marks a local hub. |
| Reciprocity | The share of a node's 8 nearest that also list it among their 8 nearest. |
| Clustering coefficient | The share of pairs of a node's neighbours that are also linked to each other: high in compact clusters, low along chains. |
| RWSE | Random-walk structural encoding: the probability that a walk starting at the node, stepping to a random neighbour each time, is back at the node after t steps (t = 2, 3, 6 and 11 are kept). |
| Feature shift | How far the test fold's average value of a feature lies from the training folds' average, in training standard deviations. |

#### Model training and evaluation (Stages 4 to 6)

| Term | Meaning |
|---|---|
| Iteration 1, iteration 2 | The two design cycles of the model: `model.ipynb` (one mode, relies on mapping attributes; retired) and `gnn_location_model.ipynb` (two modes). |
| Known POI mode | The model sees all 66 features of a node, including how the node itself was mapped. Used for mapped POIs. |
| Site mode | The node's own mapping attributes (the own-node columns) are hidden and the flag `own_attrs_hidden` is 1. The model sees only the location and its neighbourhood, as for a candidate site where no business stands yet. |
| Own-node columns | The 10 features that describe how the node itself was mapped: the 8 `attr_` features, `ctx_osm_id_rank_vs_k32` and `ctx_footprint_vs_k32`. |
| Mask rate | The share of the loss nodes put in site mode in each training epoch (0.3 in the final model). |
| Label trick | Feeding training nodes' categories to the model as inputs (a one-hot vector, zeros when hidden), so it can use its neighbours' categories. In each epoch the loss is taken only on nodes whose own category is hidden. |
| Label rate | The share of training nodes whose category is shown as an input in an epoch; drawn uniformly from 0.1 to 0.9 each epoch. |
| Label reuse, pseudo-label | After training, confident predictions (softmax ≥ τ) of nodes outside the training folds are fed back in as visible categories (pseudo-labels) and the model runs again. No true test label is used. |
| Distance weights | Each node's neighbours are averaged with weights `softmax(−distance / scale)` (scale 250 m), so near neighbours count more than far ones. |
| Residual connection, jumping knowledge | A layer adds its output to its input (`h + layer(h)`), so the node's own signal is not averaged away; the classifier reads the outputs of all layers side by side. |
| One-way node | A candidate site added to the graph so that it receives messages from its neighbours but sends none, leaving every existing POI's prediction unchanged. |
| Validation fold | The fold used to choose settings and stop training: fold (*f* + 1) mod 5 when fold *f* is the test fold. |
| Early stopping, patience | Training stops when the validation score has not improved for 50 epochs; the best epoch's weights are kept. |
| Seed, seed ensemble | A seed fixes the random start of a training run. The final model trains 5 seeds per split and averages their probabilities. |
| Out-of-fold (OOF) prediction | A node's prediction from the models that held its fold out, i.e. never trained on its area. Every labelled node has exactly one. |
| Softmax, logit | A logit is the model's raw score per category; softmax turns the 8 scores into probabilities that sum to 1. |
| Temperature scaling | Dividing the logits by one number *T* fitted on the validation fold, so that the probabilities match how often the model is right. It never changes the predicted category. |
| Precision, recall, F1 | For category *c*: precision is the share of nodes predicted *c* that are *c*; recall is the share of *c* nodes predicted *c*; F1 = 2 × precision × recall / (precision + recall). |
| Accuracy | The share of nodes given the right category. |
| Top-k accuracy | The share of nodes whose true category is among the model's *k* most likely. |
| Log loss (cross-entropy) | The mean of −ln(probability given to the true category). 0 is perfect; a uniform guess over 8 categories gives 2.08. |
| Expected calibration error (ECE) | Nodes are grouped into 15 bins by their highest probability. ECE is the bin-size-weighted mean gap between each bin's average probability and its accuracy. 0 is perfectly calibrated. |
| Calibrated | Probabilities that match frequencies: of the nodes given 30% for a category, about 30% are in it. |
| Chance level | The macro-F1 of guessing at the class frequencies: 0.125 here. |
| Fold SD | The standard deviation of a score over the 5 held-out folds. Differences smaller than about one fold SD are treated as noise. |
| Validation score | The mean of known-mode and site-mode validation macro-F1; the selection criterion of Stage 5. |
| Ablation | A run that removes or changes one part of the model, to measure what that part contributes. |
| Local features | Features computed without any other POI: coordinates, district indicators and the node's own attributes (20 columns). Used to test whether message passing can learn the neighbourhood by itself. |
| Permutation importance | The rise in log loss (or drop in macro-F1) when one feature's values are shuffled across nodes, breaking its link with the category. |
| Occlusion | Hiding one input (here, the categories of all POIs of one type) and measuring how the predictions change. |
| Deployment condition | The setting in which the tool is used: every mapped POI's category is known and visible as a neighbour category. |
| Expected and observed count | For a sub-district and category: the sum of the nodes' corrected site-mode probabilities, and the number of nodes mapped with that category. |
| Prior correction | One factor per category, fitted by iterative proportional fitting, that rescales the probabilities so that the national expected totals equal the observed totals. It removes the effect of the class weights before counting. |
| z-score | (expected − observed) / √Σ p(1 − p): how many standard deviations the gap lies from zero, if the probabilities are calibrated. |
| Gap, cluster | A gap: expected exceeds observed (z ≥ 2 and by at least 2 POIs); the locations suggest more of the category than is mapped. A cluster: the reverse (z ≤ −2). |
| Thin mapping | A sub-district whose mean `attr_quality_score` is in the lowest quarter of sub-districts. Its gaps may come from missing data. |

### D2. Codes

**Districts** (`adm1_pcode`, `adm1_idx`, and the `adm_district_*` indicator columns):

| `adm1_pcode` | `adm1_idx` | District | Nodes |
|---|---:|---|---:|
| LSA | 0 | Maseru | 2,045 |
| LSB | 1 | Butha-Buthe | 320 |
| LSC | 2 | Leribe | 812 |
| LSD | 3 | Berea | 772 |
| LSE | 4 | Mafeteng | 366 |
| LSF | 5 | Mohale's Hoek | 353 |
| LSG | 6 | Quthing | 330 |
| LSH | 7 | Qacha's Nek | 305 |
| LSJ | 8 | Mokhotlong | 368 |
| LSK | 9 | Thaba-Tseka | 412 |

`adm2_idx` numbers the 78 sub-districts 0 to 77 in the order of their `adm2_pcode`.

**Labels** (`label`):

| `label` | Category | Nodes |
|---:|---|---:|
| 0 | Infrastructure | 1,792 |
| 1 | Educational | 1,666 |
| 2 | Retail | 1,181 |
| 3 | Other | 334 |
| 4 | Food & Beverage | 293 |
| 5 | Tourism | 252 |
| 6 | Religious | 236 |
| 7 | Healthcare | 198 |
| −1 | Uncategorized (unlabelled) | 131 |

**Blocks** (`block_id`): `column_row` of the 5 km grid, i.e. `floor(x_km / 5)_floor(y_km / 5)`. For example,
`132_1333` covers easting 660–665 km and northing 6,665–6,670 km.

### D3. Categories and subcategories

The tags each subcategory is built from (Stage 2.1). "Business" marks the subcategories counted by `is_business`.

| Category | Subcategory | Nodes | Business | OSM tags |
|---|---|---:|:---:|---|
| Retail | Small grocery | 276 | ✓ | shop: kiosk, convenience, general, variety_store, greengrocer, deli, farm |
| | Supermarket & wholesale | 174 | ✓ | shop: supermarket, wholesale, department_store, mall |
| | Clothing & personal care | 106 | ✓ | shop: clothes, shoes, bag, hairdresser, beauty, cosmetics, dry_cleaning, laundry |
| | Automotive & fuel | 247 | ✓ | amenity: fuel, car_wash, car_rental, vehicle_inspection; shop: car_repair, car, car_parts, tyres, bicycle, fuel; man_made: fuel_pump |
| | Hardware & home | 84 | ✓ | shop: hardware, doityourself, furniture, kitchen, household, houseware, interior_decoration, electrical, garden_centre, agrarian, trade, gas |
| | Electronics & phones | 40 | ✓ | shop: mobile_phone, electronics, computer, hifi, call, photo, copyshop |
| | Alcohol | 67 | ✓ | shop: alcohol, beverages |
| | Butcher & bakery | 51 | ✓ | shop: butcher, bakery, seafood, flour_mill |
| | Funeral services | 30 | ✓ | shop: funeral_directors; amenity: mortuary |
| | Market | 20 | ✓ | amenity: marketplace |
| | Other retail | 86 | ✓ | shop: yes, books, stationery, gift, toys, florist, art, craft, collector, sports, outdoor, optician |
| Food & Beverage | Bar & nightlife | 125 | ✓ | amenity: bar, pub, nightclub, casino, gambling |
| | Restaurant & cafe | 120 | ✓ | amenity: restaurant, cafe; shop: coffee |
| | Fast food | 48 | ✓ | amenity: fast_food, food_court |
| Tourism | Accommodation | 174 | ✓ | tourism: guest_house, hotel, chalet, hostel, apartment, motel, camp_site, alpine_hut |
| | Attraction & information | 78 | | tourism: viewpoint, attraction, information, artwork, museum, picnic_site, ground_sign, yes |
| Educational | School | 1,579 | | amenity: school, prep_school |
| | Pre-school & childcare | 49 | | amenity: kindergarten, childcare |
| | Tertiary & training | 38 | | amenity: college, university, language_school, driving_school |
| Healthcare | Clinic & hospital | 156 | | amenity: clinic, hospital, doctors, dentist |
| | Pharmacy | 42 | | amenity: pharmacy; shop: chemist, medical_supply |
| Religious | Place of worship | 209 | | amenity: place_of_worship, monastery; man_made: cross |
| | Cemetery | 27 | | amenity: grave_yard |
| Other | Finance | 89 | ✓ | amenity: bank, atm, bureau_de_change; shop: money_lender |
| | Post office | 32 | | amenity: post_office, post_box |
| | Public admin & safety | 138 | | amenity: police, courthouse, townhall, prison, fire_station, ranger_station, security_booth, reception_desk |
| | Community & leisure | 56 | | amenity: community_centre, social_facility, library, events_venue, conference_centre, arts_centre, theatre, stage, planetarium, studio, give_box |
| | Other services | 19 | ✓ | amenity: veterinary, animal_shelter, brothel; shop: travel_agency |
| Infrastructure | Transport & parking | 339 | | amenity: parking, parking_space, taxi, bus_station, ferry_terminal, boat_storage, bicycle_rental, bicycle_repair_station, weighbridge, shelter; man_made: pier, bridge |
| | Sanitation & waste | 134 | | amenity: toilets, waste_disposal, sanitary_dump_station, waste_basket, recycling; man_made: wastewater_plant, manhole, septic_tank. (All `amenity=toilets` features were moved out as latrines, so none of the 134 nodes is a toilet.) |
| | Water supply | 341 | | amenity: water_point, drinking_water, fountain; man_made: storage_tank, water_tower, water_works, water_well, water_tap, reservoir_covered, pumping_station, water_pump, windmill |
| | Telecom & power | 475 | | amenity: telephone; man_made: mast, tower, antenna, satellite_dish, street_cabinet, utility_pole |
| | Agriculture | 123 | | man_made: kraal, silo, bunker_silo, animal_dip |
| | Industry & mining | 71 | | amenity: warehouse; man_made: works, adit, tailings_pond, chimney |
| | Minor structures | 309 | | amenity: bench, hunting_stand; man_made: survey_point, flagpole, geoglyph, symbol, advertising, monitoring_station |
| Uncategorized | Unspecified | 131 | | amenity: fixme; man_made: yes |

**OSM values that may be unfamiliar**

| Tag | Meaning |
|---|---|
| `shop=kiosk` | A small stall or booth selling everyday goods; many here are market stalls, merged by rule C. |
| `shop=general` | A general dealer selling a mix of goods. |
| `shop=yes`, `tourism=yes`, `man_made=yes` | Tagged as a shop, a tourism feature or a structure without saying which kind. `man_made=yes` is too vague to label, so it becomes Uncategorized. |
| `amenity=fixme` | A mapper's note that the feature needs checking. Uncategorized. |
| `amenity=prep_school` | A tutoring or preparatory school. |
| `amenity=shelter` | A roofed shelter, such as at a bus or taxi stop. |
| `amenity=weighbridge` | A scale for weighing vehicles. |
| `amenity=give_box` | A small public box for freely giving and taking items. |
| `man_made=kraal` | A livestock enclosure. |
| `man_made=animal_dip` | A dipping tank where livestock are treated against parasites. |
| `man_made=silo`, `bunker_silo` | Storage for grain or feed; a bunker silo is a walled open pit for silage. |
| `man_made=survey_point` | A survey marker or trigonometric beacon. |
| `man_made=geoglyph` | Large letters or a figure laid out on a hillside. |
| `man_made=street_cabinet` | A roadside utility cabinet for telecom or power equipment. |
| `man_made=adit` | A horizontal entrance to a mine. |
| `man_made=tailings_pond` | A pond holding waste from mining. |
| `tourism=chalet`, `alpine_hut` | A self-catering holiday cottage; a mountain hut. |

### D4. Columns of `points_of_interest.csv`

"Filled" is the share of the 12,536 rows with a value (from the export's column profile).

| Column | Meaning | Filled |
|---|---|---:|
| `id` | OSM id (`node/…`, `way/…` or `relation/…`); unique | 100% |
| `name` | Name as mapped, in any language | 33.9% |
| `name_en` | English name | 1.1% |
| `name_st` | Sesotho name | 1 row |
| `amenity`, `shop`, `tourism`, `man_made` | Values of the four OSM keys (D3) | 68.4%, 11.5%, 2.4%, 17.9% |
| `opening_hours` | Opening hours in OSM syntax, e.g. `Mo-Fr 08:00-17:00`, `24/7`, `sunrise-sunset` | 2.3% |
| `beds`, `rooms` | Accommodation capacity | 0%, 6 rows |
| `addr_full`, `addr_housenumber`, `addr_street`, `addr_city` | Address parts. `addr_housenumber` often holds a place description (e.g. "Maseru Mall") instead of a number | 0%, 1.3%, 4.4%, 4.4% |
| `source` | Source note for the mapping | 0% |
| `adm0_pcode`, `adm0_name` | Country: `LSO`, Lesotho | 100% |
| `adm1_pcode`, `adm1_name` | District (D2) | all but 1 row (a tunnel whose centre lies outside the boundary) |
| `adm2_pcode`, `adm2_name` | Sub-district | all but the same row |
| `adm3_*`, `adm4_*` | Lower administrative levels; not available for Lesotho | 0% |
| `name_latin` | Name converted to Latin script, taking `name_en` when present | 34.3% |
| `geometry_type` | `Point`, `Polygon` or `LineString` | 100% |
| `longitude`, `latitude` | Centroid in WGS84 degrees, 7 decimals | 100% |
| `geometry_wkt` | Full geometry as WKT | 100% |

### D5. Columns of `processed_poi.csv`

The role says how a column may be used: `id` and `info` for joining and reading results, `label` for targets,
`feature_loc` and `feature_attr` as model inputs, `graph` for building edges, `split` for cross-validation.

| Column | Role | Meaning |
|---|---|---|
| `node_id` | id | Row number 0 to 6,082; the graph node index |
| `osm_id` | id | OSM id of the representative feature |
| `osm_ids` | info | Every OSM id merged into the node, separated by `;` |
| `n_units` | info | Number of OSM features merged into the node. Not a feature (depends on the label) |
| `name` | info | Name of the node, if any. Not a feature (leaks the label) |
| `tag_key`, `tag_value` | info | The primary tag the label came from, e.g. `amenity`, `school`. Not features |
| `lon`, `lat` | info | Node position in WGS84 degrees. For a merged node, the mean of its members, so not the raw centroid |
| `nearest_town` | info | Name of the nearest data-derived town (a sub-district name, see D1 "Town") |
| `category` | label | One of the 8 categories, or `Uncategorized` |
| `label` | label | Category as an integer 0 to 7; −1 = unlabelled (D2) |
| `is_labelled` | label | 1 if `label` ≥ 0 |
| `subcategory` | label | One of 36 finer types (D3); a possible secondary target |
| `is_business` | label | 1 for the business subcategories |
| `x_km`, `y_km` | feature_loc, graph | UTM 35S easting and northing, km |
| `dist_town_km` | feature_loc | Distance to the nearest town centre, km |
| `dist_maseru_km` | feature_loc | Distance to the Maseru town centre, km |
| `is_urban` | feature_loc | 1 if at least 50 other nodes lie within 2 km |
| `density_500m`, `density_2km`, `density_10km` | feature_loc | Other nodes within 500 m, 2 km and 10 km |
| `nn1_dist_m` | feature_loc | Distance to the nearest other node, m (0 for the two nodes sharing a position) |
| `knn8_mean_dist_m` | feature_loc | Mean distance to the 8 nearest nodes, m |
| `latrines_1km` | feature_loc | Household latrines within 1 km. Concentrated in Mafeteng because of one mapping effort |
| `has_name` | feature_attr | 1 if the node is named |
| `has_address` | feature_attr | 1 if any address field is filled |
| `has_hours` | feature_attr | 1 if opening hours are tagged |
| `has_footprint` | feature_attr | 1 if a member was mapped as an outline (a building or a site such as a car park) |
| `log_footprint_m2` | feature_attr | `log1p` of the largest outline area in m²; 0 without an outline |
| `is_point` | feature_attr | 1 if a member was mapped as a point. A node can be both point and outline (738 nodes) |
| `quality_score` | feature_attr | Mean of `has_name`, `has_address`, `has_hours`, `has_footprint`: 0, 0.25, 0.5, 0.75 or 1 |
| `osm_id_rank` | feature_attr | Percentile of the smallest OSM id among the members; a bias covariate (D1) |
| `adm1_pcode`, `adm1_name`, `adm1_idx` | graph | District code, name and index 0 to 9 (D2) |
| `adm2_pcode`, `adm2_name`, `adm2_idx` | graph | Sub-district code, name and index 0 to 77 |
| `fold` | split | Spatial CV fold, 0 to 4 |
| `block_id` | split | The 5 km block the node falls in (D2) |
| `fold_boundary` | split | 1 if a node of another fold lies within 700 m |

### D6. Columns of `poi_engineered.csv`

The first 26 columns are carried over unchanged from `processed_poi.csv` and mean the same as in D5: `node_id`,
`osm_id`, `osm_ids`, `n_units`, `name`, `tag_key`, `tag_value`, `lon`, `lat`, `nearest_town`, `category`, `label`,
`is_labelled`, `subcategory`, `is_business`, `x_km`, `y_km`, `adm1_pcode`, `adm1_name`, `adm2_pcode`, `adm2_name`,
`adm1_idx`, `adm2_idx`, `fold`, `block_id`, `fold_boundary`. The other 66 columns are the model features.

**Reading a feature name**

| Part | Meaning |
|---|---|
| `loc_`, `adm_`, `struct_`, `attr_`, `ctx_` | Group: location, administrative, graph structure, the node's own mapping attributes, the surrounding area's mapping attributes |
| `log_` | Stored as `log1p` of the value (D1) |
| `_m`, `_km`, `_m2` | Unit of the value before the log: metres, kilometres, square metres |
| `_100m` … `_20km` | Search radius |
| `_k8`, `_k32`, `knn8`, `knn16` | Computed over the 8, 16 or 32 nearest other nodes; the `struct_knn8_*` columns on the k = 8 graph |
| `has_`, `is_` | Binary, 0 or 1 |
| `_share` | Fraction from 0 to 1 |
| `_pct_` | Percentile rank from 0 to 1 |
| `_vs_` | The node's value minus the mean of its surroundings |

**Features**

| Column(s) | Meaning |
|---|---|
| `loc_x_km`, `loc_y_km` | UTM 35S easting and northing, km |
| `loc_log_dist_town_km` | Distance to the nearest town centre, km |
| `loc_dist_maseru_km` | Distance to the Maseru town centre, km (not logged: already evenly spread) |
| `loc_is_urban` | 1 if at least 50 other nodes lie within 2 km |
| `loc_log_density_100m`, `_250m`, `_500m`, `_1km`, `_2km`, `_5km`, `_10km`, `_20km` | Other nodes within each radius |
| `loc_log_nn1_dist_m` | Distance to the nearest other node, m |
| `loc_log_knn8_mean_dist_m` | Mean distance to the 8 nearest nodes, m |
| `loc_log_knn16_dist_m` | Distance to the 16th nearest node, m |
| `loc_log_latrines_1km` | Household latrines within 1 km (mapping-biased to Mafeteng) |
| `loc_shape_linearity_k8`, `_k32` | Linearity of the nearest nodes' positions (D1) |
| `loc_shape_offset_k8`, `_k32` | Offset from the nearest nodes' centre (D1) |
| `loc_shape_gap_k8`, `_k32` | Widest empty direction among the nearest nodes (D1) |
| `loc_log_town_size` | Size of the nearest town: nodes within 5 km of its centre |
| `loc_log_town_rel_dist` | Distance to the nearest town centre divided by that town's radius. The ratio is ≤ 1 inside the town, i.e. a stored value ≤ 0.69 |
| `loc_log_dist_town2_km` | Distance to the second-nearest town centre, km |
| `loc_log_town_access_5km` | Gravity index of access to all towns, β = 5 km (D1) |
| `adm_log_adm2_nodes`, `adm_log_adm1_nodes` | Number of nodes in the node's sub-district and district |
| `adm_adm2_urban_share` | Share of the sub-district's nodes that are urban |
| `adm_density_pct_in_adm2` | Percentile of the node's 2 km density within its sub-district (1 = the busiest spot) |
| `adm_log_dist_adm2_centre_km` | Distance to the mean position of the sub-district's nodes, km |
| `adm_district_LSA` … `adm_district_LSK` | One 0/1 indicator per district (D2) |
| `struct_knn8_in_degree` | In-degree on the k = 8 graph (D1) |
| `struct_knn8_reciprocity` | Reciprocity on the k = 8 graph (D1) |
| `struct_knn8_clustering` | Clustering coefficient on the k = 8 graph (D1) |
| `struct_rwse_02`, `_03`, `_06`, `_11` | Random-walk return probability after 2, 3, 6 and 11 steps (D1) |
| `attr_has_name`, `attr_has_address`, `attr_has_hours`, `attr_has_footprint`, `attr_is_point`, `attr_quality_score`, `attr_log_footprint_m2` | The EDA's mapping attributes, unchanged (D5) |
| `attr_osm_id_rank` | Percentile of the smallest OSM id; mixes mapping date with point-versus-outline (D1). A bias covariate |
| `ctx_named_share_k32`, `ctx_footprint_share_k32`, `ctx_point_share_k32`, `ctx_address_share_k32`, `ctx_hours_share_k32` | Share of the 32 nearest nodes that are named, have an outline, were mapped as a point, have an address, have opening hours |
| `ctx_osm_id_rank_mean_k32`, `ctx_osm_id_rank_std_k32` | Mean and spread of `osm_id_rank` over the 32 nearest nodes |
| `ctx_osm_id_rank_vs_k32` | The node's `osm_id_rank` minus the mean of its 32 nearest |
| `ctx_footprint_vs_k32` | The node's log outline area minus the mean log outline area of the outlined nodes among its 32 nearest; 0 if the node has no outline |

### D7. Columns of `model_outputs_v2/oof_predictions.csv`

One row per labelled node (5,952). Each prediction comes from the 5-model ensemble of the split that held the node's
fold out.

| Column | Meaning |
|---|---|
| `node_id`, `osm_id`, `name`, `tag_key`, `tag_value`, `lon`, `lat`, `nearest_town`, `adm1_name`, `adm2_name` | Identity and location, as in D5 (for reading only) |
| `category`, `fold` | True category and spatial fold |
| `pred_known`, `pred_site` | Predicted category in known mode and in site mode |
| `p_known_<category>`, `p_site_<category>` | Temperature-scaled probability of each of the 8 categories in each mode; each set of 8 sums to 1 |

### D8. Columns of `model_outputs_v2/unlabelled_predictions.csv`

The 131 unlabelled nodes, with the same columns as D7 except `category` and `fold`. Each node lies outside the
training folds of every split, so its probabilities are averaged over the 5 split ensembles.

### D9. Columns of `model_outputs_v2/market_gaps.csv`

One row per sub-district and category (78 × 8 = 624 rows). Read by `predict_poi.py gaps` and `site`.

| Column | Meaning |
|---|---|
| `adm1_name`, `adm2_name`, `adm2_pcode` | District, sub-district and its P-code |
| `category` | One of the 8 categories |
| `pois` | Labelled nodes in the sub-district |
| `observed` | Nodes of this category mapped in the sub-district |
| `expected` | Sum of the nodes' prior-corrected site-mode probabilities for this category |
| `gap` | `expected − observed`: positive when the locations suggest more of the category than is mapped |
| `z` | `gap` divided by √Σ p(1 − p) (D1) |
| `signal` | `gap`, `cluster` or empty (rules in Stage 5.8; only sub-districts with at least 20 POIs are flagged) |
| `mapping` | `thin` when the sub-district's mean mapping quality is in the lowest quarter, else empty |

### D10. Contents of `gnn_dual_models.pt` and `metrics.json`

`gnn_dual_models.pt` (load with `torch.load(..., weights_only=False)`):

| Key | Content |
|---|---|
| `models` | 25 entries of `{fold, seed, state}`: the test fold of the split, the seed and the model weights |
| `cfg` | The final configuration (layer type, width, depth, dropout, mask rate, label rate, distance scale, ...) |
| `features`, `own_cols`, `binary`, `continuous` | The 66 input columns in model order, the 10 own-node columns, and the binary (kept 0/1) and continuous (standardised) columns |
| `scalers` | Per split, the mean and SD of each continuous column on the non-test folds |
| `cats`, `k` | The category order of the outputs, and k = 8 |
| `temperatures` | Per mode and split, the fitted temperature of each seed's model |
| `reuse`, `reuse_settings`, `seeds` | The label-reuse setting chosen on validation, all settings tried, and the seeds |

`metrics.json` sections: `architecture_comparison`, `selected_architecture`, `tuning`, `selected_config`, `final`
(reuse table, per-fold, pooled and per-class scores in both modes), `ablations`, `ro2_permutation_importance`,
`ro2_neighbour_influence_pp`, `ro2_site_f1_deployment`, `ro3_prior_factors` and `ro3_flagged`.

## Licence

Data © OpenStreetMap contributors, available under the Open Database Licence (ODbL 1.0). All files derived from it
in this folder fall under the same licence and must carry the same attribution.
