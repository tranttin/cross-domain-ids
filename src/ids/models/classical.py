from sklearn.pipeline import make_pipeline
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier


def build_legacy_multiclass_models(k: int = 30, seed: int = 42):
    """Exact constructors from the original CSE-CIC-IDS2018 notebook."""
    return {
        "SGD": make_pipeline(
            SelectKBest(f_classif, k=k),
            SGDClassifier(
                loss="modified_huber", penalty="l2", max_iter=1000, tol=1e-3,
                n_jobs=-1, early_stopping=True, random_state=seed,
            ),
        ),
        "LogReg": make_pipeline(
            SelectKBest(f_classif, k=k),
            LogisticRegression(
                solver="saga", multi_class="multinomial", max_iter=500,
                n_jobs=-1, random_state=seed,
            ),
        ),
        "NaiveBayes": make_pipeline(SelectKBest(f_classif, k=k), GaussianNB()),
        "DecisionTree": make_pipeline(
            SelectKBest(f_classif, k=k),
            DecisionTreeClassifier(
                max_depth=10, min_samples_split=50, ccp_alpha=0.01, random_state=seed,
            ),
        ),
        "RandomForest": make_pipeline(
            SelectKBest(f_classif, k=k),
            RandomForestClassifier(
                n_estimators=50, max_depth=10, n_jobs=-1, random_state=seed,
            ),
        ),
    }


def build_legacy_cross_domain_models(seed: int = 42):
    """Exact two classical models used in the later cross-domain notebook cell."""
    return {
        "LogReg": LogisticRegression(max_iter=2000),
        "RandomForest": RandomForestClassifier(n_estimators=200, n_jobs=-1, random_state=seed),
    }


def build_clean_binary_models(seed: int = 42):
    """Five fixed classical baselines for the journal binary benchmark."""
    return {
        "NaiveBayes": GaussianNB(),
        "SGDClassifier": SGDClassifier(
            loss="log_loss", penalty="l2", max_iter=2000, tol=1e-4,
            class_weight="balanced", random_state=seed,
        ),
        "LogisticRegression": LogisticRegression(
            solver="lbfgs", max_iter=2000, class_weight="balanced", random_state=seed,
        ),
        "DecisionTree": DecisionTreeClassifier(
            max_depth=12, min_samples_split=20, class_weight="balanced", random_state=seed,
        ),
        "RandomForest": RandomForestClassifier(
            n_estimators=200, max_depth=16, n_jobs=-1,
            class_weight="balanced_subsample", random_state=seed,
        ),
    }
