"""Domain property classifiers (§3.2.1).

Six classifiers, each matching the model and features described in the paper:
  1. Language    — Naive Bayes (NB) on character 3-gram vectors
  2. Illicitness — Random Forest (RF)
  3. Category    — RF with One-vs-Rest (6 classes)
  4. Template    — NB on CSS / DOM / TF-IDF features
  5. Tracking    — SVM on blacklisted JS function signatures
  6. Attribution — RF on address context features

Classifiers require external labeled data to train (see fit() methods).
After training, call save() / load() to persist models.
"""
from __future__ import annotations

import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.multiclass import OneVsRestClassifier
from sklearn.naive_bayes import ComplementNB
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC

from .features import AddressContext, PageFeatures

CATEGORIES = ["social_media", "marketplace", "pornography", "indexer", "crypto", "other"]


@dataclass
class DomainClassification:
    language: str = "unknown"
    language_confidence: float = 0.0
    is_illicit: bool = False
    illicit_confidence: float = 0.0
    category: str = "other"
    category_confidence: float = 0.0
    template_group: Optional[str] = None
    template_confidence: float = 0.0
    has_tracking: bool = False
    tracking_confidence: float = 0.0
    has_attribution: bool = False
    attribution_confidence: float = 0.0


# ── Language classifier ───────────────────────────────────────────────────────

class LanguageClassifier:
    """Naive Bayes on character 3-grams; supports 50 languages."""

    def __init__(self):
        self.pipeline = Pipeline([
            ("tfidf", TfidfVectorizer(analyzer="char", ngram_range=(3, 3), max_features=10_000)),
            ("clf", ComplementNB(alpha=0.1)),
        ])
        self.trained = False

    def fit(self, texts: List[str], labels: List[str]) -> "LanguageClassifier":
        self.pipeline.fit(texts, labels)
        self.trained = True
        return self

    def predict(self, text: str) -> Tuple[str, float]:
        if not self.trained:
            return "unknown", 0.0
        proba = self.pipeline.predict_proba([text])[0]
        idx = int(np.argmax(proba))
        return self.pipeline.classes_[idx], float(proba[idx])


# ── Illicitness classifier ────────────────────────────────────────────────────

class IllicitnessClassifier:
    """Random Forest on structural + content features."""

    _FEATURE_NAMES = [
        "uses_css", "uses_js", "num_chars", "num_img_tags",
        "num_button_tags", "num_input_tags", "num_external_urls",
        "num_crypto_addresses", "num_blacklisted_js",
    ]

    def __init__(self):
        self.clf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
        self.trained = False

    def _vectorize(self, f: PageFeatures) -> List[float]:
        return [
            float(f.uses_css),
            float(f.uses_js),
            float(f.num_chars),
            float(f.num_img_tags),
            float(f.num_button_tags),
            float(f.num_input_tags),
            float(f.num_external_urls),
            float(f.num_crypto_addresses),
            float(len(f.blacklisted_js_calls)),
        ]

    def fit(self, features: List[PageFeatures], labels: List[bool]) -> "IllicitnessClassifier":
        X = [self._vectorize(f) for f in features]
        self.clf.fit(X, [int(l) for l in labels])
        self.trained = True
        return self

    def predict(self, f: PageFeatures) -> Tuple[bool, float]:
        if not self.trained:
            return False, 0.0
        proba = self.clf.predict_proba([self._vectorize(f)])[0]
        return bool(proba[1] > 0.5), float(proba[1])


# ── Category classifier ───────────────────────────────────────────────────────

class CategoryClassifier:
    """Random Forest with One-vs-Rest for 6-category classification."""

    def __init__(self):
        self.tfidf = TfidfVectorizer(max_features=5_000, ngram_range=(1, 2), sublinear_tf=True)
        self.clf = OneVsRestClassifier(
            RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
        )
        self.le = LabelEncoder()
        self.trained = False

    def fit(self, texts: List[str], labels: List[str]) -> "CategoryClassifier":
        X = self.tfidf.fit_transform(texts)
        y = self.le.fit_transform(labels)
        self.clf.fit(X, y)
        self.trained = True
        return self

    def predict(self, text: str) -> Tuple[str, float]:
        if not self.trained:
            return "other", 0.0
        X = self.tfidf.transform([text])
        proba = self.clf.predict_proba(X)[0]
        idx = int(np.argmax(proba))
        label = self.le.inverse_transform([idx])[0]
        return str(label), float(proba[idx])


# ── Template classifier ───────────────────────────────────────────────────────

class TemplateClassifier:
    """
    Naive Bayes on CSS rule sequences, TF-IDF terms, and DOM tree sequences.
    Predicts which template group (mirror cluster) a domain belongs to.
    """

    def __init__(self):
        self.vec = TfidfVectorizer(max_features=5_000, analyzer="word")
        self.clf = ComplementNB(alpha=0.1)
        self.trained = False

    def _text(self, f: PageFeatures) -> str:
        css = " ".join(f.css_rule_sequences[:50])
        dom = " ".join(f.dom_tree_sequence[:50])
        tfidf = " ".join(f.tfidf_top_terms)
        return f"{css} {dom} {tfidf}"

    def fit(self, features: List[PageFeatures], template_labels: List[str]) -> "TemplateClassifier":
        """template_labels: a domain name or cluster ID per sample."""
        texts = [self._text(f) for f in features]
        X = self.vec.fit_transform(texts)
        self.clf.fit(X, template_labels)
        self.trained = True
        return self

    def predict(self, f: PageFeatures) -> Tuple[str, float]:
        if not self.trained:
            return "unique", 0.0
        X = self.vec.transform([self._text(f)])
        proba = self.clf.predict_proba(X)[0]
        idx = int(np.argmax(proba))
        return str(self.clf.classes_[idx]), float(proba[idx])


# ── Tracking classifier ───────────────────────────────────────────────────────

class TrackingClassifier:
    """SVM on blacklisted JS function presence vectors."""

    def __init__(self):
        self.pipeline = Pipeline([
            ("tfidf", TfidfVectorizer(max_features=2_000, analyzer="word")),
            ("scaler", StandardScaler(with_mean=False)),
            ("clf", SVC(probability=True, kernel="rbf", C=1.0)),
        ])
        self.trained = False

    def _text(self, f: PageFeatures) -> str:
        return " ".join(sorted(f.blacklisted_js_calls))

    def fit(self, features: List[PageFeatures], labels: List[bool]) -> "TrackingClassifier":
        texts = [self._text(f) for f in features]
        self.pipeline.fit(texts, [int(l) for l in labels])
        self.trained = True
        return self

    def predict(self, f: PageFeatures) -> Tuple[bool, float]:
        if not self.trained:
            return f.has_js_tracking, float(f.has_js_tracking)
        proba = self.pipeline.predict_proba([self._text(f)])[0]
        return bool(proba[1] > 0.5), float(proba[1])


# ── Attribution classifier ────────────────────────────────────────────────────

class AttributionClassifier:
    """Random Forest on address context features."""

    def __init__(self):
        self.clf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
        self.trained = False

    def _vectorize(self, ctx: AddressContext) -> List[float]:
        return [
            float(ctx.in_a_tag),
            float(ctx.in_url),
            float(ctx.in_form),
            float(ctx.in_footer),
            float(ctx.in_table),
            float(ctx.in_list),
            float(ctx.in_img),
            float(ctx.near_donate),
            float(ctx.near_example),
        ]

    def fit(self, contexts: List[AddressContext], labels: List[bool]) -> "AttributionClassifier":
        X = [self._vectorize(c) for c in contexts]
        self.clf.fit(X, [int(l) for l in labels])
        self.trained = True
        return self

    def predict(self, ctx: AddressContext) -> Tuple[bool, float]:
        if not self.trained:
            heuristic = ctx.near_donate and not ctx.near_example
            return heuristic, float(heuristic)
        proba = self.clf.predict_proba([self._vectorize(ctx)])[0]
        return bool(proba[1] > 0.5), float(proba[1])


# ── Unified interface ─────────────────────────────────────────────────────────

class DizzyClassifiers:
    """Unified interface to all six domain-property classifiers."""

    def __init__(self):
        self.language = LanguageClassifier()
        self.illicitness = IllicitnessClassifier()
        self.category = CategoryClassifier()
        self.template = TemplateClassifier()
        self.tracking = TrackingClassifier()
        self.attribution = AttributionClassifier()

    def classify(self, text: str, features: PageFeatures) -> DomainClassification:
        result = DomainClassification()

        result.language, result.language_confidence = self.language.predict(text)
        result.is_illicit, result.illicit_confidence = self.illicitness.predict(features)
        result.category, result.category_confidence = self.category.predict(text)
        result.has_tracking, result.tracking_confidence = self.tracking.predict(features)
        result.template_group, result.template_confidence = self.template.predict(features)

        if features.crypto_addresses:
            # Attribution: any address marked as self-attributed counts
            attr_results = [self.attribution.predict(ctx) for ctx in features.crypto_addresses]
            result.has_attribution = any(v for v, _ in attr_results)
            result.attribution_confidence = max((c for _, c in attr_results), default=0.0)

        return result

    def save(self, directory: str) -> None:
        d = Path(directory)
        d.mkdir(parents=True, exist_ok=True)
        for name in ("language", "illicitness", "category", "template", "tracking", "attribution"):
            clf = getattr(self, name)
            with open(d / f"{name}.pkl", "wb") as fh:
                pickle.dump(clf, fh)

    def load(self, directory: str) -> None:
        d = Path(directory)
        for name in ("language", "illicitness", "category", "template", "tracking", "attribution"):
            path = d / f"{name}.pkl"
            if path.exists():
                with open(path, "rb") as fh:
                    setattr(self, name, pickle.load(fh))
