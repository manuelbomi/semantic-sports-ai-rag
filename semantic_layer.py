"""
semantic_layer.py
─────────────────
A semantic layer sits between raw data and consumers (dashboards, ML models,
LLMs) and enforces *consistent definitions* of business concepts.

Without one, "engaged fan" can mean five different things depending on who
you ask. With one, it's defined once and everyone reads from the same source.

This file implements:

  Metric          → a named, versioned business measure with its computation rule
  Feature         → a derived attribute computed from raw fan/event data
  SemanticLayer   → registry of all metrics and features; single lookup point
  FeatureStore    → computes feature values for individual fan records
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Callable, Any


# ─────────────────────────────────────────────────────────────────────────────
# 1. METRIC — a named business measure
#    Think of a dbt metric or a Looker measure: one canonical definition
#    that every dashboard and model references.
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class Metric:
    name: str
    description: str
    compute: Callable[[dict], Any]   # takes a fan record, returns the metric value
    version: str = "1.0"
    tags: list[str] = field(default_factory=list)

    def evaluate(self, record: dict) -> Any:
        return self.compute(record)

    def __repr__(self):
        return f"Metric(name={self.name!r}, version={self.version})"


# ─────────────────────────────────────────────────────────────────────────────
# 2. FEATURE — a derived attribute for ML or analytics
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class Feature:
    name: str
    description: str
    compute: Callable[[dict], Any]
    dtype: str = "float"       # float, int, bool, str, categorical
    tags: list[str] = field(default_factory=list)

    def evaluate(self, record: dict) -> Any:
        return self.compute(record)


# ─────────────────────────────────────────────────────────────────────────────
# 3. SEMANTIC LAYER REGISTRY
# ─────────────────────────────────────────────────────────────────────────────

class SemanticLayer:
    """
    Central registry for all metrics and features.

    Any system that needs to know "what does engaged_fan mean?"
    asks this registry rather than implementing its own definition.
    """

    def __init__(self, name: str):
        self.name = name
        self._metrics: dict[str, Metric] = {}
        self._features: dict[str, Feature] = {}

    def register_metric(self, m: Metric):
        self._metrics[m.name] = m

    def register_feature(self, f: Feature):
        self._features[f.name] = f

    def get_metric(self, name: str) -> Metric:
        if name not in self._metrics:
            raise KeyError(f"Metric {name!r} not found — did you register it?")
        return self._metrics[name]

    def get_feature(self, name: str) -> Feature:
        if name not in self._features:
            raise KeyError(f"Feature {name!r} not found — did you register it?")
        return self._features[name]

    def compute_all(self, record: dict) -> dict:
        """Run every registered metric and feature against a fan record."""
        result = {}
        for name, metric in self._metrics.items():
            result[f"metric::{name}"] = metric.evaluate(record)
        for name, feature in self._features.items():
            result[f"feature::{name}"] = feature.evaluate(record)
        return result

    def list_metrics(self):
        return list(self._metrics.keys())

    def list_features(self):
        return list(self._features.keys())


# ─────────────────────────────────────────────────────────────────────────────
# 4. BUILD THE NBA FAN SEMANTIC LAYER
# ─────────────────────────────────────────────────────────────────────────────
#
# A fan record looks like this:
# {
#   "fan_id": "fan_0001",
#   "name": "Marcus",
#   "city": "Los Angeles",
#   "follows_team": "Lakers",
#   "events_attended_season": 12,      # total games attended this season
#   "ticket_purchases_90d": 3,         # tickets bought in last 90 days
#   "app_opens_last_7d": 5,            # mobile app opens in last 7 days
#   "merchandise_spend_ytd": 180.0,    # $ spent on merch year to date
#   "days_since_last_activity": 4,     # days since any platform interaction
#   "preferred_contact": "email",
#   "account_age_days": 820,
# }

nba_layer = SemanticLayer("nba_fan_intelligence")

# ── Metrics ──────────────────────────────────────────────────────────────────

nba_layer.register_metric(Metric(
    name="engaged_fan",
    description=(
        "True if the fan is actively engaged this season. "
        "Engagement requires meeting at least ONE of: attended 5+ events, "
        "bought 2+ tickets in last 90 days, or opened the app this week."
    ),
    compute=lambda r: (
        r.get("events_attended_season", 0) >= 5
        or r.get("ticket_purchases_90d", 0) >= 2
        or r.get("app_opens_last_7d", 0) >= 1
    ),
    tags=["engagement", "retention"],
))

nba_layer.register_metric(Metric(
    name="loyalty_tier",
    description=(
        "Fan loyalty tier based on season attendance. "
        "Platinum: 10+, Gold: 5-9, Silver: 2-4, Bronze: 0-1."
    ),
    compute=lambda r: (
        "Platinum" if r.get("events_attended_season", 0) >= 10 else
        "Gold"     if r.get("events_attended_season", 0) >= 5  else
        "Silver"   if r.get("events_attended_season", 0) >= 2  else
        "Bronze"
    ),
    tags=["loyalty", "rewards"],
))

nba_layer.register_metric(Metric(
    name="at_churn_risk",
    description=(
        "True if the fan shows churn signals: "
        "no activity in 30+ days AND no ticket purchase in 90 days."
    ),
    compute=lambda r: (
        r.get("days_since_last_activity", 0) >= 30
        and r.get("ticket_purchases_90d", 0) == 0
    ),
    tags=["churn", "retention"],
))

nba_layer.register_metric(Metric(
    name="high_value_fan",
    description="True if the fan is Platinum tier AND has spent $100+ on merchandise YTD.",
    compute=lambda r: (
        r.get("events_attended_season", 0) >= 10
        and r.get("merchandise_spend_ytd", 0.0) >= 100.0
    ),
    tags=["revenue", "vip"],
))

# ── Features (for ML models) ─────────────────────────────────────────────────

nba_layer.register_feature(Feature(
    name="recency_score",
    description="Inverse of days_since_last_activity, capped at 1.0. Recent → high score.",
    compute=lambda r: min(1.0, 1.0 / max(1, r.get("days_since_last_activity", 1))),
    dtype="float",
    tags=["recency", "rfm"],
))

nba_layer.register_feature(Feature(
    name="frequency_score",
    description="Normalised events attended this season (cap = 20).",
    compute=lambda r: min(1.0, r.get("events_attended_season", 0) / 20.0),
    dtype="float",
    tags=["frequency", "rfm"],
))

nba_layer.register_feature(Feature(
    name="monetary_score",
    description="Normalised merch spend YTD (cap = $500).",
    compute=lambda r: min(1.0, r.get("merchandise_spend_ytd", 0.0) / 500.0),
    dtype="float",
    tags=["monetary", "rfm"],
))

nba_layer.register_feature(Feature(
    name="rfm_score",
    description="Composite RFM score: average of recency, frequency, and monetary scores.",
    compute=lambda r: (
        min(1.0, 1.0 / max(1, r.get("days_since_last_activity", 1))) * (1/3) +
        min(1.0, r.get("events_attended_season", 0) / 20.0)          * (1/3) +
        min(1.0, r.get("merchandise_spend_ytd", 0.0) / 500.0)        * (1/3)
    ),
    dtype="float",
    tags=["rfm", "composite"],
))

nba_layer.register_feature(Feature(
    name="is_local_fan",
    description="True if fan's city matches the team's home city.",
    compute=lambda r: _city_matches_team(r.get("city", ""), r.get("follows_team", "")),
    dtype="bool",
    tags=["geography"],
))


# city ↔ team mapping (separate from the main graph for portability)
_TEAM_CITIES = {
    "Lakers":  "Los Angeles",
    "Celtics": "Boston",
    "Warriors": "San Francisco",
    "Bucks":   "Milwaukee",
    "Nuggets": "Denver",
}

def _city_matches_team(city: str, team: str) -> bool:
    return _TEAM_CITIES.get(team, "").lower() == city.lower()


# ─────────────────────────────────────────────────────────────────────────────
# 5. FEATURE STORE  —  compute + cache features for a batch of fans
# ─────────────────────────────────────────────────────────────────────────────

class FeatureStore:
    """
    In production this would be backed by Redis (online, low-latency) or
    a data warehouse (offline, batch). Here it's just an in-memory dict.

    Key idea: features are computed ONCE per fan per day and cached.
    Downstream systems (models, APIs, LLMs) read from the cache, not from
    raw data, so they all see the same feature values.
    """

    def __init__(self, layer: SemanticLayer):
        self._layer = layer
        self._cache: dict[str, dict] = {}

    def compute_and_store(self, fan_records: list[dict]):
        for record in fan_records:
            fid = record["fan_id"]
            self._cache[fid] = self._layer.compute_all(record)
            self._cache[fid]["_raw"] = record

    def get(self, fan_id: str) -> dict:
        return self._cache.get(fan_id, {})

    def all_fan_ids(self):
        return list(self._cache.keys())


# ─────────────────────────────────────────────────────────────────────────────
# 6. DEMO
# ─────────────────────────────────────────────────────────────────────────────

SAMPLE_FANS = [
    {
        "fan_id": "fan_0001", "name": "Marcus", "city": "Los Angeles",
        "follows_team": "Lakers", "events_attended_season": 12,
        "ticket_purchases_90d": 3, "app_opens_last_7d": 5,
        "merchandise_spend_ytd": 220.0, "days_since_last_activity": 2,
        "account_age_days": 820,
    },
    {
        "fan_id": "fan_0002", "name": "Aisha", "city": "Boston",
        "follows_team": "Celtics", "events_attended_season": 6,
        "ticket_purchases_90d": 2, "app_opens_last_7d": 3,
        "merchandise_spend_ytd": 85.0, "days_since_last_activity": 8,
        "account_age_days": 1240,
    },
    {
        "fan_id": "fan_0003", "name": "Jordan", "city": "San Francisco",
        "follows_team": "Warriors", "events_attended_season": 14,
        "ticket_purchases_90d": 4, "app_opens_last_7d": 7,
        "merchandise_spend_ytd": 410.0, "days_since_last_activity": 1,
        "account_age_days": 690,
    },
    {
        "fan_id": "fan_0004", "name": "Wei", "city": "Los Angeles",
        "follows_team": "Lakers", "events_attended_season": 1,
        "ticket_purchases_90d": 0, "app_opens_last_7d": 0,
        "merchandise_spend_ytd": 0.0, "days_since_last_activity": 45,
        "account_age_days": 120,
    },
]


if __name__ == "__main__":
    print("=" * 60)
    print(f"Semantic Layer: {nba_layer.name}")
    print(f"  Metrics:  {nba_layer.list_metrics()}")
    print(f"  Features: {nba_layer.list_features()}")

    store = FeatureStore(nba_layer)
    store.compute_and_store(SAMPLE_FANS)

    print("\n" + "=" * 60)
    print("FAN PROFILES — computed metrics + features")
    print("=" * 60)

    for fid in store.all_fan_ids():
        values = store.get(fid)
        raw = values.pop("_raw")
        name = raw["name"]
        print(f"\n  {name} ({fid})")
        for key, val in sorted(values.items()):
            prefix, metric_name = key.split("::")
            if isinstance(val, float):
                print(f"    {prefix:8s}  {metric_name:25s} = {val:.4f}")
            else:
                print(f"    {prefix:8s}  {metric_name:25s} = {val}")

    print("\n" + "=" * 60)
    print("CHURN RISK REPORT")
    print("=" * 60)
    store.compute_and_store(SAMPLE_FANS)   # re-compute after we popped _raw
    for fan in SAMPLE_FANS:
        fid = fan["fan_id"]
        vals = store.get(fid)
        if vals.get("metric::at_churn_risk"):
            print(f"  ⚠  {fan['name']} — at churn risk "
                  f"(last active {fan['days_since_last_activity']}d ago, "
                  f"0 purchases in 90d)")
