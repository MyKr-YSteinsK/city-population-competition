from __future__ import annotations

import glob
import os
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor, GradientBoostingRegressor, HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import StandardScaler


BASE = Path(__file__).resolve().parent
RAW = BASE / "raw"


def _pick(pattern: str) -> Path:
    files = sorted(RAW.glob(pattern))
    if not files:
        raise FileNotFoundError(f"No file matched {pattern!r} in {RAW}")
    return files[0]


def _numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def _city_filter(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    out["city"] = out["city"].astype(str).str.strip()
    return out[out["city"].str.match(r"^city\d+$", case=False, na=False)].copy()


def _long_income_tables(path: Path) -> list[pd.DataFrame]:
    names = [
        "income_disposable",
        "income_consumption",
        "income_urban_consumption",
        "income_rural_consumption",
        "income_urban_disposable",
        "income_rural_disposable",
    ]
    result: list[pd.DataFrame] = []
    book = pd.ExcelFile(path)
    for idx, sheet in enumerate(book.sheet_names):
        raw = pd.read_excel(path, sheet_name=sheet, header=None)
        raw = raw.dropna(how="all")
        if raw.empty or raw.shape[1] < 2:
            continue
        years = pd.to_numeric(raw.iloc[0, 1:], errors="coerce")
        rows = []
        for _, row in raw.iloc[1:].iterrows():
            city = str(row.iloc[0]).strip()
            if not city.lower().startswith("city"):
                continue
            for col_idx, year in enumerate(years, start=1):
                if pd.isna(year):
                    continue
                rows.append((city, int(year), row.iloc[col_idx]))
        if rows:
            frame = pd.DataFrame(rows, columns=["city", "year", names[idx] if idx < len(names) else f"income_{idx}"])
            frame[frame.columns[-1]] = _numeric(frame[frame.columns[-1]])
            result.append(frame)
    return result


def load_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load the target table and convert all auxiliary workbooks to city-year rows."""
    population = pd.read_excel(_pick("*人口规模*.xlsx"))
    population = population.iloc[:, :4].copy()
    population.columns = ["city", "year", "resident", "registered"]
    population = _city_filter(population)
    population["year"] = _numeric(population["year"]).astype("Int64")
    population["resident"] = _numeric(population["resident"])
    population["registered"] = _numeric(population["registered"])
    population = population.dropna(subset=["city", "year"]).copy()
    population["year"] = population["year"].astype(int)
    population = population.sort_values(["city", "year"]).reset_index(drop=True)

    frames: list[pd.DataFrame] = []

    density = pd.read_excel(_pick("*人口密度*.xlsx")).iloc[:, :3].copy()
    density.columns = ["year", "city", "density"]
    density = density[["city", "year", "density"]]
    density["year"] = _numeric(density["year"])
    density["density"] = _numeric(density["density"])
    frames.append(_city_filter(density).dropna(subset=["year"]))

    urban = pd.read_excel(_pick("*城镇化率*.xlsx")).iloc[:, :3].copy()
    urban.columns = ["city", "year", "urbanization_rate"]
    urban["year"] = _numeric(urban["year"])
    urban["urbanization_rate"] = _numeric(urban["urbanization_rate"])
    frames.append(_city_filter(urban).dropna(subset=["year"]))

    employment_path = _pick("*就业信息*.xlsx")
    unemployment = pd.read_excel(employment_path, sheet_name=0).iloc[:, :3].copy()
    unemployment.columns = ["year", "city", "unemployment_rate"]
    unemployment["year"] = _numeric(unemployment["year"])
    unemployment["unemployment_rate"] = _numeric(unemployment["unemployment_rate"])
    frames.append(_city_filter(unemployment).dropna(subset=["year"]))

    employees = pd.read_excel(employment_path, sheet_name=1).iloc[:, :3].copy()
    employees.columns = ["city", "year", "employees_number"]
    employees["year"] = _numeric(employees["year"])
    employees["employees_number"] = _numeric(employees["employees_number"])
    frames.append(_city_filter(employees).dropna(subset=["year"]))

    sectors = pd.read_excel(employment_path, sheet_name=2).iloc[:, :5].copy()
    sectors.columns = ["city", "year", "primary_employment", "secondary_employment", "tertiary_employment"]
    sectors["year"] = _numeric(sectors["year"])
    for col in sectors.columns[2:]:
        sectors[col] = _numeric(sectors[col])
    frames.append(_city_filter(sectors).dropna(subset=["year"]))

    wage = pd.read_excel(_pick("*工资水平*.xlsx"))
    first = wage.columns[0]
    wage = wage.rename(columns={first: "year_token"})
    wage = wage.melt(id_vars=["year_token"], var_name="city", value_name="average_wage")
    wage["year"] = _numeric(wage["year_token"].astype(str).str.extract(r"(\d{4})")[0])
    wage["average_wage"] = _numeric(wage["average_wage"])
    frames.append(_city_filter(wage[["city", "year", "average_wage"]]).dropna(subset=["year"]))

    age = pd.read_excel(_pick("*年龄结构*.xlsx")).iloc[:, :5].copy()
    age.columns = ["city", "year", "age_0_14", "age_15_64", "age_65_plus"]
    age["year"] = _numeric(age["year"])
    for col in age.columns[2:]:
        age[col] = _numeric(age[col])
    frames.append(_city_filter(age).dropna(subset=["year"]))

    frames.extend(_long_income_tables(_pick("*生活水平*.xlsx")))

    exog = frames[0][["city", "year"]].copy()
    for frame in frames:
        value_cols = [c for c in frame.columns if c not in {"city", "year"}]
        frame = frame[["city", "year"] + value_cols].copy()
        frame["year"] = _numeric(frame["year"])
        frame = frame.dropna(subset=["year"])
        frame = frame.groupby(["city", "year"], as_index=False)[value_cols].mean(numeric_only=True)
        exog = exog.merge(frame, on=["city", "year"], how="outer")
    exog["year"] = _numeric(exog["year"]).astype("Int64")
    exog = exog.dropna(subset=["city", "year"]).copy()
    exog["year"] = exog["year"].astype(int)
    exog = exog.sort_values(["city", "year"]).reset_index(drop=True)
    return population, exog


def repair_obvious_target_anomalies(population: pd.DataFrame) -> pd.DataFrame:
    """Repair only an isolated terminal collapse; retain genuine level shifts."""
    out = population.copy()
    mask = (out["city"].eq("city40") & out["year"].eq(2021))
    # The value 65 is an isolated terminal collapse against 255-271 in the
    # preceding five years.  Use the latest stable level, not a new trend.
    if mask.any():
        prior = out.loc[out["city"].eq("city40") & out["year"].between(2017, 2020), "registered"].dropna()
        if len(prior) >= 2 and float(prior.median()) / max(float(out.loc[mask, "registered"].iloc[0]), 1.0) > 2.0:
            out.loc[mask, "registered"] = float(prior.tail(2).median())
    return out


def _series_maps(population: pd.DataFrame, exog: pd.DataFrame):
    target_maps: dict[tuple[str, str], pd.DataFrame] = {}
    for city, group in population.groupby("city"):
        for col in ["resident", "registered"]:
            x = group[["year", col]].dropna().sort_values("year")
            target_maps[(city, col)] = x
    exog_maps: dict[tuple[str, str], pd.DataFrame] = {}
    for city, group in exog.groupby("city"):
        for col in [c for c in exog.columns if c not in {"city", "year"}]:
            x = group[["year", col]].dropna().sort_values("year")
            exog_maps[(city, col)] = x
    return target_maps, exog_maps


def _asof(mapping: dict, key: tuple[str, str], year: int) -> float:
    frame = mapping.get(key)
    if frame is None or frame.empty:
        return np.nan
    years = frame["year"].to_numpy()
    idx = np.searchsorted(years, year, side="right") - 1
    if idx < 0:
        return np.nan
    return float(frame.iloc[idx, 1])


def _exact(mapping: dict, key: tuple[str, str], year: int) -> float:
    frame = mapping.get(key)
    if frame is None or frame.empty:
        return np.nan
    hit = frame[frame["year"].eq(year)]
    return float(hit.iloc[-1, 1]) if not hit.empty else np.nan


def _history(mapping: dict, key: tuple[str, str], cutoff: int) -> pd.DataFrame:
    frame = mapping.get(key)
    if frame is None:
        return pd.DataFrame(columns=["year", key[1]])
    return frame[frame["year"] <= cutoff].copy()


def _slope(history: pd.DataFrame, value_col: str, k: int, statistic: str = "median") -> float:
    x = history[["year", value_col]].dropna().sort_values("year").tail(k + 1)
    if len(x) < 2:
        return 0.0
    slopes = np.diff(x[value_col].to_numpy(dtype=float)) / np.diff(x["year"].to_numpy(dtype=float))
    if statistic == "mean":
        return float(np.mean(slopes))
    return float(np.median(slopes))


def _linear_forecast(history: pd.DataFrame, value_col: str, cutoff: int, horizon: int, k: int) -> float:
    x = history[["year", value_col]].dropna().sort_values("year").tail(k)
    if x.empty:
        return np.nan
    if len(x) < 2:
        return float(x[value_col].iloc[-1])
    return float(np.polyval(np.polyfit(x["year"], x[value_col], 1), cutoff + horizon))


def _area_ratio(target_maps, exog_maps, city: str, cutoff: int, k: int = 1) -> float:
    resident = _history(target_maps, (city, "resident"), cutoff)
    density = _history(exog_maps, (city, "density"), cutoff)
    if resident.empty or density.empty:
        return np.nan
    x = resident.merge(density, on="year", how="inner").dropna()
    x = x[(x["density"] > 0) & (x["resident"] > 0)].sort_values("year").tail(k)
    if x.empty:
        return np.nan
    return float((x["resident"] / x["density"]).median())


def local_candidates(population, target_maps, exog_maps, city: str, target: str, cutoff: int, horizon: int = 2) -> dict[str, float]:
    history = _history(target_maps, (city, target), cutoff)
    if history.empty:
        return {}
    last = float(history[target].iloc[-1])
    result = {
        "local_last": last,
        "local_slope3": last + horizon * _slope(history, target, 3, "median"),
        "local_slope5": last + horizon * _slope(history, target, 5, "median"),
        "local_mean_slope5": last + horizon * _slope(history, target, 5, "mean"),
        "local_linear3": _linear_forecast(history, target, cutoff, horizon, 3),
        "local_linear5": _linear_forecast(history, target, cutoff, horizon, 5),
    }
    # This candidate is deliberately limited to an actually observed density
    # at cutoff+1; carrying an older density forward would falsely imply a
    # 2022 observation for cities whose density table stops earlier.
    density_next = _exact(exog_maps, (city, "density"), cutoff + 1)
    area = _area_ratio(target_maps, exog_maps, city, cutoff, k=1)
    if pd.notna(density_next) and pd.notna(area) and target == "resident":
        # Density at cutoff+1 gives a useful intermediate level; one recent
        # population increment carries it one more year to the 2023 target.
        intermediate = area * density_next
        result["density_hold"] = intermediate
        result["density_plus_slope5"] = intermediate + _slope(history, target, 5, "median")
    return result


def build_features(population: pd.DataFrame, exog: pd.DataFrame, target_maps, exog_maps, years: list[int]) -> pd.DataFrame:
    exog_cols = [c for c in exog.columns if c not in {"city", "year"}]
    cities = sorted(population["city"].dropna().unique(), key=lambda x: int(str(x).replace("city", "")))
    rows = []
    for year in years:
        for city in cities:
            row: dict[str, object] = {"city": city, "year": int(year)}
            feature_year = year - 1
            for col in ["resident", "registered"]:
                hist = _history(target_maps, (city, col), year - 2)
                for lag in [2, 3, 4, 5]:
                    row[f"{col}_lag{lag}"] = _asof(target_maps, (city, col), year - lag)
                row[f"{col}_slope3"] = _slope(hist, col, 3, "median")
                row[f"{col}_slope5"] = _slope(hist, col, 5, "median")
                row[f"{col}_mean_slope5"] = _slope(hist, col, 5, "mean")
                row[f"{col}_linear_slope5"] = (
                    np.polyfit(hist["year"].tail(5), hist[col].tail(5), 1)[0]
                    if len(hist) >= 2
                    else 0.0
                )
            for col in exog_cols:
                row[f"exog_{col}"] = _asof(exog_maps, (city, col), feature_year)
            resident_lag = row["resident_lag2"]
            density_now = row.get("exog_density", np.nan)
            area = _area_ratio(target_maps, exog_maps, city, year - 2, k=1)
            row["resident_density_proxy"] = area * density_now if pd.notna(area) and pd.notna(density_now) else np.nan
            if pd.notna(row["resident_lag2"]) and pd.notna(row["registered_lag2"]):
                row["registered_resident_ratio"] = row["registered_lag2"] / max(float(row["resident_lag2"]), 1.0)
            else:
                row["registered_resident_ratio"] = np.nan
            row["year_since_2000"] = year - 2000
            rows.append(row)
    return pd.DataFrame(rows)


def _matrices(train: pd.DataFrame, test: pd.DataFrame, feature_cols: list[str]):
    all_x = pd.concat([train[feature_cols], test[feature_cols]], ignore_index=True)
    all_x = pd.get_dummies(all_x, columns=["city"], dtype=float)
    x_train = all_x.iloc[: len(train)].copy()
    x_test = all_x.iloc[len(train) :].copy()
    imputer = SimpleImputer(strategy="median", add_indicator=True)
    x_train = imputer.fit_transform(x_train)
    x_test = imputer.transform(x_test)
    return x_train, x_test


def _model(name: str):
    if name == "ridge":
        return Ridge(alpha=20.0)
    if name == "extra_trees":
        return ExtraTreesRegressor(n_estimators=350, min_samples_leaf=3, max_features=0.75, random_state=42, n_jobs=-1)
    if name == "random_forest":
        return RandomForestRegressor(n_estimators=350, min_samples_leaf=3, max_features=0.75, random_state=42, n_jobs=-1)
    if name == "hist_gradient":
        return HistGradientBoostingRegressor(max_iter=250, learning_rate=0.04, max_leaf_nodes=15, l2_regularization=5.0, random_state=42)
    if name == "gradient_huber":
        return GradientBoostingRegressor(n_estimators=220, learning_rate=0.03, max_depth=2, loss="huber", random_state=42)
    raise KeyError(name)


def panel_prediction(train: pd.DataFrame, test: pd.DataFrame, target: str, model_name: str) -> np.ndarray:
    feature_cols = [c for c in train.columns if c not in {"resident", "registered", "year"}]
    x_train, x_test = _matrices(train, test, feature_cols)
    if model_name == "ridge":
        scaler = StandardScaler()
        x_train = scaler.fit_transform(x_train)
        x_test = scaler.transform(x_test)
    model = _model(model_name)
    model.fit(x_train, train[target].to_numpy(dtype=float))
    return model.predict(x_test)


def evaluate_and_forecast(population: pd.DataFrame, features: pd.DataFrame, target_maps, exog_maps, target: str):
    model_names = ["ridge", "extra_trees", "random_forest", "hist_gradient", "gradient_huber"]
    method_names = [
        "local_last",
        "local_slope3",
        "local_slope5",
        "local_mean_slope5",
        "local_linear3",
        "local_linear5",
        "density_hold",
        "density_plus_slope5",
    ] + [f"model_{name}" for name in model_names]
    cities = sorted(population["city"].unique(), key=lambda x: int(str(x).replace("city", "")))
    validation_years = [2017, 2018, 2019, 2020, 2021]
    records = []
    for valid_year in validation_years:
        cutoff = valid_year - 2
        train = features[features["year"].le(cutoff)].copy()
        train = train[train[target].notna()].copy()
        test = features[features["year"].eq(valid_year)].copy()
        if train.empty or test.empty:
            continue
        actual = population[["city", "year", target]].copy()
        test = test.merge(actual, on=["city", "year"], how="left", suffixes=("", "_actual"))
        for city in cities:
            row = test[test["city"].eq(city)]
            if row.empty or pd.isna(row[f"{target}_actual"].iloc[0]):
                continue
            local = local_candidates(population, target_maps, exog_maps, city, target, cutoff, horizon=2)
            for name, value in local.items():
                records.append({"city": city, "year": valid_year, "method": name, "pred": value, "actual": float(row[f"{target}_actual"].iloc[0])})
        for model_name in model_names:
            pred = panel_prediction(train, test.drop(columns=[f"{target}_actual"]), target, model_name)
            for city, value in zip(test["city"], pred):
                actual_value = test.loc[test["city"].eq(city), f"{target}_actual"].iloc[0]
                records.append({"city": city, "year": valid_year, "method": f"model_{model_name}", "pred": float(value), "actual": float(actual_value)})
    validation = pd.DataFrame(records)
    metrics = (
        validation.dropna(subset=["pred"])
        .assign(abs_error=lambda x: (x["pred"] - x["actual"]).abs(), sq_error=lambda x: (x["pred"] - x["actual"]) ** 2)
        .groupby("method")
        .agg(n=("abs_error", "size"), mae=("abs_error", "mean"), medae=("abs_error", "median"), rmse=("sq_error", lambda x: float(np.sqrt(x.mean()))))
        .sort_values(["mae", "rmse"])
    )
    city_metrics = (
        validation.dropna(subset=["pred"])
        .assign(abs_error=lambda x: (x["pred"] - x["actual"]).abs())
        .groupby(["city", "method"])
        .agg(n=("abs_error", "size"), mae=("abs_error", "mean"))
        .reset_index()
    )

    cutoff = 2021
    final_test = features[features["year"].eq(2023)].copy()
    train = features[features["year"].le(cutoff)].copy()
    train = train[train[target].notna()].copy()
    final_preds: dict[tuple[str, str], float] = {}
    for city in cities:
        local = local_candidates(population, target_maps, exog_maps, city, target, cutoff, horizon=2)
        for name, value in local.items():
            final_preds[(city, name)] = value
    final_test_for_model = final_test.copy()
    for model_name in model_names:
        pred = panel_prediction(train, final_test_for_model, target, model_name)
        for city, value in zip(final_test["city"], pred):
            final_preds[(city, f"model_{model_name}")] = float(value)

    chosen = []
    for city in cities:
        cm = city_metrics[city_metrics["city"].eq(city)].copy()
        # Require at least two validation points; otherwise use the best global method.
        cm = cm[cm["n"].ge(2)].sort_values(["mae", "method"])
        best_method = cm["method"].iloc[0] if not cm.empty else metrics.index[0]
        value = final_preds.get((city, best_method), np.nan)
        if pd.isna(value):
            for fallback in metrics.index:
                if pd.notna(final_preds.get((city, fallback), np.nan)):
                    best_method = fallback
                    value = final_preds[(city, fallback)]
                    break
        chosen.append({"city": city, "year": 2023, "pred": max(float(value), 0.0), "method": best_method})
    chosen = pd.DataFrame(chosen)
    return chosen, metrics, city_metrics, validation


def main() -> None:
    population, exog = load_data()
    population = repair_obvious_target_anomalies(population)
    target_maps, exog_maps = _series_maps(population, exog)
    all_years = sorted(set(population["year"].astype(int)) | {2023})
    features = build_features(population, exog, target_maps, exog_maps, all_years)
    for target in ["resident", "registered"]:
        features[target] = population.set_index(["city", "year"])[target].reindex(pd.MultiIndex.from_frame(features[["city", "year"]])).to_numpy()
        chosen, metrics, city_metrics, validation = evaluate_and_forecast(population, features, target_maps, exog_maps, target)
        out = chosen.rename(columns={"city": "city_id"})[["city_id", "year", "pred"]].copy()
        out["pred"] = out["pred"].round(2)
        filename = "submission_resident.csv" if target == "resident" else "submission_registered.csv"
        out.to_csv(BASE / filename, index=False, encoding="utf-8-sig")
        print(f"\n=== {target} ===")
        print(metrics.head(10).round(3).to_string())
        print("chosen methods:")
        print(chosen.to_string(index=False))
        print("predictions:")
        print(out.to_string(index=False))


if __name__ == "__main__":
    main()
