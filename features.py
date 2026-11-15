"""Build one feature row per borrower from the seven Home Credit tables."""

from pathlib import Path

import numpy as np
import pandas as pd

ID = "SK_ID_CURR"
APPLICATION = [
    "AMT_INCOME_TOTAL",
    "AMT_CREDIT",
    "AMT_ANNUITY",
    "AMT_GOODS_PRICE",
    "EXT_SOURCE_1",
    "EXT_SOURCE_2",
    "EXT_SOURCE_3",
    "DAYS_LAST_PHONE_CHANGE",
    "DAYS_ID_PUBLISH",
    "DAYS_REGISTRATION",
    "AMT_REQ_CREDIT_BUREAU_YEAR",
    "AMT_REQ_CREDIT_BUREAU_QRT",
    "AMT_REQ_CREDIT_BUREAU_MON",
    "OBS_30_CNT_SOCIAL_CIRCLE",
    "DEF_30_CNT_SOCIAL_CIRCLE",
    "OBS_60_CNT_SOCIAL_CIRCLE",
    "DEF_60_CNT_SOCIAL_CIRCLE",
]


def read_table(directory, table, columns):
    """Accept original competition CSVs or the corresponding Parquet shards."""
    csv = Path(directory) / f"{table}.csv"
    if csv.is_file():
        return pd.read_csv(csv, usecols=columns)
    folder = "application_train_dated" if table == "application_train" else table
    files = sorted((Path(directory) / folder).glob("*.parquet"))
    if not files:
        raise FileNotFoundError(f"Missing {table} under {directory}. See the README.")
    return pd.concat(
        (pd.read_parquet(f, columns=columns) for f in files), ignore_index=True
    )


def divide(numerator, denominator):
    return numerator.div(denominator.replace(0, np.nan))


def aggregate(frame, columns, prefix):
    group = frame.groupby(ID, sort=False)
    result = group[columns].agg(["min", "max", "mean", "std"])
    result.columns = [f"{prefix}_{column}_{stat}" for column, stat in result.columns]
    result[f"{prefix}_COUNT"] = group.size()
    return result.astype("float32")


def recent_means(frame, mask, columns, prefix):
    result = frame.loc[mask].groupby(ID, sort=False)[columns].mean().astype("float32")
    result.columns = [prefix + "_" + c for c in result.columns]
    return result


def bureau_features(directory):
    fields = [
        "DAYS_CREDIT",
        "DAYS_CREDIT_ENDDATE",
        "DAYS_ENDDATE_FACT",
        "CREDIT_DAY_OVERDUE",
        "AMT_CREDIT_MAX_OVERDUE",
        "AMT_CREDIT_SUM",
        "AMT_CREDIT_SUM_DEBT",
        "AMT_CREDIT_SUM_OVERDUE",
        "AMT_ANNUITY",
        "CNT_CREDIT_PROLONG",
        "DAYS_CREDIT_UPDATE",
    ]
    bureau = read_table(
        directory, "bureau", [ID, "SK_ID_BUREAU", "CREDIT_ACTIVE", *fields]
    )
    for column in ["DAYS_CREDIT_ENDDATE", "DAYS_ENDDATE_FACT"]:
        bureau[column] = bureau[column].mask(bureau[column].abs() > 40000)
    bureau["DEBT_RATIO"] = divide(bureau.AMT_CREDIT_SUM_DEBT, bureau.AMT_CREDIT_SUM)
    bureau["IS_ACTIVE"] = bureau.CREDIT_ACTIVE.eq("Active").astype("float32")
    features = aggregate(bureau, [*fields, "DEBT_RATIO", "IS_ACTIVE"], "BUREAU")
    features = features.join(
        recent_means(
            bureau,
            bureau.DAYS_CREDIT >= -365,
            ["DEBT_RATIO", "CREDIT_DAY_OVERDUE", "IS_ACTIVE"],
            "BUREAU_YEAR",
        )
    )
    balances = read_table(
        directory, "bureau_balance", ["SK_ID_BUREAU", "MONTHS_BALANCE", "STATUS"]
    )
    balances["LATE"] = balances.STATUS.isin(["1", "2", "3", "4", "5"]).astype("float32")
    balances["RECENT_LATE"] = balances.LATE.where(balances.MONTHS_BALANCE >= -12)
    loans = balances.groupby("SK_ID_BUREAU", sort=False).agg(
        LATE_MONTH_SHARE=("LATE", "mean"),
        RECENT_LATE_SHARE=("RECENT_LATE", "mean"),
        MONTHS_RECORDED=("MONTHS_BALANCE", "size"),
        EARLIEST_MONTH=("MONTHS_BALANCE", "min"),
    )
    del balances
    columns = list(loans.columns)
    loans = bureau[[ID, "SK_ID_BUREAU"]].join(loans, on="SK_ID_BUREAU", how="inner")
    return features.join(aggregate(loans, columns, "BUREAU_MONTH"))


def previous_features(directory):
    fields = [
        "AMT_ANNUITY",
        "AMT_APPLICATION",
        "AMT_CREDIT",
        "AMT_DOWN_PAYMENT",
        "AMT_GOODS_PRICE",
        "RATE_DOWN_PAYMENT",
        "DAYS_DECISION",
        "CNT_PAYMENT",
        "DAYS_FIRST_DRAWING",
        "DAYS_FIRST_DUE",
        "DAYS_LAST_DUE",
        "DAYS_TERMINATION",
    ]
    prev = read_table(
        directory, "previous_application", [ID, "NAME_CONTRACT_STATUS", *fields]
    )
    for name in fields:
        if name.startswith("DAYS_"):
            prev[name] = prev[name].mask(prev[name].eq(365243))
    prev["APPLICATION_CREDIT_RATIO"] = divide(prev.AMT_APPLICATION, prev.AMT_CREDIT)
    prev["REFUSED"] = prev.NAME_CONTRACT_STATUS.eq("Refused").astype("float32")
    prev["APPROVED"] = prev.NAME_CONTRACT_STATUS.eq("Approved").astype("float32")
    result = aggregate(
        prev, [*fields, "APPLICATION_CREDIT_RATIO", "REFUSED", "APPROVED"], "PREV"
    )
    return result.join(
        recent_means(
            prev,
            prev.DAYS_DECISION >= -365,
            ["AMT_CREDIT", "REFUSED", "APPROVED"],
            "PREV_YEAR",
        )
    )


def installment_features(directory):
    fields = [
        "NUM_INSTALMENT_VERSION",
        "NUM_INSTALMENT_NUMBER",
        "DAYS_INSTALMENT",
        "DAYS_ENTRY_PAYMENT",
        "AMT_INSTALMENT",
        "AMT_PAYMENT",
    ]
    inst = read_table(directory, "installments_payments", [ID, *fields])
    days = inst.DAYS_ENTRY_PAYMENT - inst.DAYS_INSTALMENT
    inst["DAYS_LATE"] = days.clip(lower=0).astype("float32")
    inst["WAS_LATE"] = (days > 0).astype("float32").where(days.notna())
    inst["AMOUNT_REMAINING"] = (inst.AMT_INSTALMENT - inst.AMT_PAYMENT).astype(
        "float32"
    )
    inst["PAID_RATIO"] = divide(inst.AMT_PAYMENT, inst.AMT_INSTALMENT).astype("float32")
    result = aggregate(
        inst,
        [*fields, "DAYS_LATE", "WAS_LATE", "AMOUNT_REMAINING", "PAID_RATIO"],
        "INST",
    )
    return result.join(
        recent_means(
            inst,
            inst.DAYS_INSTALMENT >= -365,
            ["WAS_LATE", "DAYS_LATE", "PAID_RATIO"],
            "INST_YEAR",
        )
    )


def pos_features(directory):
    fields = [
        "MONTHS_BALANCE",
        "CNT_INSTALMENT",
        "CNT_INSTALMENT_FUTURE",
        "SK_DPD",
        "SK_DPD_DEF",
    ]
    pos = read_table(directory, "POS_CASH_balance", [ID, *fields])
    pos["LATE_MONTH"] = (pos.SK_DPD > 0).astype("float32").where(pos.SK_DPD.notna())
    result = aggregate(pos, [*fields, "LATE_MONTH"], "POS")
    return result.join(
        recent_means(
            pos, pos.MONTHS_BALANCE >= -12, ["LATE_MONTH", "SK_DPD"], "POS_YEAR"
        )
    )


def card_features(directory):
    fields = [
        "MONTHS_BALANCE",
        "AMT_BALANCE",
        "AMT_CREDIT_LIMIT_ACTUAL",
        "AMT_DRAWINGS_ATM_CURRENT",
        "AMT_DRAWINGS_CURRENT",
        "AMT_PAYMENT_TOTAL_CURRENT",
        "AMT_INST_MIN_REGULARITY",
        "AMT_TOTAL_RECEIVABLE",
        "CNT_DRAWINGS_CURRENT",
        "SK_DPD",
        "SK_DPD_DEF",
    ]
    card = read_table(directory, "credit_card_balance", [ID, *fields])
    card["UTILIZATION"] = divide(card.AMT_BALANCE, card.AMT_CREDIT_LIMIT_ACTUAL)
    card["PAYMENT_TO_MIN"] = divide(
        card.AMT_PAYMENT_TOTAL_CURRENT, card.AMT_INST_MIN_REGULARITY
    )
    card["LATE_MONTH"] = (card.SK_DPD > 0).astype("float32").where(card.SK_DPD.notna())
    result = aggregate(
        card, [*fields, "UTILIZATION", "PAYMENT_TO_MIN", "LATE_MONTH"], "CARD"
    )
    return result.join(
        recent_means(
            card, card.MONTHS_BALANCE >= -12, ["UTILIZATION", "LATE_MONTH"], "CARD_YEAR"
        )
    )


def build_features(directory):
    data = read_table(directory, "application_train", [ID, "TARGET", *APPLICATION])
    app = data.set_index(ID, verify_integrity=True)
    y = app.pop("TARGET").astype("int8")
    app["CREDIT_TO_INCOME"] = divide(app.AMT_CREDIT, app.AMT_INCOME_TOTAL)
    app["ANNUITY_TO_INCOME"] = divide(app.AMT_ANNUITY, app.AMT_INCOME_TOTAL)
    app["CREDIT_TO_GOODS"] = divide(app.AMT_CREDIT, app.AMT_GOODS_PRICE)
    app["ANNUITY_TO_CREDIT"] = divide(app.AMT_ANNUITY, app.AMT_CREDIT)
    app = app.astype("float32")
    for make in (
        bureau_features,
        previous_features,
        installment_features,
        pos_features,
        card_features,
    ):
        history = make(directory)
        app = app.join(history, how="left", validate="one_to_one")
        print(f"{make.__name__}: {len(history):,} applicant summaries", flush=True)
    return app.replace([np.inf, -np.inf], np.nan), y
