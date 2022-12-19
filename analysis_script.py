import os
import re

import numpy as np
import pandas as pd
from sklearn.gaussian_process import GaussianProcessClassifier, GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, DotProduct, WhiteKernel
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import MinMaxScaler, StandardScaler


def parse_static_feature_csv(filepath):
    float_converter = lambda x: float(x.replace(",", ".")) if x else np.nan

    dat_df = pd.read_csv(
        filepath,
        sep=";",
        converters={
            "pHa": float_converter,
            "AumPI": float_converter,
            "FLmm": float_converter,
            "BMI": float_converter,
            "ACMPI": float_converter,
        },
    )
    dat_df = dat_df[dat_df["Farbe"] == "blau"]  # Important. Only blue cases are taken.
    print(dat_df.columns)

    ## Meta
    # Pat-Nr -> just the ID
    # Entbindungszeitpunkt -> for reference calculations, separate in NewbornBirthdate and NewbornBirthTime
    # GeburtenNummer, FallNr, SubcaseID

    ## Features for Classification
    # MatAgeYears, int, nothing missing, take
    # Parity, floaty int, 1 missing, take
    # Gravida, floaty int, nothing missing, take
    # GestAgeDays, floaty int, nothing missing, take
    # AmnioticFluidColor, 0/1 floaty int, 25 missing, take
    # spUterineSurgery, bracket numbers 1/2/3/4, 746 missing, take
    # spInfertilityTreatment, 0/1 floaty int, very few 0s, unclear if NaN is 0, 1054 missing, take
    # spLateMiscarriageStillbirth, bracket numbers 1/2, unclear if NaN is 0, 1078 missing, take
    # DM, floaty 0/1, 940 missing, take
    # Hypertonia, 1/2/3/4, unclear if NaN is 0, 1048 missing, take
    # Nicotine, mostly 1, unclear if NaN is 0, 1075 missing, take
    # BMI, float, 19 missing, take
    # FetalWeightG, floaty int, 175 missing, ultrasound, take
    # FetalWeightPc, floaty int, 177 missing, ultrasound, just percentile NO TAKE
    # BPDmm, floaty int, 171 missing, ultrasound, take
    # AUmm, floaty int, 169 missing, ultrasound, take
    # FLmm, floaty int, 169 missing, ultrasound, take
    # AumPI, float, 194 missing, ultrasound, take
    # ACMPI, float, 894 missing, ultrasound, take

    # DuctusPI, float (I believe), 1094 missing, NO TAKE
    # A-Welle, floaty int (I believe), 1094 missing linked to DuctusPI, NO TAKE

    ## Misc
    # Cervix10cmTime, date, 646 missing, take
    # UltrasoundDate, date, 158 missing, NO TAKE

    # AFI, float, 753 missing NO TAKE for now
    # SD, float, 980 missing NO TAKE for now

    ## Outcome
    # BEa, float, 5 missing, TODO contains broken values, NO TAKE
    # pHa, float, nothing missing, take
    # pHv, float, 39 missing, NO TAKE
    # Apgar1min, floaty int, nothing missing, take
    # Apgar5min, floaty int, nothing missing, take
    # Apgar10min, floaty int, nothing missing, take
    # UnplannedSectio, floaty 1, 872 missing, one_hot, take
    # PlannedSectio, floaty 1, 862 missing, one_hot, take
    # SG, floaty 1, 470 missing, one_hot, take
    # DeliveryInRange, object always T, nothing missing, remove NO TAKE

    dat_df["UnplannedSectio"] = dat_df["UnplannedSectio"].replace(np.nan, 0.0)
    dat_df["PlannedSectio"] = dat_df["PlannedSectio"].replace(np.nan, 0.0)
    dat_df["SG"] = dat_df["SG"].replace(np.nan, 0.0)

    dat_df["spInfertilityTreatment"] = dat_df["spInfertilityTreatment"].replace(
        np.nan, 0
    )
    dat_df["DM"] = dat_df["DM"].replace(np.nan, 0)
    dat_df["Nicotine"] = dat_df["Nicotine"].replace(np.nan, 0.0)
    dat_df["AmnioticFluidColor"] = dat_df["AmnioticFluidColor"].replace(np.nan, 0.0)

    # remove bracket numbers
    dat_df["spUterineSurgery"] = [
        re.sub("[(].[)]", "", val) if isinstance(val, str) else np.nan
        for val in dat_df["spUterineSurgery"]
    ]
    dat_df["spUterineSurgery"] = [
        list(map(int, val.split("/"))) if isinstance(val, str) else []
        for val in dat_df["spUterineSurgery"]
    ]
    dat_df["spUterineSurgery1"] = [
        1.0 if (1 in val) else 0.0 for val in dat_df["spUterineSurgery"]
    ]
    dat_df["spUterineSurgery2"] = [
        1.0 if (2 in val) else 0.0 for val in dat_df["spUterineSurgery"]
    ]
    dat_df["spUterineSurgery3"] = [
        1.0 if (3 in val) else 0.0 for val in dat_df["spUterineSurgery"]
    ]
    dat_df["spUterineSurgery4"] = [
        1.0 if (4 in val) else 0.0 for val in dat_df["spUterineSurgery"]
    ]

    # remove bracket numbers
    dat_df["spLateMiscarriageStillbirth"] = [
        re.sub("[(].[)]", "", val) if isinstance(val, str) else np.nan
        for val in dat_df["spLateMiscarriageStillbirth"]
    ]
    dat_df["spLateMiscarriageStillbirth"] = [
        list(map(int, val.split("/"))) if isinstance(val, str) else []
        for val in dat_df["spLateMiscarriageStillbirth"]
    ]
    dat_df["spLateMiscarriageStillbirth1"] = [
        1.0 if (1 in val) else 0.0 for val in dat_df["spLateMiscarriageStillbirth"]
    ]
    dat_df["spLateMiscarriageStillbirth2"] = [
        1.0 if (2 in val) else 0.0 for val in dat_df["spLateMiscarriageStillbirth"]
    ]

    dat_df["Hypertonia"] = [
        list(map(int, val.split("/"))) if isinstance(val, str) else []
        for val in dat_df["Hypertonia"]
    ]
    dat_df["Hypertonia1"] = [1.0 if (1 in val) else 0.0 for val in dat_df["Hypertonia"]]
    dat_df["Hypertonia2"] = [1.0 if (2 in val) else 0.0 for val in dat_df["Hypertonia"]]
    dat_df["Hypertonia3"] = [1.0 if (3 in val) else 0.0 for val in dat_df["Hypertonia"]]
    dat_df["Hypertonia4"] = [1.0 if (4 in val) else 0.0 for val in dat_df["Hypertonia"]]

    #### Added much later to understand reasons for unplanned sectio. ####
    dat_df["Reason1"] = [1.0 if val == "1" else 0.0 for val in dat_df["Reason"]]
    dat_df["Reason2a"] = [1.0 if val == "2a" else 0.0 for val in dat_df["Reason"]]
    dat_df["Reason2b"] = [1.0 if val == "2b" else 0.0 for val in dat_df["Reason"]]
    dat_df["Reason2c"] = [1.0 if val == "2c" else 0.0 for val in dat_df["Reason"]]
    dat_df["Reason2T"] = [1.0 if val == "2T" else 0.0 for val in dat_df["Reason"]]
    dat_df["Reason3"] = [1.0 if val == "3" else 0.0 for val in dat_df["Reason"]]
    dat_df["Reason4"] = [1.0 if val == "4" else 0.0 for val in dat_df["Reason"]]

    def _combine(date_string, time_string):
        if isinstance(date_string, str) and isinstance(time_string, str):
            return date_string + " " + time_string
        else:
            return np.nan

    # Merge decision date and time
    dat_df.loc[:, "OPIndicationTime"] = dat_df.apply(
        lambda row: _combine(row["OP Indication date"], row["OP Indication time "]),
        axis=1,
    )

    dat_df = dat_df.drop(
        [
            "Qualität",  # empty, probably meant for Farbe
            "Farbe",  # only blau
            "NewbornBirthDate",  # redundant with Entbindungszeitpunkt
            "NewbornBirthTime",  # redundant with Entbindungszeitpunkt
            "Kommentar1",  # not used
            "Kommentar2",  # not used
            "DeliveryInRange",  # always True
            "Hypertonia",  # split up
            "spLateMiscarriageStillbirth",  # split up
            "spUterineSurgery",  # split up
            "Reason",  # split up
            "OP Indication date",  # merged
            "OP Indication time ",  # merged
            "FetalWeightPc",  # no take
            "DuctusPI",  # no take
            "A-Welle",  # no take
            "UltrasoundDate",  # no take
            "AFI",  # no take
            "SD",  # no take
            "BEa",  # no take
            "pHv",  # no take
        ],
        axis=1,
    )

    return dat_df


def process_static_features(dat_df):
    # TODO Remove magic numbers.
    dat_df.loc[dat_df["BPDmm"] >= 120, "BPDmm"] = np.nan
    dat_df.loc[dat_df["AUmm"] >= 500, "AUmm"] = np.nan
    dat_df.loc[dat_df["AumPI"] >= 5, "AumPI"] = np.nan
    dat_df = dat_df.copy(deep=True)

    # Median imputation for now
    dat_df = dat_df.fillna(dat_df.median(numeric_only=True))
    return dat_df


# Run single regression and print coefficients
# No quality assesment (MSE or similar) whatsoever is performed
def run_regression(features, targets, test_train_split=0.8):
    assert len(features) == len(targets)
    # weights = np.array([3.0 if lab == 1.0 else 1.0 for lab in targets])

    shuffled = np.random.permutation(len(features))

    features = features[shuffled]  # Comment for random labels
    targets = targets[shuffled]
    # weights = weights[shuffle]

    # Simple regressors.
    regressor = LinearRegression()

    split = int(test_train_split * len(features))
    train_X, test_X = features[:split], features[split:]
    train_y, test_y = targets[:split], targets[split:]

    scaler_X = StandardScaler()
    scaler_y = StandardScaler()
    train_X = scaler_X.fit_transform(train_X)
    train_y = scaler_y.fit_transform(train_y.reshape(-1, 1))
    test_X = scaler_X.transform(test_X)
    test_y = scaler_y.transform(test_y.reshape(-1, 1))

    regressor.fit(train_X, train_y)

    print("Linear regression coefficients")
    print(regressor.coef_)


# Run several classifications and print average auroc.
def run_classification(features, labels, test_train_split=0.8, iterations=100):

    assert len(features) == len(labels)
    print(f"Imbalance: {1 - np.sum(labels) / len(labels)}")
    # weights = np.array([3.0 if lab == 1.0 else 1.0 for lab in labels])

    # scaler = StandardScaler()  # necessary to compare weight size
    scaler = MinMaxScaler()  # works better with the 0/1 columns

    test_auc_ls = []
    train_auc_ls = []
    score_ls = []
    label_ls = []

    for i in range(iterations):

        shuffled = np.random.permutation(len(features))

        features = features[shuffled]  # Comment for random labels
        labels = labels[shuffled]
        # weights = weights[shuffle]

        # Simple classifiers.
        classifier = LogisticRegression()
        # kernel = 1.0 * RBF() + 10.0 * WhiteKernel()
        # classifier = GaussianProcessClassifier(kernel=kernel)

        split = int(test_train_split * len(features))
        train_X, test_X = features[:split], features[split:]
        train_y, test_y = labels[:split], labels[split:]

        train_X = scaler.fit_transform(train_X)
        test_X = scaler.transform(test_X)

        # classifier.fit(train_X, train_y, sample_weight=weights[:split])
        classifier.fit(train_X, train_y)

        test_auc_ls.append(
            roc_auc_score(test_y, classifier.predict_proba(test_X)[:, 1])
        )
        train_auc_ls.append(
            roc_auc_score(train_y, classifier.predict_proba(train_X)[:, 1])
        )
        score_ls.append(classifier.predict_proba(test_X)[:, 1])
        label_ls.append(test_y)

    print(f"Average Test AuROC: {np.mean(test_auc_ls)}")
    print(f"Std Test AuROC: {np.std(test_auc_ls)}")
    print(f"Average Train AuROC: {np.mean(train_auc_ls)}")
    print(f"Std Train AuROC: {np.std(train_auc_ls)}")


if __name__ == "__main__":

    folder_path = "/mnt/volume/ctg_tue/data"
    # csv_name = "020822CSVexport.csv"
    csv_name = "141222CSVExport.csv"

    dat_df = parse_static_feature_csv(os.path.join(folder_path, csv_name))
    dat_df = process_static_features(dat_df)

    # Usually, I subselect cases based on whether I have the corresponding CTG trace.
    # This can make a difference.

    static_features = [
        "MatAgeYears",
        "Parity",
        "Gravida",
        "GestAgeDays",
        "BMI",
        "FetalWeightG",
        "BPDmm",
        "AUmm",
        "FLmm",
        "AumPI",
        "AmnioticFluidColor",
        "Nicotine",
        "DM",
        "spInfertilityTreatment",
        "spUterineSurgery1",
        "spUterineSurgery2",
        "spUterineSurgery3",
        "spUterineSurgery4",
        "spLateMiscarriageStillbirth1",
        "spLateMiscarriageStillbirth2",
        "Hypertonia1",
        "Hypertonia2",
        "Hypertonia3",
        "Hypertonia4",
    ]

    features = dat_df[static_features].to_numpy()
    phas = dat_df["pHa"].to_numpy()
    pha_labels = (phas < 7.2).astype(np.float32)
    delivery_labels = dat_df["SG"].to_numpy()

    print("pHa classification")
    run_classification(features, pha_labels)
    print("")
    print("delivery classification")
    run_classification(features, delivery_labels)
    print("")
    print("pHa regression")
    run_regression(features, phas)
