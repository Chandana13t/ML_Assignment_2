from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

from src.logger import get_logger
from src.exceptions import OTTProjectException


logger = get_logger("preprocessing", "preprocessing.log")


def preprocess_data(df):

    try:

        logger.info("Starting preprocessing pipeline")

        df = df.copy()

        # target
        logger.info("Extracting target column: drop_off")
        y = df["drop_off"]

        # drop leakage / identifier columns
        drop_cols = [
            "drop_off",
            "show_id",
            "title",
            "drop_off_probability",
            "dataset_version",
            "avg_watch_percentage",
            "cognitive_load",
            "retention_risk"
        ]


        logger.info("Dropping leakage / identifier columns")
        X = df.drop(drop_cols, axis=1, errors="ignore")

        print("Columns in X:")
        print(X.columns)


        # encode categorical features
        logger.info("Encoding categorical columns")

        for col in X.select_dtypes(include="object").columns:
            le = LabelEncoder()
            X[col] = le.fit_transform(X[col])

        # split dataset
        logger.info("Splitting dataset")

        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=0.2,
            stratify=y,
            random_state=42
        )

        # scale features
        logger.info("Scaling features")

        scaler = StandardScaler()

        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        logger.info("Preprocessing completed successfully")

        return X_train, X_test, y_train, y_test, scaler

    except Exception as e:
        logger.error("Error during preprocessing")
        raise OTTProjectException(str(e))