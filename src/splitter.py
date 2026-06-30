import pandas as pd


def split_time_series(
    featured: pd.DataFrame,
    feature_cols: list[str],
    test_days: int = 90,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series, pd.DataFrame, pd.Series, pd.Timestamp]:
    """
    Split a featured dataframe into train/test sets by time.

    Parameters
    ----------
    featured
        Output of prepare_training_frame — fully engineered + encoded.
    feature_cols
        Feature columns to include in X.
    test_days
        Number of most-recent days to reserve for testing.

    Returns
    -------
    train_df, test_df, X_train, y_train, X_test, y_test, cutoff_date

    Raises
    ------
    ValueError
        If train or test sets are empty or too small.
    """
    cutoff_date = featured["date"].max() - pd.Timedelta(days=test_days)

    train_df = featured[featured["date"] <= cutoff_date].copy()
    test_df  = featured[featured["date"] >  cutoff_date].copy()

    if len(train_df) == 0:
        raise ValueError(
            "Training dataset is empty. Reduce test_days or provide more historical data."
        )
    if len(test_df) == 0:
        raise ValueError(
            "Test dataset is empty. Reduce test_days or provide more historical data."
        )
    if len(train_df) < 30:
        raise ValueError(
            f"Only {len(train_df)} training rows available. At least 30 are recommended."
        )

    X_train = train_df[feature_cols]
    y_train = train_df["sales"]
    X_test  = test_df[feature_cols]
    y_test  = test_df["sales"]

    return train_df, test_df, X_train, y_train, X_test, y_test, cutoff_date
