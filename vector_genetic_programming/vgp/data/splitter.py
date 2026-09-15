"""
Walk-forward splitter — enforces temporal date ordering as a hard
AssertionError and slices data into train / validation / test partitions.

Design principle (D-11): the ordering check is structural, not advisory.
Raising AssertionError (not ValueError) makes it an explicit programming
contract violation — a splitter mis-configuration must not silently produce
meaningless results.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class WalkForwardSplitter:
    """
    Slice data along the time axis into non-overlapping train / val / test sets.

    Accepts both ``pd.DataFrame`` (with ``DatetimeIndex``) and
    ``np.ndarray`` (requires the ``dates`` keyword argument).

    Parameters
    ----------
    None — all configuration is passed per ``split()`` call.
    """

    def split(
        self,
        data,
        train_end: str,
        val_start: str,
        val_end: str,
        test_start: str,
        test_end: str,
        dates: pd.DatetimeIndex | None = None,
    ) -> tuple:
        """
        Split data into (train, val, test) along the time axis.

        Structural assertions
        ---------------------
        * ``val_start`` must be strictly after ``train_end`` — enforced as
          ``AssertionError`` (D-11 requirement).
        * ``test_start`` must be strictly after ``val_end`` — same.

        Parameters
        ----------
        data : pd.DataFrame or np.ndarray
            Input data.  For DataFrames the ``DatetimeIndex`` on axis 0 is
            used directly.  For ndarrays the ``dates`` kwarg is required.
        train_end : str
            Inclusive end date for the training set (ISO format, e.g. "2023-12-31").
        val_start : str
            Inclusive start date for the validation set.
        val_end : str
            Inclusive end date for the validation set.
        test_start : str
            Inclusive start date for the test set.
        test_end : str
            Inclusive end date for the test set. REQUIRED, deliberately: this
            argument used not to exist, so every test slice silently ran from
            ``test_start`` to the end of the panel. Windows that reported a
            2-month OOS period were measured on everything from their start
            date onward, making the walk-forward OOS periods nested rather than
            disjoint. A default of "to the end" is what hid that, so there is
            no default — each caller states the end date it means.
        dates : pd.DatetimeIndex | None
            Required when ``data`` is an ``np.ndarray``.  Ignored for DataFrames.

        Returns
        -------
        tuple
            ``(train_data, val_data, test_data)`` — same type as ``data``.

        Raises
        ------
        AssertionError
            If ``val_start <= train_end``, ``test_start <= val_end``, or
            ``test_end < test_start``.
        ValueError
            If ``data`` is an ``np.ndarray`` and ``dates`` is not provided.
        """
        # ----------------------------------------------------------------
        # Step 1: Structural assertions (D-11).
        # ----------------------------------------------------------------
        assert pd.Timestamp(val_start) > pd.Timestamp(
            train_end
        ), f"val_start ({val_start}) must be strictly after train_end ({train_end})"
        assert pd.Timestamp(test_start) > pd.Timestamp(
            val_end
        ), f"test_start ({test_start}) must be strictly after val_end ({val_end})"
        assert pd.Timestamp(test_end) >= pd.Timestamp(
            test_start
        ), f"test_end ({test_end}) must be on or after test_start ({test_start})"

        # ----------------------------------------------------------------
        # Step 2: DataFrame input — boolean mask on DatetimeIndex.
        # ----------------------------------------------------------------
        if isinstance(data, pd.DataFrame):
            idx = data.index
            train = data.loc[idx <= pd.Timestamp(train_end)]
            val = data.loc[(idx >= pd.Timestamp(val_start)) & (idx <= pd.Timestamp(val_end))]
            test = data.loc[(idx >= pd.Timestamp(test_start)) & (idx <= pd.Timestamp(test_end))]

            logger.info(
                "Split summary — train: %d rows, val: %d rows, test: %d rows",
                len(train),
                len(val),
                len(test),
            )
            return train, val, test

        # ----------------------------------------------------------------
        # Step 3: ndarray input — requires dates kwarg; use np.searchsorted.
        # ----------------------------------------------------------------
        if isinstance(data, np.ndarray):
            if dates is None:
                raise ValueError(
                    "dates kwarg is required when data is np.ndarray. "
                    "Pass dates=engine.dates_ from FeatureEngine.fit_transform()."
                )
            i_train_end = np.searchsorted(dates, pd.Timestamp(train_end), side="right")
            i_val_start = np.searchsorted(dates, pd.Timestamp(val_start), side="left")
            i_val_end = np.searchsorted(dates, pd.Timestamp(val_end), side="right")
            i_test_start = np.searchsorted(dates, pd.Timestamp(test_start), side="left")
            i_test_end = np.searchsorted(dates, pd.Timestamp(test_end), side="right")

            train = data[:i_train_end]
            val = data[i_val_start:i_val_end]
            test = data[i_test_start:i_test_end]

            logger.info(
                "Split summary — train: %d rows, val: %d rows, test: %d rows",
                len(train),
                len(val),
                len(test),
            )
            return train, val, test

        raise TypeError(f"data must be pd.DataFrame or np.ndarray, got {type(data).__name__}")
