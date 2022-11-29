from typing import Tuple, List

import numpy as np
import pandas as pd
import plotly.graph_objs as go
import cufflinks

cufflinks.go_offline()

import scipy.ndimage.measurements as mnts


def plot_historic_with_sets(historic_data, periods):
    """Plot historic load with indication of train/test/validation periods
    Input:
        - historic_data (pd.DataFrame): datetimeindex, columns=['load']
        - periods (dict): keys = spinup, train, test, validate, values = (start, end), values_types = datetime.date

    return
        plotly.Figure"""

    fig = (
        historic_data[["load"]]
        .dropna()
        .iplot(
            asFigure=True,
            layout=dict(
                xaxis=dict(title=""),
                yaxis=dict(title="Belasting [MW?]"),
                margin=dict(b=0, t=0, l=0, r=0),
            ),
        )
    )
    periods_options = dict(spinup="grey", train="blue", validation="green", test="red")
    # Add shapes to indicate different periods
    for name, color in periods_options.items():
        if name not in periods.keys():
            continue

        fig.add_shape(
            type="rect",
            xref="x",
            yref="paper",
            x0=periods[name][0],
            x1=periods[name][1],
            y0=0,
            y1=1,
            fillcolor=color,
            opacity=0.2,
        )
        # Add text
        fig.add_trace(
            go.Scatter(
                x=[periods[name][0] + (periods[name][1] - periods[name][0]) / 2],
                y=[historic_data["load"].max()],
                mode="text",
                text=[name],
                textposition="top center",
            )
        )

    fig.update_layout(dict(showlegend=False))
    return fig


def plot_typical_day(bdf):
    """Plot typical profile for weekday / weekendday
    Note, this does not make much sense for windpower

    input:
        bdf (pd.DataFrame): should contain datetime index and column 'load'"""

    bdf["time"] = bdf.index.time
    bdf["weekday"] = bdf.index.weekday < 5
    week_profile = bdf[bdf.weekday].pivot_table(
        index="time", values="load", aggfunc=["mean", "max", "min"]
    )
    week_profile.columns = ["week_mean", "week_max", "week_min"]
    weekend_profile = bdf[~bdf.weekday].pivot_table(
        index="time", values="load", aggfunc=["mean", "max", "min"]
    )
    weekend_profile.columns = ["weekend_mean", "weekend_max", "weekend_min"]
    profile = week_profile.merge(weekend_profile, left_index=True, right_index=True)
    fig = profile.iplot(
        asFigure=True,
        layout=dict(
            title="Typisch week/weekend profiel",
            xaxis=dict(title="Tijd vd dag [-]"),
            yaxis=dict(title="Belasting [MW]"),
        ),
    )

    # Update traces based on name
    for trace in fig.data:
        if "week" in trace["name"]:
            trace.update(line=dict(color="blue"))
        if "weekend" in trace["name"]:
            trace.update(line=dict(color="green"))
        if "min" in trace["name"] or "max" in trace["name"]:
            trace.update(line=dict(dash="dot"))

    return fig


def plot_percentiles(bdf, limits=None):
    """Generate plot of perfentile forecast with limits
    input:
        - bdf (pd.DataFrame(index=datetime, columns=[load, P..]))
        - limits (list) congestion limits used in calculation

    return:
        - plotly.Figure
    """

    # Generate traces of Percentiles, fill below
    p_plot = go.Figure()
    for i, perc in enumerate(np.sort([x for x in bdf.columns if x[0] == "q"])):
        fill = None if i == 0 else "tonexty"
        p_plot.add_trace(
            go.Scatter(
                x=bdf.index, y=bdf[perc], fill=fill, name=perc, line=dict(width=1)
            )
        )

    # Add historic load
    p_plot.add_trace(
        go.Scatter(
            x=bdf.index, y=bdf.load, name="realised", line=dict(color="red", width=2)
        )
    )

    # add upper and lower limit if not None
    shapes = []
    for lim in limits:
        if lim is not None:
            shapes.append(
                dict(
                    type="line",
                    yref="y",
                    y0=lim,
                    y1=lim,
                    xref="paper",
                    x0=0,
                    x1=1,
                    line=dict(width=1, color="rgba(255,187,0,0.9)"),
                )
            )
    p_plot.update_layout(shapes=shapes)
    p_plot.update_layout(title="Backtest - Prognoses vs Realisatie")

    return p_plot


def calculate_amount_congestion_mitigation(
    forecast_mw: pd.Series, congestion_limit_mw: float
) -> pd.Series:
    """Calculates amount of congestion mitigation.
        This fucntion asumes congestion mitigation is always applied
        when the forecasts exceeds the congestion limit.

    Args:
        forecast_mw:
            pd.Series with the forecast. Values are expected to be in MegaWatt.

        congestion_limit_mw: (float)
            The limit above (in positive case),
                    or below (in a negative case) congestion occurs (in MegaWatt).

    Returns:
        (pd.Series) with the amount of of congestion mitigation for each PTU in MegaWatt.

    """
    # Depending on the sign of the congestion limit apply a different calculation

    if np.sign(congestion_limit_mw) == 1:  # In case of a positive congestion limit
        return np.abs(np.maximum(forecast_mw - congestion_limit_mw, 0))
    else:  # In case of a negative congestion limit
        return np.abs(np.minimum(forecast_mw - congestion_limit_mw, 0))


def calculate_total_congestion_mitigation(
    congestion_analysis_dataframe: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Reports some single value metrics on the amout of congestion mitigation per quantile.

    Args:
        congestion_analysis_dataframe:
            pd.DataFrame with data about the congestion analyis
                this DataFrame (in long format) should contain:
                 - a column with the amount of congestion mitigation in MegaWatts:
                        'amount_congestion_mitigation_mw'
                 - a column with the quantile labels

    Returns:
        pd.DataFrame total_amount_congestion_mitigation_per_quantile_mw :
            Total amount of congestion mitigation specified per quantile

        pd.DataFrame total_congestion_mitigation_time_ptu:
            Total time where congestion mitigation is applied, specified per quantile

        pd.DataFrame average_duration_congestion_mitigation_period_ptu:
            Average duration of a congestion mitigation period, per quantile
    """

    # Determine total amount of congestino mitigation by pivoting into wide format (column per quantile)
    # and agregating by summing all congestion mitigation per quantile
    total_amount_congestion_mitigation_per_quantile_mw = congestion_analysis_dataframe.pivot_table(
        columns="quantile", values="amount_congestion_mitigation_mw", aggfunc=sum
    ).T.rename(
        columns={
            "amount_congestion_mitigation_mw": "total_amount_congestion_mitigation_mw"
        }
    )

    # Determine total congestion mitigation time by pivoting into wide format (column per quantile)
    # and aggrating by counting the total number of PTU's (len)
    total_congestion_mitigation_time_ptu = (
        congestion_analysis_dataframe[
            congestion_analysis_dataframe.amount_congestion_mitigation_mw
            > 0  # We only want to count PTU's that have > 0 congstion mitigation
        ]
        .pivot_table(
            columns="quantile", values="amount_congestion_mitigation_mw", aggfunc=len
        )
        .T
    ).rename(
        columns={
            "amount_congestion_mitigation_mw": "total_congestion_mitigation_time_ptu"
        }
    )

    # Determine congestion periods
    congestion_analysis_dataframe["periods"], n_periods = mnts.label(
        congestion_analysis_dataframe.amount_congestion_mitigation_mw > 0
    )

    # Determine average length of the found congestion mitigation periods
    # and convert into the proper format
    average_duration_congestion_mitigation_period_ptu = (
        congestion_analysis_dataframe[
            congestion_analysis_dataframe.amount_congestion_mitigation_mw > 0
        ][["periods", "quantile"]]
        .value_counts()
        .to_frame()
        .reset_index()
        .set_index("periods")
        .pivot_table(columns="quantile", values=0, aggfunc="mean")
        .T.rename(columns={0: "average_duration_congestion_mitigation_period_ptu"})
    )

    return (
        total_amount_congestion_mitigation_per_quantile_mw,
        total_congestion_mitigation_time_ptu,
        average_duration_congestion_mitigation_period_ptu,
    )


def calculate_effects_congestion_mitigation(
    realised_mw: pd.Series,
    amount_congestion_mitigation_mw: pd.Series,
    congestion_limit_mw: float,
) -> Tuple[pd.Series, pd.Series]:
    """Calculate amount of congestion that is prevented and missed.
    For this a mitigation is applied using the congestion mitigation folowing the forecast.

    Args:
        realised_mw: pd.Series with realised values
        amount_congestion_mitigation_mw: pd.Series with the amount of congestion mitigation per PTU
        congestion_limit_mw: float with the congestion limit that applies

    Returns:
        pd.Series with the congestion per PTU before applying the mitigation
        pd.Series with the congestion per PTU after applying the mitigation

    """
    if np.sign(congestion_limit_mw) == 1:  # In case of a positive limit
        congestion_before_mitigation_mw = np.maximum(
            realised_mw - congestion_limit_mw, 0
        )

        congestion_left_after_mitigation_mw = np.maximum(
            congestion_before_mitigation_mw - amount_congestion_mitigation_mw, 0
        )
    else:  # In case of a negative limit
        congestion_before_mitigation_mw = np.minimum(
            realised_mw - congestion_limit_mw, 0
        )

        congestion_left_after_mitigation_mw = np.minimum(
            congestion_before_mitigation_mw + amount_congestion_mitigation_mw, 0
        )
    return congestion_before_mitigation_mw, congestion_left_after_mitigation_mw


def calculate_reduced_congestion(
    congestion_before_mitigation_mw: pd.Series,
    congestion_after_mitigation_mw: pd.Series,
) -> pd.Series:
    """Calculates how much congestion is reduced after applying the mitigation

    Args:
        congestion_before_mitigation_mw:
            pd.Series with the congestion per PTU before applying the mitigation
        congestion_after_mitigation_mw:
            pd.Series with the congestion per PTU after applying the mitigation

    Returns:
        pd.Series with the absolute amount of reduced congestion per PTU.
    """
    return congestion_before_mitigation_mw.abs() - congestion_after_mitigation_mw.abs()


def calculate_missed_congestion(congestion_after_mitigation_mw: pd.Series) -> pd.Series:
    """Calculates how much congestion is missed after applying the mitigation

    Args:
        congestion_after_mitigation_mw:
            pd.Series with the congestion per PTU after applying the mitigation

    Returns: pd.Series with the absolute amount of missed congestion per PTU.

    """
    return congestion_after_mitigation_mw.abs()


def prepare_congestion_analysis_dataframe(
    forecast_mw: pd.DataFrame, congestion_limit_mw: float
) -> pd.DataFrame:
    """Selects relevant columns and converts the dataframe to a long format
     such that there is one forecast column on which calculations are easily performed

    Args:
        congestion_limit_mw: float with the congestion limit in MegaWatt
        forecast_mw:
            pd.DataFrame with the forecast that is produced by the backtest pipeline

    Returns:
        pd.DataFrame that only contains the information nescesarry for the analysis.
         This Dataframe is returned in long format
    """

    # Select quantile forecast columns
    quantile_columns = [x for x in forecast_mw.columns if x[0] == "q"]

    # Depending on the sign of the congestion limit only take the relevant quantiles
    if np.sign(congestion_limit_mw) == 1:  # In case of negative congestion limit_mw
        relevant_quantiles = [x for x in quantile_columns if float(x[10:]) >= 50]
    else:  # In case of positive congestion limit_mw
        relevant_quantiles = [x for x in quantile_columns if float(x[10:]) <= 50]

    # Slice only the relevant columns
    congestion_analysis_dataframe = forecast_mw[["realised"] + relevant_quantiles]

    # Transform to long format to do the analysis for all quantiles simultanously
    congestion_analysis_dataframe = congestion_analysis_dataframe.reset_index().melt(
        id_vars=["index", "realised"],
        var_name="quantile",
        value_name="forecast",
    )
    return congestion_analysis_dataframe.set_index("index")


def compose_congestion_analysis_report(
    congestion_analysis_dataframe: pd.DataFrame,
    total_amount_congestion_mitigation_per_quantile_mw: pd.DataFrame,
    congestion_mitigation_time_ptu: pd.DataFrame,
    congestion_periods_with_duration: pd.DataFrame,
    PTU: float,
) -> pd.DataFrame:
    """Combines all the results in one dataframe for easy output and visualisation

    Args:
        congestion_analysis_dataframe: pd.DataFrame
        total_amount_congestion_mitigation_per_quantile_mw: pd.DataFrame
        congestion_mitigation_time_ptu: pd.DataFrame
        congestion_periods_with_duration: pd.DataFrame

    Returns: pd.DataFrame with the results

    """
    # Aggregate reduced congestion per quantile
    report_dataframe = congestion_analysis_dataframe.pivot_table(
        columns="quantile", values="reduced_congestion_mw", aggfunc=sum
    ).T
    report_dataframe["reduced_congestion_mwh"] = (
        report_dataframe["reduced_congestion_mw"] * PTU
    )
    report_dataframe = report_dataframe[["reduced_congestion_mwh"]]

    # Aggregate missed congestion per quantile
    report_dataframe["missed_congestion_mwh"] = (
        congestion_analysis_dataframe.pivot_table(
            columns="quantile", values="missed_congestion_mw", aggfunc=sum
        ).T
        * PTU
    )

    # Combine results from othercongestion mitigation analyses
    report_dataframe["total_congestion_mitigation_time_ptu"] = (
        congestion_mitigation_time_ptu["total_congestion_mitigation_time_ptu"] * PTU
    )
    report_dataframe["average_mitigation_duration_ptu"] = (
        congestion_periods_with_duration[
            "average_duration_congestion_mitigation_period_ptu"
        ]
        * PTU
    )

    report_dataframe["total_amount_congestion_mitigation_per_quantile_mwh"] = (
        total_amount_congestion_mitigation_per_quantile_mw[
            "total_amount_congestion_mitigation_mw"
        ]
        * PTU
    )

    # Calculate congestion mitigation effectivity
    report_dataframe["congestion_mitigation_effectivity"] = (
        report_dataframe["reduced_congestion_mwh"]
        / report_dataframe["total_amount_congestion_mitigation_per_quantile_mwh"]
        * 100
    )

    return report_dataframe.round(2)


def report_congestion_per_month(congestion_before_mitigation_mw: pd.Series):
    report = congestion_before_mitigation_mw.to_frame()
    report["month"] = report.index.month
    return (
        report.groupby(by="month")
        .sum()
        .rename(columns={"realised": "agregated congestion before mitigation"})
    )


def congestion_effectivity_analysis(
    forecast_mw: pd.DataFrame, congestion_limits_mw, PTU
):
    """Do congestion_effectivity_analysis

    Args:
        forecast_mw:
            pd.DataFrame with the forecast that is produced by the backtest pipeline
        congestion_limits_mw:
            list of relevant congestion limits that you want to investigate

    Returns:
        pd.DataFrame with final results (using MWh instead of MW)

    """

    # Group results per congestionlimit
    overall_aggregated_report = {}
    congestion_per_month = {}

    for limit_mw in congestion_limits_mw:

        # Prepare congestion analysis dataframe
        congestion_analysis_dataframe = prepare_congestion_analysis_dataframe(
            forecast_mw, limit_mw
        )

        # Calculate the amount of congestion mitigation based on the back test forecast.
        congestion_analysis_dataframe[
            "amount_congestion_mitigation_mw"
        ] = calculate_amount_congestion_mitigation(
            congestion_analysis_dataframe["forecast"], limit_mw
        )

        # Analyse the amount of congestion mitigation that is applied
        # and condense in single metrics per quantile.
        # These metrics can later be used for the report
        (
            total_amount_congestion_mitigation_per_quantile_mw,
            congestion_mitigation_time_ptu,
            congestion_periods_with_duration,
        ) = calculate_total_congestion_mitigation(
            congestion_analysis_dataframe,
        )

        # Analyse the effect of the congestion mitigation.
        # This answers the question how much congestion was actually prevented
        # by applying the congestion mitigation
        (
            congestion_before_mitigation_mw,
            congestion_after_mitigation_mw,
        ) = calculate_effects_congestion_mitigation(
            congestion_analysis_dataframe["realised"],
            congestion_analysis_dataframe["amount_congestion_mitigation_mw"],
            limit_mw,
        )

        # Calculate for how much congestion is still occruing despite congestion mitigation
        congestion_analysis_dataframe[
            "missed_congestion_mw"
        ] = calculate_missed_congestion(congestion_after_mitigation_mw)

        # Calculate how much congestion we manged to solve by applying congestion mitigation
        congestion_analysis_dataframe[
            "reduced_congestion_mw"
        ] = calculate_reduced_congestion(
            congestion_before_mitigation_mw,
            congestion_analysis_dataframe["missed_congestion_mw"],
        )

        # Compile results and build report dataframe
        overall_aggregated_report[limit_mw] = compose_congestion_analysis_report(
            congestion_analysis_dataframe,
            total_amount_congestion_mitigation_per_quantile_mw,
            congestion_mitigation_time_ptu,
            congestion_periods_with_duration,
            PTU,
        )

        # Complile congestion per month report
        congestion_per_month[limit_mw] = report_congestion_per_month(
            congestion_before_mitigation_mw
        )

    return overall_aggregated_report, congestion_per_month
