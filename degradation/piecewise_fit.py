import numpy as np
import numpy.typing as npt
from sklearn.linear_model import LinearRegression
from typing import Callable
from pymcmcstat.MCMC import MCMC  # type:ignore
from pathlib import Path
import os
import pandas as pd
from scipy.signal import savgol_filter
from zipfile import BadZipFile
from deg_analysis import deg_interval
from typing import Optional


def fit_three_lines(
    x: npt.NDArray, y: npt.NDArray
) -> tuple[tuple[int], LinearRegression]:
    """
    Fits three lines to a curve using Linear Regression
    INPUTS:
        x: np.ndarray, domain over which to fit
        y: np.ndarray, values to fit
    OUTPUTS:
    """
    best_score = float("inf")
    best_breaks = None
    best_models = None

    # Ensure inputs are numpy arrays
    x = np.asarray(x)
    y = np.asarray(y)

    # Try all valid pairs of breakpoints: i < j
    for i in range(2, len(x) - 4):
        for j in range(i + 2, len(x) - 2):
            x1, y1 = x[:i], y[:i]
            x2, y2 = x[i:j], y[i:j]
            x3, y3 = x[j:], y[j:]

            x1r, x2r, x3r = x1.reshape(-1, 1), x2.reshape(-1, 1), x3.reshape(-1, 1)

            model1 = LinearRegression().fit(x1r, y1)
            model2 = LinearRegression().fit(x2r, y2)
            model3 = LinearRegression().fit(x3r, y3)

            y1_pred = model1.predict(x1r)
            y2_pred = model2.predict(x2r)
            y3_pred = model3.predict(x3r)

            ssr = (
                np.sum((y1 - y1_pred) ** 2)
                + np.sum((y2 - y2_pred) ** 2)
                + np.sum((y3 - y3_pred) ** 2)
            )

            if ssr < best_score:
                best_score = ssr
                best_breaks = (i, j)
                best_models = (model1, model2, model3)

    return best_breaks, best_models


def fit_two_lines(
    x: npt.NDArray, y: npt.NDArray
) -> tuple[tuple[int], LinearRegression]:
    """
    Fits two lines to a curve using Linear Regression
    INPUTS:
        x: np.ndarray, domain over which to fit
        y: np.ndarray, values to fit
    OUTPUTS:
    """

    best_break = None
    best_score = float("inf")
    best_models = (None, None)

    for i in range(2, len(x) - 2):
        x1, y1 = x[:i].reshape(-1, 1), y[:i]
        x2, y2 = x[i:].reshape(-1, 1), y[i:]

        model1 = LinearRegression().fit(x1, y1)
        model2 = LinearRegression().fit(x2, y2)

        y1_pred = model1.predict(x1)
        y2_pred = model2.predict(x2)

        ssr = np.sum((y1 - y1_pred) ** 2) + np.sum((y2 - y2_pred) ** 2)

        if ssr < best_score:
            best_score = ssr
            best_break = i
            best_models = (model1, model2)

    return best_break, best_models


def fit_cycb_regimes(
    x: npt.NDArray, y: npt.NDArray
) -> tuple[tuple[int], LinearRegression]:
    """
    Wrapper function for 'fit_two_lines' and 'fit_three_lines'
    INPUTS:
        x: np.ndarray, domain over which to fit
        y: np.ndarray, values to fit
    OUTPUTS:
    """

    best_breaks, best_models = fit_two_lines(x, y)

    # If x/y was short (len<2) error will occur
    try:
        eps = np.abs(best_models[0].coef_ - best_models[1].coef_)
    except AttributeError:
        return np.nan, (np.nan,)

    # If abs(slope) of first fit > abs(slope) of second fit => check for three regimes
    if np.abs(best_models[0].coef_) > abs(best_models[1].coef_):
        best_breaks, best_models = fit_three_lines(x, y)
        return best_breaks, best_models

    # Must check for 3 regimes first, as
    # the slope of the two misfitted lines may have been arbitrarily similar
    # If slope of two fits is similar => we say that no slow regime occured (fit one line)
    if eps > np.abs(max([fit.coef_ for fit in best_models])) / 5:
        pass
    else:
        x = x.reshape(-1, 1)
        best_models = (LinearRegression().fit(x, y),)
        best_breaks = np.nan

    return best_breaks, best_models


def sigmoid(x, weight:Optional[float]=1):
    return 1 / (1 + np.exp(-x/weight))


def model_1(data, q):
    """
    Heaviside function model with one slope switchpoint.
    Designed to interface with the package pymcmcstat
    """

    alpha_1, delta_beta_1, theta_1 = q
    x = data.xdata[0]
    y = data.ydata[0]
    x = x.reshape((x.shape[0],))
    tau_min = 2
    tau_max = x.shape[0]-2

    # ensures beta_1 < alpha_1
    beta_1 = alpha_1 - delta_beta_1

    tau_1 = tau_min + (tau_max-tau_min) *sigmoid(theta_1)

    x_idx = np.arange(x.shape[0])
    heaviside_1 = np.heaviside(x_idx - tau_1, 1)

    return (
        alpha_1 * x
        - alpha_1 * (x - tau_1) * heaviside_1
        + beta_1 * (x - tau_1) * heaviside_1
        + y[0]
    )


def model_2(data, q):
    """
    Heaviside function model with three slope switchpoints.
    Designed to interface with the package pymcmcstat
    """

    alpha_1, delta_alpha_2, delta_beta_1, delta_beta_2, theta_1, theta_2, theta_3 = q
    x = data.xdata[0]
    y = data.ydata[0]
    x = x.reshape((x.shape[0],))
    tau_min = 2
    tau_max = x.shape[0]-2
    delta = x.shape[0]//5

    # ensures tau_1 < tau_2 < tau_3 < x.shape[0]
    tau_1 = tau_min + (tau_max-tau_min- 2*delta) * sigmoid(theta_1)
    tau_2 = (tau_1 + delta) + (tau_max-tau_1-2*delta) * sigmoid(theta_2)
    tau_3 = (tau_2+delta) + (tau_max-tau_2-delta) * sigmoid(theta_3)


    #biological variation
    beta_1 = alpha_1 - delta_beta_1 #delta_beta_1 > 0
    alpha_2 = beta_1 + delta_alpha_2 #delta_alpha_1 > 0
    beta_2 = alpha_2 - delta_beta_2 #delta_beta_2 > 0

    x_idx = np.arange(x.shape[0])
    heaviside_1 = np.heaviside(x_idx - tau_1, 1)
    heaviside_2 = np.heaviside(x_idx - tau_2, 1)
    heaviside_3 = np.heaviside(x_idx - tau_3, 1)

    return (
        y[0]
        + alpha_1 * x
        + (beta_1 - alpha_1) * (x - tau_1) * heaviside_1
        + (alpha_2 - beta_1) * (x - tau_2) * heaviside_2
        + (beta_2 - alpha_2) * (x - tau_3) * heaviside_3
    )


def ssfun_1(q, data):
    """
    Error function to compare 'model_1' with experimental data.
    Designed to interface with the package pymcmcstat
    """

    ydata = data.ydata[0]
    ymodel = model_1(data, q)
    res = ymodel.reshape(ydata.shape) - ydata
    ssr = (res**2).sum(axis=0)

    return ssr


def ssfun_2(q, data):
    """
    Error function to compare 'model_2' with experimental data.
    Designed to interface with the package pymcmcstat
    """
    ydata = data.ydata[0]
    ymodel = model_2(data, q)
    res = ymodel.reshape(ydata.shape) - ydata
    ssr = (res**2).sum(axis=0)

    alpha_1, delta_alpha_2, delta_beta_1, delta_beta_2, theta_1, theta_2, theta_3 = q
    tau_min = 2
    tau_max = data.xdata[0].shape[0] - 2
    delta = data.xdata[0].shape[0] // 5

    tau_1 = tau_min + (tau_max - tau_min - 2*delta) * sigmoid(theta_1)
    tau_2 = (tau_1 + delta) + (tau_max - tau_1 - 2*delta) * sigmoid(theta_2)
    tau_3 = (tau_2 + delta) + (tau_max - tau_2 - delta) * sigmoid(theta_3)
    d_min = 20.0  # minimum acceptable segment length
    lambda_penalty = 10000.0  # strength of the penalty

    penalty = max(0, d_min - (tau_3 - tau_2))
    ssr += lambda_penalty * penalty**2
    
    return ssr


def compute_bic(mcstat, model_func) -> float:
    """
    Given a pymcmc.MCMC() instance and a model,
    estimates the bayesian information criterion of the model with it's
    estimated parameters. Where the parameters are estimated using
    Markov-chain Monte Carlo.
    """

    # Number of data points
    ndata = mcstat.data.ydata[0].size

    # Number of parameters
    k = len(mcstat.simulation_results.results["names"])

    # MAP parameter estimate (posterior mean here)
    results = mcstat.simulation_results.results
    burnin = int(results["nsimu"] / 2)
    chain = results["chain"][burnin:, :]
    map_theta = np.mean(chain, axis=0)

    # Model predictions at MAP
    y_model = model_func(mcstat.data, map_theta).flatten()

    # Observed data
    y_data = mcstat.data.ydata[0].flatten()

    # Sum of squared residuals (SSR)
    ssr = np.sum((y_data - y_model) ** 2)

    # Estimate noise variance from posterior samples
    sigma2 = np.mean(mcstat.simulation_results.results["s2chain"])

    # Compute log-likelihood assuming Gaussian noise
    logL = -0.5 * ndata * np.log(2 * np.pi * sigma2) - 0.5 * ssr / sigma2

    # Bayesian Information Criterion (BIC)
    bic = k * np.log(ndata) - 2 * logL

    return bic

'''
def initial_theta_guesses(x_max):
    """
    Compute initial guesses for theta1, theta2, theta3
    to ensure tau1 < tau2 < tau3 within (1, x_max-1).
    """
    tau_min = 2
    tau_max = x_max - 2

    # Target tau values in (tau_min, tau_max)
    tau1 = tau_min + 0.125 * (tau_max - tau_min)
    tau2 = tau_min + 0.25 * (tau_max - tau_min)
    tau3 = tau_min + 0.75 * (tau_max - tau_min)
    
    # Logit function (inverse of sigmoid)
    def logit(p):
        return np.log(p / (1 - p))
    
    # Compute thetas to match target taus
    theta1 = logit((tau1 - tau_min) / (tau_max - tau_min))
    theta2 = logit((tau2 - tau1) / (tau_max - tau1))
    theta3 = logit((tau3 - tau2) / (tau_max - tau2))
    
    return tau_min, tau_max, theta1, theta2, theta3
    '''


def bayes_fit_cycb_regimes(
    x: npt.NDArray, y: npt.NDArray, ssfuns: list[Callable], models: list[Callable]
) -> list[float]:
    """
    Computes the fit parameters of either the
    one-switchpoint or three-switchpoint heaviside
    model.
    INPUTS:
        x: domain over with to fit
        y: values to fit
        ssfuns: list containing model evalutation functions
        models: list containing models. models in this list should be in the same order as their corresponding ssfun in ssfuns
    OUTPUTS:
    """

    # Model 1 simulation
    mcstat = MCMC()
    mcstat.data.add_data_set(x, y)
    mcstat.model_settings.define_model_settings(sos_function=ssfuns[0])
    mcstat.simulation_options.define_simulation_options(nsimu=10e3, updatesigma=True)
    mcstat.parameters.add_model_parameter(name="alpha_1", theta0=-0.2, maximum=0, prior_mu=-0.2, prior_sigma=0.025) #low and narrow prior
    mcstat.parameters.add_model_parameter(name="delta_beta_1", theta0=0.5, minimum=0, prior_mu=0.5, prior_sigma=0.4) #high and wide prior
    mcstat.parameters.add_model_parameter(
        name="theta_1", theta0=0
    ) 
    mcstat.run_simulation()

    # Model 2 simulation
    mcstat_2 = MCMC()
    mcstat_2.data.add_data_set(x, y)
    mcstat_2.model_settings.define_model_settings(sos_function=ssfuns[1])
    mcstat_2.simulation_options.define_simulation_options(nsimu=10e4, updatesigma=True)
    mcstat_2.parameters.add_model_parameter(name="alpha_1", theta0=-0.2, maximum=0.0, prior_mu=-0.2, prior_sigma=0.025)
    mcstat_2.parameters.add_model_parameter(name="delta_alpha_2", theta0=0.5, minimum=0.0, prior_mu=0.5, prior_sigma=0.025)
    mcstat_2.parameters.add_model_parameter(name="delta_beta_1", theta0=0.5, minimum=0, prior_mu=0.5, prior_sigma=0.4)
    mcstat_2.parameters.add_model_parameter(name="delta_beta_2", theta0=0.5, minimum=0, prior_mu=0.5, prior_sigma=0.4)
    mcstat_2.parameters.add_model_parameter(
        name="theta_1", theta0=0, 
    )
    mcstat_2.parameters.add_model_parameter(
        name="theta_2", theta0=0, 
    )
    mcstat_2.parameters.add_model_parameter(
        name="theta_3", theta0=0, 
    )
    mcstat_2.run_simulation()

    bic1 = compute_bic(mcstat, models[0])
    bic2 = compute_bic(mcstat_2, models[1])
    delta_bic = bic1 - bic2

    # emperical bayesian information criterion threshold
    model = mcstat if delta_bic < 100 else mcstat_2

    results = model.simulation_results.results
    burnin = int(results["nsimu"] / 2)
    chain = results["chain"][burnin:, :]

    params = np.mean(chain, axis=0)
    delta = x.shape[0]//5
    if params.shape[0] == 3:
        tau_min = 5
        tau_max = x.shape[0]-5
        tau_1 = (tau_max - tau_min) * sigmoid(params[2])
        return [
            params[0],
            params[0] - params[1],
            tau_1,
        ]  # alpha_1, #beta_1, #tau_1

    elif params.shape[0] == 7:
        tau_min = 2
        tau_max = x.shape[0]-2
        tau_1 = tau_min + (tau_max - tau_min - 2*delta) * sigmoid(params[4])
        tau_2 = tau_1 + delta + (tau_max - tau_1 - 2*delta) * sigmoid(params[5])
        tau_3 = tau_2 + delta + (tau_max - tau_2 - delta) * sigmoid(params[6])

        alpha_1 = params[0]
        beta_1 = alpha_1 - params[2]
        alpha_2 = beta_1 + params[1]
        beta_2 = alpha_2 - params[3]

        return [
            alpha_1, 
            alpha_2,
            beta_1,
            beta_2,
            tau_1,
            tau_2,
            tau_3
        ]  # alpha_1, alpha_2, beta_1, beta_2, tau_1, tau_2, tau_3


if __name__ == "__main__":

    width = 21
    root_dir = Path("input/root/dir")
    inference_dirs = [
        obj.path
        for obj in os.scandir(root_dir)
        if "_inference" in obj.name and obj.is_dir()
    ]

    cycb_paths = []
    for dir in inference_dirs:
        cycb_path = [obj.path for obj in os.scandir(dir) if "chromatin" in obj.name]
        cycb_paths += cycb_path

    for cycb_path in cycb_paths:
        try:
            writer = pd.ExcelWriter(
                cycb_path, engine="openpyxl", mode="a", if_sheet_exists="replace"
            )
        except BadZipFile:
            continue

        traces = pd.read_excel(cycb_path, sheet_name=0, index_col=0).to_numpy()
        semantic = pd.read_excel(cycb_path, sheet_name=1, index_col=0).to_numpy()

        assert traces.shape == semantic.shape, f"Mismatched shapes: {traces.shape} vs {semantic.shape}"

        fit_info = []
        for i, trace in enumerate(traces):
            # cannot be smoothed or trace ends in mitosis
            if trace.shape[0] > width and semantic[i][-1] != 1:
                trace[np.isnan(trace)] = 0
                smooth_trace = savgol_filter(trace, width, 2)
            else:
                fit_info.append(np.zeros(3))
                continue

            low_bound, high_bound = deg_interval(smooth_trace, semantic[i])
            mit_neg_trace = smooth_trace[low_bound:high_bound]

            x = np.linspace(1, mit_neg_trace.shape[0], mit_neg_trace.shape[0])
            if x.shape[0] > 0:
                params = bayes_fit_cycb_regimes(
                    x, mit_neg_trace, [ssfun_1, ssfun_2], [model_1, model_2]
                )
            else:
                params = None

            if params == None:
                params = np.zeros(3)
            elif len(params) == 3:
                params[2] += low_bound
            elif len(params) == 7:
                params = [
                    param if i < 4 else param + low_bound
                    for i, param in enumerate(params)
                ]

            fit_info.append(tuple(params))

        fit_info = pd.DataFrame(fit_info)
        fit_info.to_excel(writer, sheet_name="fitting info")
        writer.close()
