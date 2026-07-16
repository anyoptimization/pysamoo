"""Default surrogate-model factories for objectives and constraints."""

from ezmodel.core.factory import models_from_clazzes
from ezmodel.models.kriging import Kriging
from ezmodel.models.rbf import RBF
from ezmodel.util.transformation.plog import Plog
from pydacefit.regr import ConstantRegression, LinearRegression, QuadraticRegression
from pymoo.util.normalization import NoNormalization


def DEFAULT_OBJ_MODELS(**defaults):

    # models_from_clazzes returns {name: model_instance}
    models = models_from_clazzes(RBF, **defaults)

    # pydacefit expects a regression *object* (not the string "constant"/"linear"/...).
    # Passing strings silently fails the fit (caught by the benchmark's raise_exception=False),
    # which used to drop the entire Kriging zoo and leave only RBF -- a large, silent quality loss.
    models["kriging-const"] = Kriging(regr=ConstantRegression())
    models["kriging-lin"] = Kriging(regr=LinearRegression())
    models["kriging-quadr"] = Kriging(regr=QuadraticRegression())

    models["kriging-const-ARD"] = Kriging(regr=ConstantRegression(), ARD=True)
    models["kriging-lin-ARD"] = Kriging(regr=LinearRegression(), ARD=True)
    models["kriging-quadr-ARD"] = Kriging(regr=QuadraticRegression(), ARD=True)

    return models


def DEFAULT_IEQ_CONSTR_MODELS(**defaults):
    models = {}

    for kernel in ["cubic", "linear", "mq"]:
        # Plog is a *target* transform for constraint violations (SACOBRA-style), so it belongs on
        # norm_y -- keep the caller's design-space norm_X instead of overwriting it.
        for label, norm in [("default", NoNormalization()), ("plog", Plog())]:
            for normalized in [False, True]:
                for tail in ["constant", "linear", "linear+quadratic"]:
                    params = dict(defaults)
                    params["kernel"] = kernel
                    params["norm_y"] = norm
                    params["normalized"] = normalized
                    params["tail"] = tail

                    model = RBF(**params)
                    models[f"rbf-{kernel}-{tail}-{label}-{normalized}"] = model

    return models


def DEFAULT_EQ_CONSTR_MODELS(**defaults):
    models = {}

    for kernel in ["cubic", "linear", "mq"]:
        for normalized in [False, True]:
            for tail in ["constant", "linear", "linear+quadratic"]:
                params = dict(defaults)
                params["kernel"] = kernel
                params["normalized"] = normalized
                params["tail"] = tail
                params["optimize"] = False

                model = RBF(**params)
                models[f"rbf-{kernel}-{tail}-{normalized}-False"] = model

    models["kriging-const"] = Kriging(regr=ConstantRegression())
    models["kriging-lin"] = Kriging(regr=LinearRegression())
    models["kriging-quadr"] = Kriging(regr=QuadraticRegression())

    return models
