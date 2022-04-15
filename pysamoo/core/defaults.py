from ezmodel.core.factory import models_from_clazzes
from ezmodel.models.kriging import Kriging
from ezmodel.models.rbf import RBF
from ezmodel.util.transformation.plog import Plog
from pymoo.util.normalization import NoNormalization


def DEFAULT_OBJ_MODELS(**defaults):

    models = models_from_clazzes(
        # YAGP,
        # BOTORCH,
        # TKGP,
        # Kriging,
        # GGP,
        # LGP,
        RBF,
        # KNN,
        # RBF3,
        # PolynomialRegression,
        # RBF2,
        # pySOTRBF,
        # SVR,
        # InverseDistanceWeighting,
        # NearestNeighbors,
        **defaults)

    models = {name: entry["model"] for name, entry in models.items()}

    # models = {}

    # for kernel in ["cubic", "linear", "mq"]:
    #     for normalized in [False, True]:
    #         for tail in ["constant", "linear", "linear+quadratic"]:
    #             params = dict(defaults)
    #             params["kernel"] = kernel
    #             params["normalized"] = normalized
    #             params["tail"] = tail
    #
    #             model = RBF(**params)
    #             models[f"rbf-{kernel}-{tail}-{normalized}"] = model

    models['kriging-const'] = Kriging(regr="constant")
    models['kriging-lin'] = Kriging(regr="linear")
    models['kriging-quadr'] = Kriging(regr="quadratic")

    models['kriging-const-ARD'] = Kriging(regr="constant", ARD=True)
    models['kriging-lin-ARD'] = Kriging(regr="linear", ARD=True)
    models['kriging-quadr-ARD'] = Kriging(regr="quadratic", ARD=True)
    # models['kriging-sine'] = Kriging(regr="sine")

    return models


def DEFAULT_IEQ_CONSTR_MODELS(**defaults):
    models = {}

    for kernel in ["cubic", "linear", "mq"]:
        for label, norm in [("default", NoNormalization()), ("plog", Plog())]:
            for normalized in [False, True]:
                for tail in ["constant", "linear", "linear+quadratic"]:
                    params = dict(defaults)
                    params["kernel"] = kernel
                    params["norm_X"] = norm
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
                for optimize in [False]:
                    params = dict(defaults)
                    params["kernel"] = kernel
                    params["normalized"] = normalized
                    params["tail"] = tail
                    params["optimize"] = optimize

                    model = RBF(**params)
                    models[f"rbf-{kernel}-{tail}-{normalized}-{optimize}"] = model

    models['kriging-const'] = Kriging(regr="constant")
    models['kriging-lin'] = Kriging(regr="linear")
    models['kriging-quadr'] = Kriging(regr="quadratic")

    return models
