import tensorflow as tf
from a3c.estimators import A3CEstimator
from acer.estimators import AcerEstimator

def get_estimator(type):
    type = type.upper()

    print "Using {} as estimator".format(type)

    if type == "A3C":
        return A3CEstimator
    elif type == "ACER":
        AcerEstimator.create_averge_network()
        return AcerEstimator
    else:
        raise TypeError("Unknown type " + type)
