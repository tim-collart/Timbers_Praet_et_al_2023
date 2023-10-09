from statsmodels.regression.linear_model import OLSResults
from statsmodels.tools.tools import add_constant
def apply_model(ins,outs):
    # get extra arguments
    model_file = pdalargs['model_file']
    in_dimension = pdalargs['in_dimension']
    out_dimension = pdalargs['out_dimension']
    # load the model
    model = OLSResults.load(model_file)
    # use the model to predict the dimension
    outs[out_dimension] = model.predict(add_constant(ins[in_dimension]))
    return True
