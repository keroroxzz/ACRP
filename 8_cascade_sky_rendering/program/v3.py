"""V3 evaluation only: fixed quality/performance grid on the selected V2 kernel.

RMSE budgets are OFFLINE evaluation gates, not online error estimators. Each
candidate uses one fixed threshold across all scenes; no reference enters render.
"""
from collections import OrderedDict

RMSE_BUDGETS=(.002,.003,.004,.005,.0075,.010,.015,.020)
THRESHOLDS=(.05,.07,.09,.12,.16,.22,.32)


def variants():
    result=OrderedDict([
        ('full',dict(levels=1)),
        ('anchor',dict(levels=3,factor=2,threshold=.04)),
        ('v2',dict(v2=True,levels=2,factor=4,threshold=.035))])
    for levels,factor in ((2,4),(3,2)):
        for threshold in THRESHOLDS:
            result['L%d-N%d-T%g'%(levels,factor,threshold)]=dict(
                v2=True,levels=levels,factor=factor,threshold=threshold)
    return result
