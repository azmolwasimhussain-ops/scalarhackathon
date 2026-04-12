def grade(pred, truth):
    if pred == truth:
        return 0.999999  # strictly less than 1.0
    elif pred in truth:
        return 0.5
    else:
        return 0.000001  # strictly greater than 0.0