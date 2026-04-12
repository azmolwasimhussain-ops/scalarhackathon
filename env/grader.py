def grade(pred, truth):
    if pred == truth:
        return 1.0
    elif pred in truth:
        return 0.5
    else:
        return 0.0