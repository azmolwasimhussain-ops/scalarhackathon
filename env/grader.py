def grade(prediction, truth):
    if prediction == truth:
        return 1.0
    elif prediction in truth:
        return 0.5
    return 0.0