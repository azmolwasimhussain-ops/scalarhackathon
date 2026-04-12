def grade(prediction, truth):
    if prediction == truth:
        return 0.99
    elif prediction in truth:
        return 0.50
    return 0.01