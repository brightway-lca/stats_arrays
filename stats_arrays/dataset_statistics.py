# Weighted formulas from
# http://en.wikipedia.org/wiki/Weighted_mean#Weighted_sample_variance

# TODO: Test against http://pygsl.sourceforge.net/README.html
# http://pygsl.sourceforge.net/reference/pygsl/node36.html


def weighted_mean(values, weights):
    assert values.shape == weights.shape
    return (values * weights).sum() / weights.sum()


def weighted_sample_variance(values, weights):
    assert values.shape == weights.shape
    return ((weights * values ** 2).sum() * weights.sum() -
            ((weights * values).sum()) ** 2) / ((weights.sum()) ** 2 -
                                                (weights ** 2).sum())


def weighted_sample_stddev(values, weights):
    return weighted_sample_variance(values, weights) ** 0.5


def test():
    from numpy import array
    a = array((1, 2, 3, 4))
    print a ** 2
    b = array((0.5, 0.5, 1, 1))
    print weighted_mean(a, b)
    print weighted_sample_stddev(a, b)

if __name__ == '__main__':
    test()
