from .uncertainty_choices import uncertainty_choices


class DataFitter(object):
    """List all distributions in uncertainty_choices, in order of goodness-of-fit"""
    def __init__(self, data):
        self.data = data
        self.results = {}
        for choice in uncertainty_choices:
            try:
                goodness = choice.fit(data)
                self.results
            except NotImplementedError:
                continue
