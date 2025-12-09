class Instance:
    def __init__(self, features):
        self.field_names = list(features.keys())

        for key in features:
            setattr(self, key, features[key])

    @property
    def features(self):
        return [getattr(self, name) for name in self.field_names[:-1]]

    @property
    def label(self):
        return getattr(self, self.field_names[-1])

    def __repr__(self):
        str = ""

        for name in self.field_names:
            str += f"{name}: {getattr(self, name)}\n"

        return str[:-1]
