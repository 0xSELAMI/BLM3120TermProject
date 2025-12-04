class Instance:
    def __init__(self, features):
        if len(list(features.keys())) != 11:
            print('[ERROR] Invalid instance initialization')
            return None

        self.field_names = features.keys()

        for key in features:
            setattr(self, key, features[key])

    def __repr__(self):
        str = ""

        for name in self.field_names:
            str += f"{name}: {getattr(self, name)}\n"

        return str[:-1]
