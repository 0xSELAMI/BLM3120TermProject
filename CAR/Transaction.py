class TransactionItem:
    def __init__(self, feature_name, rule_format):
        self.feature_name = feature_name
        self.rule_format  = rule_format

    def __repr__(self):
        return f"TransactionItem({self.rule_format})"

    def __eq__(self, other):
        if type(self) == type(other):
            return (self.feature_name == other.feature_name) and (self.rule_format == other.rule_format)

        return False

    def __hash__(self):
        return hash(repr(self))

class TransactionItemset:
    def __init__(self, items = None):
        self.items = list(items) if items else []

    def __iter__(self):
        self.iterctx = 0
        return self

    def __next__(self):
        if self.iterctx >= len(self.items):
            raise StopIteration
        item = self.items[self.iterctx]
        self.iterctx += 1
        return item

    def __repr__(self):
        ret = "TransactionItemset{"

        if not len(self.items):
            return ret + "}"

        for item in self.items:
            ret += f"{item}, "

        ret = (ret[:-2] + "}")
        return ret

    def add(self, item):
        if item in self.items:
            return self

        if not (type(item) == TransactionItem):
            self.items.append(item)
            return self

        # only add if there's no item with this feature name
        if ( item.feature_name in [i.feature_name for i in self.items] ):
            return self

        self.items.append(item)
        return self

    def __or__(self, other):
        out = TransactionItemset(self.items)

        for item in other.items:
            out.add(item)

        return out

    def __contains__(self, item):
        return item in self.items

    def issubset(self, other):
        return (set(self.items).issubset(set(other)))

    def __eq__(self, other):
        return set(self.items) == set(other.items)

    def __sub__(self, other):
        return TransactionItemset(set(self.items) - other)

    def __len__(self):
        return len(self.items)

    def __hash__(self):
        return hash(repr(self))
