import math

from collections import Counter

from common.Instance import Instance
from common.Features import FeatureType, FeatureFilter

class DatasetSchema:
    feature_types   = None
    entropy_weights = None

    def __init__(self):
        if DatasetSchema.feature_types is None or DatasetSchema.entropy_weights is None:
            raise ValueError("Dataset Schema is not configured")

    @staticmethod
    def configure_schema(dataset_contents, field_types, entropy_weights):
        DatasetSchema.entropy_weights = entropy_weights

        if type(dataset_contents) == list:
            # getting field names from first line and
            # trimming user_id's from the rest of the instances
            field_descriptions = zip(dataset_contents[0][1:], field_types)
            dataset_contents = [instance[1:] for instance in dataset_contents[1:]]

        elif type(dataset_contents) == dict:
            field_descriptions = zip(dataset_contents["field_names"], field_types)
            dataset_contents = dataset_contents["instances"]

        else:
            raise ValueError(f"Invalid dataset contents type: {type(dataset_contents)}")

        DatasetSchema.feature_types = {
            field_description[0]: FeatureType(
                field_description, [i[desc_idx] for i in dataset_contents]
                ) for desc_idx, field_description in enumerate(field_descriptions)
        }

class Dataset(DatasetSchema):
    def __init__(self, instance_array):
        super().__init__()

        self.majority_label = None
        self.entropy        = None
        self.gini           = None
        self.value_domains  = None
        self.instances      = []
        self.size           = 0

        if not instance_array or len(instance_array) == 0:
            return

        if type(instance_array[0]) == Instance:
            self.instances = instance_array
        else:
            field_names = list(self.feature_types.keys())

            for instance in instance_array:
                instance_dict = {}

                for i, f in enumerate(instance):
                    field_name = field_names[i]
                    field_type = self.feature_types[field_name].value

                    instance_dict[field_name] = field_type(int(f)) if field_type == bool else field_type(f)

                self.instances.append(Instance(instance_dict))

        self.size = len(self.instances)

        if not self.is_empty:
            self.majority_label = self.calc_majority_label()
            self.entropy        = self.calc_binary_label_entropy(self.entropy_weights)
            self.gini           = self.calc_binary_label_gini()
            self.value_domains  = self.compute_value_domains()

    @property
    def is_empty(self):
        return self.size == 0

    @property
    def is_pure(self):
        return True if self.is_empty else (self.majority_label[1] == self.size)
    
    @property
    def count_label_true(self):
        return (self.size - self.majority_label[1]) if self.majority_label[0] == False else self.majority_label[1]
    
    @property
    def count_label_false(self):
        return (self.size - self.count_label_true)

    def compute_value_domains(self):
        domains = {name: set() for name in self.feature_types.keys()}

        for inst in self.instances:
            for name in domains.keys():
                domains[name].add(getattr(inst, name))

        for name in domains.keys():
            if self.feature_types[name].is_numeric:
                domains[name] = sorted(domains[name])

        return domains

    def value_domains_repr(self):
        str = ""

        for feature_name in self.value_domains:
            str += f"{feature_name}: "
            domain = self.value_domains[feature_name]
            if self.feature_types[feature_name].is_numeric:
                str += f"range({domain[0]}, {domain[-1]})"
            else:
                str += repr(domain)
            str += '\n'

        return str[:-1]

    def calc_majority_label(self):
        counts = Counter(getattr(inst, list(self.feature_types.keys())[-1]) for inst in self.instances)
        return counts.most_common()[0]

    def calc_binary_label_entropy(self, weights):
        if self.is_pure:
            return 0

        # majority label: [0] is value, [1] is count of value
        prob = self.count_label_true / self.size
        return -( (weights[0] * prob * math.log2(prob)) + (weights[1] * (1 - prob) * math.log2(1 - prob)) )
    
    def calc_binary_label_gini(self):
        if self.is_empty:
            return 0.0

        p = self.count_label_true / self.size

        return 2 * p * (1 - p)
    
    @classmethod
    def subset_with_feature_filter(cls, dataset, feature_filters):
        instance_array = []

        if dataset.instances:
            for instance in dataset.instances:
                does_conform = True

                for feature_filter in feature_filters:
                    attrval = getattr(instance, feature_filter.feature_type.name)
                    if not eval(f'attrval {feature_filter.op} feature_filter.value'):
                        does_conform = False

                if does_conform:
                    instance_array.append(instance)

        return cls(instance_array)

    def __repr__(self):
        str = ""

        for c, i in enumerate(self.instances):
            str += f"Instance: {c}\n{i}\n\n"

        return str[:-2]
