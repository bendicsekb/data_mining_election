import math
import pandas as pd
from enum import Enum

# all the types and their respective operators
binary_operators = nominal_operators = ["==", "!="]
numerical_operators = ["<=", ">="]

descriptions_strings = set()

# enumerate on the possible rule types
class RuleType(Enum):
    UNKNOWN = 0
    BINARY = 1
    NOMINAL = 2
    NUMERICAL = 3


# Given a rule type, return a list of operators for that rule type
def get_operators_per_type(rule_type: RuleType):
    if rule_type.value is RuleType.BINARY.value:
        return binary_operators
    elif rule_type.value is RuleType.NOMINAL.value:
        return nominal_operators
    elif rule_type.value is RuleType.NUMERICAL.value:
        return numerical_operators
    else:
        raise Exception("Rule type", rule_type, " has no operators")


"""
A rule like in subgroup discovery or EMM
A rule contains an attribute, operator and value, for example:
    salary >= 3000
Where   salary is the attribute
        >= is the operator
        3000 is the value
Rules can be of type binary, nominal and numerical and their operators should be conform those types
"""
class Rule:
    def __init__(self, rule_type: RuleType, attribute: str, operator: str, value):
        if rule_type.value is RuleType.UNKNOWN.value:
            raise Exception("Rule type not defined", self.rule_type)
        else:
            self.rule_type = rule_type

        if rule_type.value is RuleType.BINARY.value and operator not in binary_operators:
            raise Exception("Operator", operator, " not a binary operator")
        elif rule_type.value is RuleType.NOMINAL.value and operator not in nominal_operators:
            raise Exception("Operator", operator, " not a nominal operator")
        elif rule_type.value is RuleType.NUMERICAL.value and operator not in numerical_operators:
            raise Exception("Operator", operator, " not a numerical operator")
        else:
            self.attribute = attribute
            self.operator = operator
            self.value = value

    def to_string(self):
        return self.attribute + " " + self.operator + " " + str(self.value)


# A description is a set of rules containing at least one rule with a quality assigned to it
class Description:
    # def __init__(self, rules: list[Rule] = None, quality: float = 0.0):
    def __init__(self, rules = None, quality: float = 0.0):
        if rules is None:
            self.rules = []
        else:
            self.rules = sorted(self.rules, key=lambda r: r.to_string())

        if quality < 0:
            raise Exception("Quality values cannot be negative")
        else:
            self.quality = quality

    def add_rule(self, rule: Rule):
        self.rules.append(rule)
        self.rules = sorted(self.rules, key=lambda r: r.to_string())

    def to_string(self):
        string = ""
        if len(self.rules) != 0:
            string = self.rules[0].to_string()
            for i in range(1, len(self.rules)):
                string = string + " and " + self.rules[i].to_string()
        return string

    def print_description(self):
        string = self.to_string()
        if string == "":
            print("Empty description")
        else:
            string = string + " with quality " + str(self.quality)
            print(string)


class DataSet:
    # def __init__(self, data, targets: list[str], descriptors: list[str]):
    def __init__(self, data, targets, descriptors):
        self.dataframe = data

        if len(targets) == 0:
            raise Exception("User has to specify the targets for a dataset")
        else:
            self.targets = targets

        if len(descriptors) == 0:
            columns = data.columns
            self.descriptors = [attr for attr in columns if attr not in targets]
        else:
            self.descriptors = descriptors

        self.descriptor_types = {}
        self.set_descriptor_types()

    def set_descriptor_types(self):
        for desc in self.descriptors:
            self.descriptor_types[desc] = RuleType.BINARY # RuleType.NUMERICAL  # TODO: improve

    def get_descriptor_type(self, descriptor):
        return self.descriptor_types[descriptor]


# reduce the input data given a description
def get_subgroup_data(description: Description, data: pd.DataFrame):
    rules = description.rules
    output_data = data
    for rule in rules:
        if rule.operator == "==":
            output_data = output_data[output_data[rule.attribute] == rule.value]
        elif rule.operator == "!=":
            output_data = output_data[output_data[rule.attribute] != rule.value]
        elif rule.operator == "<=":
            output_data = output_data[output_data[rule.attribute] <= rule.value]
        elif rule.operator == ">=":
            output_data = output_data[output_data[rule.attribute] >= rule.value]

    return output_data


# Given a dataset and seed (description) generate all possible candidates
# by dynamically discretization in the number of bins
def refine(seed: Description, data: DataSet, bins: int):
    descriptions = []

    for attribute in data.descriptors:  # new rule for every attribute
        attribute_type = data.get_descriptor_type(attribute)

        if attribute_type == RuleType.BINARY and not check_duplicate_attribute(attribute, seed):
            eq_desc, neq_desc = Description(), Description()
            for rule in seed.rules:
                eq_desc.add_rule(rule)
                neq_desc.add_rule(rule)

            eq_rule = Rule(attribute_type, attribute, "==", 1.0)
            neq_rule = Rule(attribute_type, attribute, "!=", 1.0)
            eq_desc.add_rule(eq_rule)
            neq_desc.add_rule(neq_rule)

            if eq_desc.to_string() not in descriptions_strings:
                descriptions.append(eq_desc)
                descriptions_strings.add(eq_desc.to_string())
            if neq_desc.to_string() not in descriptions_strings:
                descriptions.append(neq_desc)
                descriptions_strings.add(neq_desc.to_string())

        elif attribute_type == RuleType.NUMERICAL:
            for operator in get_operators_per_type(attribute_type):  # new rule for every operator
                if not check_duplicate_operator(operator, seed):
                    binning_intervals = discretize(seed, attribute, data.dataframe, bins)  # define binning intervals
                    for interval in binning_intervals:  # new rule for every bin
                        new_rule = Rule(attribute_type, attribute, operator, interval)

                        # create new description and add to the existing list
                        # Note: somehow passing seed.rules as argument to Description() results in incorrect descriptions
                        new_desc = Description()
                        for rule in seed.rules:
                            new_desc.add_rule(rule)
                        new_desc.add_rule(new_rule)

                        descriptions.append(new_desc)

    return descriptions


# dynamically discretize the input data with the equal-height method subjective to the description
# by returning an #bins size array containing the intervals
def discretize(description: Description, attribute: str, data: pd.DataFrame, bins: int):
    intervals = []
    if len(description.rules) == 0:
        subgroup_data = data[attribute].sort_values(ascending=False)
    else:
        subgroup_data = get_subgroup_data(description, data)[attribute].sort_values(ascending=False)

    n = len(subgroup_data.index)
    if bins > n:  # check if there are more bins than rows
        bins = n

    for i in range(1, bins+1):
        index = math.floor((n - 1) / i)
        value = subgroup_data.iloc[index]
        if value not in intervals:
            intervals.append(value)

    return intervals


# Check if the passed attribute already exists in the description
def check_duplicate_attribute(attribute: str, description: Description):
    if len(description.rules) == 0:
        return False
    for rule in description.rules:
        if attribute == rule.attribute:
            return True
    return False


def check_duplicate_operator(operator: str, description: Description):
    if len(description.rules) == 0:
        return False
    for rule in description.rules:
        if operator == rule.operator:
            return True
    return False
