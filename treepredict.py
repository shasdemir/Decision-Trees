my_data = [['slashdot', 'USA', 'yes', 18, 'None'],
           ['google', 'France', 'yes', 23, 'Premium'],
           ['digg', 'USA', 'yes', 24, 'Basic'],
           ['kiwitobes', 'France', 'yes', 23, 'Basic'],
           ['google', 'UK', 'no', 21, 'Premium'],
           ['(direct)', 'New Zealand', 'no', 12, 'None'],
           ['(direct)', 'UK', 'no', 21, 'Basic'],
           ['google', 'USA', 'no', 24, 'Premium'],
           ['slashdot', 'France', 'yes', 19, 'None'],
           ['digg', 'USA', 'no', 18, 'None'],
           ['google', 'UK', 'no', 18, 'None'],
           ['kiwitobes', 'UK', 'no', 19, 'None'],
           ['digg', 'New Zealand', 'yes', 12, 'Basic'],
           ['slashdot', 'UK', 'no', 21, 'None'],
           ['google', 'UK', 'yes', 18, 'Basic'],
           ['kiwitobes', 'France', 'yes', 19, 'Basic']]


class DecisionNode(object):
    def __init__(self, column=-1, true_value=None, results=None, true_child=None, false_child=None):
        self.column = column  # which column of the data we are testing
        self.true_value = true_value  # the value that the column must match to get a true result
        self.results = results  # dict of results for this leaf. None except leaf nodes.
        self.true_child = true_child  # ref to child DecisionNode if the test is true
        self.false_child = false_child  # false child


def divide_set(rows, column, check):
    """ Divide the set on a specific column. Can handle numeric or nominal values. Return rows as two parts, one the
    values on the specified column satisfies the check, other do not. """

    # Make a function to determine if a row is in the first group (true) or the second group (false)
    if isinstance(check, int) or isinstance(check, float):
        split_function = lambda row: row[column] >= check
    else:
        split_function = lambda row: row[column] == check

    # divide the rows and return the partition
    part1 = filter(split_function, rows)
    part2 = filter(lambda row: not split_function(row), rows)

    return part1, part2


def unique_counts(rows):
    """ Create counts of each possible result. Last column of each row is the result. """

    results = {}
    for row in rows:
        row_result = row[-1]
        if row_result not in results:
            results[row_result] = 0
        results[row_result] += 1
    return results


def gini_impurity(rows):
    """ Calculate the Gini impurity of a selection of rows. Gini impurity is the probability that a randomly placed
    item will be in the wrong category. Note: The random selection of placement is not uniform over the categories. """

    total = len(rows)
    counts = unique_counts(rows)
    imp = 0

    for k1 in counts:
        p1 = float(counts[k1]) / total
        for k2 in counts:
            if k1 != k2:
                p2 = float(counts[k2]) / total
                imp += p1 * p2
    return imp


def entropy(rows):
    """ Entropy: sum of -p(x)log(p(x)) across all possible different results. """

    from math import log
    log2 = lambda x: log(x) / log(2)

    results = unique_counts(rows)
    ent = 0.
    for res in results:
        p = float(results[res]) / len(rows)
        ent = ent - p * log2(p)
    return ent