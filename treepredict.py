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
