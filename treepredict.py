from PIL import Image, ImageDraw


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
        ent -= p * log2(p)
    return ent


def build_tree(rows, score_f=entropy):
    if len(rows) == 0:
        return DecisionNode()

    current_score = score_f(rows)

    # set up variables to track best criteria
    best_gain = 0.
    best_criteria = None
    best_sets = None

    column_count = len(rows[0]) - 1  # 0-based count
    for col in range(column_count):  # generate the list of different values in this column
        column_values = {}
        for row in rows:
            column_values[row[col]] = 1
        # try dividing the rows up for each value in this column
        for value in column_values:
            set1, set2 = divide_set(rows, col, check=value)

            # information gain
            p = float(len(set1)) / len(rows)
            gain = current_score - p * score_f(set1) - (1-p) * score_f(set2)

            if gain > best_gain and len(set1) > 0 and len(set2) > 0:
                best_gain = gain
                best_criteria = col, value
                best_sets = set1, set2

    if best_gain > 0:  # create subbranches
        true_branch, false_branch = build_tree(best_sets[0]), build_tree(best_sets[1])

        return DecisionNode(column=best_criteria[0], true_value=best_criteria[1],
                            true_child=true_branch, false_child=false_branch)
    else:
        return DecisionNode(results=unique_counts(rows))


def print_tree(tree, indent="    "):
    if tree.results is not None:  # leaf node
        print tree.results
    else:
        print str(tree.column) + ": " + str(tree.true_value) + "? "  # print the criteria

        print indent + "T ->",
        print_tree(tree.true_child, indent + "    ")
        print indent + "F ->",
        print_tree(tree.false_child, indent + "    ")


def get_width(tree):
    if tree.true_child is None and tree.false_child is None:
        return 1
    return get_width(tree.true_child) + get_width(tree.false_child)


def get_depth(tree):
    if tree.true_child is None and tree.false_child is None:
        return 1
    return max(get_depth(tree.true_child), get_depth(tree.false_child)) + 1


def draw_tree(tree, jpeg="tree.png"):
    """ Determine the appropriate total size and pass a canvas and the top node to draw_node. """

    w = get_width(tree) * 100
    h = get_depth(tree) * 100 + 120

    img = Image.new('RGB', (w, h), (255, 255, 255))
    draw = ImageDraw.Draw(img)

    draw_node(draw, tree, w/2., 20)
    img.save(jpeg, 'PNG')


def draw_node(draw, tree, x, y):
    """ Draw the nodes recursively. """

    if tree.results is None:
        # get the width of each branch
        w1 = get_width(tree.false_child) * 100
        w2 = get_width(tree.true_child) * 100

        # determine the total space required by this node
        left = x - (w1 + w2) / 2.
        right = x + (w1 + w2) / 2.

        # draw the condition string
        draw.text((x - 20, y - 20), str(tree.column) + ": " + str(tree.true_value), (0, 0, 0))

        # draw links to the branches
        draw.line((x, y, left + w1 / 2., y + 100), fill=(255, 0, 0))
        draw.line((x, y, right - w2 / 2., y + 100), fill=(255, 0, 0))

        # draw the branch nodes
        draw_node(draw, tree.false_child, left + w1 / 2., y + 100)
        draw_node(draw, tree.true_child, right - w2 / 2., y + 100)
    else:
        txt = " \n".join(["%s:%d" %v for v in tree.results.items()])
        draw.text((x - 20, y), txt, (0, 0, 0))


def classify(observation, tree):
    """ Classify observation according to the tree. """

    if tree.results is not None:
        return tree.results
    else:
        v = observation[tree.column]
        if isinstance(v, int) or isinstance(v, float):
            if v >= tree.true_value:
                branch = tree.true_child
            else:
                branch = tree.false_child
        else:
            if v == tree.true_value:
                branch = tree.true_child
            else:
                branch = tree.false_child
        return classify(observation, branch)


def prune(tree, min_gain):
    # if the branches aren't leaves, prune recursively
    if tree.true_child.results is None:
        prune(tree.true_child, min_gain)
    if tree.false_child.results is None:
        prune(tree.false_child, min_gain)

    # if both the subbranches are now leaves ,see if they should be merged
    if tree.true_child.results is not None and tree.false_child.results is not None:
        # buld a combined dataset
        tb, fb = [], []
        for v, c in tree.true_child.results.items():
            tb += [[v]] * c
        for v, c in tree.false_child.results.items():
            fb += [[v]] * c

        # test the reduction in entropy
        delta = entropy(tb + fb) - (entropy(tb) + entropy(fb)) / 2.

        if delta < min_gain:  # merge
            tree.true_child, tree.false_child = None, None
            tree.results = unique_counts(tb + fb)


def mdclassify(observation, tree):
    """ Classify observations even if there are missing data points. """

    if tree.results is not None:
        return tree.results
    else:
        v = observation[tree.column]
        if v is None:
            tr, fr = mdclassify(observation, tree.true_child), mdclassify(observation, tree.false_child)

            t_count, f_count = sum(tr.values()), sum(fr.values())

            tw = float(t_count) / (t_count + f_count)
            fw = float(f_count) / (t_count + f_count)

            result = {}
            for k, v in tr.iteritems():
                result[k] = v * tw
            for k, v in fr.iteritems():
                result[k] = result.setdefault(k, 0) + (v * fw)
            return result
        else:
            if isinstance(v, int) or isinstance(v, float):
                if v >= tree.true_value:
                    branch = tree.true_child
                else:
                    branch = tree.false_child
            else:
                if v == tree.true_value:
                    branch = tree.true_child
                else:
                    branch = tree.false_child
            return mdclassify(observation, branch)


def variance(rows):
    if len(rows) == 0:
        return 0
    data = [float(row[-1]) for row in rows]
    mean = sum(data) / len(data)
    variance = sum([(d - mean) ** 2 for d in data]) / float(len(data))
    return variance