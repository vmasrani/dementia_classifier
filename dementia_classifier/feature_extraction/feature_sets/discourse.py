import os
from dp_tree import DiscourseParseTree

DISCOURSE_RELATIONS = ["Attribution",
                       "Background",
                       "Cause",
                       "Comparison",
                       "Condition",
                       "Contrast",
                       "Elaboration",
                       "Enablement",
                       "Evaluation",
                       "Explanation",
                       "Joint",
                       "Manner-Means",
                       "Same-Unit",
                       "Summary",
                       "Temporal",
                       "TextualOrganization",
                       "Topic-Change",
                       "Topic-Comment"]

EDU_BREAK = 'EDU_BREAK'


def get_depth(dpt):
    return dpt.root.depth()


def get_discourse_relations(dpt):
    d = dict.fromkeys(DISCOURSE_RELATIONS, 0)
    for edge in dpt.getAllEdges():
        if edge != 'span':
            d[edge] += 1
    return d


# returns avg # of EDUs / utterance
def get_edu_rate(sentences):
    edus = []
    for l in sentences:
        edus.append(1 + l.count(EDU_BREAK))
    return sum(edus) / float(len(edus))


def get_all(path):
    # Extract data from discourse directory
    if os.path.exists(path):
        parsed_data = {}

        tree_dir = os.path.join(path, 'doc')
        seg_dir = os.path.join(path, 'seg')
        # Array containing full file paths
        tree_paths = [os.path.join(tree_dir, tree) for tree in os.listdir(tree_dir)]
        seg_paths = [os.path.join(seg_dir, seg) for seg in os.listdir(seg_dir)]

        for tree, seg in zip(tree_paths, seg_paths):
            with open(tree) as f:
                dpt = DiscourseParseTree(f.readlines())
            with open(seg) as f:
                sentences = f.readlines()
            key = tree.split('/')[-1].split('_')[0]
            print "Processing %s ..." % key
            features = {}
            features["depth"]    = get_depth(dpt)
            features["edu_rate"] = get_edu_rate(sentences)
            features["number_of_utterances"] = len(sentences)

            discourse_relations = get_discourse_relations(dpt)
            n_relations = float(sum(discourse_relations.values()))
            discourse_ratios = {key + "_ratio": value / n_relations for (key, value) in discourse_relations.iteritems()}
            features.update(discourse_relations)
            features.update(discourse_ratios)
            
            # Calculate type to token ratio
            dtypes = [c for c in discourse_relations.values() if c is not 0]
            features["discourse_type_token_ratio"] = len(dtypes) / float(sum(dtypes))
            
            parsed_data[key] = features
    else:
        raise IOError("File not found: " + path + " does not exist")
    return parsed_data


if __name__ == '__main__':
    get_all("../data/dbank/discourse_trees/control")
