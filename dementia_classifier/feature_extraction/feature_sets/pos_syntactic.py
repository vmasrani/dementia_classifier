import dementia_classifier.lib.SCA.L2SCA.analyzeText as at
from dementia_classifier import settings

try:
    import cPickle as pickle
except:
    import pickle

auxiliary_dependencies = frozenset([
    'auxpass',
    'cop',
    'aux',
    'xcomp'
])


class tree_node():

    def __init__(self, key, phrase=None):
        self.key = key
        self.phrase = phrase
        self.children = []

    def addChild(self, node):
        self.children.append(node)


def build_tree(parse_tree):
    node_stack = []
    build_node = False
    node_type  = None
    phrase = None
    root_node = None
    encounter_leaf = False
    for ch in parse_tree:
        # If we encounter a ( character, start building a node
        if ch == '(':
            if node_type:
                # Finished building node
                node_type = node_type.strip()
                new_node = tree_node(node_type)
                node_stack.append(new_node)
            # Reset
            encounter_leaf = False
            build_node = True
            node_type = None
            phrase = None
            continue
        if ch == ')':
            # pop from the stack and add it to the children for the node before it
            if phrase:
                new_node = tree_node(node_type, phrase)
                node_stack.append(new_node)
            popped_node = node_stack.pop()
            if len(node_stack) > 0:
                parent = node_stack[-1]
                parent.addChild(popped_node)
            else:
                root_node = popped_node
            phrase = None
            node_type = None
            build_node = False
            encounter_leaf = False
            continue
        if encounter_leaf and build_node:
            if not phrase:
                phrase = ''
            phrase += ch
            continue
        if ch.isspace():
            encounter_leaf = True
            continue
        if build_node:
            if not node_type:
                node_type = ''
            node_type = node_type + ch
            continue
    return root_node


def get_height_of_tree(tree_node):

    depths = [0]
    for children in tree_node.children:
        depths.append(get_height_of_tree(children))
    depths = map(lambda x: x + 1, depths)
    return max(depths)


def get_count_of_parent_child(child_type, parent_type, tree_node, prev_type=None):
    curr_type = tree_node.key
    count = 0
    if prev_type == parent_type and curr_type == child_type:
        count = 1
    for child in tree_node.children:
        count += get_count_of_parent_child(child_type, parent_type, child, curr_type)
    return count


def get_count_of_parent_children(child_types, parent_type, tree_node):
    count = 0
    curr_type = tree_node.key
    if not len(tree_node.children):
        return count
    curr_children = [child.key for child in tree_node.children]
    if curr_type == parent_type and set(child_types).issubset(set(curr_children)):
        count = 1
    for child in tree_node.children:
        count += get_count_of_parent_children(child_types, parent_type, child)
    return count


def get_NP_2_PRP(tree_node):
    return get_count_of_parent_child('PRP', 'NP', tree_node)


def get_ADVP_2_RB(tree_node):
    return get_count_of_parent_child('RP', 'ADVP', tree_node)


def get_NP_2_DTNN(tree_node):
    return get_count_of_parent_children(['DT', 'NN'], 'NP', tree_node)


def get_VP_2_VBG(tree_node):
    return get_count_of_parent_child('VBG', 'VP', tree_node)


def get_VP_2_VBGPP(tree_node):
    return get_count_of_parent_child(['VBG', 'PP'], 'VP', tree_node)


def get_VP_2_AUXVP(tree_node, dependents):
    return get_VP_to_aux_and_more(tree_node, "VP", dependents)


def get_VP_2_AUXADJP(tree_node, dependents):
    return get_VP_to_aux_and_more(tree_node, "ADJP", dependents)


def get_VP_to_aux_and_more(tree_node, sibling_to_check, dependents):
    count = 0
    if tree_node.key == 'VP':
        # Check children phrase to see if it is inside the aux dependencies
        child_keys = []
        aux_present = False
        for child in tree_node.children:
            if child.phrase:  # If child phrase exists
                if child.phrase in dependents:
                    aux_present = True
            child_keys.append(child.key)
        # Check for condition
        if aux_present:
            child_keys = set(child_keys)
            if sibling_to_check in child_keys:
                count += 1
    for child in tree_node.children:
        count += get_VP_to_aux_and_more(child, sibling_to_check, dependents)

    return count


def get_aux_dependency_dependent(dependencies):
    dependents_list = []
    for dependency in dependencies:
        if dependency['dep'] in auxiliary_dependencies:
            dependents_list.append(dependency['dependentGloss'])
    return dependents_list


def get_VP_2_AUX(dependencies):
    # return number of aux dependencies
    # ----------- ASSUMING that aux dependencies always have a VP as a parent node
    count = 0
    for dependency in dependencies:
        if dependency['dep'] in auxiliary_dependencies:
            count += 1
    return count


def get_VP_2_VBDNP(tree_node):
    return get_count_of_parent_child(['VBD', 'NP'], 'VP', tree_node)


def get_INTJ_2_UH(tree_node):
    return get_count_of_parent_child('UH', 'INTJ', tree_node)


def get_ROOT_2_FRAG(tree_node):
    return get_count_of_parent_child('FRAG', 'ROOT', tree_node)


def get_all_syntactics_features(sample):
    # Make a temporary file for writing to
    sample_file_name   = settings.SCA_FOLDER + 'SCA_tmp_file.txt'
    sample_output_name = settings.SCA_FOLDER + 'SCA_tmp_output.txt'
    
    rawtext = ''.join([utterance['raw'] for utterance in sample])
    
    with open(sample_file_name, 'w+') as f:
        f.write(rawtext)

    with open(sample_output_name, 'w+') as output:
        at.analyze_file(sample_file_name, output, calling_dir=settings.SCA_FOLDER)

    with open(sample_output_name, 'r') as analyzed_file:
        headers = analyzed_file.readline().split(',')[1:]  # Headers
        headers = [s.replace("/", "_").strip() for s in headers]
        data    = [s.strip() for s in analyzed_file.readline().split(',')[1:]]  # actual data

    features = dict(zip(headers, data))
    return features


def get_number_of_nodes_in_tree(root_node):
    if len(root_node.children) == 0:
        return 1
    count = 1
    for child in root_node.children:
        count += get_number_of_nodes_in_tree(child)
    return count


def get_CFG_counts(root_node, dict):
    if dict.has_key(root_node.key):
        dict[root_node.key] += 1
    if len(root_node.children) > 0:  # Child leaf
        for child in root_node.children:
            dict = get_CFG_counts(child, dict)
    return dict


def get_all_tree_features(sample):
    features = {
        'tree_height': 0,
        'NP_to_PRP': 0,
        'ADVP_to_RB': 0,
        'NP_to_DT_NN': 0,
        'VP_to_AUX_VP': 0,
        'VP_to_VBG': 0,
        'VP_to_VBG_PP': 0,
        'VP_to_AUX_ADJP': 0,
        'VP_to_AUX': 0,
        'VP_to_VBD_NP': 0,
        'INTJ_to_UH': 0,
        'ROOT_to_FRAG': 0
    }
    total_nodes = 0
    for utterance in sample:
        for tree in range(0, len(utterance['parse_tree'])):
            parse_tree = utterance['parse_tree'][tree]
            root_node = build_tree(parse_tree)
            total_nodes += get_number_of_nodes_in_tree(root_node)
            features['tree_height'] += get_height_of_tree(root_node)
            features['NP_to_PRP'] += get_NP_2_PRP(root_node)
            features['ADVP_to_RB'] += get_ADVP_2_RB(root_node)
            features['NP_to_DT_NN'] += get_NP_2_DTNN(root_node)
            features['VP_to_VBG'] += get_VP_2_VBG(root_node)
            features['VP_to_VBG_PP'] += get_VP_2_VBGPP(root_node)
            features['VP_to_VBD_NP'] += get_VP_2_VBDNP(root_node)
            features['INTJ_to_UH'] += get_INTJ_2_UH(root_node)
            features['ROOT_to_FRAG'] += get_ROOT_2_FRAG(root_node)
            # Needs special love
            dependencies = utterance['basic_dependencies'][tree]
            features['VP_to_AUX'] += get_VP_2_AUX(dependencies)
            dependents = get_aux_dependency_dependent(dependencies)
            features['VP_to_AUX_VP'] += get_VP_2_AUXVP(root_node, dependents)
            features['VP_to_AUX_ADJP'] += get_VP_2_AUXADJP(root_node, dependents)

    #================ DIVIDING BY NUMBER OF total nodes in the sample ===============#
    for k, v in features.iteritems():
        features[k] /= float(total_nodes)

    return features


def get_all_CFG_features(sample):
    total_nodes = 0
    CFG_counts = {
        "ADJP": 0,
        "ADVP": 0,
        "CONJP": 0,
        "FRAG": 0,
        "INTJ": 0,
        "LST": 0,
        "NAC": 0,
        "NP": 0,
        "NX": 0,
        "PP": 0,
        "PRN": 0,
        "PRT": 0,
        "QP": 0,
        "RRC": 0,
        "UCP": 0,
        "VP": 0,
        "WHADJP": 0,
        "WHAVP": 0,
        "WHNP": 0,
        "WHPP": 0,
        "X": 0
    }
    for utterance in sample:
        for tree in range(0, len(utterance['parse_tree'])):
            parse_tree = utterance['parse_tree'][tree]
            root_node = build_tree(parse_tree)
            total_nodes += get_number_of_nodes_in_tree(root_node)
            CFG_counts = get_CFG_counts(root_node, CFG_counts)
    # ---- Normalize by total number of constituents in the sample
    for k, v in CFG_counts.iteritems():
        CFG_counts[k] /= float(total_nodes)
    return CFG_counts


def get_all(interview):
    feature_dict = get_all_syntactics_features(interview)
    feature_dict.update(get_all_tree_features(interview))
    feature_dict.update(get_all_CFG_features(interview))
    return feature_dict


def print_tree(root_node):
    queue = []
    queue.append(root_node)

    while len(queue) != 0:
        node = queue.pop(0)  # POP first element
        print("current node = " + node.key)
        if node.phrase:
            print("phrase = " + node.phrase)
        for child in node.children:
            queue.append(child)


if __name__ == '__main__':
    with open('../stanford/processed/pickles/dbank_control.pickle', 'rb') as handle:
        control = pickle.load(handle)
    test_set = control[1:]
    features = []
    for interview in test_set:
        features.append(get_all_CFG_features(interview))
