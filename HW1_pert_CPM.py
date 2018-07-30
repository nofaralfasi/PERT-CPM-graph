import logging
import unittest
import itertools

def log_with_msg(msg):
    def log(func):
        def wrapper(self, *args, **kwargs):
            logging.getLogger(__name__ + ": ").info(msg)
            return func(self, *args, **kwargs)
        return wrapper
    return log

class Activity:
    """
        The Activity class
        -------------------
        represents the edges between each node
        Each activity has a unique name and its duration in the project
        Activities will be equal if their name and their duration is the same
    """

    @log_with_msg("Initializing Activity")
    def __init__(self, name, duration):
        self._name = name
        self._duration = duration

    @log_with_msg("Returning Activity repr")
    def __repr__(self) -> str:
        return f"<{self.name}, {self.duration} weeks>"

    @log_with_msg("Returning Activity str")
    def __str__(self) -> str:
        return f"<{self.name}, {self.duration} weeks>"

    @property
    def name(self) -> str:
        return self._name

    @name.setter
    def name(self, name):
        self._name = name

    @property
    def duration(self) -> float:
        return self._duration

    @duration.setter
    def duration(self, duration):
        self._duration = duration

    @log_with_msg("Comparing Activities")
    def __eq__(self, other) -> bool:
        return self.name == other.name and self.duration == other.duration

    def __ne__(self, other) -> bool:
        return not self == other


class Node:
    """
        The Node class
        --------------
        represents the actual node in the graph.
        knows the early and late finish times and also knows the slack time
        Each node is unique and is recognized by its number.
        A node is equal to another node if their number is equal (regardless of the other properties)
        It is set that way to keep the nodes unique
        Each node has an optional parallel node. If a node has a parallel node,
        both activities leading to those nodes must be completed together
    """

    @log_with_msg("Initializing Node")
    def __init__(self, number: int, *parallel_nodes: "List of Nodes"):
        self._number = number
        self._early_finish = 0
        self._late_finish = 0
        self._slack = 0
        self._parallel_nodes = parallel_nodes

    @log_with_msg("Returning Node repr")
    def __repr__(self) -> repr:
        return f"(Node {self.number})"

    @log_with_msg("Returning Node str")
    def __str__(self) -> str:
        string = f"(Node {self.number})"
        if not (self.late_finish == self.early_finish == 0):
            string += f"{[self.early_finish,self.late_finish]}"
        if self.has_parallel_nodes():
            string += " <---> {"
            for node in list(self.parallel_nodes)[:-1]:
                string += f"{node}, "
            string += f"{self.parallel_nodes[-1]}" + "}"
        return string

    @property
    def early_finish(self) -> float:
        return self._early_finish

    @early_finish.setter
    def early_finish(self, early_finish):
        self._early_finish = early_finish

    @property
    def late_finish(self) -> float:
        return self._late_finish

    @late_finish.setter
    def late_finish(self, late_finish):
        self._late_finish = late_finish

    @property
    def slack(self) -> float:
        return self._slack

    @slack.setter
    def slack(self, slack: float):
        self._slack = slack

    @property
    def number(self) -> int:
        return self._number

    @number.setter
    def number(self, number: int):
        self._number = number

    @property
    def parallel_nodes(self) -> tuple:
        return self._parallel_nodes

    @parallel_nodes.setter
    def parallel_nodes(self, *parallel_nodes: tuple):
        self._parallel_nodes = parallel_nodes

    @log_with_msg("Checking if Node has parallel nodes")
    def has_parallel_nodes(self) -> bool:
        return list(self.parallel_nodes) != []

    @log_with_msg("Comparing Nodes")
    def __eq__(self, other) -> bool:
        return self.number == other.number

    def __ne__(self, other) -> bool:
        return not self == other

    @log_with_msg("Hashing Node")
    def __hash__(self) -> float:
        return hash(self.number)

    # For sorting purposes
    @log_with_msg("Checking what node is bigger")
    def __lt__(self, other) -> bool:
        return self.number < other.number

class Transition:
    """
        Transition class
        ----------------
        represents the transitions from one node to another.
        keeps track of the activity, the start node and the target node
    """

    @log_with_msg("Initializing Transition")
    def __init__(self, from_node: Node, activity: Activity, to_node: Node):
        self._from_node = from_node
        self._activity = activity
        self._to_node = to_node

    @log_with_msg("Returning Transition repr")
    def __repr__(self) -> repr:
        return f"({repr(self._from_node)}, {self._activity}, {repr(self._to_node)})"

    @log_with_msg("Returning Transition str")
    def __str__(self) -> str:
        return f" {self.from_node} -> {self._activity} -> {self._to_node}"

    @property
    def from_node(self) -> Node:
        return self._from_node

    @from_node.setter
    def from_node(self, from_node: Node):
        self._from_node = from_node

    @property
    def to_node(self) -> Node:
        return self._to_node

    @to_node.setter
    def to_node(self, to_node: Node):
        self._to_node = to_node

    @property
    def activity(self) -> Activity:
        return self._activity

    @activity.setter
    def activity(self, activity: Activity):
        self._activity = activity

    @log_with_msg("Comparing Transitions")
    def __eq__(self, other) -> bool:
        return self.activity == other.activity

    def __ne__(self, other) -> bool:
        return not self == other


class Project:
    """
    The Pert class
    --------------
    The class which represents the pert, using a graph.
    The graph is a dictionary of {Node : list(Transition)} - each node with the corresponding transitions from it.
    If no graph was passed to the constructor, an empty graph is initialized
    """

    @log_with_msg("Initializing new PERT")
    def __init__(self, graph: dict = None):
        self._graph = graph if graph is not None else {}
        self._all_nodes = []
        self._all_paths = []
        self._all_transition = []
        self._all_activities = []
        self._slack_list = []
        self._isolated_list = []
        self._critical_paths = []
        self._start = None
        self._end = None
        self.update()

    @log_with_msg("Printing PERT")
    def __str__(self) -> str:
        string = '!!WARNING: Invalid Graph!!' if not self.is_valid() else ''
        for path in self.all_paths:
            string += f"\nCRITICAL PATH: " if path in self.critical_paths else f"\n" + "\t" * 3 + " " * 3
            for count, n in enumerate(path[:-1]):
                if n == self.start:
                    string += f"{([trans for trans in self.graph[path[count]] if trans.to_node == path[count + 1]])[0]}"
                elif self.end is not None and n == self.end:
                    string += f" -> {self.graph[path[count-1]][0].activity} -> {n}"
                else:
                    for trans in self.graph[n]:
                        if trans.to_node == path[count + 1]:
                            string += f"-> {trans.activity} -> {trans.to_node}"
                            break
            string += '\n'
        return string

    @property
    def graph(self) -> dict:
        return self._graph

    @graph.setter
    def graph(self, graph: dict):
        self._graph = graph
        self.update() if graph else self.__nullify_graph__()

    @property
    def all_nodes(self) -> list:
        return self._all_nodes

    @all_nodes.setter
    def all_nodes(self, all_nodes: list):
        self._all_nodes = all_nodes

    @property
    def all_paths(self) -> list:
        return self._all_paths

    @all_paths.setter
    def all_paths(self, all_paths: list):
        self._all_paths = all_paths

    @property
    def all_transition(self) -> list:
        return self._all_transition

    @all_transition.setter
    def all_transition(self, all_transition: list):
        self._all_transition = all_transition

    @property
    def all_activities(self) -> list:
        return self._all_activities

    @all_activities.setter
    def all_activities(self, all_activities: list):
        self._all_activities = all_activities

    @property
    def slack_list(self) -> list:
        return self._slack_list

    @slack_list.setter
    def slack_list(self, slack_list: list):
        self._slack_list = slack_list

    @property
    def isolated_list(self) -> list:
        return self._isolated_list

    @isolated_list.setter
    def isolated_list(self, isolated_list: list):
        self._isolated_list = isolated_list

    @property
    def critical_paths(self) -> list:
        return self._critical_paths

    @critical_paths.setter
    def critical_paths(self, critical_paths: list):
        self._critical_paths = critical_paths

    @property
    def start(self) -> Node:
        return self._start

    @start.setter
    def start(self, start: Node):
        self._start = start

    @property
    def end(self) -> Node:
        return self._end

    @end.setter
    def end(self, end: Node):
        self._end = end

    # nullifies the graph's properties
    @log_with_msg("Nullifying PERT")
    def __nullify_graph__(self):
        self.all_nodes = []
        self.all_transition = []
        self.isolated_list = []
        self.all_paths = []
        self.all_activities = []
        self.slack_list = []
        self.critical_paths = []
        self.start = None
        self.end = None

    # calculates the early finished, the late finishes, the slack times and the duration of the project
    @log_with_msg("Updating PERT")
    def update(self):
        if self.graph is not None:
            self.all_nodes = self.__get_all_nodes__()
            self.all_transition = self.__get_transition_list__()
            self.isolated_list = self.__get_isolated_nodes__()
            self.start = self.__get_start_node__()
            self.end = self.__get_end_node__()
            self.all_paths = self.__find_all_paths__(self.start)
            self.all_activities = self.__get_activities_list__()
            self.__calc_early_finishes__()
            self.__calc_late_finishes__()
            self.__calc_slack_times__()
            self.critical_paths = self.__get_critical_paths__()
            self.slack_list = self.__get_all_slacks__()

    # Return the length of the project
    @log_with_msg("Returning length of PERT")
    def __len__(self) -> float:
        return self.end.late_finish if self.graph is not None else 0

    # Returns a node from the graph which his number is node_number
    # @:param node_number - the number of the node which we want to retrieve
    @log_with_msg("Retrieving Node")
    def get_node_number(self, node_number: int) -> list or None:
        for node in self.all_nodes:
            if node.number == node_number:
                return node
        return None

    #  Adds a new activity to the project.
    #  @:param
    #  from_node - the node number from which the activity is going
    #  activity - the activity itself
    #  to_node - the node number to which the activity is going
    @log_with_msg("Adding Activity")
    def add_activity(self, from_node: int, activity: Activity, to_node: int):
        f_node = self.get_node_number(from_node)
        t_node = self.get_node_number(to_node)
        transition = Transition(f_node if f_node else Node(from_node), activity, t_node if t_node else Node(to_node))
        if transition not in self._all_transition:
            self.graph[transition.from_node] = self.graph[transition.from_node] + [
                transition] if transition.from_node in self.all_nodes else [transition]
            if transition.to_node not in self.all_nodes:
                self.graph[transition.to_node] = []
            self.update()

    # adds an arbitrary amount of transitions to the graph
    # @:param *args - list of transitions to be added to the graph
    def add_activities(self, *args: "List of Transitions"):
        for transition in args:
            self.add_activity(transition.from_node.number, transition.activity, transition.to_node.number)

    # Removes a transition from the graph which his activity is the argument passed, thus removing the activity too
    # @:param activity - the activity whom transition is deleted
    @log_with_msg("Deleting Activity")
    def del_activity(self, activity: Activity):
        for transitions in self.graph.values():
            for transition in transitions:
                if activity == transition.activity:
                    transitions.remove(transition)
        self.update()

    # Returns an activity list
    @log_with_msg("Getting Activity list")
    def __get_activities_list__(self) -> list:
        return [transition.activity for transition in self.all_transition]

    # Return a list of all nodes, including isolated nodes
    @log_with_msg("Getting all nodes")
    def __get_all_nodes__(self) -> list:
        return list(self.graph.keys()) if self.graph is not None else []

    # Returns the transition list
    @log_with_msg("Getting Transition list")
    def __get_transition_list__(self) -> list:
        return list(itertools.chain(*self.graph.values())) if self.graph is not None else []

    # Returns a list of isolated nodes =
    # nodes which none of the activities are going to, and none of the activities are going from
    @log_with_msg("Getting isolated nodes")
    def __get_isolated_nodes__(self) -> list:
        return [node for node in self.all_nodes if
                not self.graph[node] and node not in [tr.to_node for tr in self.all_transition]]

    # Returns the critical paths in the project
    # By definition - a critical path is a path which every node in it has 0 slack time
    @log_with_msg("Getting critical paths")
    def __get_critical_paths__(self) -> list:
        return [path for path in self.all_paths if not [node.slack for node in path if node.slack is not 0]]

    # Returns true if and only if this graph is valid, aka - has no cycles in it
    # NOTE : a cyclic path in the graph is for example :
    #       Node1->Node2->Node3->Node4->Node2
    @log_with_msg("Checking if valid")
    def is_valid(self) -> bool:
        return True not in [len(set(path)) < len(path) for path in self.all_paths]

    # Returns a sorted list of slack
    @log_with_msg("Getting all slack times")
    def __get_all_slacks__(self) -> list:
        return sorted([node.slack for node in self.all_nodes if node.slack is not 0], reverse=True)

    # Returns the starting node, not including isolated nodes
    @log_with_msg("Getting start nodes")
    def __get_start_node__(self) -> Node:
        for node in self.all_nodes:
            if node not in [tr.to_node for tr in self.all_transition] and node not in self.isolated_list:
                return node

    # Returns the ending node, not including isolated nodes
    # NOTICE: if the graph is cyclic, there might not be an end node, in this case, the returned value will be None
    @log_with_msg("Getting end node")
    def __get_end_node__(self) -> Node or None:
        for node in self.all_nodes:
            if not self.graph[node] and not node.has_parallel_nodes() and node not in self.isolated_list:
                return node
        return None

    # Calculates the early finishes possible
    @log_with_msg("Calculating early finishes")
    def __calc_early_finishes__(self):
        for node in list(itertools.chain(*self.all_paths)):
            for transition in self._graph[node]:
                transition.to_node.early_finish = transition.activity.duration + transition.from_node.early_finish \
                    if transition.to_node.early_finish is 0 else max(transition.to_node.early_finish,
                                                                     transition.from_node.early_finish +
                                                                     transition.activity.duration)
                for par_node in transition.to_node.parallel_nodes:
                    self.get_node_number(par_node.number).early_finish = max(transition.to_node.early_finish,
                                                                             par_node.early_finish)

    # Calculates the latest finishes possible
    @log_with_msg("Calculating late finishes")
    def __calc_late_finishes__(self):
        if self.end is not None:
            self.end.late_finish = self.end.early_finish
        for node in reversed(list(itertools.chain(*self.all_paths))):
            for transition in reversed(self.graph[node]):
                if transition.to_node.has_parallel_nodes():
                    late = min(
                        [self.get_node_number(par.number).late_finish for par in transition.to_node.parallel_nodes])
                    # if we haven't calculated late finish yet or if the late is smaller than the current late finish
                    if transition.to_node.late_finish is 0 or transition.to_node.late_finish > late:
                        transition.to_node.late_finish = late

                # if to_node.late_finish still 0, we can't compute its from_node.late_finish yet...
                if transition.to_node.late_finish is not 0:
                    transition.from_node.late_finish = transition.to_node.late_finish - transition.activity.duration \
                        if transition.from_node.late_finish is 0 and transition.from_node != self.start \
                        else min(transition.from_node.late_finish,
                                 transition.to_node.late_finish - transition.activity.duration)

    # Calculates the slack times for each node
    @log_with_msg("Calculating slack times")
    def __calc_slack_times__(self):
        for node in self.all_nodes:
            node.slack = node.late_finish - node.early_finish

    # Finds all the paths in this project
    # The search will not include paths with isolated nodes.
    @log_with_msg("Finding all paths")
    def __find_all_paths__(self, start_node: Node, path: list = None) -> list:
        graph = self.graph
        path = path if path is not None else []
        if start_node in path or not graph[start_node]:
            return [path + [start_node]]
        path = path + [start_node]
        if start_node not in graph:
            return []
        paths = []
        for transition in graph[start_node]:
            paths += [path for path in self.__find_all_paths__(transition.to_node, path)]
        return paths

    # Implementation of the contains method.
    # Returns true if and only if the item is in this graph
    # An item can be of class Node, Activity or Transition
    @log_with_msg("Checking if item is in PERT")
    def __contains__(self, item) -> bool:
        if not (isinstance(item, Node) or isinstance(item, Activity) or isinstance(item, Transition)):
            raise PermissionError("this item doesnt belong to the pert!")
        return self.get_node_number(item.number) is not None if isinstance(item, Node) else \
            item in self.all_activities if isinstance(item, Activity) else item in self.all_transition


"""
    Test Classes - for your convenience
"""


class TestPert(unittest.TestCase):
    node_list = [Node(-1)] + [Node(i) for i in range(1, 12)]
    transition_list = [
        Transition(node_list[1], Activity('Formalize specs', 6), node_list[2]),
        Transition(node_list[2], Activity('Design system', 4), node_list[3]),
        Transition(node_list[2], Activity('Certification Requirements', 3), node_list[4]),
        Transition(node_list[2], Activity('Design Software', 6), node_list[6]),
        Transition(node_list[3], Activity('Prototype System', 2), node_list[5]),
        Transition(node_list[4], Activity('Certification Documentation', 4), node_list[9]),
        Transition(node_list[9], Activity('Certification Application', 1), node_list[10]),
        Transition(node_list[10], Activity('Complete Certification', 4), node_list[11]),
        Transition(node_list[6], Activity('Code Software', 4), node_list[8]),
        Transition(node_list[8], Activity('Complete Software', 1), node_list[11]),
        Transition(node_list[5], Activity('Test System', 3), node_list[6]),
        Transition(node_list[6], Activity('Release System', 2), node_list[7]),
        Transition(node_list[7], Activity('Manufacture System', 4), node_list[11]),
    ]

    @staticmethod
    def create_new_graph():
        graph = {}
        for transition in TestPert.transition_list:
            if transition.from_node in graph.keys():
                graph[transition.from_node].append(transition)
            else:
                graph[transition.from_node] = [transition]
        graph[TestPert.node_list[11]] = []
        return graph

    def setUp(self):
        self.pert = Project(TestPert.create_new_graph())

    def tearDown(self):
        self.pert = None

    # Tests for first graph
    def test_starts(self):
        self.assertEqual([(node.early_finish, node.late_finish) for node in sorted(self.pert.all_nodes)],
                         [(0, 0), (6, 6), (10, 10), (9, 12), (12, 12),
                          (15, 15), (17, 17), (19, 20), (13, 16), (14, 17),
                          (21, 21)])

    def test_project_duration(self):
        self.assertEqual(21, len(self.pert))

    def test_isolated_activities(self):
        self.pert.graph[Node(14)] = []
        self.pert.update()
        self.assertEqual([node.number for node in self.pert.isolated_list], [14])

    def test_add_activity(self):
        self.pert.add_activity(11, Activity("Test Activity", 2), 12)
        self.assertEqual(12, self.pert.__get_end_node__().number)
        self.assertEqual(len(self.pert), 23)

    def test_del_activity(self):
        self.pert.del_activity(Activity('Design Software', 6))
        self.assertNotIn([1, 2, 6, 8, 11], [[node.number for node in path] for path in self.pert.all_paths])
        self.pert.add_activity(2, Activity('Design Software', 6), 6)
        self.pert.del_activity(Activity('Formalize specs', 6))
        self.assertEqual(2, self.pert.start.number)
        self.assertIn(1, [node.number for node in self.pert.isolated_list])

    def test_critical_path(self):
        self.assertEqual([[node.number for node in path] for path in self.pert.critical_paths],
                         [[1, 2, 3, 5, 6, 7, 11], [1, 2, 6, 7, 11]])

    def test_valid_graph(self):
        self.assertEqual(self.pert.is_valid(), True)

    def test_invalid_graph(self):
        graph = {Node(1): [Transition(Node(1), Activity("Test1", 1), Node(2))],
                 Node(2): [Transition(Node(2), Activity("Test2", 2), Node(3))],
                 Node(3): [Transition(Node(3), Activity("Test3", 3), Node(4))],
                 Node(4): [Transition(Node(4), Activity("Test4", 4), Node(2))]}
        self.pert.graph = graph
        self.assertEqual(self.pert.is_valid(), False)

    def test_slack_list(self):
        self.assertEqual(self.pert.slack_list, [3, 3, 3, 1])


class TestPert2(unittest.TestCase):
    node_list_2 = [Node(0),
                   Node(1, Node(2)),
                   Node(2),
                   Node(3),
                   Node(4, Node(3), Node(6)),
                   Node(5, Node(6)),
                   Node(6),
                   Node(7, Node(8)),
                   Node(8)]

    transition_list_2 = [
        Transition(node_list_2[0], Activity("Task1", 4), node_list_2[1]),
        Transition(node_list_2[0], Activity("Task5", 6), node_list_2[2]),
        Transition(node_list_2[0], Activity("Task9", 5), node_list_2[3]),
        Transition(node_list_2[2], Activity("Task2", 2), node_list_2[4]),
        Transition(node_list_2[3], Activity("Task6", 4), node_list_2[6]),
        Transition(node_list_2[4], Activity("Task8", 5), node_list_2[5]),
        Transition(node_list_2[4], Activity("Task10", 8), node_list_2[8]),
        Transition(node_list_2[4], Activity("Task3", 2), node_list_2[7]),
        Transition(node_list_2[5], Activity("Task4", 5), node_list_2[8]),
        Transition(node_list_2[6], Activity("Task7", 6), node_list_2[8]),
    ]

    @staticmethod
    def create_graph_with_parallels():
        graph = {}
        for transition in TestPert2.transition_list_2:
            if transition.from_node in graph.keys():
                graph[transition.from_node].append(transition)
            else:
                graph[transition.from_node] = [transition]
        graph[TestPert2.node_list_2[1]] = []
        graph[TestPert2.node_list_2[7]] = []
        graph[TestPert2.node_list_2[8]] = []
        return graph

    def setUp(self):
        self.pert = Project(TestPert2.create_graph_with_parallels())

    def tearDown(self):
        self.pert = None

    # Tests for second graph:
    def test_graph_two_starts(self):
        self.assertEqual([(node.early_finish, node.late_finish) for node in sorted(self.pert.all_nodes)],
                         [(0, 0), (4, 6), (6, 6), (8, 9), (8, 8), (13, 13), (13, 13), (10, 19), (19, 19)])

    def test_graph_two_length(self):
        self.assertEqual(len(self.pert), 19)

    def test_graph_two_critical_paths(self):
        self.assertEqual([[node.number for node in path] for path in self.pert.critical_paths],
                         [[0, 2, 4, 5, 8], [0, 2, 4, 8]])

    def test_graph_two_isolated_activities(self):
        self.assertEqual(self.pert.isolated_list, [])

    def test_graph_two_valid_graph(self):
        self.assertEqual(self.pert.is_valid(), True)

    def test_add_new_activity_to_graph_two(self):
        self.pert.add_activity(8, Activity("Task12", 5), 9)
        self.assertEqual(9, self.pert.end.number)
        self.assertNotEqual(8, self.pert.end.number)
        self.assertEqual(len(self.pert), 24)

    def test_graph_two_del_activity(self):
        self.pert.del_activity(Activity('Task3', 2))
        self.assertNotIn([0, 2, 4, 7, 8], [[node.number for node in path] for path in self.pert.all_paths])

    def test_graph_two_slack_list(self):
        self.assertEqual(self.pert.slack_list, [9, 2, 1])

    def test_empty_graph(self):
        self.pert.graph = None
        self.assertEqual(len(self.pert), 0)
        self.assertEqual(self.pert.all_nodes, [])
        self.assertEqual(self.pert.isolated_list, [])
        self.assertEqual(self.pert.slack_list, [])


def read():
    with open('HW1_Nofar_Alfasi_Pert_CPM.log') as f:
        print("log file content:")
        print(f.read())


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, filename='HW1_Nofar_Alfasi_Pert_CPM.log', filemode='w', format='%(name)s %(message)s')
    print(Project(TestPert.create_new_graph()))
    print(Project(TestPert2.create_graph_with_parallels()))
    unittest.main()
