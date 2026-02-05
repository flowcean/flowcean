#!/usr/bin/env python3
import random
import math
from flowcean.testing.generator.ddtig.domain.model_analyser.surrogate.interval import Interval, IntervalEndpoint
from flowcean.testing.generator.ddtig.infrastructure.utils import reverse_list_by_value
from flowcean.testing.generator.ddtig.infrastructure import TestLogger

class TestGenerator():
    """
    A class that generates abstract test inputs for binary decision trees.

    Attributes
    ----------
    equivalence_classes : list
        Equivalence classes extracted from the decision tree.
        
    type_specs : dict
        Input types for each feature as defined in the specifications.

    testplans : list
        List of test plans used to sample test inputs.

    n_testinputs_lst : list
        Number of test inputs to generate for each equivalence class.

    logger : TestLogger
        Logger for tracking the test input generation process.

    Methods
    -------
    generate_testinputs()
        Generates abstract test inputs based on the selected coverage strategy.
    """


    def __init__(
        self,
        equivalence_classes: list,
        type_specs: dict,
        logger: TestLogger
    ) -> None:
        """
        Initializes the Test Generator.

        Args:
            equivalence_classes : Equivalence classes extracted from the decision tree.
            type_specs : Input types for each feature from the specifications.
            logger : Logger for logging the generation process.
        """
        self.equivalence_classes = equivalence_classes
        self.type_specs = type_specs
        self.testplans = []
        self.logger = logger


    # Generates a test plan for Boundary Value Analysis (BVA).
    # BVA samples around the boundaries of each interval.
    def _bva_testplan(self, epsilon: float, eqclass_interval: Interval) -> tuple:
        if (eqclass_interval.left_endpoint == IntervalEndpoint.LEFT_OPEN or
            eqclass_interval.right_endpoint == IntervalEndpoint.RIGHT_OPEN):
            test_ranges = []
            if eqclass_interval.left_endpoint == IntervalEndpoint.LEFT_OPEN:
                min_lower = eqclass_interval.min_value - epsilon
                min_upper = eqclass_interval.min_value + epsilon
                test_ranges.append((min_lower, min_upper))
            if eqclass_interval.right_endpoint == IntervalEndpoint.RIGHT_OPEN:
                max_lower = eqclass_interval.max_value - epsilon
                max_upper = eqclass_interval.max_value + epsilon
                test_ranges.append((max_lower, max_upper))
            return tuple(test_ranges)
        else:
            return ((eqclass_interval.min_value, eqclass_interval.max_value),)
    

    # Generates a test plan for Decision Tree Coverage (DTC).
    # Ensures at least one test input covers each path in the tree.
    def _dtc_testplan(self, eqclass_interval: Interval) -> tuple:
        if eqclass_interval.left_endpoint == IntervalEndpoint.LEFT_OPEN:
            if (self.type_specs[eqclass_interval.feature]['type'] == "int"):
                lower = math.floor(eqclass_interval.min_value) + 1
            else:
                lower = math.nextafter(eqclass_interval.min_value, eqclass_interval.max_value)
        else:
            lower = eqclass_interval.min_value
        upper = eqclass_interval.max_value
        return (lower, upper)


    # Samples random values from a given interval for a specific feature.
    def _generate_randoms(self, lower, upper, n_testinputs, feature: int) -> list:
        n_testinputs = int(n_testinputs)
        if lower > upper:
            tmp = upper
            upper = lower
            lower = tmp
        if lower == upper:
            return [lower]*n_testinputs
        elif self.type_specs[feature]['type'] == "int":
            return [random.randint(math.ceil(lower), math.floor(upper)) for _ in range(n_testinputs)]
        else:
            return [random.uniform(lower, upper) for _ in range(n_testinputs)]


    # Generates test inputs for a single equivalence class.
    def _generate_testinputs_eqclass(self,
                                    n_testinputs: int,
                                    test_coverage_criterium: str,
                                    eqclass: tuple,
                                    epsilon: float = 0.0) -> list:
        input_samples = []
        testplan = []
        for interval in eqclass:
            if test_coverage_criterium == "dtc":
                lower, upper = self._dtc_testplan(interval)
                testplan.append((lower, upper))
                input_samples.append(self._generate_randoms(lower, upper, n_testinputs, interval.feature))
            else:
                test_ranges = self._bva_testplan(epsilon, interval)
                testplan.append(test_ranges)
                if (len(test_ranges) == 2):
                    half_n = n_testinputs / 2
                    if (n_testinputs % 2 != 0):
                        ns = [int(half_n+0.5), int(half_n-0.5)]
                    else:
                        ns = [half_n, half_n]
                    input_samples_sublsts = []
                    n = ns[0]
                    for lower, upper in test_ranges:
                        input_samples_sublsts += self._generate_randoms(lower, upper, n, interval.feature)
                        n = ns[1]
                    input_samples.append(input_samples_sublsts)
                else:
                    lower, upper = test_ranges[0]
                    input_samples.append(self._generate_randoms(lower, upper, n_testinputs, interval.feature))
        testinputs = list(zip(*input_samples))
        self.testplans.append(testplan)
        return testinputs
    
    
    # Computes the number of test inputs to generate per equivalence class.
    def _generate_n_testinputs_list(self,
                                   n_testinputs: int,
                                   eqclass_prio: list,
                                   inverse_alloc: bool) -> list:
        n_testinputs_lst = []
        for prio in eqclass_prio[:-1]:
            n_testinputs_lst.append(round(n_testinputs*prio))
        n_testinputs_lst.append(n_testinputs-sum(n_testinputs_lst))
        if inverse_alloc:
            n_testinputs_lst = reverse_list_by_value(n_testinputs_lst)
        return n_testinputs_lst

    
    def generate_testinputs(self,
                           test_coverage_criterium: str,
                           eqclass_prio: list,
                           n_testinputs: int,
                           inverse_alloc: bool,
                           epsilon: float) -> list:
        """
        Generates abstract test inputs for all equivalence classes.

        Args:
            test_coverage_criterium : Coverage strategy ("bva" or "dtc").
            eqclass_prio : Importance scores for each equivalence class.
            n_testinputs : Total number of test inputs to generate.
            inverse_alloc : If True, allocate more inputs to less important classes.
            epsilon : Offset for BVA sampling.

        Returns:
            List of abstract test inputs. Each test input is a tuple of feature values.
            E.g.: [(1,2,3), (11,22,33), (87,29,38)]
        """
        testinputs = []
        self.n_testinputs_lst = self._generate_n_testinputs_list(n_testinputs, eqclass_prio, inverse_alloc)
        for eqclass, n_testinputs in zip(self.equivalence_classes, self.n_testinputs_lst):
            testinputs_eqclass = self._generate_testinputs_eqclass(n_testinputs, test_coverage_criterium, eqclass, epsilon)
            testinputs += testinputs_eqclass
        if self.logger:
            self.logger.log_debug("Generated test inputs for all equivalence classes successfully.")
        return testinputs