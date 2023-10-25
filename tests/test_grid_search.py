import unittest

import pytest
from core.grid_search import GridSearch

class TestGridSearch(unittest.TestCase):
    
    def test_get_permutations(self):        
        # Test case 1: Test with two parameters
        gs = GridSearch(param1=[0, 1], param2=[0, 1])
        expected_permutations = [
            {'param1': 0, 'param2': 0},
            {'param1': 0, 'param2': 1},
            {'param1': 1, 'param2': 0},
            {'param1': 1, 'param2': 1}
        ]
        self.assertEqual(gs.get_permutations(), expected_permutations)
        
        # Test case 2: Test with one parameter
        gs = GridSearch(param1=[0, 1])
        expected_permutations = [
            {'param1': 0},
            {'param1': 1}
        ]
        self.assertEqual(gs.get_permutations(), expected_permutations)
        
        # Test case 3: Test with three parameters
        gs = GridSearch(param1=[0, 1], param2=[0, 1], param3=[0, 1])
        expected_permutations = [
            {'param1': 0, 'param2': 0, 'param3': 0},
            {'param1': 0, 'param2': 0, 'param3': 1},
            {'param1': 0, 'param2': 1, 'param3': 0},
            {'param1': 0, 'param2': 1, 'param3': 1},
            {'param1': 1, 'param2': 0, 'param3': 0},
            {'param1': 1, 'param2': 0, 'param3': 1},
            {'param1': 1, 'param2': 1, 'param3': 0},
            {'param1': 1, 'param2': 1, 'param3': 1}
        ]
        self.assertEqual(gs.get_permutations(), expected_permutations)
                
    def test_get_permutations_with_empty_params(self):
        # Test case: Test with empty parameters
        with pytest.raises(ValueError):
            gs = GridSearch()
            gs.get_permutations()

if __name__ == '__main__':
    unittest.main()