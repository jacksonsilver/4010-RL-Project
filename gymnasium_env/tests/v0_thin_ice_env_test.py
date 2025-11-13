import unittest
from gymnasium_env.envs.v0_thin_ice_env import ThinIceEnv

class TestActionMaskCalculations(unittest.TestCase):
    '''
        For an action mask: 1010, this means:
            1 - Left is available
            0 - Down is not available
            1 - Right is available
            0 - Up is not available.
        This action mask has a value of 10 (1010 in binary)
    '''

    def test_none_available_returns_only_zero(self):
        masks = ThinIceEnv.get_all_possible_masks(0) # in binary: 0000
        self.assertEqual(masks, [0])

    def test_all_available_returns_all_submasks(self):
        masks = ThinIceEnv.get_all_possible_masks(15) # in binary: 1111
        self.assertEqual(len(masks), 16)
        self.assertEqual(set(masks), set(range(16)))

    def test_left_right_available(self):
        masks = ThinIceEnv.get_all_possible_masks(10) # in binary: 1010
        expected = [10, 8, 2, 0] # expected submasks: 1010 (10), 1000 (8), 0010 (2), 0000 (0)
        self.assertEqual(masks, expected)

    def test_down_available(self):
        masks = ThinIceEnv.get_all_possible_masks(4) # in binary: 0100
        expected = [4, 0]
        self.assertEqual(masks, expected)

    def test_up_available(self):
        masks = ThinIceEnv.get_all_possible_masks(1) # in binary: 0001
        expected = [1, 0]
        self.assertEqual(masks, expected)

    def test_left_available(self):
        masks = ThinIceEnv.get_all_possible_masks(8) # in binary: 1000
        expected = [8, 0]
        self.assertEqual(masks, expected)

    def test_right_up_available(self):
        masks = ThinIceEnv.get_all_possible_masks(3) # in binary: 0011
        expected = [3, 2, 1, 0]
        self.assertEqual(masks, expected)
        self.assertEqual(len(masks), 4)

    def test_down_up_available(self):
        masks = ThinIceEnv.get_all_possible_masks(5) # in binary: 0101
        expected = [5, 4, 1, 0]
        self.assertEqual(masks, expected)
        self.assertEqual(len(masks), 4)

    def test_down_right_up_available(self):
        masks = ThinIceEnv.get_all_possible_masks(7) # in binary: 0111
        expected = [7, 6, 5, 4, 3, 2, 1, 0]
        self.assertEqual(masks, expected)
        self.assertEqual(len(masks), 8)

    def test_left_up_available(self):
        masks = ThinIceEnv.get_all_possible_masks(9) # in binary: 1001
        expected = [9, 8, 1, 0]
        self.assertEqual(masks, expected)
        self.assertEqual(len(masks), 4)

    def test_no_duplicate_submasks(self):
        for mask_val in [0, 1, 3, 7, 15]:
            masks = ThinIceEnv.get_all_possible_masks(mask_val)
            self.assertEqual(len(masks), len(set(masks)), 
                           f"Submasks for {mask_val} contain duplicates: {masks}")

class TestActionMaskToActionConversions(unittest.TestCase):
    def test_none_available_returns_only_zero(self):
        actions = ThinIceEnv.action_mask_to_actions(0) # in binary: 0000
        self.assertEqual(actions, [])

    def test_all_available_returns_all_submasks(self):
        actions = ThinIceEnv.action_mask_to_actions(15) # in binary: 1111
        self.assertEqual(len(actions), 4)
        self.assertEqual(set(actions), set(range(4)))

    def test_left_right_available(self):
        actions = ThinIceEnv.action_mask_to_actions(10) # in binary: 1010
        expected = [0, 2] # expected submasks: 1010 (10), 1000 (8), 0010 (2), 0000 (0)
        self.assertEqual(actions, expected)

    def test_down_available(self):
        actions = ThinIceEnv.action_mask_to_actions(4) # in binary: 0100
        expected = [1]
        self.assertEqual(actions, expected)

    def test_up_available(self):
        actions = ThinIceEnv.action_mask_to_actions(1) # in binary: 0001
        expected = [3]
        self.assertEqual(actions, expected)

    def test_left_available(self):
        actions = ThinIceEnv.action_mask_to_actions(8) # in binary: 1000
        expected = [0]
        self.assertEqual(actions, expected)

    def test_right_up_available(self):
        actions = ThinIceEnv.action_mask_to_actions(3) # in binary: 0011
        expected = [2, 3]
        self.assertEqual(actions, expected)

    def test_down_up_available(self):
        actions = ThinIceEnv.action_mask_to_actions(5) # in binary: 0101
        expected = [1, 3]
        self.assertEqual(actions, expected)

    def test_down_right_up_available(self):
        actions = ThinIceEnv.action_mask_to_actions(7) # in binary: 0111
        expected = [1, 2, 3]
        self.assertEqual(actions, expected)

    def test_left_up_available(self):
        actions = ThinIceEnv.action_mask_to_actions(9) # in binary: 1001
        expected = [0, 3]
        self.assertEqual(actions, expected)

    def test_no_duplicate_submasks(self):
        for mask_val in [0, 1, 3, 7, 15]:
            actions = ThinIceEnv.action_mask_to_actions(mask_val)
            self.assertEqual(len(actions), len(set(actions)), 
                           f"Submasks for {mask_val} contain duplicates: {actions}")

class TestActionMaskCreation(unittest.TestCase):
    def setUp(self):
        self.test_env = ThinIceEnv(None, "test_level.txt")
        self.level = self.test_env.level
    
    def test_none_available_only_walls(self):
        tile_position = (14, 3)
        tile = self.level.get_tile(tile_position)
        mask = self.level.get_available_actions(tile)
        self.assertEqual(0, mask)
    
    def test_none_available_mix(self):
        tile_position = (2, 3)
        tile = self.level.get_tile(tile_position)
        mask = self.level.get_available_actions(tile)
        self.assertEqual(0, mask)
    
    def test_all_available(self):
        tile_position = (4, 2)
        tile = self.level.get_tile(tile_position)
        mask = self.level.get_available_actions(tile)
        self.assertEqual(15, mask)
    
    def test_all_but_up_available(self):
        tile_position = (13, 1)
        tile = self.level.get_tile(tile_position)
        mask = self.level.get_available_actions(tile)
        self.assertEqual(14, mask)
    
    def test_left_right_available(self):
        tile_position = (7, 2)
        tile = self.level.get_tile(tile_position)
        mask = self.level.get_available_actions(tile)
        self.assertEqual(10, mask)
    
    def test_up_available(self):
        tile_position = (12, 3)
        tile = self.level.get_tile(tile_position)
        mask = self.level.get_available_actions(tile)
        self.assertEqual(1, mask)

if __name__ == '__main__':
    unittest.main()