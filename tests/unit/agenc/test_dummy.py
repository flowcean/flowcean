import unittest


class TestDummy(unittest.TestCase):
    def test_nothing(self):
        print("This is a dummy test that tests nothing")
        success = True
        self.assertTrue(success)


if __name__ == "__main__":
    unittest.main()
