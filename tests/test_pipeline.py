import unittest
import os
import sys

# Add root directory to python path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.preprocess import preprocess_text
from src.predict import EmotionPredictor

class TestNLPComponents(unittest.TestCase):

    def test_preprocess_text(self):
        raw = "I am feeling extremely excited and happy today!! http://example.com #joyful"
        cleaned = preprocess_text(raw)
        self.assertNotIn("http", cleaned)
        self.assertNotIn("!!", cleaned)
        # Verify lowercase & lemmatized words
        self.assertIn("feeling", cleaned)
        self.assertIn("excited", cleaned)

    def test_predictor_inference(self):
        predictor = EmotionPredictor()
        
        # Test Joy prediction
        res_joy = predictor.predict("I am feeling so joyful and happy with my friends!")
        self.assertIn(res_joy["emotion"], ["Joy", "Love"])
        self.assertGreater(res_joy["confidence"], 0.4)
        self.assertEqual(len(res_joy["breakdown"]), 6)

        # Test Sadness prediction
        res_sad = predictor.predict("I feel so depressed, miserable and sad.")
        self.assertEqual(res_sad["emotion"], "Sadness")

        # Test Anger prediction
        res_anger = predictor.predict("I am furious, mad, and furious with this awful service!")
        self.assertIn(res_anger["emotion"], ["Anger", "Sadness"])

if __name__ == "__main__":
    unittest.main()
