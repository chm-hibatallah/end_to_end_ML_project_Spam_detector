import unittest
import joblib

class TestSpamClassifier(unittest.TestCase):
    def setUp(self):
        self.model = joblib.load('spam_classifier.pkl')
    
    def test_spam_prediction(self):
        spam_message = "WINNER!! You have won a free ticket!"
        result = predict_spam(spam_message, self.model)
        self.assertTrue(result['is_spam'])
    
    def test_ham_prediction(self):
        ham_message = "Hey, are we meeting tomorrow?"
        result = predict_spam(ham_message, self.model)
        self.assertFalse(result['is_spam'])

if __name__ == '__main__':
    unittest.main()