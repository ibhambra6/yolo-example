#!/usr/bin/env python3
"""Unit tests for the dog/cat YOLO classifier."""

import unittest
from unittest.mock import patch, MagicMock
import sys
from pathlib import Path

# Add the parent directory to the path to import the main module
sys.path.insert(0, str(Path(__file__).parent.parent))

import dog_cat_yolo_gui as classifier


class TestDogCatClassifier(unittest.TestCase):
    """Test cases for the dog/cat classifier."""

    def setUp(self):
        """Set up test fixtures."""
        # Reset the global MODEL variable before each test
        classifier.MODEL = None

    @patch('dog_cat_yolo_gui.torch.hub.load')
    def test_load_model_default_weights(self, mock_torch_load):
        """Test loading model with default weights."""
        mock_model = MagicMock()
        mock_torch_load.return_value = mock_model
        
        classifier.load_model()
        
        mock_torch_load.assert_called_once_with("ultralytics/yolov5", "yolov5s", pretrained=True)
        mock_model.eval.assert_called_once()
        self.assertEqual(classifier.MODEL, mock_model)

    @patch('dog_cat_yolo_gui.torch.hub.load')
    @patch('dog_cat_yolo_gui.Path.exists')
    def test_load_model_custom_weights(self, mock_exists, mock_torch_load):
        """Test loading model with custom weights."""
        mock_exists.return_value = True
        mock_model = MagicMock()
        mock_torch_load.return_value = mock_model
        
        classifier.load_model()
        
        # Should load custom weights if they exist
        mock_torch_load.assert_called_once()
        args, kwargs = mock_torch_load.call_args
        self.assertEqual(args[0], "ultralytics/yolov5")
        self.assertTrue(str(args[1]).endswith("dog_cat.pt"))

    def test_load_model_idempotent(self):
        """Test that load_model doesn't reload if model is already loaded."""
        mock_model = MagicMock()
        classifier.MODEL = mock_model
        
        with patch('dog_cat_yolo_gui.torch.hub.load') as mock_torch_load:
            classifier.load_model()
            mock_torch_load.assert_not_called()

    @patch('dog_cat_yolo_gui.load_model')
    def test_predict_dog(self, mock_load_model):
        """Test prediction for dog image."""
        # Mock model and results
        mock_model = MagicMock()
        mock_results = MagicMock()
        mock_results.xyxy = [[[100, 100, 200, 200, 0.8, 16]]]  # Dog class (16) with high confidence
        mock_model.return_value = mock_results
        classifier.MODEL = mock_model
        
        result = classifier.predict("test_image.jpg")
        
        self.assertEqual(result, "Dog")
        mock_model.assert_called_once_with("test_image.jpg", size=640)

    @patch('dog_cat_yolo_gui.load_model')
    def test_predict_cat(self, mock_load_model):
        """Test prediction for cat image."""
        # Mock model and results
        mock_model = MagicMock()
        mock_results = MagicMock()
        mock_results.xyxy = [[[100, 100, 200, 200, 0.9, 15]]]  # Cat class (15) with high confidence
        mock_model.return_value = mock_results
        classifier.MODEL = mock_model
        
        result = classifier.predict("test_image.jpg")
        
        self.assertEqual(result, "Cat")

    @patch('dog_cat_yolo_gui.load_model')
    def test_predict_unknown(self, mock_load_model):
        """Test prediction for unknown/no detection."""
        # Mock model and results with no detections above threshold
        mock_model = MagicMock()
        mock_results = MagicMock()
        mock_results.xyxy = [[[100, 100, 200, 200, 0.1, 16]]]  # Low confidence
        mock_model.return_value = mock_results
        classifier.MODEL = mock_model
        
        result = classifier.predict("test_image.jpg")
        
        self.assertEqual(result, "Unknown")

    @patch('dog_cat_yolo_gui.load_model')
    def test_predict_dog_vs_cat_preference(self, mock_load_model):
        """Test that higher confidence prediction wins."""
        # Mock model with both dog and cat detections
        mock_model = MagicMock()
        mock_results = MagicMock()
        mock_results.xyxy = [[[100, 100, 200, 200, 0.7, 15],  # Cat with 0.7 confidence
                              [300, 300, 400, 400, 0.9, 16]]]  # Dog with 0.9 confidence
        mock_model.return_value = mock_results
        classifier.MODEL = mock_model
        
        result = classifier.predict("test_image.jpg")
        
        self.assertEqual(result, "Dog")  # Dog should win with higher confidence

    def test_confidence_threshold(self):
        """Test that confidence threshold is properly set."""
        self.assertEqual(classifier.CONF_THRESHOLD, 0.25)

    @patch('dog_cat_yolo_gui.subprocess.run')
    def test_fine_tune(self, mock_subprocess):
        """Test fine-tuning functionality."""
        classifier.fine_tune("test_data.yaml", epochs=10, weights="yolov5s.pt")
        
        mock_subprocess.assert_called_once()
        args = mock_subprocess.call_args[0][0]
        self.assertIn("yolov5.train", " ".join(args))
        self.assertIn("test_data.yaml", args)
        self.assertIn("10", args)


class TestApp(unittest.TestCase):
    """Test cases for the GUI application."""

    @patch('dog_cat_yolo_gui.tk.Tk.__init__')
    def test_app_initialization(self, mock_tk_init):
        """Test that the App initializes correctly."""
        mock_tk_init.return_value = None
        
        with patch.multiple('dog_cat_yolo_gui.tk',
                           Button=MagicMock(),
                           Label=MagicMock()):
            app = classifier.App()
            
            # Check that the app has the required attributes
            self.assertTrue(hasattr(app, 'upload_btn'))
            self.assertTrue(hasattr(app, 'img_lbl'))
            self.assertTrue(hasattr(app, 'pred_lbl'))


if __name__ == '__main__':
    unittest.main() 