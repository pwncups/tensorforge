import safetensors
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def test_load_safetensor(file_path):
    try:
        logger.info(f"Attempting to load file: {file_path}")
        with safetensors.safe_open(file_path, framework="pt", device="cpu") as f:
            metadata = f.metadata()
            logger.info(f"Metadata: {metadata}")

            keys = list(f.keys())
            logger.info(f"Total layers: {len(keys)}")

            for i, key in enumerate(keys[:100]):  # Only load first 10 layers as a test
                tensor = f.get_tensor(key)
                logger.info(f"Layer {i}: {key}, Shape: {tensor.shape}, Dtype: {tensor.dtype}")

        logger.info("File loaded successfully")
    except Exception as e:
        logger.error(f"Error loading file: {str(e)}")


if __name__ == "__main__":
    file_path = "C:/Users/Evan/Desktop/stage_b_lite.safetensors"
    test_load_safetensor(file_path)