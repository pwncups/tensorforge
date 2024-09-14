import safetensors.torch
from tensorforge.models.base_model import BaseModel
import logging
import torch
import datetime
from collections import OrderedDict
import numpy as np
from difflib import SequenceMatcher
import re

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class SafetensorModel(BaseModel):
    def __init__(self, file_path):
        self.file_path = file_path
        self.metadata = None
        self.layer_info = None
        self.model_type = self.classify_model()
        logger.info(f"Initialized SafetensorModel with file: {file_path}")

    def classify_model(self):
        # Implement your model classification logic here
        return "Unknown"

    def load_metadata(self):
        try:
            with safetensors.safe_open(self.file_path, framework="pt", device="cpu") as f:
                self.metadata = f.metadata()
            logger.info(f"Metadata loaded successfully: {self.metadata}")
            return self.metadata
        except Exception as e:
            logger.error(f"Error loading metadata: {str(e)}")
            raise

    def get_layer_info(self):
        if not self.layer_info:
            try:
                self.layer_info = OrderedDict()
                with safetensors.safe_open(self.file_path, framework="pt", device="cpu") as f:
                    for key in f.keys():
                        tensor = f.get_tensor(key)
                        self.layer_info[key] = {
                            'shape': tensor.shape,
                            'dtype': str(tensor.dtype),
                            'size_bytes': tensor.numel() * tensor.element_size()
                        }
                logger.info(f"Layer info loaded successfully: {len(self.layer_info)} layers")
            except Exception as e:
                logger.error(f"Error loading layer info: {str(e)}")
                raise
        return self.layer_info

    def load_layer_info_chunked(self, chunk_size=100, callback=None):
        self.layer_info = OrderedDict()
        try:
            with safetensors.safe_open(self.file_path, framework="pt", device="cpu") as f:
                keys = list(f.keys())
                total_keys = len(keys)
                for i in range(0, total_keys, chunk_size):
                    chunk = keys[i:i+chunk_size]
                    for key in chunk:
                        tensor = f.get_tensor(key)
                        self.layer_info[key] = {
                            'shape': tensor.shape,
                            'dtype': str(tensor.dtype),
                            'size_bytes': tensor.numel() * tensor.element_size()
                        }
                    if callback:
                        progress = min(100, int((i + chunk_size) / total_keys * 100))
                        callback(progress)
            logger.info(f"Layer info loaded successfully: {len(self.layer_info)} layers")
        except Exception as e:
            logger.error(f"Error loading layer info: {str(e)}")
            raise

    def get_model_structure(self):
        structure = OrderedDict()
        for key, info in self.get_layer_info().items():
            parts = key.split('.')
            current = structure
            for part in parts[:-1]:
                if part not in current:
                    current[part] = OrderedDict()
                current = current[part]
            current[parts[-1]] = info
        return structure

    def diff(self, other_model):
        if not self.layer_info or not other_model.layer_info:
            logger.error("Layer info is not loaded for one or both models")
            return None

        diff_result = {
            "similarity": 0,
            "total_params_self": 0,
            "total_params_other": 0,
            "matched_params": 0,
            "self_size": 0,
            "other_size": 0,
            "matched_layers": [],
            "unmatched_layers_self": [],
            "unmatched_layers_other": [],
            "partial_matches": []
        }

        self_layers = list(self.layer_info.items())
        other_layers = list(other_model.layer_info.items())

        for self_key, self_info in self_layers:
            best_match = None
            best_score = 0
            processed_self_key = self._preprocess_key(self_key)

            for other_key, other_info in other_layers:
                if self_info['shape'] == other_info['shape']:
                    processed_other_key = self._preprocess_key(other_key)
                    score = self._calculate_similarity(processed_self_key, processed_other_key)
                    if score > best_score:
                        best_match = (other_key, other_info)
                        best_score = score

            if best_match and best_score > 0.6:
                other_key, other_info = best_match
                diff_result["matched_layers"].append({
                    "self_key": self_key,
                    "other_key": other_key,
                    "similarity": best_score,
                    "shape": self_info['shape'],
                    "dtype": self_info['dtype']
                })
                diff_result["matched_params"] += np.prod(self_info['shape'])
                other_layers = [layer for layer in other_layers if layer[0] != other_key]
            elif best_match:
                diff_result["partial_matches"].append({
                    "self_key": self_key,
                    "other_key": best_match[0],
                    "similarity": best_score,
                    "self_shape": self_info['shape'],
                    "other_shape": best_match[1]['shape'],
                    "self_dtype": self_info['dtype'],
                    "other_dtype": best_match[1]['dtype']
                })
            else:
                diff_result["unmatched_layers_self"].append({
                    "key": self_key,
                    "shape": self_info['shape'],
                    "dtype": self_info['dtype']
                })

        for other_key, other_info in other_layers:
            diff_result["unmatched_layers_other"].append({
                "key": other_key,
                "shape": other_info['shape'],
                "dtype": other_info['dtype']
            })

        diff_result["total_params_self"] = np.sum([np.prod(v['shape'], dtype=np.int64) for v in self.layer_info.values()], dtype=np.int64)
        diff_result["total_params_other"] = np.sum([np.prod(v['shape'], dtype=np.int64) for v in other_model.layer_info.values()], dtype=np.int64)
        diff_result["self_size"] = sum(v['size_bytes'] for v in self.layer_info.values())
        diff_result["other_size"] = sum(v['size_bytes'] for v in other_model.layer_info.values())

        max_params = max(diff_result["total_params_self"], diff_result["total_params_other"])
        diff_result["similarity"] = (diff_result["matched_params"] / max_params) * 100 if max_params > 0 else 0

        layer_mapping = {match['self_key']: match['other_key'] for match in diff_result['matched_layers']}
        weight_diff_result = self._weight_diff_with_shifts(other_model, layer_mapping)

        diff_result["weight_similarity"] = weight_diff_result["similarity"]
        diff_result["weight_differences"] = weight_diff_result["differences"]

        diff_result["total_params_self"] = int(diff_result["total_params_self"])
        diff_result["total_params_other"] = int(diff_result["total_params_other"])
        diff_result["matched_params"] = int(np.sum([np.prod(match['shape'], dtype=np.int64) for match in diff_result["matched_layers"]], dtype=np.int64))

        return diff_result

    def _preprocess_key(self, key):
        prefixes_to_remove = ['model.', 'diffusion_model.']
        for prefix in prefixes_to_remove:
            if key.startswith(prefix):
                key = key[len(prefix):]
        return key

    def _calculate_similarity(self, key1, key2):
        parts1 = key1.split('.')
        parts2 = key2.split('.')

        n = min(3, len(parts1), len(parts2))
        last_parts1 = parts1[-n:]
        last_parts2 = parts2[-n:]

        similarity = sum(1 for a, b in zip(last_parts1, last_parts2) if a == b) / n

        if parts1[-1] == parts2[-1]:
            similarity += 0.2

        return min(similarity, 1.0)

    def _weight_diff_with_shifts(self, other_model, layer_mapping):
        differences = {}
        total_elements = 0
        different_elements = 0

        with safetensors.safe_open(self.file_path, framework="pt", device=device) as f1, \
                safetensors.safe_open(other_model.file_path, framework="pt", device=device) as f2:

            for self_key, other_key in layer_mapping.items():
                tensor1 = f1.get_tensor(self_key).to(device)
                tensor2 = f2.get_tensor(other_key).to(device)

                if tensor1.dtype == torch.bfloat16:
                    tensor1 = tensor1.to(torch.float32)
                if tensor2.dtype == torch.bfloat16:
                    tensor2 = tensor2.to(torch.float32)

                total_elements += tensor1.numel()

                diff = torch.abs(tensor1 - tensor2) > 1e-5
                different_elements += diff.sum().item()

                if diff.any():
                    differences[self_key] = []
                    diff_indices = diff.nonzero(as_tuple=False)
                    for idx in diff_indices[:10]:
                        idx = idx.item()
                        val1 = tensor1.flatten()[idx].item()
                        val2 = tensor2.flatten()[idx].item()
                        differences[self_key].append(f"Index {idx}: {val1} vs {val2} (in {other_key})")
                    if len(diff_indices) > 10:
                        differences[self_key].append("... (more differences)")

        similarity = 100 * (1 - different_elements / total_elements)

        return {
            "similarity": similarity,
            "differences": differences,
            "layer_mapping": layer_mapping
        }

    def extract_subset(self, subset_keys, output_path):
        with safetensors.safe_open(self.file_path, framework="pt", device="cpu") as f:
            tensors = {key: f.get_tensor(key) for key in subset_keys if key in f.keys()}

        safetensors.torch.save_file(tensors, output_path)
        return output_path

    def extract_diffusion_model(self, output_path):
        diffusion_keys = [key for key in self.layer_info.keys() if 'diffusion_model' in key]
        return self.extract_subset(diffusion_keys, output_path)

    @staticmethod
    def generate_metadata(layers):
        return {
            "created_by": "TensorForge",
            "creation_date": datetime.datetime.now().isoformat(),
            "format": "pt",
            "total_layers": str(len(layers)),
            "tensorforge_version": "1.0.0",
        }

    @staticmethod
    def save_new_model(layers, file_path, layer_sources):
        logger.info(f"Starting to save new model with {len(layers)} layers")
        tensors = {}
        source_files = set(source for source in layer_sources.values() if source)

        try:
            for source_path in source_files:
                with safetensors.safe_open(source_path, framework="pt", device="cpu") as source_file:
                    source_keys = list(source_file.keys())
                    for full_name, layer_data in layers.items():
                        if layer_data['source'] == source_path:
                            logger.info(f"Processing layer: {full_name}")
                            matching_key = SafetensorModel.find_matching_key(full_name, source_keys)
                            if matching_key:
                                tensors[full_name] = source_file.get_tensor(matching_key)
                                logger.info(f"Found matching tensor for {full_name}: {matching_key}")
                            else:
                                logger.warning(f"No matching tensor found for {full_name}, skipping")

            metadata = SafetensorModel.generate_metadata(layers)
            metadata = {k: str(v) for k, v in metadata.items()}
            logger.info("All layers processed, saving file with metadata")
            safetensors.torch.save_file(tensors, file_path, metadata)
            logger.info("File saved successfully with metadata")
        except Exception as e:
            logger.error(f"Error saving model: {str(e)}", exc_info=True)
            raise

        return file_path

    @staticmethod
    def find_matching_key(target_key, available_keys):
        if target_key in available_keys:
            return target_key

        clean_target_key = re.sub(r':\s*.*$', '', target_key)

        target_parts = clean_target_key.split('.')
        for key in available_keys:
            key_parts = key.split('.')
            if key_parts[-len(target_parts):] == target_parts:
                return key

        target_pattern = r'.*?' + r'.*?'.join(re.escape(part) for part in target_parts)
        for key in available_keys:
            if re.match(target_pattern, key):
                return key

        last_part = target_parts[-1]
        for key in available_keys:
            if key.endswith(last_part):
                return key

        return None

    @staticmethod
    def validate_saved_file(file_path):
        try:
            with safetensors.safe_open(file_path, framework="pt", device="cpu") as f:
                keys = f.keys()
                logger.info(f"Validated file: {len(keys)} layers found")
            return True
        except Exception as e:
            logger.error(f"Error validating saved file: {str(e)}")
            return False