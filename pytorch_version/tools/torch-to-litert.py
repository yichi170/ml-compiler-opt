import argparse
import os
import torch
import ai_edge_torch
from pytorch_version import specs
from pytorch_version.sched.sched_network import SchedNetwork

def load_model(model_path, problem_name):
    if problem_name == 'sched':
        observation_spec, _ = specs.get_sched_signature_spec()
        model = SchedNetwork(observation_spec=observation_spec['observation'], num_actions=256, fc_layer_params=(256, 128))
    else:
        raise ValueError(f"Unknown problem: {problem_name}")

    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    return model

def export_tflite(model, dummy_input, tflite_path):
    tflite_model = ai_edge_torch.convert(model.eval(), dummy_input)
    tflite_model.export(tflite_path)
    print(f"Exported TFLite model to: {tflite_path}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, required=True, help='Path to the .pt PyTorch model file.')
    parser.add_argument('--problem', type=str, required=True, choices=['sched'])
    parser.add_argument('--output_dir', type=str, default='converted_models')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print("Loading PyTorch model...")
    model = load_model(args.model_path, args.problem)

    # Create a dummy input based on one sample from observation_spec
    if args.problem == 'sched':
        obs_spec, _ = specs.get_sched_signature_spec()
    else:
        raise ValueError("Unknown problem")
    # Create dummy inputs
    dummy_input = {
        k: torch.randn(1, *v.shape).float()
        for k, v in obs_spec['observation'].items()
    }

    # Wrap as tuple since your model likely expects a dict input
    dummy_input_tuple = (dummy_input,)

    tflite_path = os.path.join(args.output_dir, f"{args.problem}_model.tflite")

    print("Converting to TFLite...")
    export_tflite(model, dummy_input_tuple, tflite_path)

if __name__ == "__main__":
    main()
