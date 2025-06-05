import os
import json
from pathlib import Path
import argparse

import numpy as np
from PIL import Image
from flask import Flask, render_template, request, send_from_directory, abort

import sys
# add ./src to sys.path to import datasets and transforms
sys.path.append("./src")

from datasets import FashionIQDataset
from utils import targetpad_transform


def create_app(OUTPUT_DIR: Path) -> Flask:
    # Read args.json once at startup
    ARGS_PATH = OUTPUT_DIR / "args.json"
    if not ARGS_PATH.exists():
        raise FileNotFoundError(f"Cannot find args.json at {ARGS_PATH}")

    with open(ARGS_PATH, "r") as f:
        exp_args = json.load(f)

    MODEL_TYPE = exp_args.get("model_type")
    if MODEL_TYPE in ['SEIZE-B', 'SEIZE-L', 'SEIZE-H', 'SEIZE-g', 'SEIZE-G', 'SEIZE-CoCa-B', 'SEIZE-CoCa-L']:
        preprocess = targetpad_transform(1.25, 224)
    else:
        raise ValueError(f"Model type '{MODEL_TYPE}' not supported in this app")

    # Where the images actually live:
    DATASET_PATH = Path(exp_args["dataset_path"])    # must contain `images/`

    if not (DATASET_PATH / "images").exists():
        raise FileNotFoundError(f"Image folder not found at {(DATASET_PATH/'images')}")

    # ––––––––––––––––––––––––––––––––––––––––––––
    # CREATE FLASK APP
    # ––––––––––––––––––––––––––––––––––––––––––––
    app = Flask(__name__)


    @app.route("/", methods=["GET", "POST"])
    def index():
        """Show a form to input item_idx (integer) and dress_type (string)."""
        error = None
        if request.method == "POST":
            # Parse inputs
            try:
                item_idx = int(request.form.get("item_idx", "").strip())
            except ValueError:
                error = "Item index must be an integer."
                return render_template("index.html", error=error)

            dress_type = request.form.get("dress_type", "").strip()
            if not dress_type:
                error = "Dress type cannot be empty."
                return render_template("index.html", error=error)

            # Redirect to results view (POST→GET pattern) or directly render result:
            # We'll directly call the result logic here:
            try:
                context = compute_visualization(item_idx, dress_type)
            except Exception as e:
                error = str(e)
                return render_template("index.html", error=error)

            return render_template("result.html", **context)

        # GET request: just show the form
        return render_template("index.html", error=error)


    def compute_visualization(item_idx: int, dress_type: str):
        """
        Replicates most of the logic from your visualize(...) function,
        but returns a dict of values to pass into the Jinja template.
        Raises an Exception if anything goes wrong.
        """

        # 1. Load the validation dataset (where `relative` mode is used)
        relative_val_dataset = FashionIQDataset(
            str(DATASET_PATH),
            'val',
            [dress_type],
            'relative',
            preprocess
        )

        if item_idx < 0 or item_idx >= len(relative_val_dataset):
            raise IndexError(f"item_idx {item_idx} is out of range (0 to {len(relative_val_dataset)-1}).")

        # 2. Grab the item at that index
        item = relative_val_dataset[item_idx]

        # 3. Collect item fields (omitting anything with "image" in the key)
        item_fields = []
        for k, v in item.items():
            if "image" in k:
                continue
            item_fields.append((k, v))

        # 4. Open the reference (input) image and the ground‐truth target image (PIL objects)
        reference_name = item["reference_name"]
        target_name = item["target_name"]

        ref_img_path = DATASET_PATH / "images" / f"{reference_name}.png"
        tgt_img_path = DATASET_PATH / "images" / f"{target_name}.png"

        if not ref_img_path.exists():
            raise FileNotFoundError(f"Reference image not found: {ref_img_path}")
        if not tgt_img_path.exists():
            raise FileNotFoundError(f"Ground truth image not found: {tgt_img_path}")

        # 5. Load the distance and sorted_index_names arrays
        dist_file = OUTPUT_DIR / "retrieved_index_names" / f"{dress_type}_distances.npy"
        sorted_names_file = OUTPUT_DIR / "retrieved_index_names" / f"{dress_type}_sorted_index_names.npy"

        if not dist_file.exists() or not sorted_names_file.exists():
            raise FileNotFoundError(
                f"Could not find distance/sorted files for '{dress_type}' at:\n"
                f"  {dist_file}\n  {sorted_names_file}"
            )

        distances = np.load(str(dist_file))               # shape = (N, N)
        sorted_index_names = np.load(str(sorted_names_file), allow_pickle=True)  # shape = (N, N)

        if sorted_index_names.shape[0] != len(relative_val_dataset) or distances.shape[0] != len(relative_val_dataset):
            raise ValueError("Shape mismatch: sorted_index_names or distances does not align with dataset length.")

        # 6. Find retrieval rank of ground‐truth
        #    sorted_index_names[item_idx] is a 1D array of names; find index where it matches target_name
        row_names = sorted_index_names[item_idx]  # array of all names sorted by distance for this query
        # Make sure we compare string vs string
        try:
            rank_idx = int(np.where(row_names == target_name)[0][0])
        except Exception:
            raise ValueError(f"Ground truth '{target_name}' not found in sorted_index_names for query {item_idx}.")

        # 7. Pick top_k (we’ll use 3)
        top_k = 3
        dists_row = distances[item_idx]
        sorted_indices = np.argsort(dists_row)[:top_k]
        topk_names = [str(row_names[i]) for i in range(top_k)]
        topk_dists = [float(dists_row[i]) for i in sorted_indices]
        #dists_row[sorted_indices].tolist()

        # 8. Construct URLs for images via a custom route
        #    We’ll create a route @app.route("/images/<filename>")
        input_image_url = f"/images/{reference_name}.png"
        gt_image_url = f"/images/{target_name}.png"

        topk_list = []
        for idx, (name, dist) in enumerate(zip(topk_names, topk_dists), start=1):
            topk_list.append({
                "rank": idx,
                "name": name,
                "distance": f"{dist:.4f}",
                "url": f"/images/{name}.png"
            })

        # 9. Prepare context for template
        context = {
            "item_idx": item_idx,
            "dress_type": dress_type,
            "item_fields": item_fields,      # list of (key, value)
            "input_image_url": input_image_url,
            "ground_truth_url": gt_image_url,
            "ground_truth_rank": rank_idx + 1, # 1-based
            "topk": topk_list                # list of dicts with rank, name, distance, url
        }

        return context


    @app.route("/images/<filename>")
    def serve_image(filename):
        """
        Serve images from DATASET_PATH/images/<filename>.
        """
        img_path = DATASET_PATH / "images" / filename
        if not img_path.exists():
            return abort(404)
        # send_from_directory needs folder path, so pass the parent and filename separately
        return send_from_directory(directory=str(DATASET_PATH / "images"), path=filename)

    return app

if __name__ == "__main__":
    # import argparse
    # parser = argparse.ArgumentParser()
    # parser.add_argument("--output_dir", type=str, required=True, help="Path to the experiment output directory")
    # args = parser.parse_args()

    from args import args_define
    args = args_define.args

    app = create_app(Path(args.output_dir))

    app.run(debug=True)
