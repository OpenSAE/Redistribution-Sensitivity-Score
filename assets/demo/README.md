# RSS examples

The NumPy-only [`rss.py`](../../rss.py) selector is separate from model-specific demonstrations. All runnable model demos and their recorded output are located here.

Install the optional pretrained-model dependencies from the repository root:

```bash
python -m pip install -r requirements_demo.txt
```

Run the public GPT-2 Small experiments:

```bash
python assets/demo/demo.py
python assets/demo/demo.py --layer 4
python assets/demo/demo_heldout.py
```

The first script evaluates 24 prompts across six transformation settings and five variants, comparing normalized MeanRSS and responsive-support overlap. It creates `demo_layer8.png` in the working directory by default. The [bundled layer-8 preview](demo_layer8.png) comes from a recorded demonstration, **not** from the full 27-SAE study. The held-out script tests directions on ten unseen variants per prompt and setting.

To run the fixed linear readout evaluation:

```bash
python assets/demo/demo_readout.py --example
python assets/demo/demo_readout.py --data-dir /path/to/arrays
```

`--example` is explicitly **synthetic**, not the paper's Gemma/Llama probes. Use `--example --export-example ./example_data` to inspect the required `.npy` files; real results require your own activation arrays and readout weights.

For the public selector, refer to the single [Quickstart](../../README.md#-quickstart) or [core API guide](../../docs/CORE_API.md).
