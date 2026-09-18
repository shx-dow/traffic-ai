.PHONY: test lint benchmark ablation clean

PYTEST := python -m pytest

test:
	$(PYTEST) -q

lint:
	ruff check .

benchmark:
	python -m sumo_demo.benchmark --steps 600 \
	    --out profiling/artifacts/benchmark_results.json

ablation:
	python -m sumo_demo.benchmark --ablate switch_gap_relative \
	    --ablate-out profiling/artifacts/ablation_gap_relative.json
	python -m sumo_demo.benchmark --ablate baseline_green \
	    --ablate-out profiling/artifacts/ablation_baseline_green.json
	python -m sumo_demo.benchmark --ablate fusion_weight \
	    --ablate-out profiling/artifacts/ablation_fusion.json

clean:
	rm -f profiling/artifacts/benchmark_results.json
	rm -f profiling/artifacts/ablation_*.json