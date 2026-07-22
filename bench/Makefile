
build:
	@echo ""
	@export RUSTFLAGS="-C target-cpu=native"
	cargo build --release --bin prove

run:
	@cd .. && ./benchmark.sh

build-and-run: build run
