# Multi-agent-debate

### Task 2: Reproduce Du et al. Baseline Results
```
python3 -m src.run_reproduction.py
```
for 100 questions, repeat 3 times, output directory is "data/task2"

For example to change configurations: 
```
python3 -m src.run_reproduction -q 100 -rep 2 --seed 2026 -o data/task2
```

### Task 3: Build an Empirical Game
```
python3 -m src.run_egta.py
```
for 25 questions, repeat 5 times, 3 rounds, and output directory is "data/task3"

For example to change configurations: 
```
python3 -m src.run_egta --subset 10 --repeats 1 --n_rounds 2
```

### Task 4: Mechanism Design Extension
```
python3 -m src.run_mechanism
```
for 25 questions, repeat 5 times, 3 rounds, and output directory is "data/task4"

For example to change configurations: 
```
python3 -m src.run_mechanism --subset 10 --repeats 1 --n_rounds 2
```
