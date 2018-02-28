# Benchmark CLgen inference

## Experimental Setup

### Hardware
* CPU: Intel E5-2620 @ 2.10 GHZ
* GPU: NVIDIA GTX 1080
* Memory: 32GB RAM, 8GB VRAM

### CLgen model and sampler

model.json
```
{
  "corpus": {
    "path": "~/data/kernels/github",
    "vocabulary": "greedy"
  },
  "architecture": {
    "rnn_size": 512,
    "num_layers": 2
  },
  "train_opts": {
    "epochs": 50
  }
}
```

sampler.json
```
{
  "kernels": {
    "args": null,
    "max_length": 10000,
    "temperature": 1
  },
  "sampler": {
    "static_checker": false
  }
}
```

## Results

Invocation:
```
$ timeout -s9 3h clgen s ~/data/models/github-512x2x50-greedy.json ~/data/samplers/any.json
sampling sampler[784ec28]: '__kernel void A('
/ 6449 Elapsed Time: 2:58:31
```

Runtime: 2:58:31 = 10711 seconds

Dump samples:
```
$ clgen db dump --input-samples kernels.db -d kernels
```

Count characters sampled:
```
$ wc -c kernels/*.cl | tail -n1
4982978 total
```

4982978 characters in 10711 seconds = 465 char / sec

Generation time = len(sample) / 465
